from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting the string.
    """
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(adjacent box1 location6)".
    - `args`: The expected pattern elements (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # In case the pattern has fewer arguments than the fact, only match those parts.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the 'secret_domain' (Sokoban-like) domain.

    Summary:
    This heuristic estimates the number of actions required to push all boxes
    into their goal configuration, i.e., for each box to satisfy (adjacent-2 box),
    using only the information visible in the current state and static facts.
    The heuristic is tailored for the domain where:
      - The robot is located at some location (at-robot-2).
      - Boxes are at some locations and can be pushed towards goals.
      - The goal is that all boxes have the predicate (adjacent-2 box).
    The heuristic is NOT necessarily admissible but aims to provide a good guidance for greedy best-first search.

    Assumptions:
    - Each box must be pushed to a location from which it satisfies (adjacent-2 box).
    - The robot can move between locations using clear paths (clear ?loc1 ?loc2).
    - The robot pushes boxes by first moving next to them (adjacent box location), then pushing them.
    - The heuristic assumes independence between boxes.
    - The robot can move freely on clear location pairs.
    - The cost of pushing a box to adjacency-2 (goal) is estimated by counting minimal required pushes plus robot moves.
    - Robot movement between locations is estimated via shortest paths on the graph formed by 'clear' facts.
    - The heuristic returns 0 if all goal facts are met (goal state), otherwise a finite estimate.

    Heuristic Initialization:
    - Extract boxes and locations from the facts in the task.
    - Extract static adjacency graph from (clear loc1 loc2) facts to compute shortest paths.
    - Extract static robot-box adjacency relations from (at-robot box location) facts in static (if any) - mainly ignored here, but static facts are preserved.
    - Store goal boxes as those that must satisfy (adjacent-2 box).
    - Precompute the adjacency relationships (which boxes can be adjacent to which locations).
    - Prepare data structures (e.g. adjacency matrix or dict) for efficient shortest path computations.

    Step-By-Step Thinking for Computing Heuristic:
    1. From the current state, identify:
       - Robot's current location: fact (at-robot-2 locationX).
       - For each box:
         * its current location (implied by (adjacent box loc) or (at box) for boxes not adjacent to robot).
         * whether it currently satisfies goal (adjacent-2 box).
    2. For each box that is not yet adjacent-2 (not at goal):
       - Estimate the minimal number of pushes required: at least 1 push needed to get box adjacent-2.
         (Pushing the box once presumably moves it from 'adjacent box location' to the adjacent-2 state.)
       - Estimate cost for robot to move to location adjacent to that box to push it:
         * Robot must reach a location adjacent to the box.
         * Use shortest path on the location graph to compute distance from robot to *some* location adjacent to the box.
       - Sum the cost:
         * Robot movement cost to reach pushing position.
         * One push action (push or push-2).
         * Possibly additional robot moves after push.
    3. Sum these estimates for all boxes independently.
    4. Return this sum as heuristic estimate.
    5. The heuristic is zero if all goal facts (adjacent-2 box) hold in state.

    Note:
    - This heuristic ignores possible blocking or complex interactions between boxes.
    - Robot moves are counted approximately via shortest path length.
    - Each push action assumed to cost 1.
    """

    def __init__(self, task):
        # Store the list of goal boxes (boxes with adjacent-2 facts in goals)
        self.goal_boxes = set()
        for fact in task.goals:
            parts = get_parts(fact)
            if parts[0] == "adjacent-2" and len(parts) == 2:
                self.goal_boxes.add(parts[1])

        # All boxes known from typed facts in the task
        # Since we don't have explicit type info here, infer boxes from static facts or initial state facts mentioning boxes.
        # We parse all facts mentioning boxes to find unique box names.
        self.boxes = set()
        for fact in task.facts:
            parts = get_parts(fact)
            # box is always the first or second argument for predicates on boxes
            # Specifically: (adjacent box location), (at-robot box location), (at box), (adjacent-2 box)
            pred = parts[0]
            if pred in {"adjacent", "at-robot", "adjacent-2", "at"} and len(parts) >= 2:
                self.boxes.add(parts[1])

        # Extract all locations from facts that mention locations.
        self.locations = set()
        for fact in task.facts:
            parts = get_parts(fact)
            # Look for predicates that have location arguments
            # The domain has predicates like:
            # (adjacent ?v1 - box ?v2 - location)
            # (at-robot ?v1 - box ?v2 - location)
            # (clear ?v3 - location ?v4 - location)
            # (at-robot-2 ?v2 - location)
            # We collect any object appearing in location position.

            if parts[0] == "adjacent" and len(parts) == 3:
                self.locations.add(parts[2])
            elif parts[0] == "at-robot" and len(parts) == 3:
                self.locations.add(parts[2])
            elif parts[0] == "clear" and len(parts) == 3:
                self.locations.add(parts[1])
                self.locations.add(parts[2])
            elif parts[0] == "at-robot-2" and len(parts) == 2:
                self.locations.add(parts[1])

        # Construct graph of locations from static facts (clear loc1 loc2)
        # This undirected graph will be used for shortest path computations for robot moves.
        self.location_graph = {loc: set() for loc in self.locations}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "clear" and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                if loc1 in self.locations and loc2 in self.locations:
                    self.location_graph[loc1].add(loc2)
                    self.location_graph[loc2].add(loc1)

        # Precompute box-to-adjacent-locations map from static facts (adjacent box location).
        # This gives for each box the set of locations adjacent to it.
        self.box_adjacent_locations_static = dict()
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "adjacent" and len(parts) == 3:
                box, loc = parts[1], parts[2]
                if box not in self.box_adjacent_locations_static:
                    self.box_adjacent_locations_static[box] = set()
                self.box_adjacent_locations_static[box].add(loc)

        # If some boxes never have static adjacent locations, initialize empty sets
        for box in self.boxes:
            if box not in self.box_adjacent_locations_static:
                self.box_adjacent_locations_static[box] = set()

    def shortest_path_length(self, start, goals):
        """
        Compute shortest path length from start location to any location in goals using BFS
        on the undirected location graph.

        Returns distance as integer number of steps,
        or None if no path exists.
        """
        if start in goals:
            return 0
        from collections import deque

        queue = deque([(start, 0)])
        visited = {start}
        while queue:
            loc, dist = queue.popleft()
            for neighbor in self.location_graph.get(loc, ()):
                if neighbor in goals:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return None

    def __call__(self, node):
        state = node.state

        # If goal reached, heuristic = 0.
        # Goal requires all boxes in goal_boxes have (adjacent-2 box).
        goal_reached = True
        for goal_box in self.goal_boxes:
            goal_fact = f"(adjacent-2 {goal_box})"
            if goal_fact not in state:
                goal_reached = False
                break
        if goal_reached:
            return 0

        # Identify robot location from (at-robot-2 ?loc)
        robot_location = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot-2" and len(parts) == 2:
                robot_location = parts[1]
                break

        if robot_location is None:
            # If robot location unknown, return a large heuristic to deprioritize
            return 99999

        total_estimated_cost = 0

        # For each box, find current location and if it meets goal.
        # Current box location is the location for which (adjacent box location) holds in state.
        # If no adjacent fact, but (at box) holds, we consider the box not adjacent to robot.
        # We estimate pushing cost only if box not in goal.

        for box in self.boxes:
            goal_fact = f"(adjacent-2 {box})"
            if goal_fact in state:
                # Box at goal, no cost needed.
                continue

            # Find current adjacent location(s) of box
            box_adj_locs = set()
            for fact in state:
                if match(fact, "adjacent", box, "*"):
                    parts = get_parts(fact)
                    box_adj_locs.add(parts[2])

            # If no adjacency found, fallback to abstract cost
            if not box_adj_locs:
                # We estimate the cost as 3 actions as a fallback (one move to box + one push + one move after),
                # plus robot move assuming robot location known.
                # Without knowing exact box location, assume minimal robot move = 0.
                total_estimated_cost += 3
                continue

            # Robot needs to move to a location adjacent to the box for pushing
            # Candidate robot goal locations for pushing are those adjacent locations of the box.
            # We compute the shortest distance from robot location to any of these locations.
            dist = self.shortest_path_length(robot_location, box_adj_locs)

            if dist is None:
                # If no path to any adjacent location, return a large cost
                return 99999

            # Heuristic for this box:
            # robot move cost := dist (each move counts as 1)
            # one push action (push or push-2) cost := 1
            # assume one extra robot move after pushing (approximated as 1)
            heur_box_cost = dist + 1 + 1

            total_estimated_cost += heur_box_cost

        # Return sum of estimated costs over all boxes not at goal.
        return total_estimated_cost
