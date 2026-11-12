from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(adjacent box1 location6)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the 'secret_domain' (Sokoban-like domain).

    Summary:
    This heuristic estimates the number of actions (move, push, move-2, push-2) required
    to achieve the goal condition that all boxes must be in the 'adjacent-2' relation.
    The major cost contributors are getting each box from its current position (marked by
    the predicate 'at ?box') to a state where it is adjacent-2. This requires the box to be
    pushed by the robot and the robot to be positioned correctly to do so.

    Assumptions:
    - The goal is that all boxes satisfy predicate (adjacent-2 ?box).
    - The robot can only push boxes when adjacent to them (preconditions use (at-robot ?box ?loc) and (at ?box)).
    - The robot moves between locations using 'move-2' and 'push-2', and it can 'push' boxes to advance them.
    - Static facts describe robot-box adjacency and location clearance relations.
    - The domain resembles a simplified Sokoban, where the challenge is pushing boxes to their goal adjacency-2 locations.
    - The heuristic does not try to compute full paths but estimates a minimal effort to bring boxes to adjacency-2 using static info.

    Heuristic Initialization:
    - Extract from static facts:
       * Which box is currently adjacent to which location (robot adjacency).
       * The set of all locations and their clearance relations (to infer possible robot moves).
       * The start location of the robot from facts (at-robot-2 ?loc).
    - Extract from goal the set of boxes that must have adjacent-2 true (goal boxes).
    - Extract current box positions from state (fact (at ?box)) and robot position (at-robot-2 ?loc).
    - Build adjacency mappings between boxes and locations (= possible push targets).
    - Build a graph between locations to approximate shortest distances for robot movement.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify which boxes are not yet adjacent-2 (still need pushing).
    2. For each such box:
       a) Get current location of the box (from predicate (at ?box) is true or inferred).
       b) Determine target adjacency-2 location or just assume pushing once suffices to reach adjacency-2.
          Since domain abstracts goal as (adjacent-2 ?box), heuristic assumes one push needed.
       c) Find minimal distance for the robot to get adjacent to that box to perform a push.
          This uses robot location, adjacency facts (at-robot ?box ?loc), and static adjacency relations.
       d) Heuristic cost for one box includes:
          - Robot moves needed to reach pushing position.
          - One push action.
          - Possibly one move after push to reposition.
    3. Sum the estimates for all boxes not yet in adjacency-2.
    4. Because the heuristic guides greedy best-first search, precise optimality is not mandatory.
       A consistent relative estimate helps prioritize states that reduce number of boxes still requiring pushing
       and minimize robot travel distance.

    Note:
    The heuristic may overestimate or underestimate but aims to balance informativeness and efficiency:
    - Robot movement approximated by number of moves between locations using static clearance info.
    - Each box assumed to need at least one push plus robot moves to its pushing point.
    """

    def __init__(self, task):
        """Initialize heuristic data structures from static and goal facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract boxes required to be adjacent-2 from goal
        self.goal_boxes = set()
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] == "adjacent-2" and len(parts) == 2:
                self.goal_boxes.add(parts[1])

        # Extract all boxes from static facts (those appearing in adjacency, at-robot)
        self.boxes = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "adjacent" and len(parts) == 3:
                self.boxes.add(parts[1])
            elif parts[0] == "at-robot" and len(parts) == 3:
                self.boxes.add(parts[1])
            elif parts[0] == "adjacent-2" and len(parts) == 2:
                self.boxes.add(parts[1])

        # Extract all locations from static facts
        # Locations appear in adjacency, clear predicates, at-robot-2 facts
        self.locations = set()
        for fact in self.static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "adjacent" and len(parts) == 3:
                self.locations.add(parts[2])
            elif pred == "clear" and len(parts) == 3:
                self.locations.add(parts[1])
                self.locations.add(parts[2])
            elif pred == "at-robot-2" and len(parts) == 2:
                self.locations.add(parts[1])

        # Build a map from box to the set of adjacent locations for robot adjacency (static)
        self.box_adjacent_locations = dict()
        for box in self.boxes:
            self.box_adjacent_locations[box] = set()
        for fact in self.static:
            if match(fact, "adjacent", "*", "*"):
                parts = get_parts(fact)
                box = parts[1]
                loc = parts[2]
                if box in self.box_adjacent_locations:
                    self.box_adjacent_locations[box].add(loc)

        # Build a graph of robot movements between locations based on 'clear' facts.
        # Each 'clear loc1 loc2' means robot can move-2 from loc1 to loc2.
        # We assume undirected edges (usually adjacency is symmetric).
        self.location_graph = {loc: set() for loc in self.locations}
        for fact in self.static:
            if match(fact, "clear", "*", "*"):
                parts = get_parts(fact)
                loc1 = parts[1]
                loc2 = parts[2]
                if loc1 in self.locations and loc2 in self.locations:
                    self.location_graph[loc1].add(loc2)
                    self.location_graph[loc2].add(loc1)

        # Store robot's initial location placeholder; will get from state each call
        self.initial_robot = None

    def shortest_path_length(self, graph, start, goals):
        """
        Compute shortest distance from start location to any location in goals using BFS.
        Return integer steps or None if no path found.
        """
        from collections import deque

        if start in goals:
            return 0

        visited = set([start])
        queue = deque([(start, 0)])

        while queue:
            current, dist = queue.popleft()
            for neighbor in graph[current]:
                if neighbor not in visited:
                    if neighbor in goals:
                        return dist + 1
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return None

    def __call__(self, node):
        """Estimate actions needed from current state to reach goal for all boxes."""
        state = node.state

        # Quick check: if goal reached, heuristic = 0
        if self.goals <= state:
            return 0

        # Current positions of boxes: which box is at 'at ?box'
        box_positions = {}
        for fact in state:
            if match(fact, "at", "*"):
                parts = get_parts(fact)
                box = parts[1]
                box_positions[box] = True  # 'at' means box present at some position; 
                # However, domain does not explicitly say where box is in 'at'? 
                # Because 'at ?box' only has one argument: According to domain, 
                # (at ?box) is predicate with one parameter only, meaning "the box is at some location". 
                # We have no direct location for box from (at ?box) so position inference from static adjacency.

        # Robot location: find location from (at-robot-2 ?loc)
        robot_location = None
        for fact in state:
            if match(fact, "at-robot-2", "*"):
                parts = get_parts(fact)
                robot_location = parts[1]
                break
        if robot_location is None:
            # Should never happen in valid states, fallback heuristic
            return len(self.goal_boxes) * 5

        # Find boxes that are NOT yet in adjacency-2 relation (by goal boxes minus (adjacent-2 ?box) in state)
        boxes_to_push = self.goal_boxes.copy()
        for fact in state:
            if match(fact, "adjacent-2", "*"):
                parts = get_parts(fact)
                box = parts[1]
                if box in boxes_to_push:
                    boxes_to_push.remove(box)

        # If no boxes left to push, heuristic is zero
        if not boxes_to_push:
            return 0

        # Heuristic calculation:
        # For each box not yet adjacent-2:
        # - Estimate minimal moves for robot to get adjacent to box (a location in box_adjacent_locations[box])
        # - Add cost of one push action (push or push-2)
        # - Add a small cost for repositioning, e.g. 1 move-2 action
        # Sum all these costs to estimate total future effort

        total_cost = 0
        PUSH_COST = 1  # one push action per box
        MOVE_AFTER_PUSH_COST = 1  # assume one move after push (to reposition robot)

        for box in boxes_to_push:
            # We must find the best location adjacent to box where robot can push it
            candidate_locations = self.box_adjacent_locations.get(box, set())
            if not candidate_locations:
                # No known adjacency locations means likely unreachable, assign high cost
                total_cost += 10
                continue

            dist = self.shortest_path_length(self.location_graph, robot_location, candidate_locations)
            if dist is None:
                # No path - assign large penalty
                dist = 10

            total_cost += dist + PUSH_COST + MOVE_AFTER_PUSH_COST

        return total_cost
