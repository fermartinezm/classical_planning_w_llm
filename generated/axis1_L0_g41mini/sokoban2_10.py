from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_3_5)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    Summary:
    This heuristic estimates the number of actions required to push all boxes
    from their current locations to their goal locations. It leverages the adjacency
    relations between locations to compute shortest distances and the relative
    positions of the robot and boxes to estimate the effort needed for pushing.

    Assumptions:
    - Each push action moves exactly one box from its current location to an adjacent,
      clear location.
    - The robot can move freely on clear locations (move actions), however, moving costs 
      are considered but prioritized less than pushing boxes.
    - The heuristic is not necessarily admissible but aims to be informative and efficiently computable.
    - Robot movements aiming to reach boxes or push destinations are approximated using shortest distances in adjacency graph.

    Heuristic Initialization:
    - Extract adjacency relations from static facts and build a bidirectional graph of location adjacencies.
    - Extract all box objects and their goal locations from the goal conditions.
    - Precompute shortest path distances between all pairs of locations using BFS on adjacency graph.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify the robot's current location.
    2. Identify each box's current location.
    3. Calculate for each box the shortest distance to its respective goal location.
    4. Calculate the shortest distance from the robot's location to the *push-position* for each box.
       The push-position is the location from which the robot can push the box towards the goal.
       Since Sokoban involves pushing from an adjacent location opposite the push direction,
       we approximate the minimal cost to move the robot adjacent to the box in a suitable position.
    5. For each box, estimate the minimal cost as:
       = distance(robot to push-position)
         + distance(box to goal)
         + pushing cost (assumed one action per push, length = box-to-goal distance in adjacency graph)
       Summing these estimates for all boxes gives the heuristic.
    6. Return zero if all boxes already on their goal locations.
    7. This heuristic is finite for all solvable states since distances are well-defined,
       and provides informative guidance by considering robot-box and box-goal distances.
    """

    def __init__(self, task):
        """Initialize adjacency graph, boxes, goals, and precompute shortest distances."""

        self.goals = task.goals
        self.static = task.static

        # Build adjacency graph: {location: set(neighbors)}
        self.adj_graph = {}
        for fact in self.static:
            if match(fact, "adjacent", "*", "*", "*"):
                parts = get_parts(fact)
                loc1, loc2 = parts[1], parts[2]
                self.adj_graph.setdefault(loc1, set()).add(loc2)

        # Extract all boxes from facts by scanning goals (goals are of form (at box loc_goal))
        self.box_goals = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                box, goal_loc = parts[1], parts[2]
                self.box_goals[box] = goal_loc

        # For efficient distance queries, precompute shortest paths between all locations
        self.locations = list(self.adj_graph.keys())

        # Precompute shortest distances between all location pairs using BFS
        self.distance_cache = {}
        for loc in self.locations:
            self.distance_cache[loc] = self._bfs_distances(loc)

    def _bfs_distances(self, start):
        """Compute shortest path distances from start to all reachable locations using BFS."""
        distances = {start: 0}
        queue = [start]
        while queue:
            current = queue.pop(0)
            for neighbor in self.adj_graph.get(current, ()):
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        return distances

    def _shortest_dist(self, from_loc, to_loc):
        """Return the shortest distance between two locations, or a large number if unreachable."""
        return self.distance_cache.get(from_loc, {}).get(to_loc, 99999)

    def __call__(self, node):
        """Estimate number of actions needed to solve the Sokoban state.

        Uses shortest-path distances in the adjacency graph to estimate:
        - distance for robot to reach the box push position,
        - plus number of pushes needed to get the box to its goal,
        by summing over all boxes still not at their goal.
        """

        state = node.state

        # If goal satisfied => heuristic is 0
        if self.goals <= state:
            return 0

        # Get robot location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
                break
        if robot_loc is None:
            # Defensive fallback, robot location must be in state
            return 99999

        # Get current locations of all boxes
        box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                obj, loc = parts[1], parts[2]
                if obj in self.box_goals:
                    box_locations[obj] = loc

        total_estimate = 0

        # For each box, estimate cost to get it to goal
        for box, goal_loc in self.box_goals.items():
            current_loc = box_locations.get(box, None)
            if current_loc is None:
                # Missing box location: treat as very far
                total_estimate += 1000
                continue

            # If box at goal already, no cost for it
            if current_loc == goal_loc:
                continue

            # Approximate cost:
            # 1) pushing cost = number of pushes (approx distance box to goal)
            pushes = self._shortest_dist(current_loc, goal_loc)
            if pushes == 99999:
                # unreachable goal, large heuristic to avoid dead ends
                total_estimate += 1000
                continue

            # 2) robot must get to a "push-position", that is a location adjacent to the box opposite the push direction.
            # To simplify, we approximate by:
            # - find all locations adjacent to the box's current location
            # - for each such location, compute shortest distance from robot to that location
            # - take minimum distance (closest place robot can stand to push)
            adj_positions = self.adj_graph.get(current_loc, set())
            if not adj_positions:
                # box stuck in isolated location (no adjacents)
                total_estimate += 1000
                continue

            min_robot_to_push_pos = min(
                self._shortest_dist(robot_loc, adj) for adj in adj_positions
            )

            # Heuristic cost is approx robot travel to push position + pushes to goal
            estimate = min_robot_to_push_pos + pushes

            # Adding pushes again to account for push actions themselves (each push is one action)
            # The push moves the box, so each push corresponds to at least one action,
            # and moving robot to the correct position costs actions as well.
            # We add the pushes count once more as minimal pushing steps (push actions).
            estimate += pushes

            total_estimate += estimate

        return total_estimate
