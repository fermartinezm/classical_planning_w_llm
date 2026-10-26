from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at-robot loc_4_7)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class SokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to move all boxes to their target locations.

    # Assumptions:
    - The robot can move freely between adjacent clear locations.
    - A box can only be pushed if there is a clear space in front of it.
    - The robot must be adjacent to a box to push it.

    # Heuristic Initialization
    - Extract the target locations for each box from the goal conditions.
    - Build a graph of adjacent locations from static facts.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. For each box, determine its current location and target location.
    2. Calculate the minimum number of moves required for the robot to reach the box's current location.
    3. Calculate the number of moves required to push the box to its target location.
    4. Sum these values for all boxes, adjusting for the robot's current position and any overlapping movements.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Static facts

        # Build adjacency graph from static facts
        self.adjacent = {}
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                loc1, loc2, dir = get_parts(fact)
                if loc1 not in self.adjacent:
                    self.adjacent[loc1] = {}
                self.adjacent[loc1][loc2] = dir
                if loc2 not in self.adjacent:
                    self.adjacent[loc2] = {}
                self.adjacent[loc2][loc1] = dir

        # Extract target locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

    def __call__(self, node):
        """Estimate the minimum number of actions to reach the goal state."""
        state = node.state

        # Extract current locations of boxes and robot
        current_locations = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at":
                obj, loc = args
                if obj.startswith("box"):
                    current_locations[obj] = loc
            elif predicate == "at-robot":
                loc = args[0]
                current_locations["robot"] = loc

        total_cost = 0

        # For each box, calculate the required moves
        for box in self.goal_locations:
            goal_loc = self.goal_locations[box]
            current_loc = current_locations.get(box, None)

            if current_loc is None:
                continue  # Box not present in state

            if current_loc == goal_loc:
                continue  # Box is already at target

            # Calculate distance from robot to box
            robot_loc = current_locations.get("robot", None)
            if robot_loc is None:
                continue  # Robot location not known

            # Calculate Manhattan distance between robot and box
            # Note: Sokoban grid is 2D, but we need adjacency graph
            # Here, we assume that the distance is the number of moves via adjacency
            # For simplicity, we'll use a BFS approach to find the shortest path
            # This is a simplification and may not handle all cases optimally
            path = self.bfs(robot_loc, current_loc)
            if path is None:
                continue  # No path found (should not happen in Sokoban)
            moves_to_box = len(path) - 1  # Exclude starting point

            # Calculate distance from box's current location to goal
            path_to_goal = self.bfs(current_loc, goal_loc)
            if path_to_goal is None:
                continue  # No path found (should not happen in Sokoban)
            moves_to_goal = len(path_to_goal) - 1

            # Total moves for this box: to box + push to goal
            total_cost += moves_to_box + moves_to_goal

        return total_cost

    def bfs(self, start, end):
        """
        Perform BFS to find the shortest path between two locations.
        Returns the list of locations in the path, or None if no path exists.
        """
        visited = set()
        queue = [(start, [start])]

        while queue:
            current, path = queue.pop(0)
            if current == end:
                return path
            if current in visited:
                continue
            visited.add(current)

            for neighbor in self.adjacent.get(current, []):
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))

        return None
