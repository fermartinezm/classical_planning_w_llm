from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_1_1)".
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
    - Each box must be pushed to its target location.
    - The robot can move freely to adjacent clear locations.
    - Boxes can only be pushed if there is space in front of them.

    # Heuristic Initialization
    - Extract the target location for each box from the goal conditions.
    - Build a grid map from the static facts to determine adjacency of locations.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. For each box, determine its current location and target location.
    2. If the box is already at the target, no actions are needed.
    3. Otherwise, calculate the number of actions required to move the box to the target:
       a. Move the robot to the box's current location.
       b. Push the box towards the target location.
       c. Move the box step-by-step to the target, considering the grid layout.
    4. Sum the actions required for all boxes to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Static facts

        # Extract goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

        # Build grid map from static facts
        self.grid = {}
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                loc1, loc2, dir = get_parts(fact)
                if loc1 not in self.grid:
                    self.grid[loc1] = []
                self.grid[loc1].append(loc2)
                if loc2 not in self.grid:
                    self.grid[loc2] = []
                self.grid[loc2].append(loc1)

    def __call__(self, node):
        """Estimate the minimum number of actions to reach the goal state."""
        state = node.state  # Current world state

        # Track current locations of boxes and the robot
        current_locations = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at":
                obj, loc = args
                current_locations[obj] = loc

        total_cost = 0  # Initialize action cost counter

        # For each box, calculate the cost to reach its target
        for box, goal_loc in self.goal_locations.items():
            if box not in current_locations:
                # Box is already at the target
                continue

            current_loc = current_locations[box]
            if current_loc == goal_loc:
                continue  # No cost if already at goal

            # Calculate the number of actions needed to move the box
            # Each push requires moving the robot to the box and then pushing it step by step
            # We assume each box needs at least 2 actions per distance unit (1 for moving, 1 for pushing)
            # This is a simplification and can be optimized further

            # Find the shortest path length between current and goal locations
            path_length = self._get_path_length(current_loc, goal_loc)
            if path_length is None:
                # If no path exists, the state is unsolvable (shouldn't happen in a valid problem)
                continue

            # Each step requires at least 2 actions (move to box, push it)
            total_cost += 2 * path_length

        return total_cost

    def _get_path_length(self, start, goal):
        """
        Calculate the shortest path length between two locations using BFS.
        Returns None if no path exists.
        """
        if start == goal:
            return 0

        visited = set()
        queue = [(start, 0)]

        while queue:
            current, dist = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            if current == goal:
                return dist
            if current in self.grid:
                for neighbor in self.grid[current]:
                    if neighbor not in visited:
                        queue.append((neighbor, dist + 1))

        return None  # No path found
