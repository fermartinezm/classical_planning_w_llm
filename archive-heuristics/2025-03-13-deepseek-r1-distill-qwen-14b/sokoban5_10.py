from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_2_4)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class SokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to move all boxes to their respective goal locations using the Manhattan distance.

    # Assumptions:
    - Each box must be moved to a specific goal location.
    - The grid layout is known and static.
    - The Manhattan distance provides a good estimate of the required moves.

    # Heuristic Initialization
    - Extract goal locations for each box.
    - Parse static facts to determine the coordinates of each location.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the goal location for each box.
    2. Determine the current location of each box.
    3. Calculate the Manhattan distance between the current and goal locations for each box.
    4. Sum these distances to estimate the total number of actions required.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        # Extract goal locations for each box
        self.goals = {}
        for goal in task.goals:
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                box, loc = parts[1], parts[2]
                self.goals[box] = loc

        # Parse static facts to build location coordinate map
        self.location_coords = {}
        for fact in task.static:
            if match(fact, "adjacent", "*", "*", "*"):
                continue  # Skip adjacency facts as they don't provide coordinates
            if match(fact, "clear", "*"):
                loc = get_parts(fact)[1]
                # Extract coordinates from location string (e.g., loc_2_4 -> (2,4))
                x, y = map(int, loc.split('_')[1:])
                self.location_coords[loc] = (x, y)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_distance = 0

        # For each box, calculate the distance to its goal
        for box, goal_loc in self.goals.items():
            # Find current location of the box
            current_loc = None
            for fact in state:
                if match(fact, "at", box, "*"):
                    current_loc = get_parts(fact)[2]
                    break
            if current_loc is None:
                # Box is not present, assume it's already at the goal
                continue

            # Get coordinates
            current_coords = self.location_coords.get(current_loc, (0, 0))
            goal_coords = self.location_coords.get(goal_loc, (0, 0))

            # Calculate Manhattan distance
            distance = abs(current_coords[0] - goal_coords[0]) + abs(current_coords[1] - goal_coords[1])
            total_distance += distance

        return total_distance
