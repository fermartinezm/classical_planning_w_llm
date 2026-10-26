from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at robot1 tile_0_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed for the robot to paint all required tiles.

    # Assumptions:
    - The robot can move up, down, left, or right to adjacent tiles.
    - The robot can change colors if needed, which takes one action.
    - Each tile requires exactly one action to paint if the robot is at the correct position with the correct color.

    # Heuristic Initialization
    - Extract static facts including tile connections and available colors.
    - Identify the goal tiles and their required colors.

    # Step-By-Step Thinking for Computing the Heuristic Value
    1. Extract the robot's current position and held color from the state.
    2. Identify all tiles that need to be painted and their required colors.
    3. For each unpainted tile:
       a. Calculate the Manhattan distance from the robot's current position to the tile.
       b. If the robot does not have the required color, add a color change action.
    4. Sum the movement actions and color change actions to get the total estimated actions.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and goal conditions."""
        self.goals = task.goals
        self.static = task.static

        # Extract tile connections (up, down, left, right)
        self.tile_connections = {}
        for fact in self.static:
            if fact.startswith("(up ") or fact.startswith("(down ") or fact.startswith("(left ") or fact.startswith("(right "):
                parts = get_parts(fact)
                x, y = parts[1], parts[2]
                self.tile_connections[(x, y)] = fact

        # Extract available colors
        self.available_colors = set()
        for fact in self.static:
            if match(fact, "available-color", "*"):
                color = get_parts(fact)[1]
                self.available_colors.add(color)

    def __call__(self, node):
        """Estimate the minimum cost to reach the goal state."""
        state = node.state

        # Extract robot's current position and held color
        robot_position = None
        robot_color = None
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                robot_position = get_parts(fact)[1], get_parts(fact)[2]
            if match(fact, "robot-has", "*", "*"):
                robot_color = get_parts(fact)[1]

        if not robot_position:
            return 0  # No robot position found, assume 0 cost (should not happen)

        # Identify all tiles that need to be painted and their required colors
        goal_tiles = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                goal_tiles[tile] = color

        # If all goals are already achieved, return 0
        if not goal_tiles:
            return 0

        # Calculate the heuristic cost
        total_cost = 0

        # For each goal tile, calculate the distance and check color
        for tile, required_color in goal_tiles.items():
            # Check if the tile is already painted
            if any(fact == f"(painted {tile} {required_color})" in state for fact in state):
                continue

            # Calculate Manhattan distance from robot's position to the tile
            # Extracting tile coordinates (assumes tile naming is like tile_x_y)
            x1, y1 = robot_position
            x2, y2 = tile.split('_')[1], tile.split('_')[2]
            distance = abs(int(x1) - int(x2)) + abs(int(y1) - int(y2))
            total_cost += distance

            # Check if color change is needed
            if robot_color != required_color:
                total_cost += 1  # One action to change color

        return total_cost
