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

class FloortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all required tiles by considering:
    - The distance the robot needs to move to each unpainted tile.
    - The need to change colors if the current color does not match the required color for a tile.

    # Assumptions:
    - The robot can move up, down, left, or right on the grid.
    - Each tile requires exactly one painting action.
    - Color changes are required when the robot's current color does not match the tile's required color.

    # Heuristic Initialization
    - Extracts static facts to build a grid map representing the layout of tiles.
    - Identifies available colors and their positions.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. Extract the current position of the robot.
    2. Identify all tiles that need to be painted (based on the goal state).
    3. For each unpainted tile:
       a. Calculate the Manhattan distance from the robot's current position.
       b. Add 1 action for painting the tile.
       c. If the tile's required color is different from the robot's current color, add 1 action for changing the color.
    4. Sum the actions for all tiles to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Static facts to build the grid map.
        - Available colors.
        """
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Static facts

        # Build grid map from static facts
        self.grid_map = {}
        for fact in static_facts:
            if match(fact, "up", "*", "*"):
                y, x = get_parts(fact)[1], get_parts(fact)[2]
                self.grid_map[(x, 'up')] = y
            elif match(fact, "down", "*", "*"):
                y, x = get_parts(fact)[1], get_parts(fact)[2]
                self.grid_map[(x, 'down')] = y
            elif match(fact, "left", "*", "*"):
                y, x = get_parts(fact)[1], get_parts(fact)[2]
                self.grid_map[(x, 'left')] = y
            elif match(fact, "right", "*", "*"):
                y, x = get_parts(fact)[1], get_parts(fact)[2]
                self.grid_map[(x, 'right')] = y

        # Extract available colors
        self.available_colors = {
            get_parts(fact)[1]
            for fact in static_facts
            if match(fact, "available-color", "*")
        }

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state

        # Extract robot's current position
        robot_position = None
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                _, robot = get_parts(fact)
                robot_position = get_parts(fact)[2]
                break

        if robot_position is None:
            return 0  # Robot not found, should not happen in valid state

        # Extract current color of the robot
        current_color = None
        for fact in state:
            if match(fact, "robot-has", "*", "*"):
                _, robot, color = get_parts(fact)
                if robot == robot_position.split('_')[0]:
                    current_color = color
                    break

        # Identify all tiles that need to be painted
        goal_tiles = set()
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                _, tile, color = get_parts(goal)
                goal_tiles.add((tile, color))

        # Check if all goal tiles are already painted
        if all(match(fact, "painted", "*", "*") for fact in state):
            return 0

        total_actions = 0

        # For each goal tile, calculate required actions
        for goal_tile, required_color in goal_tiles:
            # Check if the tile is already painted with the required color
            is_painted = any(
                match(fact, "painted", goal_tile, required_color) for fact in state
            )
            if is_painted:
                continue

            # Calculate Manhattan distance from robot's current position to the tile
            # Parse tile coordinates (e.g., "tile_0_1" -> (0,1))
            tile_coords = goal_tile.split('_')
            x = int(tile_coords[1])
            y = int(tile_coords[2])
            robot_coords = robot_position.split('_')[1:]
            rx = int(robot_coords[0])
            ry = int(robot_coords[1])

            distance = abs(x - rx) + abs(y - ry)

            # Add distance actions (movement)
            total_actions += distance

            # Add 1 action for painting
            total_actions += 1

            # Add 1 action for color change if needed
            if current_color != required_color:
                total_actions += 1

        return total_actions
