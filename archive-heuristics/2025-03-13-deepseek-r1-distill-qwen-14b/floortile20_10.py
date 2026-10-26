from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at tile_0_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class FloortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all required tiles with the correct colors.

    # Assumptions:
    - The robot can move up, down, left, or right to adjacent tiles.
    - Each tile can be painted only if it is clear.
    - The robot can change colors, but each color change costs one action.
    - The goal is to paint specific tiles with specific colors.

    # Heuristic Initialization
    - Extract the goal conditions to identify which tiles need to be painted and with which colors.
    - Extract static facts to determine available colors.

    # Step-by-Step Thinking for Computing Heuristic
    1. Identify the current position of the robot and the color it is holding.
    2. For each goal tile, check if it is already painted correctly. If not, and if the tile is clear, add it to the list of required tiles.
    3. Group the required tiles by their required color to minimize the number of color changes.
    4. For each color group:
       a. If the robot's current color is different from the group's color, add one action for changing the color.
       b. Calculate the Manhattan distance from the robot's current position to each tile in the group.
       c. Sum these distances and add the number of tiles in the group (each requiring a paint action).
    5. Sum the costs for all color groups to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Static facts

        # Extract goal tiles and their required colors
        self.goal_tiles = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_tiles[tile] = color

        # Extract available colors
        self.available_colors = set()
        for fact in static_facts:
            if match(fact, "available-color", "*"):
                color = get_parts(fact)[1]
                self.available_colors.add(color)

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state

        # Extract robot's current position and held color
        robot_pos = None
        held_color = None
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                robot_pos = get_parts(fact)[1:]
            if match(fact, "robot-has", "*", "*"):
                held_color = get_parts(fact)[1]

        if not robot_pos:
            return 0  # Robot position not found, assume already at goal

        # Collect required tiles that are not painted correctly and are clear
        required_tiles = []
        for tile, color in self.goal_tiles.items():
            painted = f'(painted {tile} {color})' in state
            clear = f'(clear {tile})' in state
            if not painted and clear:
                required_tiles.append((tile, color))

        if not required_tiles:
            return 0  # All goal tiles are already painted correctly

        # Group required tiles by their color
        color_groups = {}
        for tile, color in required_tiles:
            if color not in color_groups:
                color_groups[color] = []
            color_groups[color].append(tile)

        total_cost = 0
        current_color = held_color

        # Process each color group
        for color, tiles in color_groups.items():
            # Check if color is available (though problem assumes it's solvable)
            if color not in self.available_colors:
                continue  # Should not happen in solvable instances

            # Change color if necessary
            if current_color != color:
                total_cost += 1
                current_color = color

            # Calculate the Manhattan distance for each tile in the group
            # Parse robot's current position
            robot_x = int(robot_pos[0].split('_')[1])
            robot_y = int(robot_pos[0].split('_')[2])

            for tile in tiles:
                # Parse tile coordinates
                tile_parts = tile.split('_')
                x = int(tile_parts[1])
                y = int(tile_parts[2])
                distance = abs(x - robot_x) + abs(y - robot_y)
                total_cost += distance

            # Add paint actions for each tile in the group
            total_cost += len(tiles)

            # Update robot's position to the last tile in the group
            robot_x = x
            robot_y = y

        return total_cost
