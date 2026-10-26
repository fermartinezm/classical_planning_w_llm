from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions for parsing PDDL facts and tile names

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_2 black)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts is sufficient and if all specified args match
    if len(parts) < len(args) or not all(fnmatch(part, arg) for part, arg in zip(parts, args)):
         return False
    # If the fact has more parts than args, ensure the pattern allows for this (e.g., ends with *)
    if len(parts) > len(args) and (not args or args[-1] != '*'):
         return False
    return True


def get_coords(tile_name):
    """
    Parses a tile name like 'tile_r_c' into (row, col) integers.
    Assumes tile names follow the format 'tile_<row>_<col>'.
    """
    try:
        parts = tile_name.split('_')
        # Expecting format like 'tile_1_5'
        if len(parts) == 3 and parts[0] == 'tile':
            # Convert row and column parts to integers
            return (int(parts[1]), int(parts[2]))
        else:
            # Handle unexpected format
            # print(f"Warning: Unexpected tile name format: {tile_name}")
            return None
    except (ValueError, IndexError):
        # Handle cases where conversion to int fails or parts are missing
        # print(f"Warning: Could not parse tile coordinates from: {tile_name}")
        return None

def manhattan_distance(coords1, coords2):
    """
    Calculates Manhattan distance between two coordinate pairs (r1, c1) and (r2, c2).
    Returns infinity if coordinates are invalid.
    """
    if coords1 is None or coords2 is None:
        return float('inf') # Cannot calculate distance for invalid coordinates
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by summing three components: the number of tiles that need painting,
    the estimated number of color changes required, and the estimated movement cost
    based on Manhattan distance to unpainted tiles. It is designed for greedy
    best-first search and is not admissible.

    # Assumptions
    - The grid structure is implied by tile names in the format 'tile_row_col'.
    - Actions (move, paint, change_color) have a cost of 1.
    - The heuristic ignores the 'clear' precondition for movement and painting when estimating distance.
    - The robot always holds exactly one color.
    - It is efficient to paint all required tiles of one color before changing to another.
    - There is only one robot, named 'robot1'.

    # Heuristic Initialization
    - Parses the goal facts from the task to create a dictionary mapping each goal tile to its required color.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Extract Relevant Information from the State:
       - Find the robot's current location using the `(robot-at robot1 ?x)` fact.
       - Find the color the robot is currently holding using the `(robot-has robot1 ?c)` fact.
       - Build a dictionary representing the current painting status of each tile, mapping tile names to their color if painted, or 'clear' if clear.

    2. Identify Misplaced Goal Tiles:
       - Compare the current painting status of each tile with the target painting status defined in the goal.
       - A tile is considered "misplaced" if it is a goal tile (i.e., appears in the goal as `(painted tile color)`) but is currently not painted with the correct goal color (either it's clear, or painted with a different color).
       - Collect the set of misplaced goal tiles, storing them as `(tile, goal_color)` pairs.
       - Collect the set of distinct colors required for these misplaced tiles.

    3. Calculate Heuristic Components:
       - **Paint Actions:** The number of misplaced goal tiles is a lower bound on the number of `paint` actions required. Add this count to the heuristic value.
       - **Color Change Actions:** Estimate the number of `change_color` actions. This is the number of distinct colors required for the misplaced tiles. If the robot is already holding one of these required colors, assume one color change is saved for the first batch of that color. Add this adjusted count to the heuristic value (minimum 0).
       - **Movement Cost:** Estimate the movement required. Calculate the sum of the Manhattan distances from the robot's current location to *each* tile that needs painting (i.e., each tile in the set of misplaced goal tiles). Add this sum to the heuristic value. This component is a significant overestimate but helps guide the search towards states where the robot is closer to unpainted tiles.

    4. Sum the Components:
       - The total heuristic value for the state is the sum of the estimated paint actions, color change actions, and movement cost.
       - If there are no misplaced goal tiles, the state is a goal state, and the heuristic value is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal painting requirements from the task.
        """
        super().__init__(task)
        self.goal_painting = {}
        # Parse goal facts to find required painting for each tile
        for goal_fact in self.goals:
            # Use the match helper to safely parse goal facts
            if match(goal_fact, "painted", "*", "*"):
                parts = get_parts(goal_fact)
                tile, color = parts[1], parts[2]
                self.goal_painting[tile] = color

        # Static facts like adjacency (up, down, left, right) and available colors
        # are not explicitly needed for this heuristic's calculation, as Manhattan
        # distance is derived from tile naming convention, and available colors
        # are implicitly handled by the 'needed_colors' set.

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state from the current state.
        """
        state = node.state

        # 1. Extract current state information
        robot_loc = None
        robot_color = None
        current_painting = {} # {tile_name: color or 'clear'}

        for fact in state:
            # Use the match helper to safely parse state facts
            if match(fact, "robot-at", "robot1", "*"):
                robot_loc = get_parts(fact)[2]
            elif match(fact, "robot-has", "robot1", "*"):
                 robot_color = get_parts(fact)[2]
            elif match(fact, "painted", "*", "*"):
                parts = get_parts(fact)
                tile, color = parts[1], parts[2]
                current_painting[tile] = color
            elif match(fact, "clear", "*"):
                tile = get_parts(fact)[1]
                current_painting[tile] = 'clear'

        # In a valid state, robot_loc and robot_color should always be found.
        # If not, something is wrong with the state representation or domain.
        # We proceed assuming they are found.

        # 2. Identify misplaced goal tiles and needed colors
        misplaced_goal_tiles = [] # List of (tile, goal_color) tuples
        needed_colors = set()
        misplaced_tiles_set = set() # Set of tile names that need painting

        for tile, goal_color in self.goal_painting.items():
            # Check if the tile is not in the state, or is painted with the wrong color, or is clear
            # A tile is misplaced if its current state fact is NOT the goal state fact
            current_fact_for_tile = current_painting.get(tile) # Get current status (color or 'clear')

            if current_fact_for_tile != goal_color: # If current status is not the desired goal color
                 # It's a misplaced tile that needs to become (painted tile goal_color)
                 misplaced_goal_tiles.append((tile, goal_color))
                 needed_colors.add(goal_color)
                 misplaced_tiles_set.add(tile)


        # 3. If no misplaced tiles, goal is reached
        if not misplaced_goal_tiles:
            return 0

        # 4. Estimate paint actions: One paint action per misplaced tile
        num_paint_actions = len(misplaced_goal_tiles)

        # 5. Estimate color change actions: Number of distinct colors needed, minus 1 if robot has a needed color
        num_color_changes = len(needed_colors)
        # If the robot currently has a color that is needed for any misplaced tile,
        # the first batch of painting that color doesn't require a change_color action first.
        if robot_color is not None and robot_color in needed_colors:
             num_color_changes = max(0, num_color_changes - 1)


        # 6. Estimate movement cost: Sum of Manhattan distances from robot to each misplaced tile
        total_manhattan_distance = 0
        robot_coords = get_coords(robot_loc)

        # Only calculate distance if robot location coordinates are valid
        if robot_coords is not None:
            for tile in misplaced_tiles_set:
                tile_coords = get_coords(tile)
                # Add distance only if tile coordinates are valid
                dist = manhattan_distance(robot_coords, tile_coords)
                if dist != float('inf'):
                    total_manhattan_distance += dist
                # else:
                    # print(f"Warning: Could not get coordinates for tile {tile}")
                    # If any tile coords are invalid, the total distance is infinite
                    # total_manhattan_distance = float('inf')
                    # break # Exit loop early if distance is infinite


        # 7. Combine heuristic components
        # If any part of the calculation resulted in infinity (e.g., invalid coords), return infinity
        if total_manhattan_distance == float('inf'):
             return float('inf')

        heuristic_value = num_paint_actions + num_color_changes + total_manhattan_distance

        return heuristic_value
