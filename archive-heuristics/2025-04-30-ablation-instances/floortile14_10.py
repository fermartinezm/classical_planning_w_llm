from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def get_tile_coords(tile_name):
    """Parses a tile name 'tile_row_col' into a tuple (row, col)."""
    parts = tile_name.split('_')
    # Assuming format is always tile_row_col and row/col are integers
    # PDDL objects are case-insensitive, but typically lowercase.
    # Let's assume the format is consistent.
    try:
        row = int(parts[1])
        col = int(parts[2])
        return (row, col)
    except (ValueError, IndexError):
        # Handle unexpected tile name format if necessary, though problem implies consistency
        print(f"Warning: Unexpected tile name format: {tile_name}")
        return None # Or raise an error

def manhattan_distance(tile1_name, tile2_name):
    """Calculates Manhattan distance between two tiles based on their names."""
    coords1 = get_tile_coords(tile1_name)
    coords2 = get_tile_coords(tile2_name)
    if coords1 is None or coords2 is None:
        return float('inf') # Cannot calculate distance for invalid tile names
    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all goal tiles
    with their required colors. It sums the cost for each unpainted goal tile,
    considering the paint action, movement to the tile's vicinity, and color changes.

    # Assumptions
    - The grid structure is implicitly defined by tile names like 'tile_row_col'.
    - Movement cost between adjacent tiles is 1.
    - Changing color costs 1.
    - The robot always holds exactly one color (or is free-color, though free-color isn't used in actions). We assume it always has a color based on the example state and domain actions.
    - The heuristic sums costs for individual tiles and colors, which might overestimate the true cost but provides a gradient towards the goal.

    # Heuristic Initialization
    - Extracts the goal conditions (which tiles need which color).
    - Extracts static facts (like available colors), although these are not strictly used in the current heuristic calculation but are available.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize heuristic value `h = 0`.
    2. Identify the robot's current location and the color it is holding from the current state.
    3. Create a dictionary to store unpainted goal tiles, grouped by the color they need: `unpainted_tiles_by_color = {color: [tile1, tile2, ...], ...}`.
    4. Iterate through the goal conditions (`self.goals`). For each goal fact `(painted T C)`:
       - Check if `(painted T C)` is present in the current state.
       - If not, add tile `T` to the list for color `C` in `unpainted_tiles_by_color`.
    5. If `unpainted_tiles_by_color` is empty, the state is a goal state, return `h = 0`.
    6. Calculate the cost for painting and movement for each unpainted tile:
       - For each color `C` and each tile `T` in `unpainted_tiles_by_color[C]`:
         - Add 1 to `h` (cost for the `paint` action).
         - Calculate the Manhattan distance between the robot's current location and tile `T`.
         - Add the movement cost to get adjacent to `T`:
           - If distance is 0 (robot is on T), add 1 (move off T).
           - If distance is 1 (robot is adjacent), add 0.
           - If distance is > 1, add `distance - 1` (moves to get adjacent).
    7. Calculate the cost for color changes:
       - Identify the set of distinct colors needed for the unpainted tiles (`needed_colors`).
       - Find the color the robot currently has (`C_robot`).
       - If `C_robot` is one of the `needed_colors`: The robot needs to change color for each *other* needed color. Add `len(needed_colors) - 1` to `h`.
       - If `C_robot` is not one of the `needed_colors`: The robot needs to change color for *each* needed color. Add `len(needed_colors)` to `h`.
    8. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals  # Goal conditions are facts like '(painted tile_x_y color)'
        # Static facts are available in task.static, but not strictly needed for this heuristic calculation.
        # We could parse available colors here if needed, but the heuristic only cares about colors mentioned in goals.
        # Example: self.available_colors = {get_parts(fact)[1] for fact in task.static if match(fact, "available-color", "*")}

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings).

        h = 0

        # 2. Identify robot's current location and color
        robot_location = None
        robot_color = None
        # Assuming one robot, find its name and details
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'robot-at':
                # Assuming robot name is always 'robot1' or similar pattern
                # We only need the location, not the robot name itself for this heuristic
                robot_location = parts[2]
            elif parts[0] == 'robot-has':
                robot_color = parts[2]

        # Handle case where robot info isn't found (shouldn't happen in valid states)
        if robot_location is None:
             # This state is likely invalid or terminal (robot removed?), return infinity or a large value
             return float('inf') # Or a large number indicating this is a bad state

        # 3. Create dictionary for unpainted goal tiles by color
        unpainted_tiles_by_color = {} # {color: [tile1, tile2, ...]}

        # 4. Iterate through goals and identify unpainted tiles
        goal_painted_facts = {fact for fact in self.goals if match(fact, "painted", "*", "*")}

        for goal_fact in goal_painted_facts:
            if goal_fact not in state:
                # This tile is not painted correctly
                parts = get_parts(goal_fact)
                tile = parts[1]
                color = parts[2]

                if color not in unpainted_tiles_by_color:
                    unpainted_tiles_by_color[color] = []
                unpainted_tiles_by_color[color].append(tile)

        # 5. If no unpainted tiles, goal reached
        if not unpainted_tiles_by_color:
            return 0

        # 6. Calculate cost for painting and movement for each unpainted tile
        for color, tiles_to_paint in unpainted_tiles_by_color.items():
            for tile in tiles_to_paint:
                h += 1 # Cost for paint action

                # Movement cost to get adjacent to the tile
                dist = manhattan_distance(robot_location, tile)
                if dist == 0:
                    # Robot is on the tile, needs 1 move to get off to an adjacent tile
                    h += 1
                elif dist > 1:
                    # Robot is further away, needs dist - 1 moves to get adjacent
                    h += dist - 1
                # If dist == 1, robot is already adjacent, needs 0 moves.

        # 7. Calculate cost for color changes
        needed_colors = set(unpainted_tiles_by_color.keys())

        if robot_color in needed_colors:
            # Robot has one of the colors needed. Needs to change for the others.
            h += len(needed_colors) - 1
        elif robot_color is not None:
             # Robot has a color, but it's not one of the needed ones.
             # It needs to change to one of the needed colors, and then potentially others.
             # Simplification: Assume it needs to change for every needed color.
             h += len(needed_colors)
        # else: robot_color is None (free-color). Handled by assuming robot always has color.

        # 8. Return total heuristic value
        return h
