from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Import math for infinity

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle cases like '(predicate)' or '(predicate arg)'
    if not fact.startswith('(') or not fact.endswith(')'):
        # Return empty list or handle error for malformed facts
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_2 black)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    that are not currently painted correctly. It considers the number of tiles
    to paint, the number of color changes needed, and a simplified estimate
    of movement cost.

    # Assumptions
    - The goal state specifies the required color for certain tiles using the
      `(painted ?x ?c)` predicate.
    - Tiles that need to be painted according to the goal are initially `clear`.
      If a goal tile is found to be painted with a *wrong* color, the problem
      is considered unsolvable from that state, and the heuristic returns a
      very large value (infinity).
    - The robot can only hold one color at a time.
    - Movement cost between adjacent tiles is 1.
    - The grid structure is defined by `up`, `down`, `left`, `right` predicates,
      and tile names follow the pattern `tile_R_C`.

    # Heuristic Initialization
    - Parse the grid structure from static facts (`up`, `down`, `left`, `right`)
      to build a mapping from tile names (`tile_R_C`) to grid coordinates (`(R, C)`).
    - Build an adjacency map for tiles based on the grid structure.
    - Extract the goal conditions, specifically the required color for each goal tile.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the robot's current location and the color it is holding from the state.
       Note: The robot might have `free-color` instead of holding a specific color.
    2. Identify all tiles that are currently painted and their colors from the state.
    3. Determine the set of "unpainted goal tiles": these are tiles specified in the goal
       as `(painted t c)` but are not currently painted with color `c` in the state.
    4. Check for unsolvability: If any unpainted goal tile `t` is *not* `clear` in the
       current state (meaning it's painted with a wrong color), return a large heuristic value (infinity).
    5. If the set of unpainted goal tiles is empty, the heuristic is 0 (goal reached for painted facts).
    6. If there are unpainted goal tiles:
       a. Count the number of unpainted goal tiles (`num_unpainted`). Each requires a paint action (cost 1).
       b. Identify the set of distinct colors needed for these unpainted tiles (`needed_colors`).
       c. Estimate the number of color changes required (`num_color_changes`). This is approximated as the number of distinct needed colors, minus one if the robot already holds one of the needed colors. If the robot has `free-color`, it needs to acquire the first color. This estimate assumes an optimal sequence of painting colors.
       d. Estimate the movement cost. A simple non-admissible estimate is used: assume roughly one "unit" of movement is needed per unpainted tile to get into position or move between tiles.
       e. The total heuristic is the sum of these costs: `num_unpainted` (paint) + `num_color_changes` (color) + `num_unpainted` (movement).
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract goal paintings: {tile_name: color_name}
        self.goal_paintings = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted":
                if len(parts) == 3:
                    tile, color = parts[1], parts[2]
                    self.goal_paintings[tile] = color
                else:
                    print(f"Warning: Unexpected goal fact format: {goal}")


        # Build tile_to_coords map and adjacency map from static facts
        self.tile_to_coords = {}
        self.adj_map = {}

        # Collect all tile names first by looking at connectivity facts
        all_tiles = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
                 all_tiles.add(parts[1])
                 all_tiles.add(parts[2])

        # Parse tile names like tile_R_C into coordinates (R, C)
        for tile_name in all_tiles:
            try:
                # Assuming tile names are like 'tile_R_C'
                # Split by '_' and take parts after 'tile'
                parts = tile_name.split('_')
                if len(parts) == 3 and parts[0] == 'tile':
                    row_str, col_str = parts[1], parts[2]
                    self.tile_to_coords[tile_name] = (int(row_str), int(col_str))
                    self.adj_map[tile_name] = set() # Initialize adjacency set
                else:
                     print(f"Warning: Unexpected tile name format: {tile_name}")
            except ValueError:
                # Handle cases where R or C are not integers
                print(f"Warning: Could not parse coordinates from tile name: {tile_name}")
                pass # Ignore tiles that don't match the expected format

        # Build adjacency map
        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
                tile1, tile2 = parts[1], parts[2]
                # Ensure both tiles were successfully parsed and added to maps
                if tile1 in self.adj_map and tile2 in self.adj_map:
                     self.adj_map[tile1].add(tile2)
                     self.adj_map[tile2].add(tile1) # Adjacency is symmetric

    def manhattan_distance(self, tile1_name, tile2_name):
        """Calculate Manhattan distance between two tiles."""
        if tile1_name not in self.tile_to_coords or tile2_name not in self.tile_to_coords:
            # Should not happen if parsing is correct, but handle defensively
            # print(f"Warning: Cannot calculate distance for unknown tiles: {tile1_name}, {tile2_name}")
            return float('inf')
        r1, c1 = self.tile_to_coords[tile1_name]
        r2, c2 = self.tile_to_coords[tile2_name]
        return abs(r1 - r2) + abs(c1 - c2)

    # get_closest_adjacent_tile is not used in the final heuristic formula,
    # but kept for potential future use or debugging.
    def get_closest_adjacent_tile(self, from_tile_name, target_tile_name):
        """Find the minimum Manhattan distance from from_tile to any tile adjacent to target_tile."""
        if target_tile_name not in self.adj_map or not self.adj_map[target_tile_name]:
             # Target tile has no defined adjacency or is not in our map
             # print(f"Warning: Target tile {target_tile_name} has no known adjacent tiles.")
             return float('inf')

        min_dist = float('inf')
        for adj_tile in self.adj_map[target_tile_name]:
            dist = self.manhattan_distance(from_tile_name, adj_tile)
            min_dist = min(min_dist, dist)
        return min_dist


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify robot location and color
        robot_loc = None
        robot_color = None # Will be None if robot has free-color
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "robot-at":
                # Assuming only one robot
                if len(parts) == 3:
                    robot_loc = parts[2]
            elif parts and parts[0] == "robot-has":
                 # Assuming only one robot
                 if len(parts) == 3:
                    robot_color = parts[2]
            # We don't explicitly check for free-color, robot_color being None implies it.


        if robot_loc is None:
             # Robot location must be known to proceed
             # print("Warning: Robot location not found in state.")
             return float('inf') # Should not happen in valid states

        # 2. Identify current painted tiles
        current_paintings = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "painted":
                if len(parts) == 3:
                    tile, color = parts[1], parts[2]
                    current_paintings[tile] = color

        # 3. Determine unpainted goal tiles
        unpainted_goal_tiles = {} # {tile: goal_color}
        for goal_tile, goal_color in self.goal_paintings.items():
            if goal_tile not in current_paintings or current_paintings[goal_tile] != goal_color:
                unpainted_goal_tiles[goal_tile] = goal_color

        # 4. Check for unsolvability (incorrectly painted goal tile)
        # If a tile needs to be painted (is in unpainted_goal_tiles) but is NOT clear,
        # it must be painted with the wrong color, which is unsolvable.
        for tile in unpainted_goal_tiles:
             if f"(clear {tile})" not in state:
                  # Tile is not clear, and it's not painted with the correct color.
                  # This implies it's painted with a wrong color.
                  # print(f"Unsolvable state: Goal tile {tile} needs color {unpainted_goal_tiles[tile]} but is not clear.")
                  return float('inf') # Use a large number for infinity

        # 5. If the set of unpainted goal tiles is empty, heuristic is 0
        if not unpainted_goal_tiles:
            return 0

        # 6. Calculate heuristic components

        # Number of paint actions needed
        num_unpainted = len(unpainted_goal_tiles)

        # Estimate color changes
        needed_colors = set(unpainted_goal_tiles.values())
        num_color_changes = 0
        if needed_colors: # Only need color changes if there are tiles to paint
            if robot_color is None: # Robot has free-color
                 # Needs to acquire the first color, then potentially change for others
                 num_color_changes = len(needed_colors)
            elif robot_color in needed_colors:
                 # Already has one needed color, needs to change for the remaining distinct colors
                 num_color_changes = len(needed_colors) - 1
            else: # Robot has a color not needed
                 # Needs to change to one of the needed colors, then potentially change for others
                 num_color_changes = len(needed_colors)

        # Estimate movement cost
        # A simple non-admissible estimate: assume roughly one "unit" of movement
        # is needed per unpainted tile to get into position or move between tiles.
        # This is a very rough approximation.
        movement_cost = num_unpainted

        # Total heuristic cost
        # Sum of paint actions, estimated color changes, and estimated movement.
        total_cost = num_unpainted + num_color_changes + movement_cost

        return total_cost

