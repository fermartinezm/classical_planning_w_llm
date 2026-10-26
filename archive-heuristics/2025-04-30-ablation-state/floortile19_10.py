from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Import math for infinity

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args contains wildcards
    # A simpler check: just zip and compare. fnmatch handles wildcards.
    if len(parts) != len(args):
         return False # Cannot match if number of elements is different
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_tile_name(tile_name):
    """Parses a tile name like 'tile_r_c' into (row, col) coordinates."""
    try:
        parts = tile_name.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            return (int(parts[1]), int(parts[2]))
    except ValueError:
        pass # Not a tile name in the expected format
    return None # Return None if parsing fails

def manhattan_distance(coord1, coord2):
    """Calculates the Manhattan distance between two (row, col) coordinates."""
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with the correct color. It sums the cost for each unpainted goal tile,
    considering the need for a robot to reach an adjacent tile and have the
    correct color.

    # Assumptions
    - The grid structure is implied by tile names 'tile_r_c', where r and c are
      row and column indices.
    - 'up' corresponds to decreasing row index, 'down' to increasing row index,
      'left' to decreasing column index, and 'right' to increasing column index.
    - Movement cost is estimated using Manhattan distance on the grid, ignoring
      dynamic 'clear' predicates for path calculation simplicity (non-admissible).
    - A 'change_color' action is needed for each required color not currently
      held by any robot. This cost is added once per color type, not per tile.
    - Solvable instances are assumed; states where a goal tile is painted with
      the wrong color are not explicitly handled (they would likely result in
      a high heuristic value or infinity if detected).

    # Heuristic Initialization
    - Extracts goal conditions to identify which tiles need to be painted and with
      which color.
    - Parses all tile object names to build a mapping from tile name to its
      (row, col) coordinates based on the 'tile_r_c' naming convention.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  **Identify Unpainted Goal Tiles:** Determine which tiles specified in the
        goal are not currently painted with the correct color in the current state.
        Assume any goal tile not painted correctly is currently 'clear'.

    2.  **Calculate Color Cost:**
        - Identify the set of distinct colors required by the unpainted goal tiles.
        - Identify the set of distinct colors currently held by the robots.
        - The color cost is the number of required colors that are not present
          among the colors currently held by any robot. This estimates the minimum
          number of 'change_color' actions needed across all robots to acquire
          the necessary colors.

    3.  **Calculate Movement and Paint Cost per Tile:** For each unpainted goal tile:
        - A 'paint' action is required (cost 1).
        - A robot must be positioned on a tile adjacent to the target tile.
        - Estimate the minimum movement cost for *any* robot to reach *any* tile
          adjacent to the target tile. This is calculated as the minimum Manhattan
          distance between any robot's current location and any valid adjacent
          tile location of the target tile.
        - The cost for this individual tile is 1 (paint) + minimum movement distance.

    4.  **Sum Costs:** The total heuristic value is the sum of the color cost
        and the movement/paint cost calculated for each unpainted goal tile.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and tile coordinates.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static # Static facts (not used directly for grid structure, rely on naming)

        # Store goal tiles and their target colors: {tile_name: color_name}
        self.goal_tiles = {}
        for goal in self.goals:
            # Goal facts are typically (painted tile_name color_name)
            if match(goal, "painted", "*", "*"):
                _, tile_name, color_name = get_parts(goal)
                self.goal_tiles[tile_name] = color_name

        # Store tile names and their coordinates: {tile_name: (row, col)}
        self.tile_coords = {}
        # We need to get all tile objects from the task definition.
        # The task object doesn't directly expose objects by type, but we can
        # infer them from static facts or operator parameters if needed.
        # A simpler way is to assume tile names follow the 'tile_r_c' pattern
        # and parse all names found in goal_tiles or static facts.
        # Let's parse all potential tile names from goal_tiles first.
        for tile_name in self.goal_tiles:
             coords = parse_tile_name(tile_name)
             if coords is not None:
                 self.tile_coords[tile_name] = coords

        # Also parse tile names from static facts like adjacency relations
        # to ensure we capture all tiles in the grid, not just goal tiles.
        # This is safer for calculating distances.
        all_potential_tile_names = set(self.goal_tiles.keys())
        for fact in static_facts:
             parts = get_parts(fact)
             # Check if fact involves tiles (e.g., adjacency predicates)
             if len(parts) > 1 and parts[0] in ['up', 'down', 'left', 'right', 'clear', 'painted', 'robot-at']:
                 for part in parts[1:]: # Check parameters
                     if part.startswith('tile_'):
                         all_potential_tile_names.add(part)

        for tile_name in all_potential_tile_names:
             coords = parse_tile_name(tile_name)
             if coords is not None and tile_name not in self.tile_coords:
                 self.tile_coords[tile_name] = coords

        # Identify all valid tile names based on successful parsing
        self.all_tile_names = set(self.tile_coords.keys())


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of facts).

        # Parse current state to get dynamic information
        robot_locations = {} # {robot_name: tile_name}
        robot_colors = {}    # {robot_name: color_name}
        painted_tiles = {}   # {tile_name: color_name}
        clear_tiles = set()  # {tile_name}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                _, robot, location = parts
                robot_locations[robot] = location
            elif parts[0] == "robot-has":
                _, robot, color = parts
                robot_colors[robot] = color
            elif parts[0] == "painted":
                _, tile, color = parts
                painted_tiles[tile] = color
            elif parts[0] == "clear":
                _, tile = parts
                clear_tiles.add(tile)

        # 1. Identify Unpainted Goal Tiles
        unpainted_goal_tiles = {} # {tile_name: target_color}
        for tile_name, target_color in self.goal_tiles.items():
            if tile_name not in painted_tiles or painted_tiles[tile_name] != target_color:
                 # Assuming solvable instances, unpainted goal tiles must be clear
                 # assert tile_name in clear_tiles, f"Goal tile {tile_name} not clear and not painted correctly in state: {state}"
                 unpainted_goal_tiles[tile_name] = target_color

        # If all goal tiles are painted correctly, the heuristic is 0.
        if not unpainted_goal_tiles:
            return 0

        # 2. Calculate Color Cost
        required_colors = set(unpainted_goal_tiles.values())
        held_colors = set(robot_colors.values())
        # Number of colors needed that no robot currently has
        colors_to_acquire = required_colors - held_colors
        color_cost = len(colors_to_acquire) # Each needs one change_color action by some robot

        # 3. Calculate Movement and Paint Cost per Tile
        movement_paint_cost = 0

        for tile_name, target_color in unpainted_goal_tiles.items():
            tile_coords = self.tile_coords.get(tile_name)
            if tile_coords is None:
                 # Should not happen if tile names are consistent, but handle defensively
                 # This tile cannot be painted if its location is unknown.
                 # Return a very large number indicating potential unsolvability or error.
                 return math.inf # Or a large constant like 1000000

            # Find valid adjacent tiles for the current goal tile
            r_T, c_T = tile_coords
            potential_adj_coords = [(r_T-1, c_T), (r_T+1, c_T), (r_T, c_T-1), (r_T, c_T+1)]

            valid_adj_tiles = []
            for adj_tile_name, adj_coords in self.tile_coords.items():
                 if adj_coords in potential_adj_coords:
                     valid_adj_tiles.append(adj_tile_name)

            # If a tile has no adjacent tiles (e.g., 1x1 grid), it cannot be painted
            if not valid_adj_tiles:
                 return math.inf # Cannot paint this tile

            min_dist_to_adjacent = math.inf

            # Find the minimum distance from any robot to any valid adjacent tile
            for robot_name, robot_location in robot_locations.items():
                robot_coords = self.tile_coords.get(robot_location)
                if robot_coords is None:
                     # Robot location unknown, cannot calculate distance
                     continue # Skip this robot

                for adj_tile_name in valid_adj_tiles:
                    adj_coords = self.tile_coords.get(adj_tile_name)
                    if adj_coords is None:
                         continue # Adjacent tile coordinates unknown

                    dist = manhattan_distance(robot_coords, adj_coords)
                    min_dist_to_adjacent = min(min_dist_to_adjacent, dist)

            # If min_dist_to_adjacent is still infinity, no robot can reach an adjacent tile
            if min_dist_to_adjacent == math.inf:
                 return math.inf # Cannot reach this tile

            # Cost for this tile: 1 (paint action) + minimum moves to get adjacent
            movement_paint_cost += (1 + min_dist_to_adjacent)

        # 4. Sum Costs
        total_heuristic = color_cost + movement_paint_cost

        return total_heuristic

