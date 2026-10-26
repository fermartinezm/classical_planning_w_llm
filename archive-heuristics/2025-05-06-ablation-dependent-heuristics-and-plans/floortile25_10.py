import math # For infinity

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def get_coords(tile_name):
    """Parse tile name 'tile_row_col' into (row, col) tuple."""
    parts = tile_name.split('_')
    # Assuming format is always tile_row_col
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # Handle cases where row/col are not integers if necessary
            # print(f"Warning: Could not parse tile coordinates from {tile_name}")
            pass
    # Return None for unexpected formats
    # print(f"Warning: Unexpected tile name format: {tile_name}")
    return None

def manhattan_distance(coords1, coords2):
    """Calculate Manhattan distance between two coordinate tuples (row, col)."""
    if coords1 is None or coords2 is None:
        return math.inf # Cannot calculate distance if coordinates are missing
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])

class floortileHeuristic: # Assuming Heuristic base class is not strictly required for submission format, but should inherit if used in a framework
# class floortileHeuristic(Heuristic): # Use this line if inheriting from a base class
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    that are currently unpainted. It considers the cost of painting each tile,
    changing robot color when necessary, the initial movement cost for robots
    to reach the vicinity of tiles needing a specific color, and the movement
    cost to cover the area containing tiles of the same target color.

    # Assumptions
    - Tiles are arranged in a grid structure implied by 'tile_row_col' naming.
    - Tile names follow the format 'tile_row_col'.
    - Movement cost between adjacent tiles is 1 (Manhattan distance is used as a relaxation).
    - Changing color costs 1 action.
    - Painting a tile costs 1 action.
    - A robot must be at an adjacent tile to paint a target tile (this is implicitly handled by movement cost estimation).
    - Unpainted goal tiles are currently 'clear' (heuristic counts goals not met).

    # Heuristic Initialization
    - Parse goal facts to identify which tiles need which color.
    - Parse static facts to map tile names to coordinates.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Initialize the total heuristic value `h` to 0.
    2. Identify the current location and color of each robot from the state.
    3. Identify all goal facts of the form `(painted tile color)`. Store these as `goal_paintings = {tile: color}`.
    4. Identify the set of tiles that are goals but are currently *not* painted correctly. Group these by the required color: `unpainted_goal_tiles_by_color = {color: {tile}}`. A tile `t` needing color `C` is considered unpainted correctly if the fact `(painted t C)` is not present in the current state.
    5. For each color `C` that has a non-empty set of unpainted goal tiles (`tiles_for_C`):
        a. Add the number of tiles in `tiles_for_C` to `h`. This accounts for the paint action for each tile.
        b. Check if any robot currently has color `C`. If not, add 1 to `h` (representing the cost of one robot changing to color `C`). This is a simplification assuming one color change is sufficient per color needed.
        c. Estimate the movement cost for color `C`. This is broken into two parts using Manhattan distance as a relaxation:
            i. Initial approach cost: Find the minimum Manhattan distance from any robot's current location to *any* tile in `tiles_for_C`. Add this minimum distance to `h`. This estimates the cost to get a robot into the general area where painting of color `C` is needed.
            ii. Covering area cost: Find the bounding box (min/max row and column) of the tiles in `tiles_for_C`. Add the sum of the row range (`max_row - min_row`) and column range (`max_col - min_col`) to `h`. This estimates the minimum movement needed to traverse the dimensions of the area containing the tiles that need painting with color `C`.
    6. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Parse goal facts to get target paintings
        self.goal_paintings = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                self.goal_paintings[tile] = color

        # Parse static facts to map tile names to coordinates
        self.tile_coords = {}
        # Collect all tile names from static facts (directional predicates often list tiles)
        all_tiles = set()
        for fact in static_facts:
             parts = get_parts(fact)
             if len(parts) > 1 and parts[1].startswith('tile_'):
                 all_tiles.add(parts[1])
             if len(parts) > 2 and parts[2].startswith('tile_'):
                 all_tiles.add(parts[2])

        # Also add tiles from goals if they weren't in static (unlikely but safe)
        all_tiles.update(self.goal_paintings.keys())

        for tile in all_tiles:
             coords = get_coords(tile)
             if coords is not None:
                 self.tile_coords[tile] = coords


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Identify robot locations and colors
        robot_locs = {}
        robot_colors = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "robot-at" and len(parts) == 3:
                robot, loc = parts[1], parts[2]
                robot_locs[robot] = loc
            elif parts and parts[0] == "robot-has" and len(parts) == 3:
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color

        # Identify unpainted goal tiles, grouped by color
        unpainted_goal_tiles_by_color = {}
        for tile, goal_color in self.goal_paintings.items():
            # Check if the goal fact (painted tile goal_color) is NOT in the current state
            goal_fact_str = f"(painted {tile} {goal_color})"
            if goal_fact_str not in state:
                 # This tile needs painting with goal_color
                 if goal_color not in unpainted_goal_tiles_by_color:
                     unpainted_goal_tiles_by_color[goal_color] = set()
                 unpainted_goal_tiles_by_color[goal_color].add(tile)


        h = 0

        for color, tiles_for_C in unpainted_goal_tiles_by_color.items():
            if not tiles_for_C:
                continue

            # Cost for painting
            h += len(tiles_for_C)

            # Cost for color change
            has_color = any(robot_colors.get(r) == color for r in robot_colors)
            if not has_color:
                h += 1 # Assume one robot changes color

            # Cost for movement
            # Find the minimum Manhattan distance from any robot to any tile in the set.
            min_initial_dist = math.inf
            
            # Ensure we only consider robots whose location is a valid tile with known coordinates
            valid_robot_locs = {r: loc for r, loc in robot_locs.items() if loc in self.tile_coords}

            if valid_robot_locs:
                for robot, loc_R in valid_robot_locs.items():
                    loc_R_coords = self.tile_coords[loc_R]

                    min_dist_R_to_set = math.inf
                    
                    # Ensure we only consider target tiles with known coordinates
                    valid_tiles_for_C = [t for t in tiles_for_C if t in self.tile_coords]

                    if valid_tiles_for_C:
                        for tile_T in valid_tiles_for_C:
                            tile_T_coords = self.tile_coords[tile_T]
                            dist = manhattan_distance(loc_R_coords, tile_T_coords)
                            min_dist_R_to_set = min(min_dist_R_to_set, dist)

                    if min_dist_R_to_set != math.inf:
                        min_initial_dist = min(min_initial_dist, min_dist_R_to_set)

            if min_initial_dist != math.inf: # Add the cost to get near the set
                 h += min_initial_dist


            # Add cost to cover the spread of tiles for this color.
            # Calculate bounding box of tiles_for_C
            min_row = math.inf
            max_row = -math.inf
            min_col = math.inf
            max_col = -math.inf
            
            # Ensure we only consider tiles with valid coordinates
            valid_tiles_for_C = [t for t in tiles_for_C if t in self.tile_coords]

            if valid_tiles_for_C:
                for tile_T in valid_tiles_for_C:
                    r, c = self.tile_coords[tile_T]
                    min_row = min(min_row, r)
                    max_row = max(max_row, r)
                    min_col = min(min_col, c)
                    max_col = max(max_col, c)

                # Only add spread cost if there's more than one tile or the bounding box is non-zero size
                # A single tile has spread 0. Multiple tiles might have spread 0 if collinear.
                # The (max-min) + (max-min) covers the dimensions of the bounding box.
                # If there's only one tile, max=min, spread=0. This is correct.
                spread_cost = (max_row - min_row) + (max_col - min_col)
                h += spread_cost


        # The heuristic is 0 if and only if unpainted_goal_tiles_by_color is empty,
        # which means all goal paintings are in the state.
        return h

