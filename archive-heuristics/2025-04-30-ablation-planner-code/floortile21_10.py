from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # For infinity
from collections import defaultdict

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle cases like '(predicate arg1 arg2)'
    return fact[1:-1].split()

def parse_tile_name(tile_name):
    """Parses a tile name like 'tile_r_c' into (row, col) integer coordinates."""
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # Handle unexpected tile name format
            # print(f"Warning: Could not parse tile coordinates from {tile_name}")
            return None
    # Handle unexpected tile name format
    # print(f"Warning: Could not parse tile coordinates from {tile_name}")
    return None

def dist_manhattan(coords1, coords2):
    """Calculates Manhattan distance between two (row, col) coordinates."""
    if coords1 is None or coords2 is None:
        return float('inf') # Cannot calculate distance if coordinates are invalid
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    that are not yet painted correctly. It sums the estimated minimum cost for each
    unpainted goal tile independently, considering the cost of changing color,
    moving a robot to a position from which the tile can be painted, and painting the tile.

    # Assumptions
    - The grid structure defined by 'up', 'down', 'left', 'right' predicates forms a connected grid.
    - Tile names follow the format 'tile_row_col', and these correspond to (row, col) coordinates.
    - The predicates 'up', 'down', 'left', 'right' define the relative positions and possible moves/paint locations.
      Specifically, (up Y X) means Y is up from X, and a robot at X can paint Y.
      Assuming 'up' increases row index, 'down' decreases row, 'left' decreases col, 'right' increases col.
    - Unpainted goal tiles are initially 'clear' and can be painted.
    - The heuristic uses Manhattan distance on the inferred grid coordinates as a proxy for movement cost,
      ignoring obstacles (non-clear tiles) and the dynamic nature of the grid.
    - The cost for a single tile is estimated by finding the minimum cost among all robots
      to paint that specific tile, ignoring potential synergies or conflicts when
      painting multiple tiles.
    - Robots with `free-color` cannot paint or change color (based on action definitions).
      We assume valid instances where robots needed for painting have or can acquire a color.

    # Heuristic Initialization
    - Extract the goal conditions to identify which tiles need to be painted and with which colors.
    - Parse static facts ('up', 'down', 'left', 'right') to:
        - Infer the (row, col) coordinates for all tiles in the grid.
        - Build a map `adj_from` where `adj_from[T]` is the set of tiles `X` from which tile `T` can be painted.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify all goal facts of the form `(painted tile_name color_name)`. Store these as the set of target painted states (`self.goal_painted_tiles`). This is done during initialization.
    2. Identify all current facts of the form `(painted tile_name color_name)` in the current state.
    3. Determine the set of `UnpaintedGoals`: goal painted states that are not present in the current state. If this set is empty, the heuristic is 0.
    4. Identify the current location and color held by each robot from the state facts `(robot-at robot_name tile_name)` and `(robot-has robot_name color_name)`. Robots with `free-color` are noted.
    5. Initialize the total heuristic value `h = 0`.
    6. For each unpainted goal `(target_tile, target_color)` in `UnpaintedGoals`:
        a. Initialize `min_cost_for_this_tile = infinity`.
        b. For each robot `R` with current location `R_loc` and color `R_color` (or `free-color`):
            i. If robot `R` has `free-color`, it cannot paint or change color according to the domain. Skip this robot for this tile.
            ii. Calculate the estimated cost for robot `R` to paint `target_tile` with `target_color`.
            iii. Cost to get the right color: `cost_color_change = 1` if `R_color` is not `target_color`, otherwise `0`. This assumes the robot needs one change action if its current color is wrong.
            iv. Cost to move to a paint position: Find the minimum Manhattan distance from `R_loc` to any tile `X` in `self.adj_from[target_tile]`. This is the estimated number of moves required to get into a position to paint the tile. `min_dist_to_paint_pos = min(dist_manhattan(self.tile_coords[R_loc], self.tile_coords[X]) for X in self.adj_from[target_tile])`. Handle cases where `adj_from[target_tile]` is empty or coordinates are missing (should result in infinity).
            v. Cost to paint: 1 action (`paint_...`).
            vi. Total estimated cost for robot `R` to paint this tile: `total_cost_R = cost_color_change + min_dist_to_paint_pos + cost_paint`.
            vii. Update `min_cost_for_this_tile = min(min_cost_for_this_tile, total_cost_R)`.
        c. Add `min_cost_for_this_tile` to the total heuristic value `h`. If `min_cost_for_this_tile` is still infinity (e.g., no robots can paint it), `h` becomes infinity.
    7. Return `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building grid structure maps."""
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static # Static facts.

        # Extract goal painted tiles: {(tile_name, color_name)}
        self.goal_painted_tiles = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "painted" and len(parts) == 3:
                self.goal_painted_tiles.add((parts[1], parts[2]))

        # Build tile coordinate map and adjacency map for painting positions
        self.tile_coords = {} # {tile_name: (row, col)}
        self.adj_from = defaultdict(set) # {tile_name_to_paint: set_of_tile_names_to_stand_on}
        all_tiles = set()

        # Collect all tile names first
        for fact in static_facts:
             parts = get_parts(fact)
             for part in parts:
                 if part.startswith('tile_'):
                     all_tiles.add(part)
        for goal in self.goals:
             parts = get_parts(goal)
             for part in parts:
                 if part.startswith('tile_'):
                     all_tiles.add(part)

        if not all_tiles:
             # No tiles found, likely a malformed problem. Heuristic will be based on goals only.
             # print("Warning: No tiles found in static facts or goals.")
             pass # Handle gracefully later if needed

        # Use BFS to infer coordinates and build adj_from map
        if all_tiles:
            start_tile = None
            # Try to find a tile_r_c with r=0 or c=0 as a potential starting point
            for tile in all_tiles:
                coords = parse_tile_name(tile)
                if coords is not None and (coords[0] == 0 or coords[1] == 0):
                    start_tile = tile
                    break
            # If no tile_0_x or tile_x_0 found, just pick any tile
            if start_tile is None:
                 start_tile = next(iter(all_tiles))

            # Assign the starting tile coordinates based on its name
            start_coords_from_name = parse_tile_name(start_tile)
            if start_coords_from_name is None:
                 # Fallback: assign (0,0) if name parsing failed for the start tile
                 start_coords_from_name = (0,0)

            queue = [(start_tile, start_coords_from_name)]
            visited = {start_tile}
            self.tile_coords[start_tile] = start_coords_from_name

            # Define coordinate changes for each direction based on our assumption
            # (up Y X): Y is up from X. If X=(r,c), Y=(r+1, c).
            # (down Y X): Y is down from X. If X=(r,c), Y=(r-1, c).
            # (left Y X): Y is left from X. If X=(r,c), Y=(r, c-1).
            # (right Y X): Y is right from X. If X=(r,c), Y=(r, c+1).
            coord_diff = {
                'up': (1, 0),
                'down': (-1, 0),
                'left': (0, -1),
                'right': (0, 1)
            }
            # Reverse mapping for adjacency: if (DIR Y X) is true, robot at X can paint Y.
            # So, X is in adj_from[Y].

            q_idx = 0
            while q_idx < len(queue):
                current_tile, current_coords = queue[q_idx]
                q_idx += 1

                # Find neighbors from static facts
                for fact in static_facts:
                    parts = get_parts(fact)
                    if len(parts) == 3 and parts[0] in coord_diff:
                        direction, neighbor_tile, source_tile = parts
                        if source_tile == current_tile:
                            # Fact is (direction neighbor_tile current_tile)
                            # This means neighbor_tile is in 'direction' from current_tile
                            # A robot at current_tile can paint neighbor_tile
                            self.adj_from[neighbor_tile].add(current_tile)

                            if neighbor_tile not in visited:
                                visited.add(neighbor_tile)
                                diff = coord_diff[direction]
                                neighbor_coords = (current_coords[0] + diff[0], current_coords[1] + diff[1])
                                self.tile_coords[neighbor_tile] = neighbor_coords
                                queue.append((neighbor_tile, neighbor_coords))
                        elif neighbor_tile == current_tile:
                             # Fact is (direction current_tile source_tile)
                             # This means current_tile is in 'direction' from source_tile
                             # A robot at source_tile can paint current_tile
                             self.adj_from[current_tile].add(source_tile)
                             # We already processed source_tile when it was popped from the queue
                             # or it will be processed later. No need to infer coords here again.


        # Ensure all tiles found have coordinate entries, even if isolated
        for tile in all_tiles:
             if tile not in self.tile_coords:
                  # print(f"Warning: Isolated tile {tile} found. Cannot determine coordinates.")
                  self.tile_coords[tile] = None # Mark as having unknown coordinates


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Extract current painted tiles, robot locations, and robot colors from the state.
        current_painted_tiles = set()
        robot_locations = {} # {robot_name: tile_name}
        robot_colors = {}    # {robot_name: color_name or 'free-color'}

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == "painted" and len(parts) == 3:
                 current_painted_tiles.add((parts[1], parts[2]))
            elif predicate == "robot-at" and len(parts) == 3:
                 robot_name, tile_name = parts[1], parts[2]
                 robot_locations[robot_name] = tile_name
            elif predicate == "robot-has" and len(parts) == 3:
                 robot_name, color_name = parts[1], parts[2]
                 robot_colors[robot_name] = color_name
            elif predicate == "free-color" and len(parts) == 2:
                 robot_name = parts[1]
                 robot_colors[robot_name] = 'free-color' # Use a special marker

        # Identify unpainted goal tiles
        unpainted_goals = self.goal_painted_tiles - current_painted_tiles

        # If all goal tiles are painted correctly, the heuristic is 0.
        if not unpainted_goals:
            return 0

        total_heuristic_cost = 0

        # For each unpainted goal tile, estimate the minimum cost to paint it.
        for target_tile, target_color in unpainted_goals:
            min_cost_for_this_tile = float('inf')

            # Check if the target tile can be painted from anywhere (is in adj_from keys and has paint positions)
            if target_tile not in self.adj_from or not self.adj_from[target_tile]:
                 # This tile cannot be painted with the current actions/grid structure.
                 # Problem is likely unsolvable. Return infinity.
                 return float('inf')

            # Iterate through all robots to find the one that can paint this tile with minimum cost.
            for robot_name, r_loc_name in robot_locations.items():
                r_color = robot_colors.get(robot_name)

                # If robot has free-color, it cannot paint or change color. Skip it.
                if r_color == 'free-color':
                    continue

                # Cost to get the right color
                # If robot has a color but it's the wrong one, assume 1 change_color action is needed.
                cost_color_change = 0
                if r_color != target_color:
                   cost_color_change = 1

                # Cost to move to a paint position
                r_coords = self.tile_coords.get(r_loc_name)

                if r_coords is None:
                    # Robot is at a tile with unknown coordinates. Cannot calculate distance.
                    # This robot cannot paint this tile in the heuristic estimate.
                    continue

                min_dist_to_paint_pos = float('inf')
                for paint_pos_tile in self.adj_from[target_tile]:
                    paint_pos_coords = self.tile_coords.get(paint_pos_tile)
                    if paint_pos_coords is not None:
                        dist = dist_manhattan(r_coords, paint_pos_coords)
                        min_dist_to_paint_pos = min(min_dist_to_paint_pos, dist)

                # If min_dist_to_paint_pos is still infinity, it means the paint positions
                # for this tile have unknown coordinates, or the robot's location has unknown coords.
                # This robot cannot paint this tile in the heuristic estimate.
                if min_dist_to_paint_pos == float('inf'):
                    continue

                # Cost to paint the tile
                cost_paint = 1

                # Total estimated cost for this robot to paint this specific tile
                total_cost_R = cost_color_change + min_dist_to_paint_pos + cost_paint

                min_cost_for_this_tile = min(min_cost_for_this_tile, total_cost_R)

            # If min_cost_for_this_tile is still infinity, it means no robot could potentially paint this tile
            # (e.g., no robots, or all robots have free-color, or paint positions are unreachable/invalid).
            # This state might be unsolvable. Return infinity.
            if min_cost_for_this_tile == float('inf'):
                 return float('inf')

            # Add the minimum estimated cost for this tile to the total heuristic.
            total_heuristic_cost += min_cost_for_this_tile

        return total_heuristic_cost
