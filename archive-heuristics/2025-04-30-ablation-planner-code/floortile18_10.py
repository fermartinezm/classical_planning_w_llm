from heuristics.heuristic_base import Heuristic
import math # Used for float('inf')

def get_parts(fact):
    """Extract the components of a PDDL fact string."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with their target colors. It sums the estimated cost for each unpainted goal tile,
    considering the closest robot, the travel distance, the color change cost,
    and the paint action cost. It also accounts for the cost of moving an obstructing
    robot if an adjacent tile is occupied.

    # Assumptions
    - The grid structure is regular and can be inferred from tile names like 'tile_R_C'.
    - Movement cost between adjacent tiles is 1 (Manhattan distance applies).
    - Color change cost is 1.
    - Paint action cost is 1.
    - Tiles painted with the wrong color cannot be repainted (unsolvable state or not possible in domain).
    - Tiles adjacent to a target tile are either clear or occupied by another robot.
      If occupied by another robot, that robot can move away in 1 action.
    - All tiles mentioned in static facts (up, down, left, right relations) or goals follow the 'tile_R_C' naming convention.

    # Heuristic Initialization
    - Parse static facts to identify all tiles and their (row, col) coordinates.
    - Store the mapping from tile name to coordinates and vice-versa.
    - Build an adjacency list for the grid based on static facts.
    - Store the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify all goal facts `(painted T C)` that are not currently true in the state. These are the unpainted goal tiles.
    2. If there are no unpainted goal tiles, the heuristic is 0.
    3. For each robot, find its current location and the color it holds from the current state.
    4. Initialize the total heuristic value `h` to 0.
    5. For each unpainted goal tile `(T, C)`:
        a. Find all tiles `AdjT` that are adjacent to `T` using the pre-built adjacency list.
        b. Calculate the minimum cost for *any* robot to paint tile `T` with color `C`. Initialize `min_cost_to_paint_T` to infinity.
        c. For each robot `R` with location `R_loc` and color `R_color`:
            i. Calculate the minimum travel cost for robot `R` to reach a *usable* tile `AdjT` adjacent to `T`. Initialize `min_travel_cost` to infinity. A usable tile is one that is clear or occupied by another robot. It cannot be a tile that is already painted.
            ii. For each adjacent tile `AdjT` of `T`:
                - Get coordinates for `R_loc` and `AdjT` using the pre-computed maps.
                - Calculate Manhattan distance: `dist = abs(R_row - AdjT_row) + abs(R_col - AdjT_col)`.
                - Check the state of `AdjT`:
                    - If `(painted AdjT SomeColor)` is in the state (check by iterating state facts), this tile cannot be used. Skip.
                    - If `(robot-at R' AdjT)` is in the state for some robot `R'` (where `R'` is not `R`):
                        - The cost to use this tile is `dist + 1` (1 for `R'` to move away).
                        - Update `min_travel_cost = min(min_travel_cost, dist + 1)`.
                    - If `(clear AdjT)` is in the state:
                        - The cost to use this tile is `dist`.
                        - Update `min_travel_cost = min(min_travel_cost, dist)`.
            iii. If `min_travel_cost` is finite (meaning there is at least one usable adjacent tile):
                - Calculate the color change cost: `color_cost = 1` if `R_color` is not `C`, otherwise `0`.
                - The total cost for robot `R` to paint `T` is `min_travel_cost + color_cost + 1` (the final +1 is for the paint action itself).
                - Update `min_cost_to_paint_T = min(min_cost_to_paint_T, cost_for_robot)`.
        d. If `min_cost_to_paint_T` is finite, add it to the total heuristic `h`.
    6. Return `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, grid structure, and adjacency."""
        self.goals = task.goals
        static_facts = task.static

        self.tile_coords = {}
        self.coords_tile = {}
        self.available_colors = set()
        self.adj_list = {} # tile_name -> set of adjacent tile_names

        # Helper to parse tile name into coordinates
        def parse_tile_name(tile_name):
            try:
                parts = tile_name.split('_')
                if len(parts) == 3 and parts[0] == 'tile':
                    row, col = int(parts[1]), int(parts[2])
                    return (row, col)
            except ValueError:
                pass
            return None # Return None if parsing fails

        # Populate tile_coords and coords_tile from static facts
        all_tile_names = set()
        for fact in static_facts:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate in ['up', 'down', 'left', 'right']:
                # Facts are like (relation tileA tileB)
                tile_a = parts[1]
                tile_b = parts[2]
                all_tile_names.add(tile_a)
                all_tile_names.add(tile_b)
            elif predicate == 'available-color':
                self.available_colors.add(parts[1])

        # Also add tiles from goals, just in case
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == 'painted':
                 all_tile_names.add(parts[1])

        # Parse coordinates for all collected tile names
        for tile_name in all_tile_names:
            coords = parse_tile_name(tile_name)
            if coords:
                self.tile_coords[tile_name] = coords
                self.coords_tile[coords] = tile_name

        # Build adjacency list using parsed coordinates
        for tile_name, (r, c) in self.tile_coords.items():
            self.adj_list[tile_name] = set()
            # Check 4 potential neighbors by coordinate
            potential_neighbors_coords = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
            for nr, nc in potential_neighbors_coords:
                if (nr, nc) in self.coords_tile:
                    neighbor_name = self.coords_tile[(nr, nc)]
                    self.adj_list[tile_name].add(neighbor_name)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unpainted goal tiles
        unpainted_goals = [] # List of (tile_name, target_color)
        # Pre-process state for quick lookups of painted tiles
        painted_status = {} # tile_name -> color or False (if clear/occupied)
        occupied_tiles = {} # tile_name -> robot_name
        robot_colors = {} # robot_name -> color
        robot_locations = {} # robot_name -> tile_name

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'painted':
                painted_status[parts[1]] = parts[2]
            elif parts[0] == 'robot-at':
                robot_locations[parts[1]] = parts[2]
                occupied_tiles[parts[2]] = parts[1]
            elif parts[0] == 'robot-has':
                robot_colors[parts[1]] = parts[2]
            # We don't strictly need 'clear' facts in the lookup dicts
            # as we can check `f'(clear {tile_name})' in state` directly.

        robot_info = {
            robot: (robot_locations[robot], robot_colors[robot])
            for robot in robot_locations if robot in robot_colors
        }


        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'painted':
                tile_name, target_color = parts[1], parts[2]
                # Check if the goal fact is NOT in the current state
                # A tile needs painting if the goal is not met AND it's not already painted correctly.
                # If it's painted with the wrong color, it's likely an unsolvable state,
                # but our heuristic will still try to estimate cost based on reaching it.
                # We add it to unpainted goals if the exact goal fact is missing.
                if goal not in state:
                     unpainted_goals.append((tile_name, target_color))


        # 2. If no unpainted goals, return 0
        if not unpainted_goals:
            return 0

        # 4. Calculate total heuristic
        total_h = 0

        for tile_name, target_color in unpainted_goals:
            # 5a. Find adjacent tiles
            # Use the pre-built adjacency list. Handle case where tile_name might not be in adj_list (e.g., if it's a goal tile not connected to anything, unlikely in valid problems).
            adj_tiles = self.adj_list.get(tile_name, set())

            # 5b. Calculate the minimum cost for *any* robot to paint tile `T` with color `C`.
            min_cost_to_paint_T = float('inf')

            for robot, (robot_loc, robot_color) in robot_info.items():
                robot_coords = self.tile_coords.get(robot_loc)
                if robot_coords is None: continue # Should not happen in valid states

                # 5c. Calculate the minimum travel cost for robot `R` to reach a *usable* tile `AdjT` adjacent to `T`.
                min_travel_cost = float('inf')

                for adj_tile in adj_tiles:
                    adj_tile_coords = self.tile_coords.get(adj_tile)
                    if adj_tile_coords is None: continue # Should not happen

                    # Check if the adjacent tile is usable (not painted, or occupied by another robot, or clear)
                    # If it's painted, we cannot move there.
                    if adj_tile in painted_status:
                        continue # Cannot move to a painted tile

                    dist = abs(robot_coords[0] - adj_tile_coords[0]) + abs(robot_coords[1] - adj_tile_coords[1])

                    # Check if the adjacent tile is occupied by another robot
                    if adj_tile in occupied_tiles and occupied_tiles[adj_tile] != robot:
                        # Cost is distance + 1 (for the other robot to move away)
                        travel_cost = dist + 1
                        min_travel_cost = min(min_travel_cost, travel_cost)
                    elif f'(clear {adj_tile})' in state:
                        # Tile is clear, cost is just distance
                        travel_cost = dist
                        min_travel_cost = min(min_travel_cost, travel_cost)
                    # else: Tile is not painted, not occupied by another robot, and not clear.
                    # This case implies the tile is occupied by *this* robot, which is impossible
                    # as the robot is at robot_loc, not adj_tile, unless robot_loc == adj_tile,
                    # but then dist would be 0, and the robot would be at an adjacent tile already.
                    # The logic covers clear and other-robot-occupied cases.

                # 5c iii. If min_travel_cost is finite (a usable adjacent tile exists)
                if min_travel_cost != float('inf'):
                    # Calculate color change cost
                    color_cost = 1 if robot_color != target_color else 0

                    # Total cost for this robot to paint this tile
                    cost_for_robot = min_travel_cost + color_cost + 1 # +1 for the paint action

                    # Update minimum cost for this tile
                    min_cost_to_paint_T = min(min_cost_to_paint_T, cost_for_robot)

            # 5d. Add the minimum cost for this tile to the total heuristic
            # If min_cost_to_paint_T is still infinity, it means no robot can reach a usable adjacent tile for this goal.
            # This goal might be unreachable in this state. We don't add infinity, effectively ignoring this goal
            # in the sum, which is acceptable for a non-admissible heuristic.
            if min_cost_to_paint_T != float('inf'):
                 total_h += min_cost_to_paint_T

        # 6. Return total heuristic
        return total_h

