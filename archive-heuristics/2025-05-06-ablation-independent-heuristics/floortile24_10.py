import re
import logging
from heuristics.heuristic_base import Heuristic
# Assuming Task class is available in the environment where this heuristic runs
# from task import Task

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the floortile domain.

    Summary:
    The heuristic estimates the cost to reach the goal by summing up the
    estimated costs for each unsatisfied goal tile that needs to be painted.
    For each such tile, it calculates the minimum cost required for any robot
    to reach a position from which it can paint the tile with the correct color,
    plus the cost of the paint action itself. The cost for a robot includes
    Manhattan distance movement cost and a cost of 1 if the robot needs to
    change color. Obstacles (painted tiles) are ignored in distance calculation.

    Assumptions:
    - Tile names follow the format 'tile_R_C' where R and C are integers
      representing row and column.
    - The grid structure defined by 'up', 'down', 'left', 'right' predicates
      corresponds to movement between adjacent cells in a grid.
    - For solvable instances, any goal tile (painted T C) that is not satisfied
      is currently clear. Tiles painted with the wrong color are assumed to
      indicate an unsolvable state within the scope of this heuristic.

    Heuristic Initialization:
    - Parses static facts to build a mapping from tile names to (row, col)
      coordinates and vice versa.
    - Parses static facts to build a mapping from each tile to its
      "paint-adjacent" tiles (tiles from which it can be painted).
    - Stores the set of goal facts.

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize total heuristic value `h = 0`.
    2. Identify the set of unsatisfied goal facts `(painted T C)` from the task goals
       that are not present in the current state.
    3. For each unsatisfied goal fact `(painted T C)`:
        a. Extract the tile name `T` and the required color `C`.
        b. Check the state of tile `T`. Iterate through the current state facts
           to see if `T` is painted with any color or if it is clear.
        c. If `T` is painted with *any* color other than `C`, it is considered a
           dead end in this domain (no unpaint action), so return infinity.
           (If it's painted with color `C`, it's a satisfied goal and already
           filtered in step 2).
        d. If `T` is clear (or assumed clear if not painted with any color),
           find its coordinates `(TR, TC)` using the pre-calculated map.
        e. Find the set of paint-adjacent tiles `Adj(T)` using the pre-calculated map.
        f. If `Adj(T)` is empty, the tile cannot be painted, return infinity.
        g. Initialize `min_cost_for_tile_T = infinity`.
        h. Identify the current position and color for each robot from the state.
        i. For each robot `R`:
            i. Get robot `R`'s current position `pos(R)` and color `color(R)`.
            ii. Calculate the cost `color_cost_R = 1` if `color(R)` is not the required color `C`, otherwise `0`.
            iii. Initialize `min_dist_R_to_paint_adj_T = infinity`.
            iv. For each tile `X` in `Adj(T)`:
                - Get `X`'s coordinates `(XR, XC)`.
                - Get robot `R`'s coordinates `(RR, RC)`.
                - Calculate Manhattan distance `dist = abs(RR - XR) + abs(RC - XC)`.
                - Update `min_dist_R_to_paint_adj_T = min(min_dist_R_to_paint_adj_T, dist)`.
            v. If `min_dist_R_to_paint_adj_T` is not infinity:
                - Calculate `cost_R_to_reach_and_color = min_dist_R_to_paint_adj_T + color_cost_R`.
                - Update `min_cost_for_tile_T = min(min_cost_for_tile_T, cost_R_to_reach_and_color)`.
        j. If `min_cost_for_tile_T` is still infinity (no robot can reach any paint-adjacent tile), return infinity.
        k. Otherwise, add `min_cost_for_tile_T + 1` (for the paint action) to the total heuristic `h`.
    4. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals
        self.static_facts = task.static

        # Pre-process static information
        self.tile_coords = {} # tile_name -> (row, col)
        self.coords_tile = {} # (row, col) -> tile_name
        self.paint_adjacent_map = {} # tile_name -> set of paint_adjacent_tile_names

        # Extract tile names and coordinates from static adjacency facts
        # Example: '(up tile_1_1 tile_0_1)'
        tile_pattern = re.compile(r"tile_(\d+)_(\d+)")
        adj_predicates = ['up', 'down', 'left', 'right']

        all_tiles = set()
        # Collect all tile names mentioned in adjacency predicates
        for fact in self.static_facts:
            parts = fact.strip('()').split()
            if len(parts) == 3 and parts[0] in adj_predicates:
                all_tiles.add(parts[1]) # tile_y_name
                all_tiles.add(parts[2]) # tile_x_name

        # Parse coordinates for all collected tiles
        for tile_name in all_tiles:
            match = tile_pattern.match(tile_name)
            if match:
                r, c = int(match.group(1)), int(match.group(2))
                self.tile_coords[tile_name] = (r, c)
                self.coords_tile[(r, c)] = tile_name
            else:
                 logging.warning(f"Tile name {tile_name} does not match expected pattern 'tile_R_C'.")

        # Initialize paint-adjacent map for all known tiles
        for tile_name in self.tile_coords:
             self.paint_adjacent_map[tile_name] = set()

        # Build paint-adjacent map from static adjacency facts
        for fact in self.static_facts:
            parts = fact.strip('()').split()
            if len(parts) == 3:
                pred = parts[0]
                tile_y_name = parts[1] # The tile being painted/moved to
                tile_x_name = parts[2] # The tile the robot is at

                # Robot at tile_x_name can paint tile_y_name if (pred tile_y_name tile_x_name) is true
                # So, tile_x_name is paint-adjacent to tile_y_name
                if pred in adj_predicates:
                     if tile_y_name in self.paint_adjacent_map:
                         self.paint_adjacent_map[tile_y_name].add(tile_x_name)
                     else:
                          # This tile_y_name was mentioned in a static fact but wasn't
                          # added to tile_coords? This indicates an issue with initial parsing.
                          logging.warning(f"Tile {tile_y_name} from fact {fact} not found in tile_coords during paint-adjacent map building.")


    def __call__(self, node):
        state = node.state
        h = 0
        state_set = set(state) # Convert frozenset to set for faster lookups

        # Find current robot positions and colors
        robot_info = {} # robot_name -> {'pos': tile_name, 'color': color_name}
        for fact in state_set:
            parts = fact.strip('()').split()
            if len(parts) == 3:
                if parts[0] == 'robot-at':
                    robot_name = parts[1]
                    tile_name = parts[2]
                    if robot_name not in robot_info:
                        robot_info[robot_name] = {}
                    robot_info[robot_name]['pos'] = tile_name
                elif parts[0] == 'robot-has':
                    robot_name = parts[1]
                    color_name = parts[2]
                    if robot_name not in robot_info:
                        robot_info[robot_name] = {}
                    robot_info[robot_name]['color'] = color_name

        # Iterate through goal facts
        for goal_fact in self.goals:
            # Check if goal fact is already satisfied
            if goal_fact in state_set:
                continue

            # Goal fact is not satisfied, must be (painted T C)
            # Assuming goal facts are only of the form (painted T C)
            parts = goal_fact.strip('()').split()
            if len(parts) != 3 or parts[0] != 'painted':
                 # Should not happen based on domain/problem structure
                 logging.warning(f"Unexpected goal fact format: {goal_fact}")
                 continue

            tile_name_T = parts[1]
            color_C = parts[2]

            # Check the state of tile T
            is_painted_wrong_color = False
            is_clear = False

            # Check if the tile is painted with *any* color
            painted_fact_prefix = f'(painted {tile_name_T} '
            is_painted = False
            for fact in state_set:
                 if fact.startswith(painted_fact_prefix):
                      is_painted = True
                      # If painted, check if it's the wrong color
                      if fact != goal_fact:
                           is_painted_wrong_color = True
                      break # Found painted status, no need to check other painted facts for this tile

            # If not painted, check if it's clear
            if not is_painted:
                 if f'(clear {tile_name_T})' in state_set:
                      is_clear = True
                 else:
                      # Tile is neither painted nor clear. Invalid state?
                      # In a valid floortile state, a tile is either clear or painted.
                      # If it's not painted (and not a goal state), it must be clear.
                      # We already checked if it's painted above. If we are here, it's not painted.
                      # So it must be clear.
                      is_clear = True # Assume clear if not painted

            if is_painted_wrong_color:
                 # Tile is painted with the wrong color. Assuming unsolvable in this domain.
                 return float('inf')

            # If we reach here, the tile needs painting and is assumed clear.

            tile_T_coords = self.tile_coords.get(tile_name_T)
            if tile_T_coords is None:
                 # Should not happen if tile is mentioned in goal and static facts
                 logging.warning(f"Goal tile {tile_name_T} not found in static facts/tile_coords map.")
                 return float('inf') # Or handle appropriately

            paint_adj_tiles_T = self.paint_adjacent_map.get(tile_name_T, set())
            if not paint_adj_tiles_T:
                 # No tile from which T can be painted. Likely unsolvable.
                 logging.warning(f"Goal tile {tile_name_T} has no paint-adjacent tiles defined in static facts.")
                 return float('inf')

            min_cost_for_tile_T = float('inf')

            # Find the minimum cost for any robot to paint tile T
            for robot_name, info in robot_info.items():
                robot_pos_name = info.get('pos')
                robot_color = info.get('color')

                if robot_pos_name is None or robot_color is None:
                    # Robot info incomplete (e.g., robot exists but not at a tile or has no color?)
                    logging.warning(f"Incomplete info for robot {robot_name} in state.")
                    continue

                robot_coords = self.tile_coords.get(robot_pos_name)
                if robot_coords is None:
                    logging.warning(f"Robot {robot_name} at unknown tile {robot_pos_name}.")
                    continue # Cannot calculate distance

                color_cost_R = 1 if robot_color != color_C else 0

                min_dist_R_to_paint_adj_T = float('inf')

                # Find minimum distance from robot's current position to any paint-adjacent tile of T
                for paint_adj_tile_name in paint_adj_tiles_T:
                    paint_adj_coords = self.tile_coords.get(paint_adj_tile_name)
                    if paint_adj_coords is None:
                        logging.warning(f"Paint-adjacent tile {paint_adj_tile_name} for {tile_name_T} not found in static facts/tile_coords map.")
                        continue # Cannot calculate distance

                    # Manhattan distance between robot's current position and the paint-adjacent tile
                    dist = abs(robot_coords[0] - paint_adj_coords[0]) + abs(robot_coords[1] - paint_adj_coords[1])
                    min_dist_R_to_paint_adj_T = min(min_dist_R_to_paint_adj_T, dist)

                if min_dist_R_to_paint_adj_T != float('inf'):
                    # Cost for this robot to reach a paint-adjacent tile and have the correct color
                    cost_R_to_reach_and_color = min_dist_R_to_paint_adj_T + color_cost_R
                    min_cost_for_tile_T = min(min_cost_for_tile_T, cost_R_to_reach_and_color)

            # If no robot can reach a paint-adjacent tile (e.g., grid disconnected, or no robots?)
            if min_cost_for_tile_T == float('inf'):
                 # This tile cannot be painted by any robot. Likely unsolvable.
                 return float('inf')

            # Add cost for this tile (min movement/color + paint action) to total heuristic
            h += min_cost_for_tile_T + 1

        return h
