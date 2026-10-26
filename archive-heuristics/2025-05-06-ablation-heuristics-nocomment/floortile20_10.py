from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Used for float('inf')

def get_parts(fact):
    """Helper function to split a PDDL fact string into its components."""
    # Remove surrounding parentheses and split by spaces
    return fact[1:-1].split()

def match(fact, *args):
    """Helper function to check if a fact matches a pattern."""
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments
    if len(parts) != len(args):
        return False
    # Use fnmatch for flexible matching (e.g., '*' wildcard)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Floortile domain.

    Estimates the cost to reach the goal state by summing up the minimum
    estimated costs for each unpainted or incorrectly painted goal tile.
    The cost for a single tile is estimated based on the distance of the
    closest robot, the need for a color change, and the paint action.
    Designed for greedy best-first search (non-admissible).
    """

    def __init__(self, task):
        """
        Initializes the heuristic by pre-calculating static information.

        Heuristic Initialization:
        - Stores the goal state requirements (which tiles need which color).
        - Parses tile names from static facts and goals to build a mapping
          from tile name strings (e.g., 'tile_1_2') to grid coordinates (row, col).
          Assumes tile names follow the 'tile_r_c' format.
        - Identifies available colors from static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Store goal requirements: {tile_name: required_color}
        self.goal_tiles = {}
        for goal in self.goals:
            # Goal facts are typically (painted tile color)
            if match(goal, "painted", "*", "*"):
                _, tile, color = get_parts(goal)
                self.goal_tiles[tile] = color

        # 2. Build tile coordinates map: {tile_name: (row, col)}
        self.tile_coords = {}
        all_potential_tile_names = set()
        # Collect all potential tile names from static facts and goals
        for fact in static_facts | self.goals:
             parts = get_parts(fact)
             for part in parts:
                 if part.startswith('tile_'):
                     all_potential_tile_names.add(part)

        # Parse coordinates for all identified tiles
        for obj_name in all_potential_tile_names:
            if obj_name.startswith('tile_'):
                try:
                    # Assumes tile names are in the format 'tile_row_col'
                    parts = obj_name.split('_')
                    # Ensure there are enough parts and the row/col are digits
                    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                        row = int(parts[1])
                        col = int(parts[2])
                        self.tile_coords[obj_name] = (row, col)
                    else:
                         # Handle unexpected tile name format
                         print(f"Warning: Unexpected tile name format, cannot parse coordinates: {obj_name}")
                except (ValueError, IndexError):
                    # Handle potential errors during parsing
                    print(f"Warning: Error parsing coordinates for tile: {obj_name}")
                    pass # Skip tiles with unparseable names

        # 3. Identify available colors
        self.available_colors = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "available-color", "*")
        }

    def manhattan_distance(self, tile1_name, tile2_name):
        """Calculates the Manhattan distance between two tiles."""
        if tile1_name not in self.tile_coords or tile2_name not in self.tile_coords:
            # This indicates a tile name was found in state/goals but not in static/goals during init parsing.
            # Should ideally not happen in valid PDDL instances, but handle defensively.
            # Returning infinity implies this tile is unreachable/problematic.
            return float('inf')

        r1, c1 = self.tile_coords[tile1_name]
        r2, c2 = self.tile_coords[tile2_name]
        return abs(r1 - r2) + abs(c1 - c2)

    def __call__(self, node):
        """
        Computes the heuristic value for the given state.

        Step-By-Step Thinking for Computing Heuristic:
        1. Identify the current location and color of each robot in the state.
        2. Identify the current painted status of each tile in the state.
        3. Initialize the total heuristic value to 0.
        4. Iterate through each goal tile (T) and its required color (C) from `self.goal_tiles`.
        5. For the current goal tile (T) and required color (C):
           a. Check if the tile T is already painted with color C in the current state.
              If yes, this goal is satisfied for this tile; continue to the next goal tile.
           b. Check if the tile T is painted with *any other* color in the current state.
              If yes, the state is likely unsolvable in this domain (no unpaint action).
              Return infinity as the heuristic value to prune this branch.
           c. If the tile T is not painted correctly (i.e., it's clear or painted wrong,
              and we've handled the wrong color case), it needs to be painted.
              Initialize a variable `min_cost_for_tile` to infinity. This will store
              the minimum estimated cost to paint this specific tile by any robot.
           d. Iterate through each robot (R) and its current location (L_R) and color (C_R)
              obtained from the state.
              i. Calculate the cost for robot R to change color to C, if needed:
                 `color_change_cost = 1` if `C_R != C`, otherwise `0`.
              ii. Calculate the Manhattan distance `d` between the robot's location L_R
                  and the goal tile T using the pre-calculated coordinates.
              iii. Estimate the number of move actions required for robot R to get
                   to a tile adjacent to T from which it can paint T.
                   - If the robot is currently *at* tile T (`d == 0`), it must first
                     move to a neighbor (1 move) before painting.
                   - If the robot is not at tile T (`d > 0`), it needs `d - 1` moves
                     to reach a tile adjacent to T.
                   Let `moves_to_adjacent = 1` if `d == 0`, else `d - 1`.
              iv. The paint action itself costs 1.
              v. The total estimated cost for robot R to paint tile T is
                 `color_change_cost + moves_to_adjacent + 1`.
              vi. Update `min_cost_for_tile = min(min_cost_for_tile, total_cost_R)`.
           e. After checking all robots, if `min_cost_for_tile` is still infinity,
              it means this goal tile is unreachable (e.g., no robots, or missing
              coordinate data). Return infinity.
           f. Add `min_cost_for_tile` to the `total_heuristic`.
        6. Return the `total_heuristic`.

        Assumptions:
        - Tile names follow the format 'tile_row_col' allowing coordinate extraction.
        - Solvable instances do not require unpainting tiles; goal tiles are either
          clear or painted with the correct color in solvable states. If a goal tile
          is found painted with the wrong color, the heuristic returns infinity.
        - The grid structure defined by 'up', 'down', 'left', 'right' predicates
          corresponds to a regular grid where Manhattan distance is a reasonable
          estimate for movement cost to an adjacent tile.
        - The heuristic calculates the cost for each unsatisfied goal tile
          independently, considering the best robot for that tile, without
          considering potential conflicts or shared costs between tiles or robots.
        """
        state = node.state

        # 1 & 2. Get current robot locations/colors and tile painted status
        robot_info = {} # {robot_name: {'loc': tile_name, 'color': color_name}}
        current_painted_tiles = {} # {tile_name: color_name}

        for fact in state:
            parts = get_parts(fact)
            if match(fact, "robot-at", "*", "*"):
                _, robot, tile = parts
                if robot not in robot_info:
                    robot_info[robot] = {'loc': None, 'color': None}
                robot_info[robot]['loc'] = tile
            elif match(fact, "robot-has", "*", "*"):
                _, robot, color = parts
                if robot not in robot_info:
                    robot_info[robot] = {'loc': None, 'color': None}
                robot_info[robot]['color'] = color
            elif match(fact, "painted", "*", "*"):
                _, tile, color = parts
                current_painted_tiles[tile] = color
            # We don't strictly need clear tiles list for this heuristic logic

        total_heuristic = 0

        # 4. Iterate through each goal tile
        for goal_tile, required_color in self.goal_tiles.items():
            # 5a. Check if already satisfied
            if goal_tile in current_painted_tiles and current_painted_tiles[goal_tile] == required_color:
                continue # Goal already satisfied for this tile

            # 5b. Check if painted with wrong color (unsolvable state assumption)
            if goal_tile in current_painted_tiles and current_painted_tiles[goal_tile] != required_color:
                 # If a goal tile is wrongly painted, it's likely unsolvable.
                 # Return infinity to guide search away.
                 return float('inf')

            # 5c. Tile needs painting (it must be clear if solvable and not painted correctly)
            min_cost_for_tile = float('inf')

            # 5d. Iterate through each robot
            for robot, info in robot_info.items():
                robot_loc = info.get('loc') # Use .get for safety, though should be present
                robot_color = info.get('color') # Use .get for safety

                if robot_loc is None or robot_color is None:
                    # Skip robots whose location or color isn't fully known in the state (shouldn't happen in valid states)
                    continue

                # i. Color change cost
                color_change_cost = 1 if robot_color != required_color else 0

                # ii. Calculate Manhattan distance
                dist = self.manhattan_distance(robot_loc, goal_tile)
                if dist == float('inf'):
                     # Cannot calculate distance (e.g., missing tile coords)
                     continue # Skip this robot for this tile

                # iii. Estimate moves to get adjacent
                # If robot is at the target tile (dist 0), it needs 1 move away to a neighbor.
                # If robot is not at the target tile (dist > 0), it needs dist - 1 moves to reach a neighbor.
                moves_to_adjacent = 1 if dist == 0 else dist - 1
                # Ensure moves is non-negative (only relevant if dist=0, handled above)
                moves_to_adjacent = max(0, moves_to_adjacent)


                # v. Total estimated cost for this robot for this tile
                # Cost = Color Change + Moves to Adjacent + Paint Action
                cost_R = color_change_cost + moves_to_adjacent + 1

                # vi. Update minimum cost for this tile
                min_cost_for_tile = min(min_cost_for_tile, cost_R)

            # 5e. Add minimum cost for this tile to total heuristic
            if min_cost_for_tile == float('inf'):
                 # This tile is unreachable by any robot (e.g., no robots found, or missing tile coords)
                 # In a solvable state, this shouldn't happen. Return inf.
                 return float('inf')

            total_heuristic += min_cost_for_tile

        # 6. Return total heuristic
        return total_heuristic
