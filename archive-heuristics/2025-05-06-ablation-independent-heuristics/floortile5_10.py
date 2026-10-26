from heuristics.heuristic_base import Heuristic
import re
import logging

class floortileHeuristic(Heuristic):
    """
    Summary:
    This heuristic estimates the cost to reach the goal state in the floortile
    domain. It is domain-dependent and designed for greedy best-first search,
    so it is not necessarily admissible. The heuristic calculates the sum of
    estimated costs for each unsatisfied goal tile. For each unsatisfied goal
    tile, it finds the minimum cost among all robots to paint that tile with
    the required color. The cost for a robot to paint a tile includes the
    Manhattan distance the robot needs to travel to a position from which it
    can paint the tile, plus the cost of changing color if necessary, plus the
    cost of the paint action itself.

    Assumptions:
    - Tile names follow the format 'tile_R_C' where R and C are integers
      representing row and column.
    - The grid is connected as defined by the 'up', 'down', 'left', 'right'
      predicates in the static facts.
    - Problem instances are solvable, meaning no goal tile is initially painted
      with the wrong color. If a tile that needs to be painted is found to be
      painted with the wrong color, the heuristic returns infinity.
    - All colors required by the goal are available (implied by domain).

    Heuristic Initialization:
    The constructor pre-processes the static information from the task:
    1. It identifies all tiles and parses their names ('tile_R_C') to store
       their grid coordinates (R, C) in a dictionary `self.tile_coords`. It
       does this by iterating through all possible facts in the task, assuming
       any fact containing a tile name will reveal that tile.
    2. It builds an adjacency map `self.paint_from` where keys are tiles that
       need to be painted, and values are lists of tiles from which a robot
       can paint the key tile (based on 'up', 'down', 'left', 'right' static
       predicates and paint action definitions). Specifically, if `(up y x)`
       is a static fact, a robot at `x` can paint `y` using `paint_up`. So,
       `x` is added to the list for `y` in `self.paint_from`. Similar logic
       applies to 'down', 'left', and 'right'.
    3. It stores the goal facts.
    4. It stores available colors (though not strictly used in the current
       heuristic calculation, it's good practice to extract static info).

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Identify all goal facts of the form `(painted T C)`.
    2. For each such goal fact, check if it is satisfied in the current state.
       If `(painted T C)` is in the state, this goal is satisfied.
       If `(painted T C)` is NOT in the state:
         a. Check if the tile T is painted with a different color C'. Iterate
            through the state to find any fact `(painted T C')` where C' != C.
            If found, the state is a dead end. Return infinity.
         b. Check if the tile T is clear. If `(clear T)` is NOT in the state,
            and it's not painted correctly (checked in step 2a), it must be
            painted with the wrong color (based on assumptions). This case is
            covered by step 2a.
         c. If T is clear and not painted with the wrong color, it is an
            unsatisfied goal tile that needs painting with color C. Add T and C
            to a dictionary `unsatisfied_goals_dict`.
    3. Identify the current location and color of each robot from the state.
       Store this information in a dictionary `robot_info`.
    4. If there are no unsatisfied goals, the state is a goal state. Return 0.
    5. Initialize the total heuristic value to 0.
    6. For each tile `tile_T` and its required color `required_color_C` in
       `unsatisfied_goals_dict`:
         a. Initialize the minimum cost for this tile (`min_cost_for_tile`) to infinity.
         b. Get the list of tiles `possible_paint_from_tiles` from which `tile_T`
            can be painted (using the `self.paint_from` map). If this list is
            empty, the tile cannot be painted, implying an unsolvable state.
            Return infinity.
         c. Get the coordinates for `tile_T`. If not found, return infinity.
         d. For each robot `robot_R` and its information (`location`, `color`)
            in `robot_info`:
              i. Get the robot's current location `robot_location_L` and color
                 `robot_color`. If information is incomplete, skip this robot.
             ii. Get the coordinates for `robot_location_L`. If not found, skip.
            iii. Calculate the minimum Manhattan distance from `robot_location_L`
                 to any tile in `possible_paint_from_tiles`. Let this be
                 `min_dist_from_R_to_paint_pos`. If no valid paint-from tiles
                 had coordinates, this remains infinity; skip this robot for
                 this tile.
             iv. Calculate the cost for robot R to paint tile T:
                 `Cost(R, T) = min_dist_from_R_to_paint_pos + (1 if robot_color != required_color_C else 0) + 1`
                 (move cost + color change cost + paint action cost).
              v. Update `min_cost_for_tile = min(min_cost_for_tile, Cost(R, T))`.
         e. If, after checking all robots, `min_cost_for_tile` is still infinity,
            it means no robot can paint this tile. This implies an unsolvable
            state. Return infinity.
         f. Add `min_cost_for_tile` to the total heuristic value.
    7. Return the total heuristic value.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals
        self.tile_coords = {}
        # paint_from[tile_to_be_painted] = [tile_robot_must_be_at_1, tile_robot_must_be_at_2, ...]
        self.paint_from = {}
        self.available_colors = set()

        # Regex to parse tile names like 'tile_R_C'
        self.tile_name_pattern = re.compile(r"tile_(\d+)_(\d+)")

        # 1. Identify tiles and their coordinates
        # Iterate through all facts in the task to find tile objects
        # task.facts contains all possible ground facts in the domain
        for fact_str in task.facts:
            # Simple parsing: remove brackets and split
            parts = fact_str[1:-1].split()
            # Check if any part looks like a tile name
            for part in parts:
                match = self.tile_name_pattern.match(part)
                if match:
                    tile_name = part
                    row, col = int(match.group(1)), int(match.group(2))
                    self.tile_coords[tile_name] = (row, col)

        # 2. Build the paint_from map and find available colors
        # Iterate through static facts defining adjacency and available colors
        for fact_str in task.static:
            parts = fact_str[1:-1].split()
            if len(parts) == 3: # Adjacency predicates have 3 parts: predicate tile1 tile2
                pred, tile1, tile2 = parts
                # Paint actions: paint_DIR ?r ?y ?x ?c requires robot-at ?r ?x and DIR ?y ?x
                # So, robot at x paints y. We want map from y (tile to paint) to x (robot location)
                if pred in ['up', 'down', 'left', 'right']:
                    tile_to_paint = tile1 # ?y
                    robot_location_for_paint = tile2 # ?x
                    # Ensure both tiles exist in our coordinate map (they should if parsed from task.facts)
                    if tile_to_paint in self.tile_coords and robot_location_for_paint in self.tile_coords:
                         self.paint_from.setdefault(tile_to_paint, []).append(robot_location_for_paint)
                    else:
                         logging.warning(f"Static fact {fact_str} involves unknown tile(s).")

            elif len(parts) == 2: # available-color predicate has 2 parts: predicate color
                 pred, color_name = parts
                 if pred == 'available-color':
                    self.available_colors.add(color_name)


    def __call__(self, node):
        state = node.state
        unsatisfied_goals_dict = {} # {tile_name: color_name}
        robot_info = {} # {robot_name: {'location': tile_name, 'color': color_name}}

        # Check for dead ends and identify unsatisfied goals
        goal_tiles_set = set() # Keep track of tiles mentioned in goals
        for goal_fact_str in self.goals:
            # Only consider painted goals
            if goal_fact_str.startswith('(painted '):
                parts = goal_fact_str[1:-1].split()
                if len(parts) == 3:
                    _, tile_T, required_color_C = parts
                    goal_tiles_set.add(tile_T)
                    if goal_fact_str not in state:
                        # This goal is not satisfied. Add to unsatisfied list.
                        unsatisfied_goals_dict[tile_T] = required_color_C

        # Check for dead ends (tiles painted with the wrong color)
        for state_fact_str in state:
            if state_fact_str.startswith('(painted '):
                state_parts = state_fact_str[1:-1].split()
                if len(state_parts) == 3:
                    _, state_tile, state_color = state_parts
                    # If this painted tile is a goal tile AND the color is wrong
                    if state_tile in goal_tiles_set and state_tile in unsatisfied_goals_dict:
                         if state_color != unsatisfied_goals_dict[state_tile]:
                              # Tile is painted, it's a goal tile, and the color is not the required one
                              # This is a dead end
                              return float('inf')
                         # Note: If state_tile is a goal tile and painted with the correct color,
                         # it would not be in unsatisfied_goals_dict in the first place.
                         # If state_tile is NOT a goal tile, we don't care if it's painted.

        # Extract robot info
        for state_fact_str in state:
            parts = state_fact_str[1:-1].split()
            if len(parts) >= 2: # robot-at (2 args), robot-has (2 args)
                pred = parts[0]
                if pred == 'robot-at' and len(parts) == 3:
                    robot_name, tile_name = parts[1], parts[2]
                    if robot_name not in robot_info:
                        robot_info[robot_name] = {}
                    robot_info[robot_name]['location'] = tile_name
                elif pred == 'robot-has' and len(parts) == 3:
                    robot_name, color_name = parts[1], parts[2]
                    if robot_name not in robot_info:
                        robot_info[robot_name] = {}
                    robot_info[robot_name]['color'] = color_name
                # free-color is not used in actions, ignore it if present

        # If there are no unsatisfied goals, the heuristic is 0
        if not unsatisfied_goals_dict:
            return 0

        # Calculate total heuristic
        total_heuristic = 0

        for tile_T, required_color_C in unsatisfied_goals_dict.items():
            min_cost_for_tile = float('inf')

            # Get possible paint-from tiles for tile_T
            possible_paint_from_tiles = self.paint_from.get(tile_T, [])
            if not possible_paint_from_tiles:
                 # This tile cannot be painted from anywhere based on static facts.
                 # This implies an unsolvable state if it's a goal.
                 logging.warning(f"Goal tile {tile_T} cannot be painted from any known adjacent tile.")
                 return float('inf')

            coords_T = self.tile_coords.get(tile_T)
            if coords_T is None:
                 # Should not happen based on init, but safety check
                 logging.warning(f"Coordinates not found for goal tile {tile_T}.")
                 return float('inf')

            for robot_R, info in robot_info.items():
                robot_location_L = info.get('location')
                robot_color = info.get('color')

                if robot_location_L is None or robot_color is None:
                    # Robot state is incomplete, cannot use this robot
                    continue

                coords_L = self.tile_coords.get(robot_location_L)
                if coords_L is None:
                    logging.warning(f"Coordinates not found for robot location {robot_location_L}.")
                    continue

                # Calculate minimum distance from robot's current location
                # to any valid paint-from tile for tile_T
                min_dist_from_R_to_paint_pos = float('inf')
                for adj_tile in possible_paint_from_tiles:
                    coords_adj = self.tile_coords.get(adj_tile)
                    if coords_adj is None:
                         logging.warning(f"Coordinates not found for paint-from tile {adj_tile}.")
                         continue

                    dist = abs(coords_L[0] - coords_adj[0]) + abs(coords_L[1] - coords_adj[1])
                    min_dist_from_R_to_paint_pos = min(min_dist_from_R_to_paint_pos, dist)

                # If no valid paint-from tiles had coordinates, min_dist_from_R_to_paint_pos is still inf
                if min_dist_from_R_to_paint_pos == float('inf'):
                     continue # Cannot paint this tile from any known position using this robot

                # Calculate cost for this robot to paint this tile
                color_change_cost = 1 if robot_color != required_color_C else 0
                paint_action_cost = 1

                cost_R_T = min_dist_from_R_to_paint_pos + color_change_cost + paint_action_cost

                min_cost_for_tile = min(min_cost_for_tile, cost_R_T)

            # If min_cost_for_tile is still inf, it means no robot can paint this tile
            # This implies the state is unsolvable (e.g., no robots, or grid is disconnected)
            if min_cost_for_tile == float('inf'):
                 logging.warning(f"No robot can paint goal tile {tile_T}.")
                 return float('inf')

            total_heuristic += min_cost_for_tile

        return total_heuristic
