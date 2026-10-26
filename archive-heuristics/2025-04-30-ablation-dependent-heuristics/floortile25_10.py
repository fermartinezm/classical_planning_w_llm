import re

class floortileHeuristic:
    """
    Domain-dependent heuristic for the Floortile domain.

    Summary:
        Estimates the cost to reach the goal by summing, for each unpainted
        goal tile, the minimum cost for any robot to paint that tile.
        The cost for a robot to paint a tile includes:
        1. Cost to change color if the robot doesn't have the required color (1 action).
        2. Estimated moves to reach a tile adjacent to the target tile. This is
           estimated using Manhattan distance on the grid: (Manhattan distance
           to target tile - 1) if distance > 0, and 1 if distance == 0.
           This estimate ignores obstacles (non-clear tiles).
        3. Cost of the paint action itself (1 action).

    Assumptions:
        - Tile names follow the pattern 'tile_row_col' where row and col are integers.
        - The grid structure implied by 'up', 'down', 'left', 'right' predicates
          corresponds to the row/col naming convention.
        - All goal tiles are reachable by at least one robot in a solvable problem
          instance, ignoring the 'clear' predicate for distance estimation.
        - There is at least one robot in the problem if the goal is not already met.
        - Goal states only require tiles to be painted; they do not impose
          conditions on robot locations or the 'clear' status of non-goal tiles.
        - Tiles that are goal tiles are initially clear and are not painted
          with a wrong color in any reachable state (as painting a painted tile
          is not possible).

    Heuristic Initialization:
        - Parses the goal facts to identify which tiles need to be painted
          and with which color, storing this in `self.goal_painted`.
        - Identifies all tile objects by parsing initial state and static facts.
        - Parses tile names ('tile_row_col') to extract row and column coordinates,
          storing this mapping in `self.tile_coords`.

    Step-By-Step Thinking for Computing Heuristic:
        1. Check if the current state is a goal state using `self.task.goal_reached(state)`. If yes, return 0.
        2. Initialize the total heuristic value `h` to 0.
        3. Extract the current location and color of each robot from the state facts. Store this in a dictionary `robot_info = {robot_name: {'loc': tile_name, 'color': color_name}}`.
        4. Extract the set of currently painted tiles and their colors from the state facts. Store this as a set of facts `current_painted_facts`.
        5. If there are no robots and the goal is not already met, the problem is unsolvable; return infinity.
        6. Iterate through each required goal painting `(tile, color)` stored in `self.goal_painted`.
        7. For the current goal tile `tile` and required `color`:
            a. Construct the goal fact string `'(painted tile color)'`.
            b. If this fact is already present in `current_painted_facts`, this goal is met for this tile; continue to the next goal tile.
            c. If the goal fact is not met, this tile needs to be painted. Find the minimum effort required by *any* robot to paint this tile.
            d. Initialize `min_effort_for_tile` to infinity.
            e. Get the coordinates `(tr, tc)` for the goal `tile` from `self.tile_coords`. If coordinates are missing (should not happen in valid instances based on assumptions), this tile is problematic. Skip this tile or handle as unreachable.
            f. Iterate through each robot `robot` and its information `info` in `robot_info`.
            g. Get the robot's current location `r_loc` and color `r_color` from `info`. Ensure both are present.
            h. Get the coordinates `(rlr, rlc)` for the robot's location `r_loc` from `self.tile_coords`. If coordinates are missing, skip this robot.
            i. Calculate the Manhattan distance between the robot's location and the target tile: `dist = abs(tr - rlr) + abs(tc - rlc)`.
            j. Estimate the number of move actions needed for the robot to reach a tile adjacent to the target tile (`X`) from which it can paint the target tile (`tile`).
               - If the robot is at the target tile (`dist == 0`), it needs 1 move to get to an adjacent tile `X`.
               - If the robot is adjacent to the target tile (`dist == 1`), it is already at an adjacent tile `X`, needing 0 moves.
               - If the robot is further (`dist > 1`), it needs `dist - 1` moves to reach an adjacent tile `X` on a shortest path towards `tile`.
               This can be summarized as `moves_to_adjacent = (dist - 1) if dist > 0 else 1`.
            k. Calculate the cost for the robot to have the correct color: `color_cost = 1` if `r_color` is not equal to the required `goal_color`, otherwise `0`.
            l. Calculate the total effort for this robot to be ready to paint (moves + color change): `effort = moves_to_adjacent + color_cost`.
            m. Update `min_effort_for_tile = min(min_effort_for_tile, effort)`.
            n. After checking all robots, if `min_effort_for_tile` is still infinity (implies no robots could paint this tile, likely due to missing coordinates or no robots existing), this part of the goal is unreachable.
            o. If `min_effort_for_tile` is infinity, return infinity immediately as the problem is unsolvable.
            p. Otherwise, add the minimum effort (`min_effort_for_tile`) plus the cost of the paint action (1) to the total heuristic value `h`.
        8. Return the final calculated heuristic value `h`.
    """
    def __init__(self, task):
        self.task = task
        self.goal_painted = {}
        self.tile_coords = {}
        self._parse_static_info()

    def _parse_static_info(self):
        """
        Parses static facts and initial state to extract goal information
        and tile coordinates from names.
        """
        # Extract goal painted facts
        for fact in self.task.goals:
            # Goal facts are expected to be of the form '(painted tile color)'
            parts = fact.strip("()").split()
            if len(parts) == 3 and parts[0] == 'painted':
                tile = parts[1]
                color = parts[2]
                self.goal_painted[tile] = color

        # Extract tile objects and parse coordinates from names
        all_objects = set()
        # Objects are listed in the instance file, but not directly in task.facts
        # We can find objects by looking at arguments of predicates in init/static/goal
        for fact in self.task.initial_state | self.task.static | self.task.goals:
             parts = fact.strip("()").split()
             # Skip predicate name (parts[0])
             for part in parts[1:]:
                 all_objects.add(part)

        tile_name_pattern = re.compile(r'tile_(\d+)_(\d+)')

        for obj in all_objects:
            match = tile_name_pattern.match(obj)
            if match:
                try:
                    row = int(match.group(1))
                    col = int(match.group(2))
                    self.tile_coords[obj] = (row, col)
                except ValueError:
                    # Should not happen if regex matches digits
                    pass # Ignore objects that don't match the tile pattern or have non-integer coords


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.
        """
        # Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        h = 0
        robot_info = {}
        current_painted_facts = set()

        # Extract robot locations and colors from the current state
        # Extract current painted tiles from the current state
        for fact in state:
            parts = fact.strip("()").split()
            if len(parts) == 3:
                if parts[0] == 'robot-at':
                    robot, tile = parts[1], parts[2]
                    if robot not in robot_info:
                        robot_info[robot] = {}
                    robot_info[robot]['loc'] = tile
                elif parts[0] == 'robot-has':
                    robot, color = parts[1], parts[2]
                    if robot not in robot_info:
                        robot_info[robot] = {}
                    robot_info[robot]['color'] = color
                elif parts[0] == 'painted':
                     current_painted_facts.add(fact)

        # If there are no robots, and the goal is not reached, the problem is unsolvable.
        # Return infinity in this case.
        if not robot_info and not self.task.goal_reached(state):
             return float('inf')


        # Iterate through goal painted tiles
        for goal_tile, goal_color in self.goal_painted.items():
            goal_fact = f'(painted {goal_tile} {goal_color})'

            # If the tile is already painted correctly, no cost
            if goal_fact in current_painted_facts:
                continue

            # Tile needs to be painted goal_color
            # Find the minimum effort for any robot to paint this tile
            min_effort_for_tile = float('inf')

            # Get target tile coordinates
            goal_coords = self.tile_coords.get(goal_tile)

            if goal_coords is None:
                 # Goal tile coordinates not found. Treat as unreachable.
                 # This implies an invalid instance or unexpected tile naming.
                 # If even one goal tile is unreachable, the problem is unsolvable.
                 return float('inf')

            (tr, tc) = goal_coords

            for robot, info in robot_info.items():
                # Ensure robot has location and color info in the current state
                if 'loc' not in info or 'color' not in info:
                    continue # Skip robots whose state is incomplete

                r_loc = info['loc']
                r_color = info['color']

                # Get robot location coordinates
                robot_coords = self.tile_coords.get(r_loc)

                if robot_coords is None:
                    # Robot is on a tile without coordinates? Skip this robot for this calculation.
                    continue

                (rlr, rlc) = robot_coords

                # Calculate Manhattan distance
                dist = abs(tr - rlr) + abs(tc - rlc)

                # Estimate moves needed to reach a tile adjacent to the target tile
                # This is the number of moves from R_loc to a tile X adjacent to T.
                # If dist == 0 (R at T), moves = 1 (move to adjacent X)
                # If dist > 0 (R not at T), moves = dist - 1 (moves towards T until adjacent X)
                moves_to_adjacent = (dist - 1) if dist > 0 else 1

                # Calculate color change cost
                color_cost = 1 if r_color != goal_color else 0

                # Total effort for this robot to be ready to paint (moves + color change)
                effort = moves_to_adjacent + color_cost

                min_effort_for_tile = min(min_effort_for_tile, effort)

            # If min_effort_for_tile is still inf, it means no robot could paint this tile.
            # This implies this part of the goal is unreachable by any robot found.
            # This could happen if robot_info was empty, or if all robots were on tiles
            # without coordinates, or if the goal tile had no coordinates (handled above).
            # If any goal tile is unreachable, the problem is unsolvable.
            if min_effort_for_tile == float('inf'):
                 return float('inf')
            else:
                 h += min_effort_for_tile + 1

        return h
