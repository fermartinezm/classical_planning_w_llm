class floortileHeuristic:
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with their specified colors. It considers the cost of getting the correct color,
    moving a robot to an adjacent tile, and performing the paint action for each
    tile that is not yet painted correctly. States where a goal tile is painted
    with the wrong color are penalized heavily as they represent dead ends.

    # Assumptions
    - Tiles are arranged in a grid, and their names follow the pattern 'tile_R_C'
      where R and C are integers representing row and column.
    - Movement between adjacent tiles (up, down, left, right) corresponds to
      unit distance in the grid (Manhattan distance).
    - A tile painted with the wrong color cannot be repainted unless it becomes clear,
      and there are no actions to make a painted tile clear. Thus, a goal tile
      painted with the wrong color indicates a dead end.
    - All colors required by the goal are available.
    - At least one robot exists in the problem.

    # Heuristic Initialization
    - Extracts the required color for each goal tile from the task's goal conditions.
    - Extracts the set of available colors from static facts.
    - Parses all tile names found in the initial state and static facts to create
      a mapping from tile name to its (row, column) coordinates for Manhattan distance calculation.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location and held color for each robot.
    2. Identify the current state of each tile (painted with which color, or clear).
    3. Initialize total heuristic cost to 0 and a penalty counter for wrongly painted goal tiles to 0.
    4. Iterate through each goal tile and its required color:
        a. Check if the tile is currently painted with the required color. If yes, this goal is satisfied for this tile; continue to the next goal tile.
        b. Check if the tile is currently painted with a *different* color. If yes, this tile is wrongly painted, indicating a dead end for this goal. Increment the penalty counter and continue to the next goal tile.
        c. If the tile is not painted with the required color (and not wrongly painted, implying it must be clear or unpainted, assumed clear for painting):
            i. Calculate the cost to paint this tile:
                - Paint action cost: 1.
                - Color cost: 1 if no robot currently holds the required color, 0 otherwise.
                - Movement cost: Find the minimum Manhattan distance from any robot's current location to *any* tile adjacent to the goal tile.
                    - If a robot is at the goal tile (distance 0), it needs 1 move to get adjacent.
                    - If a robot is distance `d > 0` away, it needs `d - 1` moves to get adjacent.
                    - The minimum of these values over all robots is the movement cost.
            ii. Add the paint cost, color cost, and minimum movement cost to the total heuristic cost.
    5. After iterating through all goal tiles, add a large penalty (e.g., 100 times the number of wrongly painted tiles) to the total cost.
    6. Return the total cost.
    """

    @staticmethod
    def get_parts(fact):
        """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
        # Handle potential empty fact string or malformed fact
        if not fact or not fact.startswith('(') or not fact.endswith(')'):
            return []
        return fact[1:-1].split()

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, available colors,
        and tile coordinates.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Extract goal tiles and their required colors
        self.goal_tiles = {}
        for goal in self.goals:
            parts = self.get_parts(goal)
            if parts and parts[0] == "painted":
                if len(parts) == 3: # Ensure format is (painted tile color)
                    self.goal_tiles[parts[1]] = parts[2]
                # else: print(f"Warning: Unexpected goal format: {goal}") # Suppress print in final code


        # Extract available colors
        self.available_colors = set()
        for fact in self.static_facts:
             parts = self.get_parts(fact)
             if parts and parts[0] == "available-color" and len(parts) == 2:
                 self.available_colors.add(parts[1])


        # Extract all tile names and parse coordinates
        all_tile_names = set()
        # Collect objects from initial state and static facts
        for fact in task.initial_state | self.static_facts:
             parts = self.get_parts(fact)
             for part in parts:
                 if part.startswith('tile_'):
                     all_tile_names.add(part)

        self.tile_coords = {}
        for name in all_tile_names:
            try:
                parts = name.split('_')
                # Expecting format like 'tile_R_C'
                if len(parts) == 3 and parts[0] == 'tile':
                    r = int(parts[1])
                    c = int(parts[2])
                    self.tile_coords[name] = (r, c)
            except (ValueError, IndexError):
                # Ignore terms that don't fit the expected tile name pattern
                pass

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0
        wrongly_painted_penalty = 0

        # Extract relevant state information
        robot_locs = {}
        robot_colors = {}
        current_painted_tiles = {} # {tile: color}
        current_clear_tiles = set()

        for fact in state:
            parts = self.get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "robot-at" and len(parts) == 3:
                robot_locs[parts[1]] = parts[2]
            elif predicate == "robot-has" and len(parts) == 3:
                robot_colors[parts[1]] = parts[2]
            elif predicate == "painted" and len(parts) == 3:
                current_painted_tiles[parts[1]] = parts[2]
            elif predicate == "clear" and len(parts) == 2:
                current_clear_tiles.add(parts[1])

        # Iterate through goal tiles and calculate cost/penalty
        for tile, required_color in self.goal_tiles.items():
            if tile in current_painted_tiles:
                actual_color = current_painted_tiles[tile]
                if actual_color == required_color:
                    # Goal satisfied for this tile
                    continue
                else:
                    # Tile painted with wrong color - dead end for this goal
                    wrongly_painted_penalty += 1
                    continue # Cannot paint this tile correctly from this state

            # If not painted correctly, check if it's clear and paintable
            if tile in current_clear_tiles:
                # Tile is clear and needs painting. Calculate cost.
                paint_cost = 1

                # Cost to get the required color
                color_cost = 1 # Assume need to change color
                has_color = False
                for robot, color in robot_colors.items():
                    if color == required_color:
                        has_color = True
                        break
                if has_color:
                    color_cost = 0
                # We assume required_color is available if it's in the goal.

                # Cost to move a robot adjacent to the tile
                min_move_cost = float('inf')
                tile_coords = self.tile_coords.get(tile)

                if tile_coords is None:
                     # Should not happen if init parsing is correct, but handle defensively
                     wrongly_painted_penalty += 1
                     continue # Cannot calculate move cost for unknown tile

                tile_r, tile_c = tile_coords

                if not robot_locs:
                    # No robots available to paint this tile
                    wrongly_painted_penalty += 1 # Treat as unsolvable tile
                    continue

                for robot, r_loc in robot_locs.items():
                    r_coords = self.tile_coords.get(r_loc)
                    if r_coords is None:
                        # Robot location not parsed? Should not happen.
                        continue # Skip this robot

                    r_r, r_c = r_coords
                    dist = abs(r_r - tile_r) + abs(r_c - tile_c)

                    # Moves needed to get adjacent: dist-1 if dist > 0, 1 if dist == 0
                    moves_to_adjacent = dist - 1 if dist > 0 else 1

                    min_move_cost = min(min_move_cost, moves_to_adjacent)

                # Add cost for this clear goal tile
                # Only add if a robot can actually reach it (min_move_cost is not inf)
                if min_move_cost != float('inf'):
                    total_cost += color_cost + min_move_cost + paint_cost
                else:
                    # Cannot reach tile with any robot (e.g., disconnected, no robots)
                    wrongly_painted_penalty += 1


            else:
                # Tile is a goal tile, not painted correctly, and not clear.
                # This implies it must be painted with a wrong color (covered by the first check)
                # or in an invalid state. Treat as wrongly painted.
                # This case should ideally not be reached if the first check is done correctly,
                # but serves as a fallback for robustness.
                 wrongly_painted_penalty += 1


        # Add penalty for wrongly painted tiles or unreachable clear goal tiles
        total_cost += wrongly_painted_penalty * 100 # Large penalty per wrong tile

        return total_cost
