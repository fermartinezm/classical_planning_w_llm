import math
from heuristics.heuristic_base import Heuristic

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Floortile domain.

    Summary:
        This heuristic estimates the cost to reach the goal by summing the estimated
        cost for each unpainted goal tile. For each unpainted goal tile, it calculates
        the minimum cost among all robots to paint that tile. The cost for a robot
        to paint a tile is estimated as:
        1 (for the paint action)
        + 1 (if the robot needs to change color)
        + Manhattan distance from the robot's current location to the closest tile
          adjacent to the target tile.
        This heuristic is non-admissible as it ignores potential conflicts between
        robots and assumes movement is always possible along the shortest path
        (ignoring the 'clear' predicate for movement).

    Assumptions:
        - Tile names follow the format 'tile_R_C' where R and C are integers representing row and column.
        - The grid structure is defined by 'up', 'down', 'left', 'right' predicates in static facts.
        - Goal states only require painting tiles, not clearing them or changing colors of already painted tiles.
        - Solvable instances do not require repainting a tile that is already painted with the wrong color.
        - Robots always hold a color.

    Heuristic Initialization:
        - Parses static facts to build a map from tile names ('tile_R_C') to their (row, col) coordinates.
        - Parses static facts to build an adjacency map for tiles based on 'up', 'down', 'left', 'right' predicates.
        - Parses goal facts to identify the set of tiles that need to be painted and their target colors.

    Step-By-Step Thinking for Computing Heuristic:
        1. Initialize the total heuristic value `h` to 0.
        2. Identify the current location and color held by each robot in the current state.
        3. Identify the set of tiles that are already painted in the current state.
        4. Iterate through each goal requirement `(goal_tile, goal_color)` stored during initialization.
        5. If `goal_tile` is not yet painted with `goal_color` in the current state:
            a. Initialize `min_cost_for_tile` to infinity.
            b. Find the set of tiles adjacent to `goal_tile` using the precomputed adjacency map.
            c. Iterate through each robot:
                i. Get the robot's current location (`robot_loc`) and color (`robot_color`).
                ii. Initialize `robot_cost` to 0.
                iii. If `robot_color` is not equal to `goal_color`, add 1 to `robot_cost` (for the change_color action).
                iv. Initialize `min_move_cost` to infinity.
                v. Iterate through each tile adjacent to `goal_tile`:
                    - Calculate the Manhattan distance between `robot_loc` and the adjacent tile.
                    - Update `min_move_move_cost` with the minimum distance found so far.
                vi. If `min_move_cost` is still infinity (meaning no adjacent tiles or distance calculation failed), this tile might be unreachable by this robot in this state (or the grid is weird). Handle this case (e.g., continue to next robot or set robot_cost to infinity). Assuming valid grids, min_move_cost will be finite for connected components.
                vii. Add `min_move_cost` to `robot_cost`.
                viii. Add 1 to `robot_cost` (for the paint action).
                ix. Update `min_cost_for_tile` with the minimum of its current value and `robot_cost`.
            d. If `min_cost_for_tile` is still infinity after checking all robots, the goal is likely unreachable (e.g., no robots, or no path). This should ideally not happen in solvable instances. For safety, we return infinity.
            e. Add `min_cost_for_tile` to the total heuristic value `h`.
        6. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static_facts = task.static

        # Data structures for precomputation
        self.tile_coords = {} # tile_name -> (row, col)
        self.tile_adj_map = {} # tile_name -> set of adjacent_tile_names
        self.goal_painted_tiles = set() # set of (tile_name, color_name)

        self._precompute_static_info()
        self._parse_goals()

    def _parse_fact(self, fact):
        """Removes parentheses and splits a PDDL fact string into predicate and arguments."""
        # Example: '(predicate arg1 arg2)' -> ('predicate', ['arg1', 'arg2'])
        parts = fact[1:-1].split()
        if not parts:
            return None, []
        return parts[0], parts[1:]

    def _parse_tile_name(self, tile_name):
        """Parses a tile name like 'tile_R_C' into (row, col) integer coordinates."""
        try:
            parts = tile_name.split('_')
            if len(parts) == 3 and parts[0] == 'tile':
                return (int(parts[1]), int(parts[2]))
        except (ValueError, IndexError):
            # print(f"Warning: Could not parse tile name {tile_name}") # Optional debug
            pass
        return None # Indicate parsing failed

    def _manhattan_distance(self, tile1_name, tile2_name):
        """Calculates the Manhattan distance between two tiles based on their coordinates."""
        coord1 = self.tile_coords.get(tile1_name)
        coord2 = self.tile_coords.get(tile2_name)
        if coord1 is None or coord2 is None:
            # This implies a tile name from state/goals wasn't found in static facts,
            # which shouldn't happen in valid problems, but handle defensively.
            return math.inf
        r1, c1 = coord1
        r2, c2 = coord2
        return abs(r1 - r2) + abs(c1 - c2)

    def _precompute_static_info(self):
        """Extracts tile coordinates and adjacency information from static facts."""
        all_tiles = set()
        for fact in self.static_facts:
            pred, args = self._parse_fact(fact)
            if pred in ['up', 'down', 'left', 'right'] and len(args) == 2:
                # e.g., (up tile_1_1 tile_0_1) means tile_1_1 is adjacent to tile_0_1
                tile1, tile2 = args
                all_tiles.add(tile1)
                all_tiles.add(tile2)
                self.tile_adj_map.setdefault(tile1, set()).add(tile2)
                self.tile_adj_map.setdefault(tile2, set()).add(tile1) # Adjacency is symmetric
            # Add other static predicates if needed, e.g., available-color
            # For this heuristic, only grid structure is needed from static facts.

        # Parse coordinates for all found tiles
        for tile_name in all_tiles:
            coords = self._parse_tile_name(tile_name)
            if coords is not None:
                self.tile_coords[tile_name] = coords
            # else: print(f"Warning: Could not parse coordinates for tile {tile_name}") # Optional debug

    def _parse_goals(self):
        """Extracts the required painted state for goal tiles."""
        # Goals are typically a list or set of facts
        # If the goal is a conjunction (and ...), task.goals is a set of facts
        # If the goal is a single fact, task.goals is that fact string
        goal_facts = self.goals
        if isinstance(goal_facts, str): # Handle single goal fact
             goal_facts = {goal_facts}

        for goal_fact in goal_facts:
            pred, args = self._parse_fact(goal_fact)
            if pred == 'painted' and len(args) == 2:
                tile, color = args
                self.goal_painted_tiles.add((tile, color))
            # else: print(f"Warning: Unexpected goal fact format: {goal_fact}") # Optional debug

    def __call__(self, node):
        """
        Computes the heuristic value for the given state.

        Args:
            node: The search node containing the state.

        Returns:
            An estimate of the cost to reach the goal state.
        """
        state = node.state

        # 2. Identify current robot location and color
        robot_info = {} # robot_name -> {'location': tile_name, 'color': color_name}
        for fact in state:
            pred, args = self._parse_fact(fact)
            if pred == 'robot-at' and len(args) == 2:
                robot, location = args
                robot_info.setdefault(robot, {})['location'] = location
            elif pred == 'robot-has' and len(args) == 2:
                robot, color = args
                robot_info.setdefault(robot, {})['color'] = color

        # 3. Identify currently painted tiles
        current_painted_tiles = set() # set of (tile_name, color_name)
        for fact in state:
            pred, args = self._parse_fact(fact)
            if pred == 'painted' and len(args) == 2:
                tile, color = args
                current_painted_tiles.add((tile, color))

        # 1. Initialize heuristic
        h = 0

        # 4. Iterate through each goal requirement
        for goal_tile, goal_color in self.goal_painted_tiles:
            # 5. If goal_tile is not yet painted correctly
            if (goal_tile, goal_color) not in current_painted_tiles:
                min_cost_for_tile = math.inf

                # Check if the tile is painted with the wrong color
                is_wrongly_painted = False
                for fact in state:
                    pred, args = self._parse_fact(fact)
                    if pred == 'painted' and len(args) == 2 and args[0] == goal_tile and args[1] != goal_color:
                         is_wrongly_painted = True
                         break

                if is_wrongly_painted:
                    # Problem is likely unsolvable if a goal tile is painted wrong
                    return math.inf

                # Find adjacent tiles for the goal tile
                adj_tiles = self.tile_adj_map.get(goal_tile, set())

                if not adj_tiles:
                     # If a goal tile has no adjacent tiles, it cannot be painted. Unsolvable.
                     # This indicates a problem with the PDDL instance/grid definition.
                     # print(f"Warning: Goal tile {goal_tile} has no adjacent tiles.") # Optional debug
                     return math.inf

                # c. Iterate through each robot
                for robot_name, info in robot_info.items():
                    robot_loc = info.get('location')
                    robot_color = info.get('color')

                    # Ensure robot info is complete (should be if state is valid)
                    if robot_loc is None or robot_color is None:
                        # print(f"Warning: Incomplete info for robot {robot_name} in state.") # Optional debug
                        continue # Skip this robot

                    robot_cost = 0
                    # iii. If robot needs to change color
                    if robot_color != goal_color:
                        robot_cost += 1 # change_color action cost

                    # iv. Find min move cost to an adjacent tile
                    min_move_cost = math.inf
                    for adj_tile in adj_tiles:
                        dist = self._manhattan_distance(robot_loc, adj_tile)
                        min_move_cost = min(min_move_cost, dist)

                    # Check if a path exists (Manhattan distance is finite)
                    if min_move_cost == math.inf:
                         # Cannot reach any adjacent tile (e.g., disconnected grid)
                         # This robot cannot paint this tile.
                         continue # Skip this robot

                    # vii. Add move cost
                    robot_cost += min_move_cost
                    # viii. Add paint action cost
                    robot_cost += 1

                    # ix. Update min_cost_for_tile
                    min_cost_for_tile = min(min_cost_for_tile, robot_cost)

                # d. Check if any robot can paint the tile
                if min_cost_for_tile == math.inf:
                    # No robot can reach and paint this tile. Unsolvable.
                    return math.inf

                # e. Add min cost for this tile to total
                h += min_cost_for_tile

        # 6. Return total heuristic value
        return h
