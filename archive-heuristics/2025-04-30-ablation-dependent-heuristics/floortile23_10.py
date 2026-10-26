from collections import deque
import math

class floortileHeuristic:
    """
    Domain-dependent heuristic for the floortile domain.

    Summary:
        Estimates the cost to reach the goal state by summing the estimated
        costs for each tile that needs to be painted according to the goal
        but is not yet painted correctly in the current state.
        For each such tile, the estimated cost is the minimum cost among all
        robots to reach a position from which the tile can be painted, plus
        the cost to change color if necessary, plus the paint action cost.
        The movement cost for a robot is estimated using BFS on the grid,
        considering only currently clear tiles as traversable destinations
        (except the robot's starting tile, which is traversable from).
        Tiles occupied by robots that need painting incur an additional cost
        of 1 (for the robot to move off) before they can be painted.

    Assumptions:
        - The PDDL domain follows the structure described (tiles named tile_r_c,
          up/down/left/right predicates define a grid).
        - The grid is connected.
        - Solvable problems do not have goal tiles initially painted with the wrong color.
        - A robot always holds exactly one color.
        - The predicates (robot-at r x) and (clear x) are mutually exclusive for any given tile x.

    Heuristic Initialization:
        The constructor precomputes static information from the task, including:
        - Mapping between tile names (strings like 'tile_r_c') and grid coordinates (tuples like (r, c)).
        - Adjacency list for the grid graph based on the up/down/left/right predicates.
        - The set of available colors.
        - The set of robot names.
        - The goal state requirements (which tiles need which color).

    Step-By-Step Thinking for Computing Heuristic:
        1. Initialize total heuristic value `h` to 0.
        2. Extract current state information: robot locations, robot colors,
           painted tiles (and their colors), and clear tiles.
        3. Identify tiles that need painting: Iterate through the goal requirements.
           For each tile specified in the goal:
           - Check if it is already painted with the correct color in the current state. If yes, it contributes 0 to the heuristic.
           - Check if it is painted with a wrong color. If yes, the problem is likely unsolvable in this domain; return infinity.
           - If it is not painted correctly, add it to a dictionary of tiles needing paint, mapping tile name to the required color.
        4. If any tile was painted with the wrong color, return infinity.
        5. If no tiles need painting, the current state is a goal state; return 0.
        6. For each tile `T` that needs painting with color `C` (from the dictionary created in step 3):
           - Determine if `T` is currently occupied by a robot. If yes, add 1 to the total heuristic `h` (representing the cost to move the robot off `T` to make it clear).
           - Calculate the minimum cost for *any* robot to paint tile `T` with color `C`, assuming `T` is clear (or becomes clear after the move-off action if it was occupied).
             - Initialize minimum robot paint cost for tile `T` to infinity.
             - Find the set of tiles adjacent to `T`.
             - Identify which of these adjacent tiles are currently `clear`. These are the potential target locations for a robot to paint `T` from.
             - If there are no clear adjacent tiles, no robot can paint `T` in the current state; this tile is unreachable. Mark the problem as unsolvable and break the loop.
             - For each robot `R` at location `L` with color `Color_R`:
               - Calculate color change cost: 1 if `Color_R` is not `C`, else 0.
               - Calculate movement cost: Perform a Breadth-First Search (BFS) on the grid graph starting from `L`. The BFS can only traverse edges leading to tiles that are currently `clear` in the state. The target nodes for the BFS are the clear adjacent tiles identified earlier. The movement cost is the shortest distance found by the BFS to any such target tile. If no clear adjacent tile is reachable via clear tiles from `L`, the movement cost is infinity for this robot.
               - If movement cost is finite, calculate the total cost for robot `R` to paint `T` (assuming `T` is clear): `color_change_cost + movement_cost + 1` (the +1 is for the paint action itself).
               - Update the minimum robot paint cost for tile `T` with the minimum found across all robots.
           - If the minimum robot paint cost for tile `T` is still infinity (meaning no robot could reach a clear adjacent tile), the state might be a dead end; mark the problem as unsolvable and break the loop.
           - Add the calculated minimum robot paint cost for tile `T` to the total heuristic `h`.
        7. If the problem was marked as unsolvable during the loop, return infinity.
        8. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic with static task information.

        Args:
            task: The planning task object.
        """
        self.task = task
        self.tile_coords = {} # tile_name -> (row, col)
        self.coord_tiles = {} # (row, col) -> tile_name
        self.adjacency = {} # tile_name -> list of adjacent tile_names
        self.goal_dict = {} # tile_name -> goal_color
        self.available_colors = set()
        self.robots = set() # set of robot names

        # Collect all facts that might contain object names
        all_relevant_facts = task.initial_state | task.static | task.goals

        tile_names_set = set()
        for fact in all_relevant_facts:
             parts = fact.replace('(', '').replace(')', '').split()
             for part in parts:
                 if part.startswith('tile_'):
                     tile_names_set.add(part)
                 # Simple way to identify robots from common names
                 elif part.startswith('robot'):
                     self.robots.add(part)

        # Refine available colors from static facts (most reliable source)
        self.available_colors = set()
        for fact in task.static:
            if fact.startswith('(available-color '):
                 parts = fact.split()
                 if len(parts) > 1: self.available_colors.add(parts[1][:-1]) # Remove ')'

        # Process tile names to get coords and populate adjacency structure
        for tile_name in tile_names_set:
            try:
                # Assuming tile names are strictly 'tile_r_c' format
                _, r_str, c_str = tile_name.split('_')
                r, c = int(r_str), int(c_str)
                self.tile_coords[tile_name] = (r, c)
                self.coord_tiles[(r, c)] = tile_name
                if tile_name not in self.adjacency:
                    self.adjacency[tile_name] = []
            except ValueError:
                # Ignore parts that look like tiles but don't follow the format
                pass

        # Build adjacency list from static facts
        for fact in task.static:
            parts = fact.replace('(', '').replace(')', '').split()
            if len(parts) == 3 and parts[0] in ['up', 'down', 'left', 'right']:
                pred, tile1, tile2 = parts
                # (pred y x): y is relative to x. Robot at x paints y. Robot moves x to y.
                # Adjacency is symmetric.
                if tile1 in self.adjacency and tile2 in self.adjacency: # Ensure tiles were parsed
                     if tile2 not in self.adjacency[tile1]:
                         self.adjacency[tile1].append(tile2)
                     if tile1 not in self.adjacency[tile2]:
                         self.adjacency[tile2].append(tile1)

        # Parse goal facts
        for goal_fact in task.goals:
            if goal_fact.startswith('(painted '):
                parts = goal_fact.replace('(', '').replace(')', '').split()
                if len(parts) == 3:
                    _, tile_name, color_name = parts
                    if tile_name in self.tile_coords: # Ensure it's a valid tile
                        self.goal_dict[tile_name] = color_name

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        Args:
            state: The current state (frozenset of facts).

        Returns:
            The estimated number of actions to reach the goal, or math.inf
            if the state is estimated to be unsolvable.
        """
        # Extract current state information
        current_robot_locations = {} # robot_name -> tile_name
        current_robot_colors = {} # robot_name -> color_name
        current_painted_tiles = {} # tile_name -> color_name
        current_clear_tiles = set() # set of tile_names that are clear

        for fact in state:
            if fact.startswith('(robot-at '):
                parts = fact.split()
                if len(parts) == 3:
                    robot_name = parts[1]
                    tile_name = parts[2][:-1] # Remove ')'
                    current_robot_locations[robot_name] = tile_name
            elif fact.startswith('(robot-has '):
                parts = fact.split()
                if len(parts) == 3:
                    robot_name = parts[1]
                    color_name = parts[2][:-1] # Remove ')'
                    current_robot_colors[robot_name] = color_name
            elif fact.startswith('(painted '):
                parts = fact.split()
                if len(parts) == 3:
                    tile_name = parts[1]
                    color_name = parts[2][:-1] # Remove ')'
                    current_painted_tiles[tile_name] = color_name
            elif fact.startswith('(clear '):
                parts = fact.split()
                if len(parts) == 2:
                    tile_name = parts[1][:-1] # Remove ')'
                    current_clear_tiles.add(tile_name)

        # Identify tiles that need painting
        tiles_needing_paint_info = {} # {tile_name: goal_color}
        unsolvable = False

        for tile_name, goal_color in self.goal_dict.items():
            is_painted_correctly = False
            is_painted_wrongly = False

            if tile_name in current_painted_tiles:
                if current_painted_tiles[tile_name] == goal_color:
                    is_painted_correctly = True
                else:
                    is_painted_wrongly = True

            if is_painted_wrongly:
                unsolvable = True
                break
            elif not is_painted_correctly:
                tiles_needing_paint_info[tile_name] = goal_color

        if unsolvable:
            return math.inf # Use math.inf for infinity

        # If no tiles need painting, it's a goal state
        if not tiles_needing_paint_info:
            return 0

        # Calculate cost for each tile needing paint
        total_estimated_cost = 0

        for tile_name, goal_color in tiles_needing_paint_info.items():

            # Check if the tile is currently occupied by a robot
            is_occupied = False
            for robot_loc in current_robot_locations.values():
                if robot_loc == tile_name:
                    is_occupied = True
                    break

            # Cost to make the tile clear if it's occupied
            # This is the cost of the robot moving off the tile.
            cost_to_clear = 1 if is_occupied else 0

            # Calculate cost to paint this tile assuming it is clear and reachable
            # This is the minimum cost for any robot to reach a clear adjacent tile
            # with the correct color and paint.
            min_robot_paint_cost_when_clear = math.inf

            # Find adjacent tiles for painting
            adjacent_tiles = self.adjacency.get(tile_name, [])

            # Identify which of these adjacent tiles are currently clear.
            # These are the potential target locations for a robot to paint T from.
            target_adjacent_clear_tiles = [
                adj_t for adj_t in adjacent_tiles if adj_t in current_clear_tiles
            ]

            # If there are no clear adjacent tiles, no robot can paint this tile yet.
            # This tile is unreachable in the current state.
            if not target_adjacent_clear_tiles:
                 unsolvable = True
                 break # Cannot paint this tile

            for robot_name, robot_loc in current_robot_locations.items():
                robot_color = current_robot_colors.get(robot_name)

                color_cost = 1 if robot_color != goal_color else 0

                # Calculate movement cost: Shortest path from robot_loc to any tile
                # in target_adjacent_clear_tiles. BFS on clear tiles + robot_loc.
                movement_cost = math.inf
                q_bfs = deque([(robot_loc, 0)])
                visited_bfs = {robot_loc}

                # Perform BFS
                while q_bfs:
                    curr_tile_bfs, dist_bfs = q_bfs.popleft()

                    # Check if this tile is one of the target tiles
                    if curr_tile_bfs in target_adjacent_clear_tiles:
                        movement_cost = dist_bfs
                        break # Found shortest path to a clear adjacent tile

                    # Explore neighbors
                    for neighbor_tile_bfs in self.adjacency.get(curr_tile_bfs, []):
                        # Can move to neighbor_tile_bfs if it is clear
                        if neighbor_tile_bfs in current_clear_tiles and neighbor_tile_bfs not in visited_bfs:
                            visited_bfs.add(neighbor_tile_bfs)
                            q_bfs.append((neighbor_tile_bfs, dist_bfs + 1))

                # End of BFS for movement cost

                if movement_cost != math.inf:
                    # Cost for this robot to paint this tile (assuming tile_name is clear)
                    robot_paint_cost = color_cost + movement_cost + 1 # +1 for the paint action
                    min_robot_paint_cost_when_clear = min(min_robot_paint_cost_when_clear, robot_paint_cost)

            # min_robot_paint_cost_when_clear is the cost to paint the tile *if* it is clear and reachable.

            if min_robot_paint_cost_when_clear == math.inf:
                 # This should have been caught by the check before the robot loop,
                 # but double-check here. If no robot could reach a clear adjacent tile.
                 unsolvable = True
                 break

            # Total cost for this tile is cost to clear (if occupied) + cost to paint when clear
            total_estimated_cost += cost_to_clear + min_robot_paint_cost_when_clear

        if unsolvable:
            return math.inf

        return total_estimated_cost
