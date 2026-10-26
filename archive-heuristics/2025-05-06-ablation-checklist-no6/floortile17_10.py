from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import math

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    Estimates the cost to paint all goal tiles that are not yet painted
    correctly. For each unpainted goal tile, it estimates the minimum cost
    for any robot to reach a clear adjacent tile with the correct color
    and paint it.

    Heuristic components for an unpainted goal tile T needing color C:
    1. Penalty if T is painted with the wrong color or is not clear.
    2. For each robot R:
       a. Cost to get color C (1 if R has a color to change from, 0 if R already has C, infinity otherwise).
       b. Cost to move from R's location to the closest clear tile adjacent to T.
          (Calculated using BFS on clear tiles).
       c. Cost of the paint action (1).
    3. The minimum of (a + b + c) over all robots is the cost for tile T.
    4. Sum the costs for all unpainted goal tiles.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting grid structure, goal tiles,
        and available colors from static facts and goals.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Preprocess static facts to build grid structure
        self.adj_list = {} # tile_name -> list of neighbor_tile_names

        # Build adjacency list based on spatial relations
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] in ['up', 'down', 'left', 'right']:
                # (predicate tile_Y tile_X) means tile_Y is in the direction 'predicate' from tile_X
                # e.g., (up tile_1_1 tile_0_1) means tile_1_1 is up from tile_0_1
                # Add adjacency in both directions
                tile_y, tile_x = parts[1], parts[2]
                self.adj_list.setdefault(tile_x, []).append(tile_y)
                self.adj_list.setdefault(tile_y, []).append(tile_x)

        # Store goal tiles and their target colors
        self.goal_tiles = {} # tile_name -> target_color
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'painted':
                tile_name, color = parts[1], parts[2]
                self.goal_tiles[tile_name] = color

        # Store available colors (not strictly needed for this heuristic logic, but good practice)
        self.available_colors = {get_parts(fact)[1] for fact in self.static_facts if get_parts(fact)[0] == 'available-color'}


    def bfs_distance(self, start_tile, target_tiles_set, clear_tiles_set):
        """
        Find the shortest path distance (number of moves) from start_tile to any tile
        in target_tiles_set, where all intermediate nodes and the target node
        must be in clear_tiles_set. The start_tile itself does not need to be clear
        to start the BFS, as the move action precondition is on the destination.
        Returns float('inf') if no path exists.
        """
        # If start is a target, distance is 0 moves.
        # This case is handled implicitly by the BFS loop structure if start_tile is added to queue first.
        # However, if start_tile is a target, it must also be in clear_tiles_set by definition of target_tiles_set.
        # If robot is already at a clear adjacent tile (start_tile is in target_tiles_set), distance is 0.
        if start_tile in target_tiles_set:
             return 0

        q = deque([(start_tile, 0)])
        visited = {start_tile}

        while q:
            current_tile, dist = q.popleft()

            # Explore neighbors reachable by moving (destination must be clear)
            if current_tile not in self.adj_list:
                 continue # Tile has no neighbors defined

            for neighbor in self.adj_list.get(current_tile, []):
                # Can only move *to* clear tiles
                if neighbor in clear_tiles_set and neighbor not in visited:
                    if neighbor in target_tiles_set: # Found a target
                        return dist + 1 # Distance is number of moves
                    visited.add(neighbor)
                    q.append((neighbor, dist + 1))

        return math.inf # No path found

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Extract current state information
        robot_locations = {} # robot_name -> tile_name
        robot_colors = {} # robot_name -> color
        clear_tiles = set() # set of tile_name
        painted_tiles = {} # tile_name -> color

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'robot-at':
                robot_locations[parts[1]] = parts[2]
            elif parts[0] == 'robot-has':
                robot_colors[parts[1]] = parts[2]
            elif parts[0] == 'clear':
                clear_tiles.add(parts[1])
            elif parts[0] == 'painted':
                painted_tiles[parts[1]] = parts[2]

        total_heuristic = 0
        LARGE_PENALTY = 1000 # Penalty for seemingly unreachable goal components

        # Identify unpainted goal tiles and calculate cost for each
        for goal_tile, target_color in self.goal_tiles.items():
            # Check if the goal tile is already satisfied
            if goal_tile in painted_tiles and painted_tiles[goal_tile] == target_color:
                continue # This goal is met

            # Check for blocking conditions (wrong color or not clear)
            if goal_tile in painted_tiles and painted_tiles[goal_tile] != target_color:
                 # Painted wrong color - this goal is unreachable from this state
                 total_heuristic += LARGE_PENALTY
                 continue

            if goal_tile not in clear_tiles:
                 # Not clear and not correctly painted - this goal is unreachable from this state
                 total_heuristic += LARGE_PENALTY
                 continue

            # Tile is clear and needs painting with target_color
            # Find potential paint locations adjacent to the goal tile
            # These are the tiles from which a robot can paint the goal_tile.
            # They are simply the neighbors of the goal_tile in the grid.
            potential_paint_locations = self.adj_list.get(goal_tile, [])

            # Filter potential paint locations to find those that are currently clear.
            # A robot must move *to* a clear tile to paint.
            clear_paint_locations = {loc for loc in potential_paint_locations if loc in clear_tiles}

            min_robot_cost_for_tile = math.inf

            # Consider each robot
            for robot_name, robot_loc in robot_locations.items():
                robot_color = robot_colors.get(robot_name) # Get color robot is holding

                # Cost to get the correct color
                color_cost = math.inf # Assume infinite cost initially
                can_paint_with_color = False

                if robot_color == target_color:
                    color_cost = 0
                    can_paint_with_color = True
                elif robot_color is not None: # Robot has *some* color, can change
                    color_cost = 1 # Cost of change_color action
                    can_paint_with_color = True
                # else: robot_color is None, cannot change color, cannot paint (color_cost remains inf)


                # Cost to move from robot_loc to the closest clear paint location
                move_cost = math.inf

                # Only calculate move cost if robot can potentially get the color needed
                if can_paint_with_color:
                    if clear_paint_locations:
                        # BFS finds shortest path from robot_loc to any tile in clear_paint_locations
                        # traversing only through clear tiles.
                        move_cost = self.bfs_distance(robot_loc, clear_paint_locations, clear_tiles)
                    # else: move_cost remains inf (no clear adjacent tiles to move to)

                # Total cost for this robot to paint this tile
                # Need move_cost + color_cost + paint_action_cost (1)
                # Only add paint_action_cost if move and color are possible (move_cost is finite)
                if move_cost != math.inf:
                    robot_cost = move_cost + color_cost + 1
                    min_robot_cost_for_tile = min(min_robot_cost_for_tile, robot_cost)
                # else: robot_cost remains inf

            # Add the minimum cost for this tile to the total heuristic
            if min_robot_cost_for_tile == math.inf:
                 # If no robot can paint this tile (e.g., no path to clear adjacent tile, or no robot has/can get the color)
                 total_heuristic += LARGE_PENALTY
            else:
                 total_heuristic += min_robot_cost_for_tile

        return total_heuristic
