from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace and parentheses
    fact = fact.strip()
    if fact.startswith('(') and fact.endswith(')'):
        fact = fact[1:-1]
    return fact.split()

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing three components:
    1. The number of tiles that need to be painted (i.e., are currently clear but are goal tiles).
    2. The number of distinct colors required by these clear goal tiles that no robot currently possesses.
    3. An estimate of the movement cost, calculated as the sum, over all clear goal tiles, of the minimum distance from any robot to any clear tile from which that goal tile can be painted.

    # Assumptions
    - Tiles are arranged in a grid structure, and movement is restricted to adjacent tiles (up, down, left, right).
    - The grid structure is defined by the static `up`, `down`, `left`, `right` predicates.
    - A tile is either `clear` or `painted` with a single color.
    - A tile that is painted with the wrong color cannot be repainted or cleared (unsolvable state).
    - Robots can only paint tiles adjacent to them using specific actions (paint_up, paint_down, paint_left, paint_right), which require the robot to be at a specific relative position to the tile being painted.
    - Robots can only move to `clear` tiles.
    - All tiles mentioned in the domain/problem are part of a connected grid (or reachable parts are relevant).
    - The heuristic assumes that if a goal tile is clear, it is eventually possible to make the required adjacent tile clear for a robot to move onto (this might not hold in complex blocking scenarios, but is a reasonable approximation). If no required clear tile is reachable from any robot, the heuristic returns a large penalty.

    # Heuristic Initialization
    - The heuristic extracts the goal conditions to identify which tiles need to be painted and with which color.
    - It parses the static facts (`up`, `down`, `left`, `right`) to build a graph representation of the tile grid and to identify the required robot location for painting each tile.
    - It precomputes the shortest path distances between all pairs of tiles using Breadth-First Search (BFS) on the tile graph.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location and color of each robot.
    2. Identify the current state of each tile (clear or painted with a specific color).
    3. Initialize the heuristic value `h` to 0.
    4. Identify the set of goal tiles that are currently `clear` and need painting, and the set of colors required for these tiles. Simultaneously, check for unsolvable states where a goal tile is painted with the wrong color. If an unsolvable state is detected, return a large penalty.
    5. Add the count of clear goal tiles needing paint to `h` (representing the minimum number of paint actions).
    6. Add the count of distinct colors required by these clear goal tiles that are not currently held by any robot to `h` (representing necessary `change_color` actions).
    7. Calculate the movement cost:
        a. For each tile `T` that is a clear goal tile needing paint:
            i. Find the set of tiles `Required_Xs` such that being at any `X` in `Required_Xs` allows painting `T` (based on static `up`/`down`/`left`/`right` facts and paint action preconditions).
            ii. Find all tiles `X` in `Required_Xs` that are currently `clear` in the state.
            iii. If there are no required clear tiles reachable from any robot, this state is likely unsolvable or requires complex clearing. Assign a large penalty to `h` and return.
            iv. Find the minimum shortest path distance from *any* robot's current location to *any* of the required clear tiles found in step 7.a.ii.
            v. Add this minimum distance to `h`.
    8. Return the final value of `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building the
        tile graph, and precomputing distances.
        """
        self.goals = task.goals
        self.static = task.static

        # Store goal tiles and their required colors
        self.goal_tiles = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_tiles[tile] = color

        # Build tile adjacency map and list of all tiles
        self.adj_map = {}
        self.all_tiles = set()
        # Store required robot location for painting a tile
        # required_paint_pos[painted_tile] = set(required_robot_positions)
        self.required_paint_pos = {}

        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] in ["up", "down", "left", "right"]:
                # Example: (up tile_1_1 tile_0_1) means tile_1_1 is UP from tile_0_1
                # paint_up action paints Y (tile_1_1) when robot is at X (tile_0_1) and (up Y X) is true.
                # So, if (up Y X) is true, X is a required position to paint Y using paint_up.
                # Similarly for down, left, right.
                painted_tile, robot_pos_for_paint = parts[1], parts[2]

                self.adj_map.setdefault(painted_tile, []).append(robot_pos_for_paint)
                self.adj_map.setdefault(robot_pos_for_paint, []).append(painted_tile) # Assume grid connections are symmetric

                self.all_tiles.add(painted_tile)
                self.all_tiles.add(robot_pos_for_paint)

                self.required_paint_pos.setdefault(painted_tile, set()).add(robot_pos_for_paint)


        # Remove duplicates from adjacency lists
        for tile in self.adj_map:
            self.adj_map[tile] = list(set(self.adj_map[tile]))

        # Precompute all-pairs shortest paths using BFS
        self.distances = {}
        for start_tile in self.all_tiles:
            self.distances[start_tile] = self._bfs(start_tile)

    def _bfs(self, start_tile):
        """Performs BFS from a start tile to find distances to all other tiles."""
        distances = {tile: float('inf') for tile in self.all_tiles}
        distances[start_tile] = 0
        queue = deque([start_tile])
        # Use a set for visited for faster lookups
        visited = {start_tile}

        while queue:
            current_tile = queue.popleft()

            # Get neighbors from the adjacency map
            neighbors = self.adj_map.get(current_tile, [])

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_tile] + 1
                    queue.append(neighbor)

        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Identify current robot locations and colors
        robot_locations = {}
        robot_colors = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif parts[0] == "robot-has":
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color

        # Identify current tile states (clear or painted)
        painted_tiles = {}
        clear_tiles = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "painted":
                tile, color = parts[1], parts[2]
                painted_tiles[tile] = color
            elif parts[0] == "clear":
                tile = parts[1]
                clear_tiles.add(tile)

        h = 0
        clear_goal_tiles_needing_paint = []
        needed_colors_for_clear_goals = set()

        # Component 1 (Paint actions) & Check for unsolvable states (wrong color)
        for goal_tile, goal_color in self.goal_tiles.items():
             # Check if this goal is satisfied
             goal_fact = f"(painted {goal_tile} {goal_color})"
             if goal_fact not in state:
                 # Goal is not satisfied. Is it painted wrong?
                 is_painted_wrong = False
                 if goal_tile in painted_tiles and painted_tiles[goal_tile] != goal_color:
                     is_painted_wrong = True

                 if is_painted_wrong:
                     # Tile is painted with the wrong color, and there's no unpaint action.
                     # This state is likely unsolvable.
                     return 1000000 # Return a large finite number for greedy search

                 # If not painted correctly and not painted wrong, it must be clear (in a valid state)
                 # So it needs painting.
                 h += 1 # Cost for paint action
                 clear_goal_tiles_needing_paint.append(goal_tile)
                 needed_colors_for_clear_goals.add(goal_color)

        # If all painted goals are satisfied, h is already 0.
        if not clear_goal_tiles_needing_paint:
             return 0

        # Component 2: Color changes needed
        current_robot_colors = set(robot_colors.values())
        num_missing_colors = len(needed_colors_for_clear_goals - current_robot_colors)
        h += num_missing_colors # Cost for change_color actions

        # Component 3: Movement cost
        movement_cost = 0
        # For each clear goal tile, find the minimum distance from any robot
        # to any clear tile from which the goal tile can be painted.
        for goal_tile in clear_goal_tiles_needing_paint:
            min_dist_to_required_clear_pos = float('inf')

            # Find required robot positions for painting this tile
            required_positions = self.required_paint_pos.get(goal_tile, set())

            # Find required positions that are currently clear
            required_clear_positions = [
                pos for pos in required_positions
                if f"(clear {pos})" in state
            ]

            # If no required position is clear, a robot cannot move there to paint.
            # This state might be unsolvable or require complex clearing actions not modeled simply.
            # Assign a large penalty if no required clear position is reachable from any robot.
            # We check reachability below.

            found_reachable_required_clear = False
            for robot_location in robot_locations.values():
                # robot_location is guaranteed to be in self.all_tiles if the initial state is valid
                # and move actions only move between connected tiles.
                # required_clear_positions are guaranteed to be in self.all_tiles if adj_map is built correctly.
                # So, distances[robot_location][pos] should always exist, though it might be inf.
                for required_pos in required_clear_positions:
                     dist = self.distances[robot_location][required_pos]
                     if dist != float('inf'):
                         min_dist_to_required_clear_pos = min(min_dist_to_required_clear_pos, dist)
                         found_reachable_required_clear = True

            if not found_reachable_required_clear:
                 # Cannot reach any required clear position from any robot.
                 # This state is likely unsolvable or requires complex coordination/clearing.
                 # Assign a large penalty.
                 return 1000000

            movement_cost += min_dist_to_required_clear_pos

        h += movement_cost

        return h
