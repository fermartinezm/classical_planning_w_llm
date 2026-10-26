# Assuming Heuristic base class is available in this path
# from heuristics.heuristic_base import Heuristic

import collections
from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs_dynamic(adj_list, start_node, clear_tiles):
    """
    Performs BFS starting from start_node, only traversing edges (u, v) where v is in clear_tiles.
    adj_list: full grid adjacency list {node: {neighbor1, ...}}
    start_node: the node to start BFS from
    clear_tiles: set of tile names that are currently clear
    Returns: dict {node: distance} for reachable nodes.
    """
    distances = {node: float('inf') for node in adj_list}
    
    # If the start node is not in the graph keys, it means it has no defined neighbors
    # via up/down/left/right. It can't move anywhere, so only itself is reachable at dist 0.
    if start_node not in adj_list:
         # Check if it's at least a recognized tile object
         if start_node in distances:
              distances[start_node] = 0
              return distances
         else:
              # Start node is not even a recognized tile object? Cannot compute distances.
              return {}


    queue = [(start_node, 0)]
    visited = {start_node}
    distances[start_node] = 0

    q = collections.deque(queue)

    while q:
        (current_node, dist) = q.popleft()

        # current_node is guaranteed to be in adj_list keys here because we check start_node
        # and only add neighbors from adj_list.
        for neighbor in adj_list[current_node]:
            # Can only move to a neighbor if it is clear
            if neighbor in clear_tiles and neighbor not in visited:
                visited.add(neighbor)
                distances[neighbor] = dist + 1
                q.append((neighbor, dist + 1))

    return distances


class floortileHeuristic: # Inherit from Heuristic if available
    """
    A domain-dependent heuristic for the Floortile domain.

    Estimates the cost by summing the minimum estimated cost for each
    unpainted goal tile. The cost for a tile includes changing robot color,
    moving the robot to an adjacent tile from which the target tile can be
    painted, and the paint action itself. Movement cost is estimated using
    BFS on currently clear tiles.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information:
        - Tile coordinates (optional, mainly for debugging/understanding).
        - Full grid adjacency list.
        - Required paint locations for each tile.
        - Available colors.
        - Goal painted facts.
        """
        # super().__init__(task) # Call superclass init if inheriting
        self.task = task # Store task object

        self.goals = task.goals # Store goals
        self.static = task.static # Store static facts

        self.tile_coords = {}
        self.coords_tile = {}
        self.full_grid_adj = {}
        self.paint_locations_map = {}
        self.available_colors = set()
        self.goal_painted_facts = set()

        # Collect all objects that are tiles
        all_tile_objects = set()
        # Look in initial state for robot locations, clear tiles, painted tiles
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts and parts[0] in ["robot-at", "clear", "painted"]:
                 if len(parts) > 1 and parts[1].startswith("tile_"):
                     all_tile_objects.add(parts[1])
                 if len(parts) > 2 and parts[2].startswith("tile_"):
                     all_tile_objects.add(parts[2])
        # Look in goals for painted tiles
        for goal in task.goals:
             parts = get_parts(goal)
             if parts and parts[0] == "painted":
                 if len(parts) > 1 and parts[1].startswith("tile_"):
                     all_tile_objects.add(parts[1])
        # Look in static facts for tile relationships (up/down/left/right)
        for fact in task.static:
             parts = get_parts(fact)
             if parts and parts[0] in ["up", "down", "left", "right"]:
                 if len(parts) > 1 and parts[1].startswith("tile_"):
                     all_tile_objects.add(parts[1])
                 if len(parts) > 2 and parts[2].startswith("tile_"):
                     all_tile_objects.add(parts[2])


        # Parse tile names and coordinates, initialize adjacency list
        for obj in all_tile_objects:
            if obj.startswith("tile_"):
                try:
                    # Assuming format tile_row_col
                    parts = obj.split('_')
                    # PDDL object names are case-insensitive, but typically lowercase.
                    # Ensure consistent parsing.
                    row = int(parts[1])
                    col = int(parts[2])
                    self.tile_coords[obj] = (row, col)
                    self.coords_tile[(row, col)] = obj
                    self.full_grid_adj[obj] = set() # Initialize adjacency list
                except (IndexError, ValueError):
                    # Ignore objects that look like tiles but don't match format
                    # print(f"Warning: Could not parse tile object name: {obj}") # Avoid printing in heuristic
                    pass # Ignore objects that don't fit the expected tile format

        # Build full grid adjacency list and paint locations map from static facts
        for fact in self.static:
            parts = get_parts(fact)
            if len(parts) == 3:
                pred, obj1, obj2 = parts
                # Check if objects are known tiles before adding to grid graph/paint map
                is_obj1_tile = obj1 in self.full_grid_adj
                is_obj2_tile = obj2 in self.full_grid_adj

                if pred in ["up", "down", "left", "right"]:
                    if is_obj1_tile and is_obj2_tile:
                        # Add bidirectional edges for movement
                        self.full_grid_adj[obj1].add(obj2)
                        self.full_grid_adj[obj2].add(obj1)

                if pred == "up":
                    # obj1 is up from obj2. Robot at obj2 paints obj1.
                    if is_obj1_tile: # Ensure obj1 is a known tile (the one being painted)
                        if obj1 not in self.paint_locations_map:
                            self.paint_locations_map[obj1] = set()
                        self.paint_locations_map[obj1].add(obj2) # obj2 is the paint location
                elif pred == "down":
                    # obj1 is down from obj2. Robot at obj2 paints obj1.
                     if is_obj1_tile: # Ensure obj1 is a known tile (the one being painted)
                        if obj1 not in self.paint_locations_map:
                            self.paint_locations_map[obj1] = set()
                        self.paint_locations_map[obj1].add(obj2) # obj2 is the paint location

            elif len(parts) == 2 and parts[0] == "available-color":
                 self.available_colors.add(parts[1])

        # Extract goal painted facts
        for goal in self.goals:
            parts = get_parts(goal)
            if len(parts) == 3 and parts[0] == "painted":
                self.goal_painted_facts.add((parts[1], parts[2]))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        robot_locations = {}
        robot_colors = {}
        clear_tiles = set()
        current_painted_facts = set()

        # Parse current state
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            if parts[0] == "robot-at" and len(parts) == 3:
                robot_locations[parts[1]] = parts[2]
            elif parts[0] == "robot-has" and len(parts) == 3:
                robot_colors[parts[1]] = parts[2]
            elif parts[0] == "clear" and len(parts) == 2:
                clear_tiles.add(parts[1])
            elif parts[0] == "painted" and len(parts) == 3:
                current_painted_facts.add((parts[1], parts[2]))

        # Identify unpainted goal tiles and check for wrongly painted tiles
        needed_painting = []
        wrongly_painted_exists = False
        for (tile, color) in self.goal_painted_facts:
            if (tile, color) not in current_painted_facts:
                needed_painting.append((tile, color))
            # Check if the tile is painted with a different color
            # Iterate through current painted facts to find if this tile is painted wrongly
            for (t_state, c_state) in current_painted_facts:
                 if t_state == tile and c_state != color:
                      wrongly_painted_exists = True
                      break # Found wrong color for this tile
            if wrongly_painted_exists: break # Found wrongly painted tile overall

        if wrongly_painted_exists:
             # If any tile is painted with the wrong color, it's likely unsolvable
             # given the domain actions (no unpaint/clear painted tile action).
             return float('inf') # Return a large value

        if not needed_painting:
            return 0 # Goal reached

        total_estimated_cost = 0

        # Pre-calculate BFS distances from each robot's location to all reachable clear tiles
        distances_from_robots = {}
        for r_name, r_loc_name in robot_locations.items():
             # BFS starts from the robot's current location
             # Ensure the robot's location is a known node in the grid graph
             # The bfs_dynamic function handles the case where start_node is not in adj_list keys.
             distances_from_robots[r_name] = bfs_dynamic(self.full_grid_adj, r_loc_name, clear_tiles)


        # Estimate cost for each unpainted goal tile independently
        for (tile_name, color) in needed_painting:
            min_cost_for_tile = float('inf')

            # Find the locations a robot needs to be at to paint this tile
            paint_locations = self.paint_locations_map.get(tile_name, set())

            if not paint_locations:
                 # This tile cannot be painted by any robot with current actions (missing up/down relation)
                 # If a goal tile has no paint location defined, the problem is likely unsolvable.
                 return float('inf')

            for r_name, r_loc_name in robot_locations.items():
                r_color = robot_colors.get(r_name) # Get robot's current color

                # Cost to get the correct color
                color_cost = 0
                if r_color != color:
                    if color in self.available_colors:
                        color_cost = 1 # 1 action for change_color
                    else:
                        # Robot cannot get the required color
                        continue # This robot cannot paint this tile

                # Cost to move to a paint location
                min_move_cost = float('inf')
                robot_distances = distances_from_robots.get(r_name, {})

                for p_loc_name in paint_locations:
                    # Check if the paint location is reachable by the robot via clear tiles
                    if p_loc_name in robot_distances:
                         dist = robot_distances[p_loc_name]
                         if dist != float('inf'):
                              min_move_cost = min(min_move_cost, dist)

                if min_move_cost != float('inf'):
                     # Total cost for this robot to paint this tile
                     # color_cost + move_cost + paint_action_cost (1)
                     cost_for_this_robot = color_cost + min_move_cost + 1
                     min_cost_for_tile = min(min_cost_for_tile, cost_for_this_robot)

            if min_cost_for_tile == float('inf'):
                 # If no robot can paint this tile, the state is likely unsolvable
                 return float('inf')

            total_estimated_cost += min_cost_for_tile

        return total_estimated_cost
