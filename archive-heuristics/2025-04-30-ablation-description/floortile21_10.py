from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at robot1 tile_0_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing the estimated costs
    for each tile that needs to be painted according to the goal. The cost for a single
    tile is estimated as the minimum number of move actions required for any robot to
    reach a position from which it can paint the tile, plus one action for painting,
    plus an estimated cost for acquiring the correct color.

    # Assumptions
    - The goal only requires tiles to be painted.
    - Tiles painted with the wrong color in the current state cannot be repainted
      (as there are no unpaint actions), making the state unsolvable if a goal tile
      is painted incorrectly.
    - The grid structure is defined by 'up', 'down', 'left', 'right' facts, and
      movement cost between adjacent tiles is 1.
    - Painting a tile requires the robot to be at an adjacent tile (specifically,
      the tile "below" for paint_up, or "above" for paint_down) and have the correct color.
    - The cost of changing color is 1. A robot changing color can potentially satisfy
      the color requirement for multiple goal tiles.

    # Heuristic Initialization
    - Extract goal conditions to identify which tiles need to be painted and with which colors.
    - Build an adjacency graph of the tiles based on 'up', 'down', 'left', 'right' facts
      to calculate movement distances.
    - Precompute the set of valid "paint-from" locations for each tile based on 'up' and 'down' facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal conditions of the form `(painted T C)`.
    2. For each goal `(painted T C)`:
       - If `(painted T C)` is already true in the current state, this goal is satisfied (cost 0).
       - If `(painted T C')` where `C' != C` is true, the state is likely unsolvable in this domain; return infinity.
       - If `(clear T)` is true, this tile needs to be painted. Add it to a list of pending goal tiles.
    3. If any goal tile is painted the wrong color, return infinity.
    4. Calculate the cost for acquiring necessary colors:
       - Determine the set of distinct colors required for all pending clear goal tiles.
       - Determine the set of colors currently held by robots.
       - The number of colors that are needed but not held is a lower bound on color change actions. Add this count to the total cost.
    5. For each pending clear goal tile `T` with target color `C`:
       - Find the set of locations `PaintFroms` from which `T` can be painted (i.e., tiles `Adj` such that `(up T Adj)` or `(down T Adj)` is true).
       - Calculate the minimum movement distance from any robot's current location to any location in `PaintFroms` using BFS on the tile graph.
       - If no robot can reach any `PaintFroms` location, the state is likely unsolvable; return infinity.
       - Add this minimum distance plus 1 (for the paint action) to the total cost.
    6. The total heuristic value is the sum of the color acquisition cost and the movement/paint costs for all pending clear goal tiles.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building the tile graph,
        and precomputing paint-from locations.
        """
        # Ensure task.goals is a set of goal facts, not a single (and ...) fact
        # The provided task representation seems to handle this correctly.
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations and colors for painted tiles
        self.goal_painted_tiles = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted" and len(parts) == 3:
                 tile, color = parts[1], parts[2]
                 self.goal_painted_tiles[tile] = color


        # Build the tile graph and paint-from locations
        self.graph = {}
        self.paint_from_locations = {} # {target_tile: set(robot_location_to_paint_from)}
        self.all_tiles = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts:
                continue # Skip malformed facts

            predicate = parts[0]
            if predicate in ["up", "down", "left", "right"]:
                # These predicates define adjacency for movement
                if len(parts) == 3:
                    tile1, tile2 = parts[1], parts[2]
                    self.all_tiles.add(tile1)
                    self.all_tiles.add(tile2)
                    self.graph.setdefault(tile1, set()).add(tile2)
                    self.graph.setdefault(tile2, set()).add(tile1) # Adjacency is symmetric

                # These predicates also define paint-from locations
                if predicate == "up" and len(parts) == 3:
                    # (up tile_Y tile_X) means tile_Y is up from tile_X.
                    # Robot at tile_X can paint tile_Y using paint_up.
                    target_tile, robot_location = parts[1], parts[2]
                    self.paint_from_locations.setdefault(target_tile, set()).add(robot_location)
                elif predicate == "down" and len(parts) == 3:
                    # (down tile_Y tile_X) means tile_Y is down from tile_X.
                    # Robot at tile_X can paint tile_Y using paint_down.
                    target_tile, robot_location = parts[1], parts[2]
                    self.paint_from_locations.setdefault(target_tile, set()).add(robot_location)

        # Ensure all tiles mentioned in goals or static facts are in the graph keys
        # Even if a tile has no connections, it should be a key in the graph dict
        for tile in self.all_tiles:
             self.graph.setdefault(tile, set())
        for tile in self.goal_painted_tiles:
             self.graph.setdefault(tile, set())


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Identify current robot locations and colors
        robot_locations = {}
        robot_colors = {}
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            if parts[0] == "robot-at" and len(parts) == 3:
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif parts[0] == "robot-has" and len(parts) == 3:
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color

        pending_goal_tiles = {} # {tile: target_color} for tiles that need painting
        wrongly_painted_tiles = set()

        # Check each goal tile's status
        for tile, target_color in self.goal_painted_tiles.items():
            is_goal_satisfied = False
            is_clear = False
            is_wrongly_painted = False

            # Check current state for this tile
            # We need to iterate through state facts related to this tile's status
            tile_status_facts = [
                fact for fact in state
                if get_parts(fact) and get_parts(fact)[1] == tile
                and get_parts(fact)[0] in ["painted", "clear"]
            ]

            for fact in tile_status_facts:
                parts = get_parts(fact)
                predicate = parts[0]

                if predicate == "painted" and len(parts) == 3:
                    current_color = parts[2]
                    if current_color == target_color:
                        is_goal_satisfied = True
                    else:
                        is_wrongly_painted = True
                    # Once painted status is found, we know it's not clear (mutually exclusive in domain)
                    break

                if predicate == "clear":
                    is_clear = True
                    # Continue checking in case it's also painted (shouldn't happen in valid states, but defensive)


            if is_goal_satisfied:
                continue # Goal met for this tile

            if is_wrongly_painted:
                wrongly_painted_tiles.add(tile)
                # If a goal tile is painted the wrong color, the problem is likely unsolvable
                # in this domain as there's no way to unpaint/repaint.
                # We can return infinity early, or add it to a set and check later.
                # Let's add and check later to be explicit.

            if is_clear:
                 # Tile is clear and needs painting
                 pending_goal_tiles[tile] = target_color


        # If any goal tile is painted the wrong color, the state is likely unsolvable
        if wrongly_painted_tiles:
            return float('inf')

        total_cost = 0

        # Estimate color change cost
        needed_colors = set(pending_goal_tiles.values())
        held_colors = set(robot_colors.values())
        # Simple estimate: count colors needed that no robot currently holds
        colors_to_acquire = needed_colors - held_colors
        total_cost += len(colors_to_acquire)


        # Estimate movement and paint cost for each pending clear goal tile
        for tile, target_color in pending_goal_tiles.items():
            required_paint_from_locs = self.paint_from_locations.get(tile, set())

            # If a tile needs painting but cannot be painted from any adjacent tile
            # according to the static facts, something is wrong with the problem definition
            # or the state is unreachable. Treat as unsolvable.
            if not required_paint_from_locs:
                 # This shouldn't happen in valid problems where goal tiles are paintable
                 # but as a safeguard:
                 # print(f"Warning: Tile {tile} needs painting but has no defined paint-from locations.")
                 return float('inf')


            min_dist_to_paint_from = float('inf')

            # Calculate min distance from any robot to any required paint-from location
            for robot, robot_loc in robot_locations.items():
                # A robot might be at a location not in the graph if the initial state
                # includes a tile not connected to anything.
                if robot_loc not in self.graph:
                     # This robot cannot move, and thus cannot reach a paint location
                     continue

                # Perform BFS from the robot's current location
                distances = self._bfs(robot_loc)

                # Find the minimum distance from this robot to any required paint-from location
                dist_from_robot = float('inf')
                for paint_loc in required_paint_from_locs:
                    if paint_loc in distances:
                        dist_from_robot = min(dist_from_robot, distances[paint_loc])

                min_dist_to_paint_from = min(min_dist_to_paint_from, dist_from_robot)

            # If no robot can reach any paint-from location for this tile
            if min_dist_to_paint_from == float('inf'):
                 # This tile cannot be painted by any robot from the current state
                 return float('inf') # Unsolvable state

            # Cost for this tile: movement to paint-from location + 1 (paint action)
            total_cost += min_dist_to_paint_from + 1

        # The heuristic value is the total estimated cost
        return total_cost

    def _bfs(self, start_node):
        """
        Performs a Breadth-First Search starting from start_node to find distances
        to all reachable nodes in the tile graph.
        Returns a dictionary mapping reachable node to its distance from start_node.
        """
        # Ensure start_node is actually in the graph
        if start_node not in self.graph:
             # Cannot start BFS from a node not in the graph
             return {}

        distances = {start_node: 0}
        queue = deque([start_node])
        visited = {start_node}

        while queue:
            current_node = queue.popleft()
            current_dist = distances[current_node]

            # current_node is guaranteed to be in self.graph keys here because we checked start_node
            # and only add neighbors that are in the graph during graph construction.
            # Using .get is still safer if graph construction might be imperfect.
            for neighbor in self.graph.get(current_node, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)

        return distances
