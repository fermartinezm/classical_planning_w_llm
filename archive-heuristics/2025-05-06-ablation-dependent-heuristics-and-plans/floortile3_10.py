from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
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


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with the correct color. It sums the costs for each individual unpainted goal tile,
    considering the paint action, the minimum movement cost for the closest robot
    to reach an adjacent tile from which painting is possible, and the cost to
    acquire the necessary color if no robot currently holds it.

    # Assumptions
    - Tiles that need painting are initially clear or painted with the wrong color.
    - If a tile is painted with the wrong color, the problem is likely unsolvable
      as there is no unpaint action in the domain. The heuristic assigns a high
      cost in this case to prune such branches.
    - Movement between adjacent tiles takes 1 action.
    - Changing color takes 1 action.
    - Painting a tile takes 1 action.
    - Multiple robots can work in parallel, but the heuristic sums costs independently
      per tile and color need, which makes it non-admissible but aims to guide
      greedy search effectively.
    - The grid structure defined by up/down/left/right predicates is static.

    # Heuristic Initialization
    - Build the tile adjacency graph based on up/down/left/right predicates. This graph
      represents possible robot movements.
    - Precompute shortest path distances between all pairs of tiles using BFS on the
      movement graph.
    - For each tile, identify the set of adjacent tiles from which a robot can paint it
      (paint neighbors).
    - Extract goal conditions, specifically the target color for each tile that needs
      to be painted.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Extract the current location of each robot and the color it is holding (if any).
    2. Extract which tiles are currently painted and with which color, and which tiles
       are clear.
    3. Initialize the total heuristic cost `h` to 0.
    4. Initialize a set `needed_colors_set` to keep track of all colors required by
       at least one unpainted goal tile.
    5. Iterate through each goal condition `(painted T C)` where `T` is a tile and `C` is a color:
       a. Check if the tile `T` is already painted with color `C` in the current state. If yes, this goal is satisfied for this tile; continue to the next goal.
       b. If `T` is painted with a *different* color `C'` (i.e., `T` is in `painted_tiles` and `painted_tiles[T] != C`), return a very large number (e.g., 1,000,000). This indicates a state that is likely a dead end in valid problem instances.
       c. If `T` is `clear` (i.e., `T` is in `clear_tiles`):
          i. Add 1 to `h` for the paint action required for tile `T`.
          ii. Add color `C` to the `needed_colors_set`.
          iii. Find the set of tiles `PaintNeighbors(T)` from which a robot can paint `T`. This set was precomputed during initialization.
          iv. For each robot `R`, calculate the shortest path distance from its current location `Loc(R)` to the closest tile in `PaintNeighbors(T)`. Use the precomputed distances.
          v. Find the minimum of these distances over all robots. Let this be `min_moves`.
          vi. Add `min_moves` to `h`. If no robot can reach any paint neighbor (distance is infinity), return a large number.
    6. After iterating through all goal tiles, iterate through the `needed_colors_set`.
    7. For each color `C` in `needed_colors_set`:
       a. Check if any robot currently holds color `C`.
       b. If no robot holds color `C`, add 1 to `h`. This represents the cost for one robot to perform a `change_color` action to acquire this color. This cost is added only once per color needed across all unpainted tiles.
    8. Return the total heuristic cost `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the grid graph and precomputing distances.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the movement graph and paint neighbor mapping
        self.move_graph = {}
        self.paint_neighbors = {} # Maps a tile to the set of tiles from which it can be painted
        all_tiles = set()

        # Parse adjacency facts to build the graph and paint neighbors
        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
                direction, tile1, tile2 = parts # (dir tile1 tile2) means tile1 is dir from tile2
                all_tiles.add(tile1)
                all_tiles.add(tile2)

                # Movement graph (undirected edges for bidirectional movement)
                self.move_graph.setdefault(tile1, []).append(tile2)
                self.move_graph.setdefault(tile2, []).append(tile1)

                # Paint neighbors mapping
                # If (dir tile1 tile2) is true, robot at tile2 can paint tile1 using paint_dir
                self.paint_neighbors.setdefault(tile1, []).append(tile2)
                # If (inverse_dir tile2 tile1) were true, robot at tile1 could paint tile2
                # The PDDL defines both directions explicitly (up/down, left/right are inverses)
                # So, if (up t1 t2) exists, (down t2 t1) also exists.
                # Robot at t2 paints t1 (up). Robot at t1 paints t2 (down).
                # So t2 is paint neighbor for t1, and t1 is paint neighbor for t2.
                # The bidirectional addition below is correct for paint neighbors too.
                self.paint_neighbors.setdefault(tile2, []).append(tile1)


        self.all_tiles = list(all_tiles) # Store as list for consistent ordering

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_node in self.all_tiles:
            self.distances[start_node] = self._bfs(start_node)

        # Store goal tiles and their target colors for quick lookup
        self.goal_painted_tiles = {}
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                _, tile, color = get_parts(goal)
                self.goal_painted_tiles[tile] = color

    def _bfs(self, start_node):
        """
        Perform BFS starting from start_node to find distances to all reachable nodes.
        Returns a dictionary {node: distance}.
        """
        distances = {node: float('inf') for node in self.all_tiles}
        if start_node not in self.all_tiles:
             # Should not happen if start_node comes from self.all_tiles, but good practice
             return distances

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()
            current_dist = distances[current_node]

            for neighbor in self.move_graph.get(current_node, []):
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
        return distances

    def get_distance(self, tile1, tile2):
        """Helper to get precomputed distance, returns infinity if no path."""
        return self.distances.get(tile1, {}).get(tile2, float('inf'))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Extract current state information
        robot_locations = {} # {robot: tile}
        robot_colors = {} # {robot: color or None}
        painted_tiles = {} # {tile: color}
        clear_tiles = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "robot-at" and len(parts) == 3:
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif predicate == "robot-has" and len(parts) == 3:
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
            elif predicate == "painted" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                painted_tiles[tile] = color
            elif predicate == "clear" and len(parts) == 2:
                tile = parts[1]
                clear_tiles.add(tile)
            # 'available-color' and 'free-color' are not needed from state

        total_cost = 0
        needed_colors_set = set() # Colors required for at least one unpainted goal tile

        # Iterate through goal tiles that need to be painted
        for goal_tile, goal_color in self.goal_painted_tiles.items():
            # Check if goal tile is already painted correctly
            if goal_tile in painted_tiles and painted_tiles[goal_tile] == goal_color:
                continue # Goal satisfied for this tile

            # Check if goal tile is painted with the wrong color
            if goal_tile in painted_tiles and painted_tiles[goal_tile] != goal_color:
                # Problem is likely unsolvable if a goal tile is painted wrong
                return 1000000 # Assign a very high cost

            # If tile is clear and needs painting
            if goal_tile in clear_tiles:
                total_cost += 1 # Cost for the paint action

                # Add the required color to the set of needed colors
                needed_colors_set.add(goal_color)

                # Find the minimum moves for the closest robot to reach a paint neighbor
                min_moves_for_tile = float('inf')
                paint_neighbor_tiles = self.paint_neighbors.get(goal_tile, [])

                if not paint_neighbor_tiles:
                     # This tile has no paintable neighbors defined in static facts
                     # Problem likely unsolvable
                     return 1000000

                # Iterate through all robots to find the one closest to a paint neighbor
                for robot, robot_loc in robot_locations.items():
                    min_dist_from_robot_to_paint_neighbors = float('inf')
                    for pn_tile in paint_neighbor_tiles:
                        dist = self.get_distance(robot_loc, pn_tile)
                        min_dist_from_robot_to_paint_neighbors = min(min_dist_from_robot_to_paint_neighbors, dist)

                    min_moves_for_tile = min(min_moves_for_tile, min_dist_from_robot_to_paint_neighbors)

                if min_moves_for_tile == float('inf'):
                     # No robot can reach any paint neighbor of this tile
                     # Problem likely unsolvable
                     return 1000000

                total_cost += min_moves_for_tile

        # Add cost for acquiring colors that no robot currently has
        for color_C in needed_colors_set:
            has_color = False
            for robot_color in robot_colors.values():
                if robot_color == color_C:
                    has_color = True
                    break
            if not has_color:
                total_cost += 1 # Cost for one robot to change color to color_C

        return total_cost
