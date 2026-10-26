import math
from collections import deque

class floortileHeuristic:
    """
    Domain-dependent heuristic for the floortile domain.

    Summary:
        This heuristic estimates the cost to reach a goal state by summing
        the minimum estimated costs for each individual goal tile that is
        not yet painted with the correct color. The estimated cost for a
        single tile is the minimum cost for any robot to acquire the
        necessary color, move to an adjacent tile, and paint the goal tile.

    Assumptions:
        - The grid structure defined by the static 'up', 'down', 'left',
          and 'right' predicates forms a connected grid.
        - Tile names follow a convention (e.g., tile_row_col) that allows
          deriving grid coordinates, although the heuristic computes
          coordinates based on adjacency relations directly.
        - Solvable states do not have goal tiles that are painted with a
          color different from the required goal color, as there is no
          action to unpaint a tile. The heuristic does not explicitly
          handle this unsolvable case with infinity, assuming it won't
          occur in solvable problem instances or reachable states thereof.
        - Robots always possess exactly one color at any time (no 'free-color'
          state is explicitly handled as it's unused in the provided domain).

    Heuristic Initialization:
        The constructor analyzes the static facts from the task definition.
        It builds:
        1. An adjacency list (`self.adj`) representing the undirected grid
           connections between tiles based on 'up', 'down', 'left', 'right'
           relations.
        2. A coordinate map (`self.coords`) assigning a (row, column) tuple
           to each tile. This is done by performing a Breadth-First Search
           starting from an arbitrary tile, using the directed adjacency
           relations ('up' means decreasing row, 'right' means increasing col, etc.)
           to determine neighbor coordinates.

    Step-By-Step Thinking for Computing Heuristic:
        1.  Identify the set of 'unpainted goal tiles'. These are tiles `T`
            for which the goal state requires `(painted T C)` for some color `C`,
            but the current state does *not* contain the fact `(painted T C)`.
        2.  If the set of unpainted goal tiles is empty, the heuristic value is 0
            (the goal is reached).
        3.  Extract the current location and color held by each robot from the
            current state facts.
        4.  Initialize the total heuristic value to 0.
        5.  For each unpainted goal tile `T` that needs to be painted with color `C`:
            a.  Find all tiles `L_adj` that are adjacent to `T` using the
                precomputed adjacency list (`self.adj`). A robot must move
                to one of these adjacent tiles to paint `T`.
            b.  Calculate the minimum cost for *any* robot `R` to paint tile `T`
                with color `C`. Initialize `min_cost_for_tile` to infinity.
            c.  For each robot `R`:
                i.  Determine the cost for robot `R` to acquire color `C`. This is 1
                    if robot `R` currently holds a different color, and 0 if it
                    already holds color `C`.
                ii. Determine the minimum movement cost for robot `R` to reach *any*
                    tile `L_adj` adjacent to `T`. This is calculated as the minimum
                    Manhattan distance between robot `R`'s current location and
                    each tile in the set `L_adj`. Manhattan distance is computed
                    using the precomputed coordinate map (`self.coords`).
                iii. The total estimated cost for robot `R` to paint tile `T` is
                     the color acquisition cost + the minimum movement cost + 1
                     (for the paint action itself).
                iv. Update `min_cost_for_tile` with the minimum of its current value
                    and the cost calculated for robot `R`.
            d.  Add `min_cost_for_tile` to the total heuristic value.
        6.  Return the total heuristic value.
    """
    def __init__(self, task):
        self.task = task
        self.adj = {} # {tile: set(adjacent_tiles)}
        self.coords = {} # {tile: (row, col)}
        self._build_grid()

    def _build_grid(self):
        """
        Parses static facts to build the grid structure (adjacency and coordinates).
        """
        directed_adj = {} # {tile: {direction: neighbor_tile}}
        tiles = set()
        # Mapping from PDDL direction predicate to the relative coordinate change
        # and the reverse direction predicate.
        # (pred tile1 tile2) means tile1 is pred of tile2.
        # If tile2 is at (r, c), tile1 is at (r + dr, c + dc).
        # Example: (up tile_1_1 tile_0_1) means tile_1_1 is UP from tile_0_1.
        # If tile_0_1 is (r, c), tile_1_1 is (r-1, c). So dr=-1, dc=0.
        # The reverse of 'up' is 'down'.
        direction_info = {
            'up': ((-1, 0), 'down'),
            'down': ((1, 0), 'up'),
            'left': ((0, -1), 'right'),
            'right': ((0, 1), 'left'),
        }

        for fact_string in self.task.static:
            pred, args = self._parse_fact(fact_string)
            if pred in direction_info:
                tile1, tile2 = args
                tiles.add(tile1)
                tiles.add(tile2)
                # Store relation from tile2 to tile1
                directed_adj.setdefault(tile2, {})[pred] = tile1
                # Store reverse relation from tile1 to tile2
                reverse_pred = direction_info[pred][1]
                directed_adj.setdefault(tile1, {})[reverse_pred] = tile2

        if not tiles:
             # No tiles found in static facts, likely an empty or malformed problem
             return

        # Use BFS to build coordinates and undirected adjacency
        start_tile = next(iter(tiles)) # Pick an arbitrary tile
        queue = deque([(start_tile, (0, 0))])
        self.coords = {start_tile: (0, 0)}
        self.adj = {tile: set() for tile in tiles} # Initialize adjacency for all found tiles

        while queue:
            tile, (r, c) = queue.popleft()

            for direction, neighbor in directed_adj.get(tile, {}).items():
                # Add undirected adjacency
                self.adj[tile].add(neighbor)
                self.adj[neighbor].add(tile) # Ensure reverse is also added

                if neighbor not in self.coords:
                    # Calculate neighbor coordinates based on direction from tile
                    (dr, dc), _ = direction_info[direction]
                    nr, nc = r + dr, c + dc
                    self.coords[neighbor] = (nr, nc)
                    queue.append((neighbor, (nr, nc)))

    def _parse_fact(self, fact_string):
        """
        Parses a PDDL fact string into a (predicate, [args]) tuple.
        e.g., '(robot-at robot1 tile_0_4)' -> ('robot-at', ['robot1', 'tile_0_4'])
        """
        # Remove parentheses and split by space
        parts = fact_string[1:-1].split()
        predicate = parts[0]
        args = parts[1:]
        return predicate, args

    def manhattan_distance(self, tile1, tile2):
        """
        Calculates the Manhattan distance between two tiles using their coordinates.
        Returns infinity if coordinates are not found (should not happen in a valid grid).
        """
        if tile1 not in self.coords or tile2 not in self.coords:
            # This indicates an issue with grid building or unexpected tile names
            # Return infinity as a large penalty.
            return math.inf

        r1, c1 = self.coords[tile1]
        r2, c2 = self.coords[tile2]
        return abs(r1 - r2) + abs(c1 - c2)

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.
        """
        # Identify unpainted goal tiles: (tile, color) tuples
        unpainted_goals = []
        for goal_fact in self.task.goals:
            pred, args = self._parse_fact(goal_fact)
            if pred == 'painted':
                tile, color = args
                # If the goal fact '(painted tile color)' is not in the current state
                if goal_fact not in state:
                     unpainted_goals.append((tile, color))

        # If all goal painted facts are true, the heuristic is 0
        if not unpainted_goals:
            return 0

        # Extract robot info: {robot_name: {'location': tile_name, 'color': color_name}}
        robot_info = {}
        for fact_string in state:
            pred, args = self._parse_fact(fact_string)
            if pred == 'robot-at':
                robot, location = args
                robot_info.setdefault(robot, {})['location'] = location
            elif pred == 'robot-has':
                robot, color = args
                robot_info.setdefault(robot, {})['color'] = color

        # Calculate total heuristic
        total_heuristic = 0

        # For each goal tile that needs painting
        for tile_to_paint, required_color in unpainted_goals:
            # Find tiles adjacent to the one that needs painting
            adjacent_tiles = self.adj.get(tile_to_paint, set())
            if not adjacent_tiles:
                 # If a goal tile has no adjacent tiles in the grid, it's unreachable.
                 # This indicates an unsolvable problem or a tile outside the defined grid.
                 # Return infinity.
                 return math.inf

            min_cost_for_tile = math.inf

            # Consider each robot as a potential candidate to paint this tile
            for robot, info in robot_info.items():
                current_location = info.get('location')
                current_color = info.get('color')

                # Ensure robot location and color info is available
                if current_location is None or current_color is None:
                    continue # Skip robots with incomplete state info

                # Cost to get the required color: 1 if color change is needed, 0 otherwise
                color_cost = 1 if current_color != required_color else 0

                # Minimum cost to move the robot to any adjacent tile
                min_move_cost_to_adj = math.inf
                for adj_tile in adjacent_tiles:
                    move_cost = self.manhattan_distance(current_location, adj_tile)
                    min_move_cost_to_adj = min(min_move_cost_to_adj, move_cost)

                # If no adjacent tile is reachable (e.g., disconnected grid segment),
                # min_move_cost_to_adj remains infinity.
                if min_move_cost_to_adj == math.inf:
                    cost_for_robot = math.inf
                else:
                    # Total estimated cost for this robot to paint this tile:
                    # Color change (0 or 1) + Movement cost + Paint action (1)
                    cost_for_robot = color_cost + min_move_cost_to_adj + 1

                # Update the minimum cost found so far for painting this tile
                min_cost_for_tile = min(min_cost_for_tile, cost_for_robot)

            # Add the minimum cost required to paint this specific goal tile
            # If min_cost_for_tile is infinity (e.g., no robots, or tile unreachable),
            # the total heuristic becomes infinity.
            total_heuristic += min_cost_for_tile

        return total_heuristic
