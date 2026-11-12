from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate obj1 obj2)".
    - `args`: Pattern elements, wildcards '*' allowed.
    - Returns True if fact matches the pattern, False otherwise.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (w133ormt) domain.

    Summary:
        This heuristic estimates the minimal number of actions required to achieve the goal state.
        The domain involves robots positioned on tiles, each with a current color (vfphwgc6),
        where the goal is to paint certain tiles with certain colors. Robots can move between tiles,
        change the color they hold, and paint tiles to change their color.

    Assumptions:
        - There is at least one robot.
        - Robots hold exactly one color at any time.
        - The goal requires certain tiles to have a certain color (yh6q5j8r fact).
        - Paintable tiles are those that initially have the predicate (d7ekk7vb) true.
        - Robots can move between tiles via adjacency relations (gk9u3vg3 and related).
        - Changing robot color uses action z7pcmkeu.
        - Painting a tile requires the robot to be on a tile adjacent (gk9u3vg3) to the tile and holding the required color.
        - Robots hold exactly one color at once, so color changes may be necessary.
        
    Heuristic Initialization:
        - Extract from static facts the adjacency maps between tiles for robot movement.
        - Extract the list of all robots.
        - Extract the list of all colors.
        - Extract initial facts that indicate tiles (d7ekk7vb), static color facts (cdb9op3o).
        - Store goal conditions requiring tiles painted with certain colors (yh6q5j8r).
        - Precompute reverse adjacency maps for bidirectional movement.

    Step-By-Step Thinking for Computing Heuristic:
        1. Check if the state already satisfies the goal exactly -> heuristic = 0.
        2. For each tile in the goal requiring a certain color:
           - If that tile already has the correct color in the current state (yh6q5j8r tile color),
             no action needed for that tile.
           - Otherwise, painting is required.
        3. For each tile to paint, estimate cost as:
           - Minimal cost for a robot to reach an adjacent tile of the tile needing painting.
           - Plus minimal number of color changes for robot to hold the needed color.
           - Plus one paint action.
        4. Since multiple robots exist, estimate minimal cost among all robots.
           - For efficiency, compute shortest distances from robot positions to all tiles using BFS.
           - Cost for moving robots is number of moves needed to get adjacent to the tile.
           - Cost for color changes: 1 if robot's current color differs from required color, else 0.
           - Painting action cost is counted as 1.
        5. Sum the minimal painting costs for all tiles to be painted.
        6. Return this sum as the heuristic estimate.

        Note:
        - The heuristic is not necessarily admissible but aims to be informative and fast.
        - The heuristic is 0 if and only if all goal painting conditions are met.
        - Distances are computed based on static adjacency relations.
    """

    def __init__(self, task):
        # Store goal conditions
        self.goals = task.goals   # a frozenset of goal facts (strings)

        # Extract static facts
        static = task.static

        # Extract all tiles (bffr59sc)
        self.tiles = set()
        # Extract robots (ioytrsy9)
        self.robots = set()
        # Extract colors (vfphwgc6)
        self.colors = set()

        # Extract tiles from static or from facts mentioning tiles
        # The predicate d7ekk7vb indicates a tile
        for fact in static:
            if match(fact, "d7ekk7vb", "*"):
                parts = get_parts(fact)
                self.tiles.add(parts[1])

        # Extract colors from cdb9op3o static facts
        for fact in static:
            if match(fact, "cdb9op3o", "*"):
                parts = get_parts(fact)
                self.colors.add(parts[1])

        # Extract robot objects from fact signatures in the domain (ioytrsy9 type)
        # Since no direct static fact for robots, infer from facts mentioning robots (gk9u3vg3, i4u94rl9)
        for fact in static:
            parts = get_parts(fact)
            # gk9u3vg3 has signature (gk9u3vg3 robot tile)
            if match(fact, "gk9u3vg3", "*", "*"):
                self.robots.add(parts[1])
            # i4u94rl9 robot color
            if match(fact, "i4u94rl9", "*", "*"):
                self.robots.add(parts[1])

        # Adjacency relations between tiles, to compute shortest distances
        # gk9u3vg3 ?robot ?tile in preconditions is not adjacency
        # But predicates xr2t1c1d, oh0smogc, w224no0l, g7ya35is represent adjacency between tiles.
        # Pattern in domain:
        #  - xr2t1c1d and oh0smogc are inverse (tile from->to),
        #  - w224no0l, g7ya35is also possibly represent adjacency in other directions.

        # We'll build a graph of tile adjacencies based on these predicates:
        # xr2t1c1d, oh0smogc, w224no0l, g7ya35is
        self.adjacency = {tile: set() for tile in self.tiles}

        # Build undirected graph for tile adjacency from static facts
        for fact in static:
            parts = get_parts(fact)
            pred = parts[0]

            if pred in {"xr2t1c1d", "oh0smogc", "w224no0l", "g7ya35is"}:
                tile1 = parts[1]
                tile2 = parts[2]
                # Add edges in both directions to make undirected graph for shortest path calculations
                if tile1 in self.tiles and tile2 in self.tiles:
                    self.adjacency[tile1].add(tile2)
                    self.adjacency[tile2].add(tile1)

        # Precompute set of goal tile-color pairs needed
        self.goal_tile_colors = set()
        for goal in self.goals:
            if match(goal, "yh6q5j8r", "*", "*"):  # (yh6q5j8r tile color)
                parts = get_parts(goal)
                tile = parts[1]
                color = parts[2]
                self.goal_tile_colors.add((tile, color))

    def _bfs_distances_from(self, start_tile):
        """
        Compute shortest distances from start_tile to all other tiles using BFS.

        Returns a dict mapping tile -> distance (number of edges).
        """
        from collections import deque

        dist = {start_tile: 0}
        queue = deque([start_tile])

        while queue:
            tile = queue.popleft()
            current_dist = dist[tile]
            for neighbor in self.adjacency[tile]:
                if neighbor not in dist:
                    dist[neighbor] = current_dist + 1
                    queue.append(neighbor)

        return dist

    def __call__(self, node):
        state = node.state

        # 1) If goal already reached, heuristic = 0
        if self.goals <= state:
            return 0

        # 2) Extract information from current state
        # For each robot store its current tile and color held
        robot_positions = {}  # robot -> tile
        robot_colors = {}     # robot -> color

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "gk9u3vg3":  # (gk9u3vg3 robot tile)
                robot_positions[parts[1]] = parts[2]
            elif pred == "i4u94rl9": # (i4u94rl9 robot color)
                robot_colors[parts[1]] = parts[2]

        # If for some robot not found position or color (should not happen), fill with dummy
        for r in self.robots:
            if r not in robot_positions:
                robot_positions[r] = None
            if r not in robot_colors:
                robot_colors[r] = None

        # 3) Find tiles currently painted with a color (yh6q5j8r tile color)
        painted_tiles = {}
        for fact in state:
            if match(fact, "yh6q5j8r", "*", "*"):
                parts = get_parts(fact)
                tile = parts[1]
                color = parts[2]
                painted_tiles[tile] = color

        # 4) Identify tiles that need painting: those that differ from goal or missing in painted_tiles
        tiles_to_paint = []
        for tile, goal_color in self.goal_tile_colors:
            current_color = painted_tiles.get(tile)
            if current_color != goal_color:
                tiles_to_paint.append((tile, goal_color))

        # If no tiles to paint, return 0
        if not tiles_to_paint:
            return 0

        # 5) Precompute BFS distances from all robot positions to all tiles
        # For efficiency, compute BFS once per robot and cache results
        robot_dists = {}
        for r in self.robots:
            pos = robot_positions.get(r)
            if pos not in self.tiles:
                # Robot not on a known tile - to avoid crash, assign large empty dist map
                robot_dists[r] = {}
            else:
                robot_dists[r] = self._bfs_distances_from(pos)

        # 6) For each tile to paint:
        #    - Find minimal cost among all robots of:
        #      movement cost to get adjacent to tile to paint (i.e. robot must be on a tile adjacent to tile to paint),
        #      color change cost (0 if robot holds needed color else 1),
        #      plus 1 painting action.
        #    - Sum these minimal costs.

        total_heuristic = 0

        for tile, needed_color in tiles_to_paint:
            # Find tiles adjacent to tile (where robot must stand to paint)
            # A tile is adjacent if it appears in adjacency[tile] (undirected)
            adjacent_tiles = self.adjacency.get(tile, set())
            if not adjacent_tiles:
                # No adjacency to tile? Painting impossible, assign high cost
                # To keep heuristic finite, assign large cost:
                total_heuristic += 50
                continue

            min_cost_for_tile = float("inf")

            for r in self.robots:
                dist_map = robot_dists[r]
                robot_color = robot_colors.get(r)
                color_change_cost = 0 if robot_color == needed_color else 1
                # Find minimal distance to any adjacent tile
                min_dist_to_adjacent = float("inf")
                for adj_tile in adjacent_tiles:
                    d = dist_map.get(adj_tile)
                    if d is not None and d < min_dist_to_adjacent:
                        min_dist_to_adjacent = d

                if min_dist_to_adjacent == float("inf"):
                    # Robot cannot reach any adjacent tile; ignore this robot for this tile
                    continue

                # Cost = moves + color change + paint action (1)
                cost = min_dist_to_adjacent + color_change_cost + 1

                if cost < min_cost_for_tile:
                    min_cost_for_tile = cost

            # If no robot can reach the tile, assign large penalty
            if min_cost_for_tile == float("inf"):
                min_cost_for_tile = 50

            total_heuristic += min_cost_for_tile

        return total_heuristic
