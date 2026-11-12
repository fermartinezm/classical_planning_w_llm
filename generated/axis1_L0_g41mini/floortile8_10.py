from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract parts of a PDDL fact string by removing parentheses and splitting."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    Summary:
    This heuristic estimates the number of actions required to reach the goal state from a given state.
    Actions include moving robots to target tiles, changing the color held by robots, and painting tiles.
    The heuristic is designed for greedy best-first search guidance and is not guaranteed to be admissible,
    but it aims to be informative and efficiently computable.

    Assumptions:
    - Multiple robots may exist, each holding exactly one color at a time.
    - The robot must be adjacent (up/down/left/right) to a clear tile to paint it.
    - Painting a tile consumes one action per tile.
    - Changing the robot's color to the required color takes one action.
    - Moving the robot between adjacent tiles requires one action per step.
    - The floor is represented by tiles connected by static relations (up/down/left/right).
    - Each tile may require painting in a specific color as per the goal.
    - Tiles that are already painted with the correct color in the current state do not require painting.
    - The heuristic considers only the minimal necessary movements and color changes ignoring robot coordination and possible blockages.
    - Robots cannot paint occupied or non-clear tiles.
    - Changing color is always possible if the color is available.

    Heuristic Initialization:
    - Extract all required paint goals: mapping from tiles to their target colors.
    - Extract the static layout of the board as adjacency mappings for navigation.
    - Extract the available colors (needed for color change validity).
    - Determine the identity and initial location of robots and the colors they currently carry.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify all tiles that must be painted and are not yet painted with the goal color.
    2. For each such tile, determine candidate robots that can paint it (i.e. robots that can hold the goal color).
       Since robots can change color, all robots are considered candidates.
    3. For each tile, compute the minimal cost to paint it by the best suitable robot:
       a) Compute the minimal navigation cost for the robot to reach an adjacent tile of the target tile from its current position.
          Adjacent means one step in up/down/left/right direction where the tile is clear (for painting).
       b) If the robot does not currently hold the required color, add 1 action to change color.
       c) Add 1 action to paint the tile.
    4. The heuristic sums the minimal costs per tile, assuming robots paint tiles independently and ignoring possible contention.
    5. To improve efficiency, distances between tiles are computed via breadth-first search within the static adjacency graph.
    6. When multiple robots exist, the heuristic takes the minimum painting cost across all robots for each tile and sums these minimal costs.
    7. The heuristic returns 0 if all painting goals are already met.

    This heuristic balances accuracy and efficiency and generally leads to fewer expanded nodes by encouraging painting 
    tiles in an order minimizing robot movements and color changes.
    """

    def __init__(self, task):
        """Initialize internal data structures for fast heuristic computation."""
        self.goals = task.goals  # Set of facts that must hold (painted tiles with specific colors).

        # Extract painting goals: dict {tile: color}
        self.paint_goals = {}
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] == "painted":
                tile, color = parts[1], parts[2]
                self.paint_goals[tile] = color

        # Extract static adjacency: for each tile, store neighbors accessible by move or paint
        self.adjacency = {}  # dict {tile: set of neighbor_tiles}
        for fact in task.static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred in ("up", "down", "left", "right"):
                src, tgt = parts[1], parts[2]
                self.adjacency.setdefault(src, set()).add(tgt)

        # Available colors (not strictly needed for heuristic but read for completeness)
        self.available_colors = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "available-color":
                self.available_colors.add(parts[1])

    def __call__(self, node):
        """
        Compute heuristic value estimating number of actions to reach the goal.

        Steps:
        - Identify unpainted or incorrectly painted tiles that need painting.
        - Identify robot positions and held colors.
        - For each tile to paint, compute minimal cost over all robots:
          movement_cost (to adjacent tile) + color_change (if needed) + 1 (paint action).
        - Sum minimal costs for all such tiles.
        - Return 0 if goal already reached.
        """
        state = node.state

        # If goals are already satisfied (painted tiles with correct colors), cost is 0.
        # Quick check: all paint goals satisfied in state:
        goal_satisfied = True
        for tile, req_color in self.paint_goals.items():
            # painted tile with req_color must be in state
            if f"(painted {tile} {req_color})" not in state:
                goal_satisfied = False
                break
        if goal_satisfied:
            return 0

        # Extract robot positions and held colors: dict robots -> (pos_tile, held_color)
        robots = {}
        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "robot-at":
                robot, tile = parts[1], parts[2]
                robots.setdefault(robot, {})["pos"] = tile
            elif pred == "robot-has":
                robot, color = parts[1], parts[2]
                robots.setdefault(robot, {})["color"] = color

        # For each tile to paint, verify if painting is needed (not already painted correctly)
        tiles_to_paint = []
        for tile, goal_color in self.paint_goals.items():
            if f"(painted {tile} {goal_color})" not in state:
                tiles_to_paint.append((tile, goal_color))

        # Precompute adjacency including only clear neighbors for painting.
        # Actually, the static adjacency gives all neighbors. 
        # For painting, robot must stand on tile adjacent to tile to paint.
        # However, painting precondition requires that target tile is clear; the tile to stand can be any.
        # We use adjacency as-is for movement cost to reach adjacent tile.

        # Precompute for quick access which tiles are clear in current state.
        clear_tiles = {
            fact[7:-1]  # remove "(clear " prefix and closing ")"
            for fact in state
            if fact.startswith("(clear ")
        }

        # Define helper function to find minimal distance from start tile to any tile in targets_set (adjacent to a given tile).
        def bfs_min_dist(start_tile, target_tiles):
            """
            Return shortest distance on adjacency graph from start_tile to any tile in target_tiles set.
            Returns infinity if no path exists.
            """
            if start_tile in target_tiles:
                return 0
            visited = {start_tile}
            frontier = [(start_tile, 0)]
            while frontier:
                current, dist = frontier.pop(0)
                neighbors = self.adjacency.get(current, set())
                for neighbor in neighbors:
                    if neighbor in visited:
                        continue
                    # Robot can only move on clear tiles, so require neighbor to be clear.
                    if neighbor not in clear_tiles:
                        continue
                    if neighbor in target_tiles:
                        return dist + 1
                    visited.add(neighbor)
                    frontier.append((neighbor, dist + 1))
            return float("inf")

        # For painting a tile, robot must stand on a neighbor tile that is clear.
        # Find all clear neighbors of the tile to paint:
        def get_adjacent_clear_tiles(tile):
            neighbors = set()
            # To find neighbors, look for tiles x where adjacency[x] contains tile, or adjacency[tile] contains x.
            # Since adjacency is one-way (e.g. up tile y), also consider reverse links for completeness.
            # The static facts define directional adjacency predicates, but moves/actions involve adjacency.
            # To simplify, consider neighbors from adjacency dict keys and also reverse edges.
            # We collect neighbors from adjacency[tile], plus tiles that have tile in their adjacency.

            # adjacency[tile]: tiles reachable from tile (neighboring tiles)
            direct_neighbors = self.adjacency.get(tile, set())

            # reverse neighbors: tiles from which we can get to tile
            reverse_neighbors = {t for t, nbrs in self.adjacency.items() if tile in nbrs}

            all_neighbors = direct_neighbors | reverse_neighbors

            # We must only consider tiles that are clear.
            for nbr in all_neighbors:
                if nbr in clear_tiles:
                    neighbors.add(nbr)

            return neighbors

        total_cost = 0

        for tile_needed, color_needed in tiles_to_paint:
            # For this tile, find minimal action cost among all robots.

            # Candidate stand positions adjacent to tile_needed:
            adj_clear_tiles = get_adjacent_clear_tiles(tile_needed)
            if not adj_clear_tiles:
                # No adjacent clear tile to paint from, painting not possible => very large heuristic.
                # To keep heuristic finite, assign large cost.
                total_cost += 1000
                continue

            best_cost_for_tile = float("inf")

            for robot, info in robots.items():
                robot_pos = info.get("pos")
                robot_color = info.get("color")
                if robot_pos is None or robot_color is None:
                    # Incomplete info, skip.
                    continue

                # Find shortest distance from robot_pos to any adj_clear_tile.
                dist = bfs_min_dist(robot_pos, adj_clear_tiles)

                if dist == float("inf"):
                    # Robot cannot reach adj tile, skip.
                    continue

                # Actions required:
                #  - dist moves to get adjacent
                #  - 1 if color change needed (only if robot_color != color_needed)
                #  - 1 paint action
                color_change_cost = 0 if robot_color == color_needed else 1
                cost = dist + color_change_cost + 1

                if cost < best_cost_for_tile:
                    best_cost_for_tile = cost

            # If no robot can paint tile (best_cost_for_tile still inf), assign large penalty.
            if best_cost_for_tile == float("inf"):
                best_cost_for_tile = 1000

            total_cost += best_cost_for_tile

        return total_cost
