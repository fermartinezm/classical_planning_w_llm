from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the floortile domain.

    Summary
    -------
    Estimates the number of actions required to reach a goal state by:
    1. Counting tiles that still need to be painted with the correct color.
    2. Estimating the minimal robot movement needed to reach and paint all these tiles,
       including necessary robot color changes.

    Assumptions
    -----------
    - Each tile must be painted with exactly one goal color.
    - Painting can only occur from tiles adjacent to the target tile in up/down/left/right directions.
    - The robot can only paint adjacent clear tiles.
    - The robot can hold only one color at a time and must change color explicitly.
    - The domain may have multiple robots, but the heuristic uses the minimal cost over all robots.
    - Moving between adjacent tiles costs 1 move action; painting costs 1 action; changing color costs 1 action.
    - The heuristic is non-admissible but computationally efficient.
    - If multiple tiles require the same color, the robot can paint them consecutively without changing color.

    Heuristic Initialization
    ------------------------
    From the static facts:
    - Extract tile connectivity for all four directions (up, down, left, right) to build adjacency.
    From the task goals:
    - Extract the required color for each tile goal fact.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For each call on a given state:
    1. Identify which tiles still need painting by comparing the painted facts in the state to the goal.
    2. For each tile still to paint, get the color it needs.
    3. Identify the positions and currently held colors of all robots.
    4. For each robot:
       a. Calculate minimal move cost as the shortest distance from the robot's current position 
          to a tile adjacent to the tile needing painting.
          (Robot cannot paint on the tile it is on; must be adjacent.)
       b. Accumulate color change cost if the robot's held color differs from the required tile color.
       c. Add painting action cost (1 per tile).
    5. For the overall heuristic, take the minimal total cost among all robots.
    6. If no tiles remain to paint, heuristic is zero.

    Implementation details:
    - Movement cost: shortest path between robot position and an adjacent tile of the tile to paint.
      This is approximated by using adjacency relations.
    - To reduce computation, precompute adjacency for all tiles.
    - For each tile to paint, find all tiles adjacent to it where the robot can stand to paint.
    - Use BFS to find shortest distances from robot position to those paint-position tiles.
    - Sum costs over all tiles to paint, accounting for color changes between tiles.
      Here, color change cost is counted pessimistically: one color change per tile if color differs
      from current or last painted color.
      (Better color sequencing is complex; this heuristic just sums conservatively.)

    The heuristic overestimates but efficiently guides search,
    minimizing node expansions by encouraging painting nearby tiles of the same color in succession.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract adjacency from static facts
        # adjacency maps tile -> set of adjacent tiles reachable by robot (must be clear or exist as adjacency)
        # Because moves are up/down/left/right, combine those relations.
        self.adjacency = {}  # tile -> set of adjacent tiles

        for fact in self.static:
            pred, *args = get_parts(fact)
            if pred in ("up", "down", "left", "right"):
                tile_from, tile_to = args
                # Since move actions require (robot-at ?r ?x) and (up ?y ?x), moves are from tile_to -> tile_from,
                # but edge direction is from tile_to to tile_from (robot moves from x to y if up y x).
                # So the adjacency edge is from tile_to to tile_from to represent where robot can move.
                # We want adjacency that is: from tile to neighbors the robot can move to.
                # So add edge tile_to -> tile_from
                self.adjacency.setdefault(tile_to, set()).add(tile_from)

        # Extract goal tiles and their colors (from painted tile color facts)
        self.tile_goal_colors = {}
        for goal_fact in self.goals:
            pred, *args = get_parts(goal_fact)
            if pred == "painted":
                tile, color = args
                self.tile_goal_colors[tile] = color

    def __call__(self, node):
        state = node.state

        # Helper: check if tile is painted with needed color
        # painted(tile, color) required but may or may not be in state
        # For efficiency: build dict tile -> color painted in current state
        painted_tiles = {}
        for fact in state:
            pred, *args = get_parts(fact)
            if pred == "painted":
                tile, color = args
                painted_tiles[tile] = color

        # Identify which tiles still need painting and their goal colors
        tiles_to_paint = []
        for tile, goal_color in self.tile_goal_colors.items():
            if painted_tiles.get(tile, None) != goal_color:
                tiles_to_paint.append((tile, goal_color))

        # If no tiles to paint, heuristic is zero
        if not tiles_to_paint:
            return 0

        # Find all robots with their positions and held color
        robots = []
        # For robot positions: (robot-at ?r ?x)
        # For robot colors: (robot-has ?r ?c)
        robot_positions = {}
        robot_colors = {}
        for fact in state:
            pred, *args = get_parts(fact)
            if pred == "robot-at":
                r, tile = args
                robot_positions[r] = tile
            elif pred == "robot-has":
                r, color = args
                robot_colors[r] = color

        # If no robots found, return large heuristic (cannot paint)
        if not robot_positions:
            # Return a large number to discourage or indicate unsolvable state
            return 9999

        # Precompute for each tile to paint the possible robot positions adjacent to it
        # For painting the tile, robot must be on an adjacent tile (adjacent by up/down/left/right)
        # That tile must also be clear in the state, but we don't know clear tiles for all,
        # since state is dynamic and could grow large. We conservatively assume robot can stand on any adjacent tile,
        # since moves require clear tiles.

        # However, state only lists clear tiles by "(clear tileX)", so gather clear tiles from state
        clear_tiles = set()
        for fact in state:
            if fact.startswith("(clear "):
                # fact example: "(clear tile_1_1)"
                _, tile = get_parts(fact)
                clear_tiles.add(tile)

        # Build reverse adjacency mapping: for each tile, its neighbors (precomputed)
        # We already have adjacency: tile -> neighbors robot can move to.
        # We want adjacency for painting: tiles adjacent to tile_to_paint that are accessible.

        # A tile from which robot can paint tile_to_paint is any tile adjacent to tile_to_paint in adjacency or reverse adjacency:
        # Because adjacency is directional from tile_to->tile_from for movement,
        # For painting adjacency, both sides count (robot at tile adjacent horizontally or vertically to target tile)
        # so we consider both adjacency edges and their reverses to find neighbors.

        # Build all adjacency directions combined:
        all_adjacency = {}  # tile -> set of neighbors (undirected)
        for tile, neighbors in self.adjacency.items():
            all_adjacency.setdefault(tile, set()).update(neighbors)
            for neigh in neighbors:
                all_adjacency.setdefault(neigh, set()).add(tile)

        # For BFS distance calculations, consider only walkable tiles:
        # We conservatively use all tiles that are keys in all_adjacency as walkable (plus any tile in clear_tiles, which should match)
        walkable_tiles = set(all_adjacency.keys()) | clear_tiles

        # For each tile to paint, find its adjacents that are walkable and clear (robot can move there):
        paint_adjacent_tiles = {}
        for tile, _color in tiles_to_paint:
            candidates = set()
            for neigh in all_adjacency.get(tile, set()):
                # The robot must be able to be on that tile to paint tile
                if neigh in walkable_tiles and neigh in clear_tiles:
                    candidates.add(neigh)
            paint_adjacent_tiles[tile] = candidates

        # To estimate move cost, we need shortest path from robot positions to any of those candidates
        # We'll perform BFS from robot position and get distance to closest paint-adjacent tile per tile to paint
        # Movement cost is number of moves needed (each move costs 1)

        # Define a BFS function for distances on the fly (tiles are nodes, neighbors are from all_adjacency)

        def bfs_distance(start, goals):
            """Return shortest distance from start tile to any of the goal tiles or None if unreachable."""
            if start in goals:
                return 0
            visited = {start}
            queue = [(start, 0)]
            while queue:
                current, dist = queue.pop(0)
                neighbors = all_adjacency.get(current, set())
                for neigh in neighbors:
                    if neigh not in visited and neigh in walkable_tiles:
                        if neigh in goals:
                            return dist + 1
                        visited.add(neigh)
                        queue.append((neigh, dist + 1))
            return None  # unreachable

        # Now, for each robot, compute total cost to paint all tiles_to_paint.
        # A simple heuristic approach:
        # - Sum costs per tile to paint:
        #   For each tile:
        #     1. Move cost from robot (or last painted tile position) to a paint-adjacent tile
        #     2. Add color change cost if robot/last color differs from tile color (cost=1)
        #     3. Add paint cost (1)
        #
        # We do this assuming independent painting of tiles in any order, using robot start position and color.
        # To keep it efficient, do a greedy approximation where robot paints tiles in the order given, updating position/color after each tile.

        min_total_cost = float("inf")

        for robot, start_pos in robot_positions.items():
            held_color = robot_colors.get(robot, None)  # can be None if not held

            total_cost = 0
            current_pos = start_pos
            current_color = held_color

            # For heuristic, try a simple greedy: paint tiles in given order minimizing distance from current_pos to next tile's painting position

            # Track which tiles remain to paint for this robot, copying tiles_to_paint for iteration
            remaining = tiles_to_paint[:]

            # We will do a simple nearest-next-tile approach with color change consideration
            while remaining:
                # For each candidate tile, compute distance from current_pos to any paint-adjacent tile, plus color change cost
                best_tile = None
                best_cost = None
                best_paint_pos = None  # adjacent tile from which painting happens
                best_color_change_cost = None
                best_move_cost = None

                for idx, (tile, tile_color) in enumerate(remaining):
                    candidates = paint_adjacent_tiles.get(tile, set())
                    if not candidates:
                        # if no place to stand and paint, consider this tile as unreachable, so large cost
                        move_cost = None
                    else:
                        move_cost = bfs_distance(current_pos, candidates)
                    if move_cost is None:
                        # unreachable, skip tile for this iteration
                        continue
                    # color change cost is 1 if current_color != tile_color else 0
                    color_change_cost = 0 if tile_color == current_color else 1
                    cost = move_cost + color_change_cost + 1  # +1 painting action

                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best_tile = idx
                        best_move_cost = move_cost
                        best_color_change_cost = color_change_cost
                        # store one candidate position at minimal BFS distance:
                        # pick first candidate with distance = move_cost
                        # Since bfs_distance returns minimal dist but not node, approximate by trying candidates for adjacency:
                        best_paint_pos = None
                        # We'll pick one candidate where distance from current_pos == move_cost by simple workaround:
                        # Run BFS again to find actual path is expensive; we pick any candidate for heuristic:
                        # We pick first candidate for simplicity:
                        if candidates:
                            best_paint_pos = next(iter(candidates))

                if best_cost is None:
                    # no reachable tile remains, break with large cost
                    total_cost = float("inf")
                    break

                # Update total_cost and robot state
                total_cost += best_cost
                # Update position to the tile from which robot paints (stands on adjacent tile)
                current_pos = best_paint_pos
                # Update color after painting
                _, painted_color = remaining[best_tile]
                current_color = painted_color
                # Remove painted tile
                del remaining[best_tile]

            if total_cost < min_total_cost:
                min_total_cost = total_cost

        # Return min total cost over all robots
        # Should always be finite if at least one robot can reach paintable tiles
        return min_total_cost if min_total_cost != float("inf") else 9999
