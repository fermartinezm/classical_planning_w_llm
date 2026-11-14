from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain 'color' domain.

    Summary:
    This heuristic estimates the minimal number of actions required to paint all goal tiles.
    Painting a tile requires the robot to be "at" a required color, and the tile to be "down" and "available".
    The robot can move colors upward along tiles, move down colors along tiles, and paint a tile, 
    changing certain facts in the state. The heuristic estimates the cost by measuring how many goal tiles remain unpainted,
    estimating the number of color moves and paint actions required.

    Assumptions:
    - Each goal tile (tile-2) must be painted (predicate painted(tile)).
    - Painting a tile consumes the "free-color" fact for some color, so fresh "free-colors" may be needed.
    - The robot can move "available-color" facts upward along tile chains using "change_color" actions,
      and can "move_down" colors between tile and color objects.
    - The robot must be "robot-at" a color-2 and color to paint a tile.
    - The robot can only paint tiles that are "down".
    - Changing colors and moving down actions progress the robot's ability to paint goal tiles.
    - The heuristic is not admissible but should give a close estimate of progress and is efficiently computable.

    Heuristic Initialization:
    - Extract goal tiles that require painting.
    - Build a graph of tiles connected by the "up" relation (from static facts).
      This helps estimate distance in tile steps needed to move colors to correct tiles.
    - Extract sets of tiles currently painted.
    - Extract current free-color colors and robot positions for reference.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify the set of goal tiles that are not yet painted.
    2. For each unpainted goal tile:
       - Attempt to estimate the minimal number of actions to get a suitable color available at the tile:
         a) Compute (approximate) the distance along "up" edges from tiles where the robot's "available-color" facts currently are,
            to the tile beneath the goal tile (since painting uses predicates on tile-2 that are "down" tiles).
         b) The robot needs to move colors upward along tiles to reach the relevant tile.
         c) Paint action requires a free color at the robot's position and the robot to be at compatible colors.
    3. Sum such distances (each distance corresponds roughly to moves and color shifts) plus one paint action per unpainted tile.
    4. If no paths or available colors exist to support these moves, fallback to a conservative count of unpainted tiles times at least one paint action.
    5. Return this heuristic value, zero when all goal tiles are painted.

    This heuristic estimates how many color movements plus paint actions are minimally necessary to reach the goal painted tiles,
    thereby guiding the search efficiently by focusing on progressing towards enabling paint actions for all goal tiles.
    """

    def __init__(self, task):
        self.goals = task.goals  # goal facts (painted tiles)
        self.static = task.static  # static facts (e.g., up relations)

        # Extract all goal tiles to paint (from painted predicates)
        self.goal_tiles = set()
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] == "painted" and len(parts) == 2:
                self.goal_tiles.add(parts[1])

        # Extract the "up" tile relationships from static facts: build adjacency for graph
        # We'll create a graph mapping each tile to the tile above it (following up edges)
        # Note: "up" predicate is (up tile tile)
        self.up_graph = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "up" and len(parts) == 3:
                lower_tile = parts[1]
                upper_tile = parts[2]
                # We store: lower_tile -> upper_tile
                # Also keep reverse mapping for distance calculations if needed
                self.up_graph.setdefault(lower_tile, set()).add(upper_tile)

        # To estimate distance against the "available-color" tiles, we also will build a reverse "down" graph,
        # since robot can move colors upward via change_color that requires (up ?v7 ?v8).

        # Get all tiles in up_graph as a set (for validation)
        self.all_tiles = set()
        for lower, uppers in self.up_graph.items():
            self.all_tiles.add(lower)
            self.all_tiles.update(uppers)

    def __call__(self, node):
        state = node.state

        # Extract painted tiles in current state
        painted_tiles = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "painted" and len(parts) == 2:
                painted_tiles.add(parts[1])

        # If all goal tiles painted => heuristic is 0
        unpainted_tiles = self.goal_tiles - painted_tiles
        if not unpainted_tiles:
            return 0

        # Extract tiles currently available with colors (from "available-color" predicate)
        # predicate: (available-color robot tile)
        # Note: From domain, "available-color" has parameters (?v1 - robot ?v2 - tile)
        # but domain types are unusual, sometimes tile or color? 
        # Heuristic will trust first arg is 'robot/color-2' and second is tile or tile-2.

        available_color_tiles = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "available-color" and len(parts) == 3:
                # parts: ['available-color', robot-color2, tile]
                # The third argument could be a tile (tile or tile-2)
                available_color_tiles.add(parts[2])

        # Extract robot-at facts: robot positions (robot-at color-2 color)
        # used to estimate which colors robot currently controls
        robot_at_colors = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at" and len(parts) == 3:
                # parts[1] = robot color-2; parts[2] = color
                robot_at_colors.add( (parts[1], parts[2]) )

        # Extract free colors
        free_colors = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "free-color" and len(parts) == 2:
                free_colors.add(parts[1])

        # Extract tiles that are "down" (must be painted tiles - tile-2 presumably)
        down_tiles = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "down" and len(parts) == 2:
                down_tiles.add(parts[1])

        # --- Heuristic calculation ---

        # We need a function to estimate minimal "color moves" from tiles with available colors,
        # via the "up" relation, to the tile immediately below the goal tile (since painting uses tile-2),
        # We approximate the distance in tile moves (number of change_color actions) needed.

        # Because paint_up action requires:
        # (available-color ?v1 ?v2) and (available-color ?v6 ?v2) and
        # (robot-at ?v1 ?v3) and (free-color ?v3) and (down ?v6) and then paints ?v6,
        # meaning the robot color and some other color must be available at the same tile,
        # which is likely the tile below the tile-2 being painted.

        # We will:
        # 1. For each unpainted tile-2 goal, find the tile (tile-2) below which it is painted.
        #    From domain, painted facts refer to tile-2 objects, while "available-color" and "up" relations are on tile and tiles.
        #    The paint_up parameters include ?v2 (a tile) and ?v6 (tile-2).
        #    The paint_up precondition requires (available-color ?v1 ?v2) and (available-color ?v6 ?v2), so the tile (tile) needs colors.
        #
        # Thus we need to approximate for each painted tile-2 goal the costs to get available colors at the required tile.
        #
        # Here, since we do not have explicit link from tile-2 to tile, but from instance examples, tile-2 objects (e.g. tile1, tile5)
        # correspond to tiles that are themselves linked via up/down relations.
        #
        # Let's try: each goal tile-2 corresponds to a tile in the world where painting needs to happen.
        # We try to find a reachable tile (tile) that corresponds to the tile-2 or is related to it.
        #
        # For the heuristic, we will treat that the painting needs to happen at the tile-2's associated tile.
        # We assume that the tile-2's name is the same as a corresponding tile or it is available as key in down_tiles.
        #
        # If the exact connection is missing, we fallback to heuristic = unpainted_tiles count.

        # For each unpainted tile-2 goal, estimate minimal distance between available-color tiles and painting tile
        # via shortest path along "up" edges (assuming color moves go upwards).
        # Because "change_color" moves available color from tile t to tile t' where (up t t').

        # We'll implement a BFS search from available_color_tiles upwards to find the minimal distance to tile corresponding to goal tile.
        # Because painting uses available-color at the tile where paint_up happens, i.e. tile-2 and tile.
        # We try to find distance from available colors to the tile related to goal tile.

        # Since we do not have direct mapping of tile-2 <-> tile, we apply a heuristic:
        # Assume that tile-2 = tile object (same name), or if not present, find closest tile in up_graph.

        def bfs_min_distance(sources, target):
            """
            BFS minimal distance from any tile in sources to target tile in the up_graph
            moving "up" edges (sources to target).
            Returns infinity if target not reachable from any source.
            Distances count number of change_color actions.
            """
            from collections import deque

            if target in sources:
                return 0

            visited = set(sources)
            queue = deque( (tile, 0) for tile in sources )
            while queue:
                current, dist = queue.popleft()
                if current == target:
                    return dist
                # Move upward
                for nxt in self.up_graph.get(current, set()):
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append( (nxt, dist + 1) )
            return float('inf')

        heuristic_value = 0

        for tile2_goal in unpainted_tiles:
            # The goal tile is tile-2 (like tile1), so we try to find color availability at tile with same name (tile) or closest tile.

            # We treat tile2_goal as tile for distance calculation (heuristic assumptions)
            tile_goal = tile2_goal

            # If tile_goal not in known tiles, fallback heuristic += 2 (one paint + one move assumed)
            if tile_goal not in self.all_tiles and tile_goal not in available_color_tiles:
                heuristic_value += 2  # paint + at least one color move assumed
                continue

            dist_to_color = bfs_min_distance(available_color_tiles, tile_goal)
            if dist_to_color == float('inf'):
                # No path to bring color to required tile, fallback penalty
                dist_to_color = 3  # arbitrary cost of signaling difficulty

            # We add the distance for change_color moves plus one paint action at that tile
            heuristic_value += dist_to_color + 1  # +1 for paint_up action itself

        # Return the accumulated estimate
        # This heuristic can be zero only if no unpainted tiles remain,
        # and is finite otherwise assuming the domain is solvable.
        return heuristic_value
