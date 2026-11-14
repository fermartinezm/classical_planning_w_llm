from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing the surrounding parentheses
    and splitting the string into tokens.
    """
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    Wildcards '*' are supported in the pattern.

    :param fact: fact string like "(predicate arg1 arg2)"
    :param args: pattern elements (predicate and arguments)
    :return: True if fact matches pattern, False otherwise
    """
    parts = get_parts(fact)
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (w133ormt).

    Summary
    -------
    This heuristic estimates the number of actions needed to reach the goal state by focusing on the tiles that need to be painted with the correct color.
    The main difficulty in the domain is to move the robot(s) around the bffr59sc tiles,
    pick up the needed color, and paint the tiles with that color.

    Assumptions
    -----------
    - The robot can move between tiles by changing its position predicate (gk9u3vg3).
    - Robots can carry colors (i4u94rl9 predicate) from a set of available colors (cdb9op3o).
    - Tiles are either blank (d7ekk7vb) or painted with a color (yh6q5j8r).
    - Painting a tile requires the robot on the "location" tile and carrying the correct color.
    - Movement actions are limited to neighboring tiles (connected by predicates xr2t1c1d, oh0smogc, w224no0l, and g7ya35is).
    - To paint a tile, the tile must be blank (d7ekk7vb), and the target color must be carried.
    - The heuristic does not need to be admissible but should be efficient and informative.

    Heuristic Initialization
    ------------------------
    - Parse static predicates to build a graph of tile adjacencies to allow shortest path calculations.
    - Extract available colors from cdb9op3o predicates.
    - Identify tiles neighbors from adjacency relations.
    - Store goal tile-color pairs from the goal conditions.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify all goal tile and target color pairs the heuristic needs to satisfy.

    2. For each goal tile-color pair, check if the tile is already painted with the target color (yh6q5j8r predicate).
       If yes, no further cost for that tile is added.

    3. For tiles not yet correctly painted:
       - The tile must be painted by some robot carrying the correct color.
       - Painting requires the tile to be blank (d7ekk7vb).
       - The robot(s) need to:
         a) reach the tile,
         b) carry the appropriate color (which can be carried if the robot is on the location of the color (cdb9op3o) and performs color pick-up action z7pcmkeu),
         c) paint the tile from the robot's position.

    4. To estimate cost for a tile unpainted or painted with wrong color:
       - Calculate shortest distance from all robots' current tile positions to a color object tile (cdb9op3o), plus one action to pick the color.
       - Then, from that color pickup tile, compute shortest distance to the goal tile.
       - Add one action for painting the tile.
       - Take the minimum cost over all robots and all available colors.
         (Preferably only consider the target color for that tile's painting to avoid unnecessary overestimations.)

    5. Sum these minimum costs for all tiles that are not correctly painted.

    6. Return this sum as the heuristic value. It is 0 only when all goal tiles have the correct color painted.

    Note:
    - All distances in tiles are computed using BFS on the undirected adjacency graph constructed from static predicates.
    - Movement actions correspond to changing gk9u3vg3 predicate.
    - Changing color carried requires picking up a color (z7pcmkeu action).
    - Painting a tile requires one action (c1qh1p5g, cadwr4wd).
    """
    def __init__(self, task):
        """
        Initialize heuristic by:
        - Extracting static adjacency relations to build tile graph.
        - Extracting color object tiles.
        - Extracting goal tile-color pairs.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract all objects of type bffr59sc (tiles) from the static facts (anything appearing in adjacency predicates)
        self.tiles = set()
        self.colors = set()
        self.tile_neighbors = {}

        # Parse colors: all cdb9op3o ?color are color-object locations
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "cdb9op3o":
                self.colors.add(parts[1])

        # Extract tile neighbors from adjacency predicates:
        # adjacency predicates: xr2t1c1d, oh0smogc, w224no0l, g7ya35is
        # These seem to represent directed edges; we treat the graph undirected for shortest paths.
        adjacency_preds = {"xr2t1c1d", "oh0smogc", "w224no0l", "g7ya35is"}
        neighbors = {}
        for fact in self.static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred in adjacency_preds:
                tile1 = parts[1]
                tile2 = parts[2]
                self.tiles.add(tile1)
                self.tiles.add(tile2)
                if tile1 not in neighbors:
                    neighbors[tile1] = set()
                if tile2 not in neighbors:
                    neighbors[tile2] = set()
                # undirected edges
                neighbors[tile1].add(tile2)
                neighbors[tile2].add(tile1)
        self.tile_neighbors = neighbors

        # Extract goal tile-color pairs from goals (yh6q5j8r tile color)
        self.goal_tile_colors = []
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "yh6q5j8r" and len(parts) == 3:
                self.goal_tile_colors.append((parts[1], parts[2]))

    def shortest_path_length(self, start, goal):
        """
        Compute shortest path distance (in number of steps) between tiles start and goal using BFS on tile adjacency graph.
        Returns an integer distance or None if no path found.
        """
        if start == goal:
            return 0
        visited = {start}
        frontier = [start]
        dist = 0
        while frontier:
            dist += 1
            next_frontier = []
            for t in frontier:
                for nbr in self.tile_neighbors.get(t, []):
                    if nbr == goal:
                        return dist
                    if nbr not in visited:
                        visited.add(nbr)
                        next_frontier.append(nbr)
            frontier = next_frontier
        return None

    def __call__(self, node):
        """
        Compute the heuristic estimate for the node's state.

        Returns 0 if all goal tile-color pairs are satisfied.
        Otherwise returns sum over tiles of estimated minimal cost:
        moving robot to the color, picking it, moving to tile, painting tile.
        """
        state = node.state

        # Helper: find robot locations (gk9u3vg3 robot tile)
        robot_positions = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "gk9u3vg3":
                robot = parts[1]
                tile = parts[2]
                robot_positions[robot] = tile

        # Helper: find which color each robot currently carries (i4u94rl9 robot color)
        robot_colors = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "i4u94rl9":
                robot = parts[1]
                color = parts[2]
                robot_colors[robot] = color

        # Helper: find tiles currently painted with which colors (yh6q5j8r tile color)
        tile_colors = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "yh6q5j8r":
                tile = parts[1]
                color = parts[2]
                tile_colors[tile] = color

        # Helper: track which tiles are blank (d7ekk7vb tile)
        blank_tiles = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "d7ekk7vb":
                blank_tiles.add(parts[1])

        # Costs for primitive actions:
        # - move robot to adjacent tile: 1 action per step
        # - change robot carried color: 1 action (z7pcmkeu)
        # - paint tile: 1 action

        total_estimated_cost = 0

        # For each goal tile-color pair, compute minimal cost to fulfill it if not already done
        for (goal_tile, goal_color) in self.goal_tile_colors:
            # Check if tile is already painted with the correct color
            if tile_colors.get(goal_tile) == goal_color:
                continue  # no cost for this tile

            # Otherwise, must paint the tile with correct color

            # We assume painting acts on blank tile; if the tile is not blank, it will have to be "reset" by some action sequence
            # but in problem instance, probably painting overwrites previous paint or tile can be painted only if blank.
            # We count at least one paint action.
            paint_cost = 1

            # For each robot, compute minimal cost to:
            # 1) move to a color tile of goal_color or currently carry that color
            #    (if robot already carries that color, no color pickup needed)
            # 2) move to goal tile
            # 3) paint the goal tile

            min_cost_for_tile = float("inf")

            # List of locations of color objects of the goal_color
            color_positions = [color_pos for fact in self.static if match(fact, "cdb9op3o", goal_color)]
            # We assume that cdb9op3o goal_color facts identify color objects.
            # But from domain file, cdb9op3o is a unary predicate on colors only -> color objects are colors themselves.
            # Actually, the cdb9op3o predicate of form (cdb9op3o white) means white and black are color objects.
            # So color "locations" for picking up colors are not spatial, but carried by the robot.
            # However, changing color requires the robot to pick up a different color at the robot's current tile.
            # So to change color carried, robot must be on the same tile as the color object (cdb9op3o color) - but colors have no tile
            # Therefore, from predicates, (cdb9op3o white) does not assign a location, it simply lists available colors by those facts.
            # Looking at action z7pcmkeu:
            # precondition (i4u94rl9 ?robot ?color1) and (cdb9op3o ?color2)
            # effect robot changes carried color from ?color1 to ?color2.
            # So color changes happen by replacing carried color with any available color. There is no location for a color object.
            # Hence changing color (z7pcmkeu) can be done anywhere robot currently is.
            # So no need to move to a color tile for pickup; only one action needed to pick color.

            # Therefore cost to get correct color:
            #   if robot already carries correct color: 0
            #   else: 1 action to switch color (z7pcmkeu).

            # So the cost to get correct color is 0 or 1, no spatial movement required.

            for robot, robot_tile in robot_positions.items():
                # Cost to switch color if needed
                color_change_cost = 0 if robot_colors.get(robot) == goal_color else 1

                # Cost to move from robot_tile to goal_tile by shortest path
                dist = self.shortest_path_length(robot_tile, goal_tile)
                if dist is None:
                    # No path found, ignore this robot
                    continue

                # Total cost: move to tile + possible color change + paint
                cost = dist + color_change_cost + paint_cost

                if cost < min_cost_for_tile:
                    min_cost_for_tile = cost

            # If no robot can reach (e.g. disconnected graph), use large cost but finite
            if min_cost_for_tile == float("inf"):
                min_cost_for_tile = 1000  # Large penalty, but finite

            total_estimated_cost += min_cost_for_tile

        # Return heuristic cost, zero if goals already satisfied
        return total_estimated_cost
