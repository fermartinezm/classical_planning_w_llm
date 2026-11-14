from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting the string.
    
    :param fact: A fact string like '(predicate arg1 arg2 ...)'
    :return: List of components: [predicate, arg1, arg2, ...]
    """
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a pattern with wildcards.
    
    :param fact: Fact string like '(painted tile_1_1 white)'
    :param args: Pattern components, e.g. "painted", "*", "white"
    :return: True if fact matches pattern, else False
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the 'floortile' domain.

    Summary:
    --------
    This heuristic estimates the number of actions required to paint all tiles 
    according to the goal specifications from the current state. It accounts 
    for the robot's positions, tile adjacencies, paint colors needed, and color 
    changes. It is designed to guide a greedy best-first search effectively, 
    prioritizing states closer to completion, though not necessarily admissible.

    Assumptions:
    ------------
    - The domain goal consists exclusively of (painted tile color) predicates.
    - The robot can only paint clear tiles adjacent (up/down/left/right) to its current location.
    - The robot must have the correct paint color equipped to paint a tile.
    - Changing color costs one action and is necessary whenever the desired paint color differs from the robot's current color.
    - Moving the robot one tile costs one action.
    - Painting a tile costs one action.
    - Tiles are painted once to reach the goal color; repainting is not required.
    - There is at least one robot in the domain (can generalize for multiple robots by considering minimal cost among them).
    - Clear tiles are those that have not been painted yet and are free to move into.

    Heuristic Initialization:
    -------------------------
    - Extract tile adjacency relations (up, down, left, right) from static facts, and build a graph for fast shortest path computations.
    - Extract goal tiles and their required colors from goal conditions.
    - Identify all available colors.
    - Extract robot(s) current color(s).
    - Extract robot(s) current locations.
    - Store which tiles are painted and which colors they have from the current state.

    Step-By-Step Thinking for Computing Heuristic:
    ----------------------------------------------
    1. Identify all tiles that still need to be painted or repainted to their goal color:
       - Tiles not painted at all in the current state.
       - Tiles painted with a different color than required in the goal.

    2. For each such tile:
       - Estimate cost of reaching a robot to any adjacent clear tile from its current location.
         Adjacent tiles are those from which the robot can paint this tile (adjacent by up/down/left/right).
       - Estimate the minimal number of required robot move actions to get adjacent.
       - Include cost of painting action (1 action).
       - Include cost of color change actions if the robot currently does not have the required color:
         - If multiple tiles require the same color consecutively, color change cost can be amortized by grouping. But for heuristic efficiency, assume worst case: color change before painting each tile if needed.
         - To keep heuristic efficient, count one color change per different color needed from the robot's current paint. If multiple tiles have the same color, only one change is counted for that color.
         - Since multiple robots may exist with different colors and positions, take minimum over all robots.

    3. Sum the minimal costs for all unpainted/mispainted tiles.

    4. To approximate color change overhead efficiently:
       - Count distinct colors needed minus one (assuming the robot can arrange paint actions to minimize changes).
       - Multiply by 1 action per color change.

    5. The heuristic returns the sum of:
       - Minimal move + paint actions needed to paint all missing tiles
       - Color change actions count (estimated as number of distinct needed colors, minus one if robot already has a needed color)

    Note:
    -----
    The heuristic may underestimate move costs slightly by assuming robot can "teleport" between tiles for painting consecutive tiles, but reflects action costs sufficiently to guide search.

    """

    def __init__(self, task):
        self.goals = task.goals  # goal facts (painted tile color)
        self.static = task.static  # static facts (adjacency and available-color)
        
        # Build tile adjacency map: tile -> set of adjacent tiles (reachable by move)
        self.adjacency = {}
        tiles = set()
        for fact in self.static:
            # We consider up/down/left/right as edges in both directions
            if match(fact, "up", "*", "*"):
                child, parent = get_parts(fact)[1:]  # (up tile_y tile_x): tile_y is above tile_x
                tiles.update([parent, child])
                self.adjacency.setdefault(parent, set()).add(child)
            elif match(fact, "down", "*", "*"):
                parent, child = get_parts(fact)[1:]
                tiles.update([parent, child])
                self.adjacency.setdefault(parent, set()).add(child)
            elif match(fact, "left", "*", "*"):
                right, left = get_parts(fact)[1:]
                tiles.update([left, right])
                self.adjacency.setdefault(left, set()).add(right)
            elif match(fact, "right", "*", "*"):
                left, right = get_parts(fact)[1:]
                tiles.update([left, right])
                self.adjacency.setdefault(left, set()).add(right)

        # Because the domain defines movement with down->up, right->left etc. edges,
        # adjacency edges have been added for one direction only, add reverse edges for movement:
        # We'll add all missing reverse edges:
        for tile in list(self.adjacency):
            for adj in list(self.adjacency[tile]):
                self.adjacency.setdefault(adj, set()).add(tile)

        self.tiles = tiles

        # Extract available colors
        self.available_colors = set()
        for fact in self.static:
            if match(fact, "available-color", "*"):
                color = get_parts(fact)[1]
                self.available_colors.add(color)

        # Parse goal painted tiles and their target colors
        self.goal_painted = {}
        for fact in self.goals:
            if match(fact, "painted", "*", "*"):
                parts = get_parts(fact)
                tile, color = parts[1], parts[2]
                self.goal_painted[tile] = color

    def __call__(self, node):
        state = node.state

        # If goal reached, heuristic is zero.
        # Check if all goal painted facts are in state.
        if self.goal_painted.items() <= {
            (get_parts(f)[1], get_parts(f)[2])
            for f in state
            if match(f, "painted", "*", "*")
        }:
            return 0

        # Extract robot info: their current location(s) and color(s).
        robots = {}
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                robot, location = get_parts(fact)[1], get_parts(fact)[2]
                if robot not in robots:
                    robots[robot] = {"location": location, "color": None}
                else:
                    robots[robot]["location"] = location
            elif match(fact, "robot-has", "*", "*"):
                robot, color = get_parts(fact)[1], get_parts(fact)[2]
                if robot not in robots:
                    robots[robot] = {"location": None, "color": color}
                else:
                    robots[robot]["color"] = color

        # Determine painted tiles and their colors from state
        painted_tiles = {}
        for fact in state:
            if match(fact, "painted", "*", "*"):
                tile, color = get_parts(fact)[1], get_parts(fact)[2]
                painted_tiles[tile] = color

        # Determine clear tiles from state (needed to move into or adjacent)
        clear_tiles = set()
        for fact in state:
            if match(fact, "clear", "*"):
                clear_tiles.add(get_parts(fact)[1])

        # For each tile that still needs painting or repainting:
        # Identify tiles where (painted tile color) != goal or tile not painted.
        to_paint_tiles = []
        for tile, goal_color in self.goal_painted.items():
            painted_color = painted_tiles.get(tile, None)
            if painted_color != goal_color:
                to_paint_tiles.append((tile, goal_color))

        # If no tiles to paint, heuristic is zero
        if not to_paint_tiles:
            return 0

        # Precompute shortest paths for robot moves using BFS on adjacency graph.
        # We'll need to compute shortest distance from robot location to tiles adjacent to the tile to be painted.
        # For efficiency, compute shortest paths once per robot location to all tiles.
        from collections import deque

        def shortest_distances(start):
            """
            Compute shortest distance in tile graph from start to every reachable tile.
            Return dict tile->distance.
            """
            dist = {start: 0}
            queue = deque([start])
            while queue:
                current = queue.popleft()
                for neighbor in self.adjacency.get(current, []):
                    if neighbor not in dist:
                        dist[neighbor] = dist[current] + 1
                        queue.append(neighbor)
            return dist

        # For all robots, compute shortest distances from their locations to all tiles.
        robot_dists = {}
        for robot, info in robots.items():
            if info["location"] is not None:
                robot_dists[robot] = shortest_distances(info["location"])

        # Identify sets of tiles adjacent to each tile, i.e., positions from which robot can paint that tile.
        # Condition for painting is robot-at at an adjacent tile (via up/down/left/right edges)
        tile_adjacent_positions = {}
        for tile in self.tiles:
            # Find all tiles T s.t. painted action can be done at tile from T (meaning tile adjacent to T)
            adj_positions = set()
            # Check all tiles if tile is adjacent in adjacency edges
            for pos in self.adjacency:
                if tile in self.adjacency.get(pos, []):
                    # The robot can paint tile when robot is at pos
                    adj_positions.add(pos)
            tile_adjacent_positions[tile] = adj_positions

        # Now compute heuristic:
        # For each tile to paint, compute minimal cost over all robots:
        # cost = minimal moves from robot location to an adjacent tile that is clear + 
        #        1 paint action + 
        #        (cost of changing color if needed)
        # Color change cost will be handled separately.
        # We'll gather all colors to paint for overall color change cost estimation.

        # To avoid overestimating color changes, track which colors are needed.
        needed_colors = set(goal_color for _, goal_color in to_paint_tiles)

        total_move_and_paint_cost = 0
        # For each tile to paint, minimal cost among all robots
        for tile, goal_color in to_paint_tiles:
            adj_positions = tile_adjacent_positions.get(tile, set())
            # Only consider adjacent tiles that are clear for robot to stand on
            reachable_positions = adj_positions & clear_tiles
            if not reachable_positions:
                # Possibly no clear position adjacent - estimate as large cost (we consider heuristic finite so use large number)
                minimal_cost = 20  # Arbitrary large cost; could be tuned
            else:
                minimal_cost = None
                for robot, info in robots.items():
                    if info["location"] is None:
                        continue
                    distances = robot_dists.get(robot, {})
                    for pos in reachable_positions:
                        dist = distances.get(pos, None)
                        if dist is not None:
                            cost = dist + 1  # dist moves to reach adjacency + 1 paint action
                            if minimal_cost is None or cost < minimal_cost:
                                minimal_cost = cost
                # If no robot can reach any adjacent clear tile (e.g., disconnected), assign large cost
                if minimal_cost is None:
                    minimal_cost = 20

            total_move_and_paint_cost += minimal_cost

        # Now handle color changes:
        # If the robot(s) currently have one of the needed colors, then number of color changes is number of distinct needed colors - 1.
        # Otherwise, number of distinct needed colors.
        # If multiple robots, consider minimal number of color changes needed assuming one robot does all work.
        robot_current_colors = {info["color"] for info in robots.values() if info["color"] is not None}

        # Count color changes as:
        # distinct needed colors - number of colors already held by some robot intersecting needed_colors (0 or 1, since robot has only one color)
        color_changes = len(needed_colors)
        # If any robot currently holds a needed color, reduce color_changes by one:
        if robot_current_colors & needed_colors:
            color_changes -= 1

        # Each color change costs one action:
        color_change_cost = max(color_changes, 0)

        heuristic_value = total_move_and_paint_cost + color_change_cost
        return heuristic_value
