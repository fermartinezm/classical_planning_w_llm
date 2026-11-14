from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern with wildcards.

    Args:
        fact (str): The fact string, e.g., "(predicate arg1 arg2)".
        args (str): Pattern to match with wildcards (*).

    Returns:
        bool: True if fact matches pattern, else False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (color) domain.

    Summary:
    --------
    This heuristic estimates the number of actions required to reach a goal state
    by approximating the steps needed to paint all required tiles. The main effort
    involves bringing a suitable robot (color-2) coordinated with a color to a tile-2 
    that is currently unpainted and "down" predicate true, then performing the paint_up 
    action.

    Assumptions:
    ------------
    - Painting each goal tile requires exactly one paint_up action.
    - To paint a tile, the robot must have the correct available-color on an upward tile.
    - Robots move colors up along "up" relations by change_color actions.
    - Robots can only paint a tile if (available-color robot tile), (robot-at robot color), (free-color color), and (down tile) hold.
    - Robots start at some color and some tile with some available-color facts.
    - Only one paint_up is needed per tile.
    - Movement of colors along tiles and robot's color movements cost one action each.
    - Free colors are a resource that must be available to paint.
    - The heuristic is not admissible but tries to approximate minimal steps.

    Heuristic Initialization:
    --------------------------
    - Extract goal tiles to be painted.
    - Extract static "up" relations to build a graph representing tile adjacency upwards.
    - Build a reversed adjacency graph ("down" edges) for fast traversal.
    - Extract all color-2 robots and their starting positions (colors).
    - Extract which colors are currently free.
    - Store these for efficient lookup during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
    -----------------------------------------------
    1. Identify goal tiles that are currently NOT painted yet.
    2. For each such tile, estimate the cost to paint it:
        a. Find all robots (color-2) that could potentially paint it.
        b. For each robot, estimate cost to bring the robot’s available-color to the tile
           (via change_color actions following "up" edges).
        c. Estimate cost to move the robot to a color associated with the tile to enable painting.
           Robots are located at some colors via "robot-at" facts.
        d. Ensure the robot has a free color to paint.
        e. Sum the estimated costs: 
           - color changes (number of change_color actions needed to align available-color with tile),
           - robot movements (if robot color != tile required color),
           - one paint_up action.
    3. Take the minimal among the costs for all robots for each tile.
    4. Sum these minimal costs for all unpainted goal tiles to produce the heuristic.
    5. If no unpainted tiles remain (goal state), heuristic = 0.

    Note:
    -----
    The heuristic is computationally efficient by using BFS distance computations on the tile graph 
    and by skipping expensive searches. It is not guaranteed to be admissible, but aims to guide 
    the search by realistic action cost approximations, minimizing node expansions.
    """

    def __init__(self, task):
        """Extract static info and goal-related objects for heuristic calculations."""
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state

        # Extract goal tiles that must be painted (predicate: painted tile-2)
        self.goal_tiles = set()
        for goal_fact in self.goals:
            if match(goal_fact, "painted", "*"):
                parts = get_parts(goal_fact)
                self.goal_tiles.add(parts[1])

        # Build graph of tiles connected by "up" relation: up(tile_i) -> tile_j
        self.up_graph = {}
        self.down_graph = {}
        for fact in self.static:
            if match(fact, "up", "*", "*"):
                parts = get_parts(fact)
                t_from, t_to = parts[1], parts[2]
                self.up_graph.setdefault(t_from, set()).add(t_to)
                self.down_graph.setdefault(t_to, set()).add(t_from)

        # Store all tiles in the domain seen in "up" and "down" predicates (for BFS)
        self.all_tiles = set(self.up_graph.keys()).union(set(self.down_graph.keys()))

    def __call__(self, node):
        """Estimate number of actions needed to paint all remaining goal tiles.

        Args:
            node: Search node containing current state.

        Returns:
            int: Estimated cost to reach goal state.
        """
        state = node.state

        # If goal reached, cost is zero
        if self.goals <= state:
            return 0

        # Identify painted tiles (painted predicate)
        painted_tiles = set()
        for fact in state:
            if match(fact, "painted", "*"):
                painted_tiles.add(get_parts(fact)[1])

        # Compute unpainted goal tiles needing painting
        unpainted_goal_tiles = self.goal_tiles - painted_tiles
        if not unpainted_goal_tiles:
            return 0  # Already painted

        # Parse robot data:
        # robot-at(robot: color-2, color)
        robot_positions = {}  # robot -> color they are at (color)
        for fact in state:
            # predicate is robot-at, expects: (robot-at ?v1 - color-2 ?v3 - color)
            if match(fact, "robot-at", "*", "*"):
                parts = get_parts(fact)
                robot_positions[parts[1]] = parts[2]

        # available-color(robot or tile, tile)
        # This predicate shows which robot/tile has which available-color on which tile.
        # According to domain: (available-color ?v1 - robot ?v2 - tile)
        available_color_facts = set()
        for fact in state:
            if match(fact, "available-color", "*", "*"):
                parts = get_parts(fact)
                available_color_facts.add((parts[1], parts[2]))

        # free-color(color)
        free_colors = set()
        for fact in state:
            if match(fact, "free-color", "*"):
                free_colors.add(get_parts(fact)[1])

        # Also note: paint_up requires robot-at(robot color-2, color),
        # available-color(robot tile), available-color(tile tile),
        # free-color(color), down(tile-2)

        # Extract tiles with down predicate (tiles that can be painted)
        down_tiles = set()
        for fact in state:
            if match(fact, "down", "*"):
                down_tiles.add(get_parts(fact)[1])

        # --- Helper functions ---

        def bfs_distance(start_set, goal_set, graph):
            """
            BFS from multiple start nodes to find shortest distance to any goal node.

            Args:
                start_set (set[str]): Starting nodes
                goal_set (set[str]): Goal nodes
                graph (dict[str, set[str]]): Adjacency list graph

            Returns:
                int or None: shortest distance, or None if no path
            """
            from collections import deque

            visited = set()
            queue = deque()

            for s in start_set:
                queue.append((s, 0))
                visited.add(s)

            while queue:
                node, dist = queue.popleft()
                if node in goal_set:
                    return dist
                for neighbor in graph.get(node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
            return None

        # --- Heuristic calculation ---

        total_cost = 0

        # For each unpainted goal tile estimate minimal cost to paint it.
        for goal_tile in unpainted_goal_tiles:

            # Painting requires that (down goal_tile)
            # Check if tile is down; if not, painting might be impossible or cost infinite
            if goal_tile not in down_tiles:
                # Cannot paint tile; assign high cost to push solver away
                total_cost += 1000
                continue

            # Possible available-colors that can be on tiles related to goal_tile:
            # Find tiles connected upwards/downwards related to goal_tile
            # We want to find from which tiles a robot can bring an available-color upwards to goal_tile (via change_color)
            # Since change_color acts on (available-color robot tile) and (up tile tile2),
            # to get available-color on goal_tile, available-color must be on a tile reachable below via up edges.

            # So compute all reachable tiles below goal_tile backward along "up" edges for available-color reachability
            # BFS backwards along up edges is done on the down_graph
            reachable_tiles = set()
            from collections import deque

            queue = deque([goal_tile])
            reachable_tiles.add(goal_tile)
            while queue:
                current = queue.popleft()
                for below_tile in self.down_graph.get(current, []):
                    if below_tile not in reachable_tiles:
                        reachable_tiles.add(below_tile)
                        queue.append(below_tile)

            # Potential start points for available-color on tiles
            potential_available_color_tiles = reachable_tiles

            # Also consider available-color facts on robots as well (robot name or tile name)
            # Because available-color can be (available-color robot tile)
            # And robots move colors up.

            # For heuristic:
            # Find all (available-color robot tile) with tile in potential_available_color_tiles
            # and robot from robot_positions keys (robots are color-2 objects)
            candidates = []

            # Gather all robots that have available colors on tiles in reachable tiles
            for robot, avail_tile in available_color_facts:
                # robot here can be robot name (color-2) or tile; we are interested in robot names that are in robot_positions
                if robot in robot_positions and avail_tile in potential_available_color_tiles:
                    candidates.append((robot, avail_tile))

            # If no candidate robots found, then no known way to bring color to goal tile, assign high cost
            if not candidates:
                # Possibly available-color only on tiles, try to find any available-color on any reachable tile regardless of robot
                # This heuristic ignores such complicated cases, assign high cost
                total_cost += 500
                continue

            min_robot_cost = None

            for robot, avail_tile in candidates:
                # 1. Compute cost of bringing avail_color from avail_tile up to goal_tile
                # This is the BFS distance from avail_tile to goal_tile along up_graph edges
                dist_up = bfs_distance({avail_tile}, {goal_tile}, self.up_graph)
                if dist_up is None:
                    # Cannot move the available-color from avail_tile up to goal_tile
                    continue

                # 2. Compute cost for robot to move from current robot position (color) to the color required for painting
                # Robot position color:
                robot_color = robot_positions.get(robot)
                if robot_color is None:
                    # No info about robot position, assign high cost
                    continue

                # We must estimate robot movement cost between colors.
                # The robot-at predicate uses "color-2" robot and "color" as second arg.
                # But the domain does not define relations between colors explicitly.
                # Assume cost 0 if robot_color == goal_tile color; else cost 1 for any move
                # Since we have no info about robot color adjacency, assign cost 0 if same color, else 1 action for move_down (robot movement)
                # The robot must be at a color matching tile to paint.

                # The standard action to move is move_down:
                # Preconditions: (available-color robot tile), (available-color ?color tile), it moves robot-at between colors.
                # But domain is ambiguous; for heuristic assign simple cost:
                robot_move_cost = 0 if robot_color == goal_tile else 1

                # 3. Check if robot's color is free (free-color)
                # paint_up requires free-color of the robot's color to be true.
                # If robot_color is not free, we count a cost of 1 to free it (approximate)
                free_color_cost = 0 if robot_color in free_colors else 1

                # 4. Painting itself costs 1 paint_up action
                paint_cost = 1

                # 5. Sum total estimated cost for this robot to paint goal_tile
                cost = dist_up + robot_move_cost + free_color_cost + paint_cost

                if min_robot_cost is None or cost < min_robot_cost:
                    min_robot_cost = cost

            # If no robot could paint this tile, assign high cost
            if min_robot_cost is None:
                total_cost += 500
            else:
                total_cost += min_robot_cost

        return total_cost
