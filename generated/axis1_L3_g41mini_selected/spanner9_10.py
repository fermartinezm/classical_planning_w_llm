from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(available-color color1 tile2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary:
    ----------
    This heuristic estimates the number of remaining actions needed to paint all tiles 
    required by the goal in the secret_domain, where a robot (color-2) must move "up" 
    through tiles (tile) changing its available color stepwise and paint certain tiles 
    (tile-2) that are "down" from these tiles using the "paint_up" action.

    Assumptions:
    ------------
    - Painting a tile requires the robot's color to be "free" and "robot-at" relationships satisfied.
    - The robot can move "up" through tiles by changing its available color from one tile to the next.
    - Only tiles mentioned in goals as painted (painted tile-2) need consideration.
    - Each paint action can be done only when the robot has the corresponding color free and is effectively positioned.
    - Movement to the next tile via "change_color" requires the robot's available color to be positioned correctly.
    - Only one robot (color-2) is active.
    - The heuristic does not have to be admissible but should be informative and efficiently computable.

    Heuristic Initialization:
    ---------------------------
    - Extract the goal tiles that need to be painted.
    - Extract the static "up" and "down" relationships between tiles.
    - Identify the tiles linked to tile-2 tiles that need painting.
    - Extract the set of all robot colors (color-2 objects).
    - Precompute tile adjacency for efficient traversal upwards.
    
    Step-By-Step Thinking for Computing Heuristic:
    -----------------------------------------------
    1. Identify which goal tiles (tile-2) remain unpainted in the current state.
    2. For each unpainted goal tile (tile-2), find:
       a) Its associated "down" tile (tile) which must be available-color for the robot to paint it.
    3. For each robot (color-2), determine its current available color tile.
    4. For each goal paint-unpainted tile:
       - Find the minimal number of "change_color" actions needed for some robot to reach the specific tile 
         from its current position by moving "up" stepwise through tiles.
         This models how many color changes are needed to get the robot's available color aligned with the tile.
       - Add 1 action for the "paint_up" action itself.
    5. Since multiple robots and multiple goal tiles can exist, take the minimal cost among all robots for each tile.
    6. Sum these minimal costs over all unpainted goal tiles.
    7. This sum estimates how many total steps are needed to paint all required tiles, considering movement and painting.
    8. Return 0 if all goal tiles are painted.

    Note:
    - This heuristic is not admissible but aims to guide search efficiently.
    - The heuristic cost accounts for the cost to move up the tile chain plus painting action per tile.
    """
    def __init__(self, task):
        # Store goal tiles that need to be painted (type tile-2)
        self.goal_painted_tiles = set()
        for g in task.goals:
            parts = get_parts(g)
            if parts[0] == "painted" and len(parts) == 2:
                self.goal_painted_tiles.add(parts[1])

        # Store static facts for lookups
        self.up_graph = dict()  # adjacency from tile -> tile (up)
        self.down_of_tile2 = dict()  # tile2 -> tile (down)
        self.tile2_for_down = dict()  # tile -> tile2 inverse mapping

        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "up" and len(parts) == 3:
                self.up_graph.setdefault(parts[1], set()).add(parts[2])
            elif parts[0] == "down" and len(parts) == 2:
                tile2, tile = parts[1], parts[2]
                self.down_of_tile2[tile2] = tile
                self.tile2_for_down[tile] = tile2

        # Cache robots of type color-2 (from initial state objects)
        # Since the domain uses typed objects:
        # We'll extract from initial state's "available-color" facts the color-2 robots
        # "available-color ?v1 - robot ?v2 - tile" with ?v1 typed as color-2
        self.color2_robots = set()
        for fact in task.initial_state:
            if match(fact, "available-color", "*", "*"):
                parts = get_parts(fact)
                # First argument is the robot (color-2)
                self.color2_robots.add(parts[1])

        # Precompute all tile adjacency by expanding 'up' transitively to speed BFS searches
        # We'll keep adjacency as given for BFS to find paths

    def __call__(self, node):
        state = node.state

        # Quick check: are all goal tiles painted? If yes, heuristic = 0.
        painted_tiles = set()
        for fact in state:
            if match(fact, "painted", "*"):
                parts = get_parts(fact)
                painted_tiles.add(parts[1])

        unpainted_goals = self.goal_painted_tiles - painted_tiles
        if not unpainted_goals:
            return 0

        # Extract current available-color tile for each robot
        # We expect facts: (available-color ?robot ?tile)
        robot_positions = dict()
        for fact in state:
            if match(fact, "available-color", "*", "*"):
                parts = get_parts(fact)
                robot, tile = parts[1], parts[2]
                if robot in self.color2_robots:
                    robot_positions[robot] = tile

        # For BFS path search
        def shortest_up_path(start, goal):
            """
            Find shortest number of 'change_color' steps moving upward from start tile to goal tile
            using 'up' adjacency relation.
            Returns:
                int: number of steps (edges) or None if unreachable
            """
            if start == goal:
                return 0
            from collections import deque
            queue = deque([(start, 0)])
            visited = {start}
            while queue:
                current, dist = queue.popleft()
                for neighbor in self.up_graph.get(current, []):
                    if neighbor == goal:
                        return dist + 1
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
            return None

        total_cost = 0

        for tile2 in unpainted_goals:
            # Get the tile under this tile2 where the robot must be positioned (available-color)
            if tile2 not in self.down_of_tile2:
                # Defensive fallback: if no down tile found, assume paint is impossible = large cost
                total_cost += 1000
                continue
            required_tile = self.down_of_tile2[tile2]

            # For each robot, compute cost to reach required_tile from current available-color tile
            min_cost = None
            for r, pos in robot_positions.items():
                change_steps = shortest_up_path(pos, required_tile)
                if change_steps is None:
                    # Can't reach this tile from current position of this robot => ignore this robot
                    continue
                # One paint_up action required after reaching position
                cost = change_steps + 1
                if min_cost is None or cost < min_cost:
                    min_cost = cost

            # If no robot can reach it, high penalty to discourage that branch
            if min_cost is None:
                min_cost = 1000
            total_cost += min_cost

        return total_cost
