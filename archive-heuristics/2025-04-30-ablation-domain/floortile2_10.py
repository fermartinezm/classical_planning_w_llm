from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Utility function to check if a PDDL fact matches a given pattern.
    - `fact`: The fact as a string (e.g., "(at robot1 tile_0_1)").
    - `args`: The pattern to match (e.g., "at", "*", "tile_0_1").
    - Returns `True` if the fact matches the pattern, `False` otherwise.
    """
    parts = fact[1:-1].split()  # Remove parentheses and split into individual elements.
    # Check if all parts match the corresponding args pattern.
    # zip stops at the shortest iterable, so this works even if len(args) < len(parts)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with their target colors. It considers the number of tiles to paint, the need
    for robots to acquire the correct colors, and the minimum movement cost for
    robots to reach the unpainted goal tiles.

    # Assumptions
    - Tiles are arranged in a grid connected by up/down/left/right relations.
    - Robots can move between adjacent tiles (cost 1).
    - Robots can change color if the new color is available (cost 1).
    - Robots can paint a clear tile if they are at the tile and have the correct color (cost 1).
    - Goal tiles are initially clear and not painted with the wrong color.
    - All goal tiles and initial robot locations are part of the connected grid defined by adjacency facts.

    # Heuristic Initialization
    - Extracts goal conditions to identify which tiles need which colors.
    - Parses static facts (up, down, left, right) to build an adjacency graph of the tile grid.
    - Pre-calculates shortest distances between all pairs of tiles using Breadth-First Search (BFS) on the adjacency graph.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal tiles that are not yet painted with their required color.
    2. If no such tiles exist, the state is a goal state, and the heuristic is 0.
    3. Extract the current location and color of each robot from the state.
    4. Calculate the "paint cost": This is simply the number of unpainted goal tiles, as each requires one `paint` action.
    5. Calculate the "change-color cost": For each color required by at least one unpainted goal tile, check if *any* robot currently possesses that color. If no robot has the required color, at least one `change-color` action is necessary *somewhere* by *some* robot to make that color available. Add 1 to the cost for each such required color that no robot currently has. This is a lower bound on color changes needed across all robots.
    6. Calculate the "movement cost": For each unpainted goal tile, find the minimum distance from *any* robot's current location to that tile using the pre-calculated distances. Sum these minimum distances. This is a relaxed estimate of the total movement needed, assuming tasks can be distributed among robots.
    7. The total heuristic value is the sum of the paint cost, change-color cost, and movement cost.
    8. If any unpainted goal tile is unreachable from all robots (based on pre-calculated distances), the state is likely unsolvable, and the heuristic returns infinity.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building the grid graph."""
        self.goals = task.goals
        self.static_facts = task.static

        # Extract goal tiles and their required colors
        self.goal_tiles = {} # tile -> color
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'painted':
                tile, color = parts[1], parts[2]
                self.goal_tiles[tile] = color

        # Build adjacency graph from static facts
        self.adjacency = {}
        all_tiles_set = set()
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] in ['up', 'down', 'left', 'right']:
                t1, t2 = parts[1], parts[2]
                self.adjacency.setdefault(t1, set()).add(t2)
                self.adjacency.setdefault(t2, set()).add(t1)
                all_tiles_set.add(t1)
                all_tiles_set.add(t2)

        # Add goal tiles to the set of all tiles if they are not in adjacency facts
        # This ensures we calculate distances to goal tiles even if they are isolated
        # (though problems with isolated goal tiles might be unsolvable).
        all_tiles_set.update(self.goal_tiles.keys())

        # Pre-calculate distances between all pairs of tiles using BFS
        self.distances = {} # (tile1, tile2) -> distance
        all_tiles = list(all_tiles_set)

        for start_tile in all_tiles:
            self.distances[(start_tile, start_tile)] = 0
            queue = deque([(start_tile, 0)])
            visited = {start_tile}

            while queue:
                current_tile, dist = queue.popleft()

                # Only explore neighbors if the current tile has adjacency info
                if current_tile in self.adjacency:
                    for neighbor in self.adjacency[current_tile]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.distances[(start_tile, neighbor)] = dist + 1
                            queue.append((neighbor, dist + 1))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unpainted goal tiles
        unpainted_goal_tiles = {} # tile -> required_color
        for tile, color in self.goal_tiles.items():
            # Check if the tile is already painted with the correct color
            if f'(painted {tile} {color})' not in state:
                 # We assume states don't have tiles painted with the wrong color
                 unpainted_goal_tiles[tile] = color

        # 2. If no such tiles exist, the state is a goal state, and the heuristic is 0.
        if not unpainted_goal_tiles:
            return 0

        # 3. Extract robot locations and colors
        robot_info = {} # robot -> {'location': tile, 'color': color}
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "robot-at", "*", "*"):
                robot, tile = parts[1], parts[2]
                robot_info.setdefault(robot, {})['location'] = tile
            elif match(fact, "robot-has", "*", "*"):
                robot, color = parts[1], parts[2]
                robot_info.setdefault(robot, {})['color'] = color

        # 4. Calculate paint cost
        paint_cost = len(unpainted_goal_tiles)

        # 5. Calculate change-color cost
        change_color_cost = 0
        needed_colors = set(unpainted_goal_tiles.values())
        robots_with_color = {info['color'] for info in robot_info.values()}

        for color in needed_colors:
            if color not in robots_with_color:
                change_color_cost += 1 # At least one robot needs to acquire this color

        # 6. Calculate movement cost
        movement_cost = 0
        for tile in unpainted_goal_tiles:
            min_dist_to_tile = float('inf')
            for robot, info in robot_info.items():
                robot_loc = info['location']
                # Find distance from robot_loc to tile
                # Use .get with default for safety in case robot_loc or tile wasn't in the initial all_tiles_set
                dist = self.distances.get((robot_loc, tile), float('inf'))
                min_dist_to_tile = min(min_dist_to_tile, dist)

            # 8. Check for unreachable tiles
            if min_dist_to_tile == float('inf'):
                 # A goal tile is unreachable from any robot based on the grid graph.
                 # This state is likely unsolvable.
                 return float('inf') # Return infinity or a large number

            movement_cost += min_dist_to_tile

        # 7. Total heuristic
        total_cost = paint_cost + change_color_cost + movement_cost

        return total_cost
