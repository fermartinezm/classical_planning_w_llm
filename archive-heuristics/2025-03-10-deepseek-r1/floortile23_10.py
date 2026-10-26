from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()

def match(fact, *args):
    """Check if a PDDL fact matches a given pattern with wildcards."""
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class Floortile23Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles with their specified colors. 
    It considers movement steps for robots to reach adjacent tiles, necessary color changes, and the paint actions.

    # Assumptions
    - Robots can move freely between adjacent tiles (ignoring whether tiles are clear).
    - Color changes are always possible if the color is available.
    - Each tile is painted from an adjacent tile, requiring one action.
    - The grid is structured such that Manhattan distance accurately reflects movement steps.

    # Heuristic Initialization
    - Extracts goal conditions to determine required colors for each tile.
    - Parses static facts to build a coordinate system for tiles.
    - Identifies available colors from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each goal tile, check if it's already painted correctly.
    2. For each unpainted/miscolored tile:
        a. Determine adjacent tiles using coordinates.
        b. For each robot, calculate the Manhattan distance to each adjacent tile.
        c. Compute the minimal movement steps, color change cost, and paint action.
        d. Sum the minimal cost across all robots for the tile.
    3. Aggregate costs for all tiles to form the heuristic value.
    """

    def __init__(self, task):
        """Initialize by extracting goal conditions, tile coordinates, and available colors."""
        self.goal_painted = {}
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'painted':
                self.goal_painted[parts[1]] = parts[2]

        self.tile_coords = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] in ['up', 'down', 'left', 'right']:
                for tile in [parts[1], parts[2]]:
                    if tile not in self.tile_coords:
                        coords = self._parse_tile_coords(tile)
                        if coords:
                            self.tile_coords[tile] = coords

        self.available_colors = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'available-color':
                self.available_colors.add(parts[1])

    def _parse_tile_coords(self, tile_name):
        """Extract (x, y) coordinates from a tile name (e.g., 'tile_1_2' -> (1, 2))."""
        if not tile_name.startswith('tile_'):
            return None
        parts = tile_name.split('_')
        if len(parts) != 3:
            return None
        try:
            return (int(parts[1]), int(parts[2]))
        except ValueError:
            return None

    def __call__(self, node):
        """Compute the heuristic estimate for the given state."""
        state = node.state
        total_cost = 0

        robots = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'robot-at':
                robot = parts[1]
                pos = parts[2]
                robots[robot] = {'pos': pos}
            elif parts[0] == 'robot-has':
                robot = parts[1]
                robots.setdefault(robot, {})['color'] = parts[2]

        for tile, req_color in self.goal_painted.items():
            current_color = None
            for fact in state:
                if match(fact, 'painted', tile, '*'):
                    current_color = get_parts(fact)[2]
                    break
            if current_color == req_color:
                continue

            tx, ty = self.tile_coords.get(tile, (None, None))
            if tx is None or ty is None:
                continue

            adj_tiles = [
                (tx + 1, ty), (tx - 1, ty),
                (tx, ty + 1), (tx, ty - 1)
            ]
            valid_adj = [f'tile_{x}_{y}' for x, y in adj_tiles if f'tile_{x}_{y}' in self.tile_coords]

            min_tile_cost = float('inf')
            for robot, data in robots.items():
                pos = data.get('pos')
                if pos not in self.tile_coords:
                    continue
                rx, ry = self.tile_coords[pos]
                color = data.get('color', None)

                min_dist = min(
                    (abs(rx - ax) + abs(ry - ay) for adj in valid_adj for ax, ay in [self.tile_coords[adj]]),
                    default=float('inf')
                )

                color_cost = 0 if color == req_color else 1
                cost = min_dist + color_cost + 1
                if cost < min_tile_cost:
                    min_tile_cost = cost

            if min_tile_cost != float('inf'):
                total_cost += min_tile_cost

        return total_cost
