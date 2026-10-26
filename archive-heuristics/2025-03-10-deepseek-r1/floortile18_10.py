from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    Estimates the number of actions needed to paint all goal tiles by considering minimal movement and color change costs for each robot. The heuristic assumes a grid layout and uses Manhattan distance for movement estimation.

    # Assumptions
    - Tile names follow the pattern 'tile_row_col'.
    - Robots can change colors if available.
    - Movement is based on Manhattan distance in a grid.
    - Paint actions require adjacency to the target tile.

    # Heuristic Initialization
    - Extracts available colors from static facts.
    - Builds adjacency map for each tile using direction predicates.
    - Parses tile coordinates from their names.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each goal tile not yet painted:
        a. Find adjacent positions (x) from which it can be painted.
        b. For each robot, compute minimal distance to any x (Manhattan).
        c. Add color change cost if the robot's color differs.
        d. Add paint action cost.
        e. Use the minimal robot cost for each tile.
    2. Sum all minimal tile costs.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        self.available_colors = set()
        self.adjacency_map = {}  # Maps tile to adjacent tiles (robot positions)
        self.tile_coords = {}    # Maps tile to (row, col)

        # Parse static facts
        for fact in self.static:
            parts = fact.strip('()').split()
            if parts[0] == 'available-color':
                self.available_colors.add(parts[1])
            elif parts[0] in ['up', 'down', 'left', 'right']:
                y, x = parts[1], parts[2]
                self.adjacency_map.setdefault(y, []).append(x)

        # Collect all tiles and parse coordinates
        all_tiles = set(self.adjacency_map.keys())
        for adj_list in self.adjacency_map.values():
            all_tiles.update(adj_list)
        for tile in all_tiles:
            coords = tile.split('_')[1:3]
            if len(coords) == 2 and coords[0].isdigit() and coords[1].isdigit():
                self.tile_coords[tile] = (int(coords[0]), int(coords[1]))

    def __call__(self, node):
        state = node.state

        # Extract required tiles and colors
        required = {}
        for goal in self.goals:
            if goal.startswith('(painted '):
                parts = goal[1:-1].split()
                required[parts[1]] = parts[2]

        # Current painted tiles
        current_painted = {}
        for fact in state:
            if fact.startswith('(painted '):
                parts = fact[1:-1].split()
                current_painted[parts[1]] = parts[2]

        # Unpainted or incorrect tiles
        unpainted = []
        for tile, color in required.items():
            if current_painted.get(tile) != color:
                unpainted.append((tile, color))

        # Robots' current positions and colors
        robots = {}
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'robot-at':
                robot = parts[1]
                robots[robot] = {'pos': parts[2], 'color': None}
            elif parts[0] == 'robot-has':
                robot = parts[1]
                if robot not in robots:
                    robots[robot] = {'pos': None, 'color': parts[2]}
                else:
                    robots[robot]['color'] = parts[2]

        total_cost = 0

        for tile, req_color in unpainted:
            if tile not in self.adjacency_map or not self.adjacency_map[tile]:
                total_cost += 1000  # Penalize unreachable
                continue

            min_cost = float('inf')
            for robot in robots.values():
                pos = robot.get('pos')
                color = robot.get('color')
                if not pos or pos not in self.tile_coords:
                    continue

                # Minimal distance to adjacent x
                min_dist = min(
                    (abs(self.tile_coords[pos][0] - self.tile_coords[x][0]) +
                     abs(self.tile_coords[pos][1] - self.tile_coords[x][1]))
                    for x in self.adjacency_map[tile] if x in self.tile_coords
                ) if self.adjacency_map[tile] else float('inf')

                if min_dist == float('inf'):
                    continue

                # Color change cost
                color_cost = 1 if color != req_color and req_color in self.available_colors else 0

                total = min_dist + color_cost + 1  # Paint action
                if total < min_cost:
                    min_cost = total

            total_cost += min_cost if min_cost != float('inf') else 1000

        return total_cost
