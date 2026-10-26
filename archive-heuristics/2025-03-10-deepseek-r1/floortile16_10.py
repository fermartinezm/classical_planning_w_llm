from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class floortile16Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles correctly. It calculates the minimal cost for each tile by considering robot movement, color changes, and painting actions, using Manhattan distance for movement estimation.

    # Assumptions
    - The grid is regular, allowing Manhattan distance to approximate movement steps.
    - Robots can move to adjacent tiles without considering intermediate obstacles.
    - Each tile can be painted once a robot is on an adjacent tile with the correct color.

    # Heuristic Initialization
    - Extracts goal conditions to identify required tiles and their colors.
    - Parses tile coordinates from their names (e.g., 'tile_3_2' becomes (3,2)).
    - Builds adjacency lists using static 'up', 'down', 'left', and 'right' facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each goal tile, check if it's already correctly painted; if not, add to required tiles.
    2. For each required tile:
        a. Find adjacent tiles from static data.
        b. For each robot, compute minimal Manhattan distance to any adjacent tile.
        c. Add color change cost if the robot's color doesn't match the goal.
        d. Sum movement, color change, and painting costs.
        e. Select the minimal cost across all robots for the tile.
    3. Sum all minimal tile costs for the total heuristic value.
    """

    def __init__(self, task):
        self.goal_painted = {}
        for goal in task.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'painted':
                self.goal_painted[parts[1]] = parts[2]

        self.tile_coords = {}
        self.adjacent = {}

        for fact in task.static:
            parts = fact[1:-1].split()
            if parts[0] in ['up', 'down', 'left', 'right']:
                from_tile = parts[2]
                to_tile = parts[1]
                if from_tile not in self.adjacent:
                    self.adjacent[from_tile] = set()
                self.adjacent[from_tile].add(to_tile)

        for tile in self.adjacent.keys():
            _, x, y = tile.split('_')
            self.tile_coords[tile] = (int(x), int(y))
        for adj_list in self.adjacent.values():
            for tile in adj_list:
                if tile not in self.tile_coords:
                    _, x, y = tile.split('_')
                    self.tile_coords[tile] = (int(x), int(y))

    def __call__(self, node):
        state = node.state
        required = []
        for tile, color in self.goal_painted.items():
            if f'(painted {tile} {color})' not in state:
                required.append((tile, color))
        if not required:
            return 0

        robots = {}
        for fact in state:
            parts = fact[1:-1].split()
            if not parts:
                continue
            if parts[0] == 'robot-at':
                robot = parts[1]
                tile = parts[2]
                if robot not in robots:
                    robots[robot] = {'tile': None, 'color': None}
                robots[robot]['tile'] = tile
            elif parts[0] == 'robot-has':
                robot = parts[1]
                color = parts[2]
                if robot not in robots:
                    robots[robot] = {'tile': None, 'color': None}
                robots[robot]['color'] = color

        total_cost = 0
        for tile, req_color in required:
            adj_tiles = self.adjacent.get(tile, set())
            if not adj_tiles:
                continue
            min_cost = float('inf')
            for robot in robots.values():
                if not robot['tile'] or not robot['color']:
                    continue
                current_tile = robot['tile']
                if current_tile not in self.tile_coords:
                    continue
                x2, y2 = self.tile_coords[current_tile]
                min_dist = float('inf')
                for adj in adj_tiles:
                    if adj not in self.tile_coords:
                        continue
                    x1, y1 = self.tile_coords[adj]
                    dist = abs(x1 - x2) + abs(y1 - y2)
                    if dist < min_dist:
                        min_dist = dist
                if min_dist == float('inf'):
                    continue
                color_cost = 0 if robot['color'] == req_color else 1
                cost = min_dist + color_cost + 1
                if cost < min_cost:
                    min_cost = cost
            if min_cost != float('inf'):
                total_cost += min_cost
        return total_cost
