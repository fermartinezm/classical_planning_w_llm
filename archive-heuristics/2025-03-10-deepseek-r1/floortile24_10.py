from heuristics.heuristic_base import Heuristic
from collections import defaultdict
import math

class floortile24Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    Estimates the number of actions needed to paint all goal tiles. For each unpainted tile,
    the heuristic calculates the minimal cost considering robot movement to an adjacent tile,
    required color changes, and the painting action. The total heuristic is the sum of these
    minimal costs across all unpainted tiles.

    # Assumptions
    - Robots can move freely between adjacent tiles (ignoring 'clear' preconditions for efficiency).
    - Each color change is considered once per tile if needed, not optimized across multiple tiles.
    - Tiles are processed independently, assuming parallel execution where possible.

    # Heuristic Initialization
    - Extracts goal tiles and their required colors from the task.
    - Builds adjacency lists for each tile using static 'up', 'down', 'left', 'right' facts.
    - Identifies available colors from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each goal tile not yet painted:
        a. Determine required color.
        b. Find adjacent tiles from which the tile can be painted.
        c. For each robot, compute the minimal distance to any adjacent tile.
        d. Calculate cost (movement + paint + color change if needed).
        e. Take the minimal cost across all robots.
    2. Sum all minimal costs for all unpainted tiles.
    """

    def __init__(self, task):
        self.goal_tiles = {}
        for goal in task.goals:
            parts = goal.strip('()').split()
            if parts[0] == 'painted':
                tile, color = parts[1], parts[2]
                self.goal_tiles[tile] = color

        self.adjacent = defaultdict(dict)
        for fact in task.static:
            parts = fact.strip('()').split()
            if parts[0] == 'up':
                y, x = parts[1], parts[2]
                self.adjacent[x]['up'] = y
            elif parts[0] == 'down':
                y, x = parts[1], parts[2]
                self.adjacent[x]['down'] = y
            elif parts[0] == 'left':
                y, x = parts[1], parts[2]
                self.adjacent[x]['left'] = y
            elif parts[0] == 'right':
                y, x = parts[1], parts[2]
                self.adjacent[x]['right'] = y

    def __call__(self, node):
        state = node.state
        robots = defaultdict(dict)
        for fact in state:
            parts = fact.strip('()').split()
            if parts[0] == 'robot-at':
                robot, tile = parts[1], parts[2]
                robots[robot]['position'] = tile
            elif parts[0] == 'robot-has':
                robot, color = parts[1], parts[2]
                robots[robot]['color'] = color

        total_cost = 0
        for tile, req_color in self.goal_tiles.items():
            if f'(painted {tile} {req_color})' in state:
                continue  # Already painted

            adjacent_tiles = []
            for direction in ['up', 'down', 'left', 'right']:
                adj = self.adjacent.get(tile, {}).get(direction)
                if adj:
                    adjacent_tiles.append(adj)
            if not adjacent_tiles:
                continue  # No adjacent tiles (invalid state)

            min_tile_cost = math.inf
            for robot in robots.values():
                pos = robot.get('position')
                color = robot.get('color')
                if not pos or not color:
                    continue

                try:
                    x1, y1 = map(int, pos.split('_')[1:])
                except:
                    continue  # Invalid tile format

                min_dist = math.inf
                for adj in adjacent_tiles:
                    try:
                        x2, y2 = map(int, adj.split('_')[1:])
                        dist = abs(x1 - x2) + abs(y1 - y2)
                        min_dist = min(min_dist, dist)
                    except:
                        continue  # Invalid adjacent tile

                if min_dist == math.inf:
                    continue

                color_cost = 0 if color == req_color else 1
                cost = min_dist + 1 + color_cost  # Move + paint + color change
                min_tile_cost = min(min_tile_cost, cost)

            if min_tile_cost != math.inf:
                total_cost += min_tile_cost

        return total_cost
