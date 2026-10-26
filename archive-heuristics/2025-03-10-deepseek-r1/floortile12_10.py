from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict

class floortile12Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles with their respective colors. It considers the minimal movement steps for robots to reach adjacent tiles, color changes, and painting actions.

    # Assumptions
    - The grid is a regular layout where tile names encode their coordinates (e.g., tile_i_j).
    - Movement between tiles uses Manhattan distance as an approximation.
    - Robots can change color to any available color, which are static.
    - Only one robot can paint a tile, and parallelism is approximated by summing individual minimal costs.

    # Heuristic Initialization
    - Extract goal conditions for each tile's required color.
    - Build an adjacency map for each tile from static up/down/left/right facts.
    - Extract available colors from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each goal tile not yet painted correctly:
        a. Find its required color.
        b. Determine adjacent tiles from which it can be painted.
    2. For each robot:
        a. Calculate the minimal Manhattan distance to any adjacent tile of the goal.
        b. Add cost for color change if the robot's current color differs from the goal.
        c. Add one for the paint action.
    3. Sum the minimal cost across all robots and tiles.
    """

    def __init__(self, task):
        self.goal_painted = {}
        for goal in task.goals:
            if goal.startswith('(painted'):
                parts = goal[1:-1].split()
                tile = parts[1]
                color = parts[2]
                self.goal_painted[tile] = color

        self.adjacency_map = defaultdict(list)
        for fact in task.static:
            if fact.startswith('(up') or fact.startswith('(down') or fact.startswith('(left') or fact.startswith('(right'):
                parts = fact[1:-1].split()
                y = parts[1]
                x = parts[2]
                self.adjacency_map[y].append(x)

        self.available_colors = set()
        for fact in task.static:
            if fact.startswith('(available-color'):
                parts = fact[1:-1].split()
                self.available_colors.add(parts[1])

    def __call__(self, node):
        state = node.state
        robots = {}
        for fact in state:
            parts = fact[1:-1].split()
            if not parts:
                continue
            if parts[0] == 'robot-at':
                robot = parts[1]
                tile = parts[2]
                robots[robot] = {'pos': tile, 'color': None}
            elif parts[0] == 'robot-has':
                robot = parts[1]
                color = parts[2]
                if robot in robots:
                    robots[robot]['color'] = color

        current_painted = {}
        for fact in state:
            if fact.startswith('(painted'):
                parts = fact[1:-1].split()
                tile = parts[1]
                color = parts[2]
                current_painted[tile] = color

        total_cost = 0

        for tile, goal_color in self.goal_painted.items():
            if current_painted.get(tile) == goal_color:
                continue

            if goal_color not in self.available_colors:
                continue

            adjacent_x = self.adjacency_map.get(tile, [])
            if not adjacent_x:
                continue

            min_cost = float('inf')
            for robot in robots.values():
                if not robot['pos'] or robot['color'] is None:
                    continue

                try:
                    rx, ry = map(int, robot['pos'].split('_')[1:3])
                except:
                    continue

                min_dist = float('inf')
                for x in adjacent_x:
                    try:
                        xx, xy = map(int, x.split('_')[1:3])
                    except:
                        continue
                    distance = abs(rx - xx) + abs(ry - xy)
                    if distance < min_dist:
                        min_dist = distance

                if min_dist == float('inf'):
                    continue

                color_change = 1 if robot['color'] != goal_color else 0
                cost = min_dist + color_change + 1

                if cost < min_cost:
                    min_cost = cost

            if min_cost != float('inf'):
                total_cost += min_cost

        return total_cost
