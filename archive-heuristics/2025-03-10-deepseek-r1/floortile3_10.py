import fnmatch
from collections import defaultdict
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

def match(fact, *args):
    parts = get_parts(fact)
    return all(fnmatch.fnmatch(part, arg) for part, arg in zip(parts, args))

class floortile3Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles with their respective colors. It considers the minimal movement steps for robots to reach adjacent tiles, necessary color changes, and painting actions.

    # Assumptions
    - Tiles are named in the format tile_X_Y, allowing extraction of coordinates.
    - Robots can move freely between tiles (ignoring other robots' positions for efficiency).
    - Color changes are possible if the required color is available.
    - Each paint action requires the robot to be on an adjacent tile, which is clear.

    # Heuristic Initialization
    - Extract goal conditions (tile-color pairs).
    - Build a reverse adjacency map to find tiles adjacent to each target tile.
    - Identify available colors from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each goal tile not yet painted correctly:
        a. Find adjacent tiles from which the robot can paint it.
        b. For each robot:
            i. Calculate the minimal Manhattan distance to any adjacent tile.
            ii. Determine if a color change is needed.
            iii. Compute total cost (distance + color change + paint action).
        c. Sum the minimal cost across all robots for this tile.
    2. Sum costs for all such tiles to get the heuristic value.
    """

    def __init__(self, task):
        self.goal_painted = {}
        for goal in task.goals:
            if match(goal, 'painted', '*', '*'):
                parts = get_parts(goal)
                tile = parts[1]
                color = parts[2]
                self.goal_painted[tile] = color

        self.reverse_adjacency_map = defaultdict(list)
        self.available_colors = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] in ['up', 'down', 'left', 'right']:
                y = parts[1]
                x = parts[2]
                self.reverse_adjacency_map[y].append(x)
            elif parts[0] == 'available-color':
                self.available_colors.add(parts[1])

    def _manhattan_distance(self, tile1, tile2):
        """Compute Manhattan distance between two tiles based on their coordinates."""
        x1, y1 = map(int, tile1.split('_')[1:3])
        x2, y2 = map(int, tile2.split('_')[1:3])
        return abs(x1 - x2) + abs(y1 - y2)

    def __call__(self, node):
        state = node.state
        current_painted = {}
        current_robot_positions = {}
        current_robot_colors = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'painted':
                current_painted[parts[1]] = parts[2]
            elif parts[0] == 'robot-at':
                current_robot_positions[parts[1]] = parts[2]
            elif parts[0] == 'robot-has':
                current_robot_colors[parts[1]] = parts[2]

        total_cost = 0
        for tile, req_color in self.goal_painted.items():
            if current_painted.get(tile) != req_color:
                adj_tiles = self.reverse_adjacency_map.get(tile, [])
                if not adj_tiles:
                    continue  # Skip if no adjacent tiles (unlikely)
                min_cost = float('inf')
                for robot in current_robot_positions:
                    robot_pos = current_robot_positions[robot]
                    robot_color = current_robot_colors.get(robot, None)
                    if robot_color is None:
                        continue  # Should not happen in valid states

                    # Calculate minimal distance to any adjacent tile
                    min_dist = min((self._manhattan_distance(robot_pos, adj) for adj in adj_tiles), default=float('inf'))
                    if min_dist == float('inf'):
                        continue

                    # Check if color change is needed and possible
                    color_change = 0
                    if robot_color != req_color:
                        if req_color in self.available_colors:
                            color_change = 1
                        else:
                            continue  # Required color not available, skip this robot

                    cost = min_dist + color_change + 1  # +1 for paint action
                    if cost < min_cost:
                        min_cost = cost
                if min_cost != float('inf'):
                    total_cost += min_cost
                # Else: no robot can paint this tile, but heuristic assumes solvable
        return total_cost
