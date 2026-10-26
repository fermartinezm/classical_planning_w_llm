from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

def match(fact, *args):
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def get_tile_coords(tile_name):
    parts = tile_name.split('_')
    if len(parts) < 3:
        return (0, 0)
    x = int(parts[1])
    y = int(parts[2])
    return (x, y)

class floortile10Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the FloorTile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles with their respective colors. It considers the movement cost for robots to reach adjacent tiles, color changes, and the painting action.

    # Assumptions
    - The grid is a 2D layout where tiles are connected in four directions (up, down, left, right).
    - Robots can move freely between adjacent tiles.
    - Each paint action requires the robot to be adjacent to the target tile and have the correct color.

    # Heuristic Initialization
    - Extract the goal conditions to determine which tiles need to be painted and their required colors.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Identify Unpainted Tiles**: Check which goal tiles are not yet painted correctly.
    2. **Robot States**: Track each robot's current position and color.
    3. **Movement Cost**: For each unpainted tile, calculate the minimal Manhattan distance from each robot's current position to any adjacent tile of the target.
    4. **Color Change**: Determine if a robot needs to change color to match the target tile's required color.
    5. **Action Sum**: Sum the minimal movement, color change (if needed), and paint action for each tile across all robots.
    """

    def __init__(self, task):
        self.goal_tiles = {}
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'painted':
                tile = parts[1]
                color = parts[2]
                self.goal_tiles[tile] = color

    def __call__(self, node):
        state = node.state
        robots = {}
        # Collect robot positions and colors
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'robot-at':
                robot = parts[1]
                tile = parts[2]
                robots[robot] = {'tile': tile, 'color': None}
            elif parts[0] == 'robot-has':
                robot = parts[1]
                color = parts[2]
                if robot in robots:
                    robots[robot]['color'] = color

        required_tiles = []
        for tile, req_color in self.goal_tiles.items():
            painted_correct = any(fact == f'(painted {tile} {req_color})' for fact in state)
            if not painted_correct:
                required_tiles.append((tile, req_color))

        total_cost = 0

        for tile_name, req_color in required_tiles:
            tile_coords = get_tile_coords(tile_name)
            adjacent_coords = [
                (tile_coords[0] + 1, tile_coords[1]),
                (tile_coords[0] - 1, tile_coords[1]),
                (tile_coords[0], tile_coords[1] + 1),
                (tile_coords[0], tile_coords[1] - 1),
            ]
            min_cost = float('inf')
            for robot in robots.values():
                if not robot['tile']:
                    continue
                current_coords = get_tile_coords(robot['tile'])
                min_dist = min(
                    abs(current_coords[0] - adj[0]) + abs(current_coords[1] - adj[1])
                    for adj in adjacent_coords
                )
                color_change = 1 if robot['color'] != req_color else 0
                cost = min_dist + color_change + 1  # 1 for paint action
                if cost < min_cost:
                    min_cost = cost
            if min_cost != float('inf'):
                total_cost += min_cost

        return total_cost
