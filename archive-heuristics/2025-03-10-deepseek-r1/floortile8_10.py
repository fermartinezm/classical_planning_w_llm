from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def parse_coords(tile_name):
    parts = tile_name.split('_')
    x = int(parts[1])
    y = int(parts[2])
    return (x, y)

class floortile8Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all required tiles by considering:
    - The minimal distance for robots to reach adjacent tiles from which they can paint.
    - The paint action itself.
    - Color changes required if no robot has the needed color.

    # Assumptions
    - Robots can move freely between tiles (ignoring 'clear' status for efficiency).
    - Tiles are arranged in a grid, with coordinates parsed from their names (e.g., tile_1_2).
    - Each paint action requires one step.
    - A color change is needed once per required color not currently held by any robot.

    # Heuristic Initialization
    - Extract goal conditions to determine required tiles and their colors.
    - Build adjacency map for each tile (tiles from which it can be painted).
    - Collect available colors from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each tile not yet painted correctly:
        a. Find adjacent tiles from which it can be painted.
        b. Compute minimal Manhattan distance from any robot to each adjacent tile.
        c. Take the smallest distance and add 1 for the paint action.
    2. Sum all these values for all required tiles.
    3. For each required color not held by any robot and available, add 1 for a color change.
    """

    def __init__(self, task):
        self.goal_painted = {}
        for goal in task.goals:
            if isinstance(goal, str) and goal.startswith('(painted '):
                parts = goal[1:-1].split()
                tile = parts[1]
                color = parts[2]
                self.goal_painted[tile] = color
            elif isinstance(goal, (list, tuple)) and goal[0] == 'painted':
                tile = goal[1]
                color = goal[2]
                self.goal_painted[tile] = color

        self.adjacent_to = {}
        for fact in task.static:
            if isinstance(fact, str):
                parts = fact[1:-1].split()
            else:
                parts = fact
            if parts[0] in ['up', 'down', 'left', 'right']:
                y = parts[1]
                x = parts[2]
                self.adjacent_to.setdefault(y, set()).add(x)

        self.available_colors = set()
        for fact in task.static:
            if isinstance(fact, str):
                parts = fact[1:-1].split()
            else:
                parts = fact
            if parts[0] == 'available-color':
                self.available_colors.add(parts[1])

    def __call__(self, node):
        state = node.state

        unpainted = []
        for tile, color in self.goal_painted.items():
            painted_fact = f'(painted {tile} {color})'
            if painted_fact not in state:
                unpainted.append((tile, color))

        if not unpainted:
            return 0

        current_robot_positions = {}
        current_robot_colors = {}
        for fact in state:
            if isinstance(fact, str):
                parts = fact[1:-1].split()
            else:
                parts = fact
            if parts[0] == 'robot-at':
                robot = parts[1]
                tile = parts[2]
                current_robot_positions[robot] = tile
            elif parts[0] == 'robot-has':
                robot = parts[1]
                color = parts[2]
                current_robot_colors[robot] = color

        sum_cost = 0
        required_colors = set()

        for tile, color in unpainted:
            required_colors.add(color)
            adjacent_x = self.adjacent_to.get(tile, set())

            if not adjacent_x:
                sum_cost += 1000
                continue

            min_distance = float('inf')
            for x in adjacent_x:
                try:
                    x_coords = parse_coords(x)
                except:
                    continue

                for robot, robot_tile in current_robot_positions.items():
                    try:
                        robot_coords = parse_coords(robot_tile)
                    except:
                        continue

                    distance = abs(robot_coords[0] - x_coords[0]) + abs(robot_coords[1] - x_coords[1])
                    if distance < min_distance:
                        min_distance = distance

            if min_distance == float('inf'):
                sum_cost += 1000
            else:
                sum_cost += min_distance + 1

        color_change_cost = 0
        for color in required_colors:
            if color not in current_robot_colors.values() and color in self.available_colors:
                color_change_cost += 1

        return sum_cost + color_change_cost
