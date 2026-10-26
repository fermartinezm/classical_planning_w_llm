from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class FloortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all tiles correctly by considering the minimal cost for each tile, which includes movement, color change (if needed), and painting.

    # Assumptions:
    - The robot can move up, down, left, right, and paint the tile it's on.
    - The robot can change colors, which takes one action.
    - Each tile that isn't painted correctly contributes to the cost based on the closest robot's position and color.

    # Heuristic Initialization
    - Extract the goal conditions (which tiles need which colors).
    - Parse static facts to map each tile to its (x, y) coordinates.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each tile that needs to be painted correctly:
       a. Check if it's already painted correctly. If yes, skip.
       b. For each robot, calculate the Manhattan distance from the robot's current position to the tile.
       c. Determine if the robot's current color matches the required color. If not, add one action for changing color.
       d. Calculate the total cost for this robot to paint the tile: distance + color change (if needed) + 1 (paint action).
       e. Track the minimal cost across all robots for this tile.
    2. Sum the minimal costs for all tiles to get the total heuristic value.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Parse static facts to get tile coordinates
        self.tile_positions = {}
        for fact in self.static:
            if fact.startswith('(up'):
                _, y, x = fact[1:-1].split()
                y_parts = y.split('_')
                x_parts = x.split('_')
                y_x = int(y_parts[1])
                y_y = int(y_parts[2])
                x_x = int(x_parts[1])
                x_y = int(x_parts[2])
                self.tile_positions[y] = (y_x, y_y)
                self.tile_positions[x] = (x_x, x_y)
            elif fact.startswith('(down'):
                _, x, y = fact[1:-1].split()
                x_parts = x.split('_')
                y_parts = y.split('_')
                x_x = int(x_parts[1])
                x_y = int(x_parts[2])
                y_x = int(y_parts[1])
                y_y = int(y_parts[2])
                self.tile_positions[x] = (x_x, x_y)
                self.tile_positions[y] = (y_x, y_y)
            elif fact.startswith('(left'):
                _, x, y = fact[1:-1].split()
                x_parts = x.split('_')
                y_parts = y.split('_')
                x_x = int(x_parts[1])
                x_y = int(x_parts[2])
                y_x = int(y_parts[1])
                y_y = int(y_parts[2])
                self.tile_positions[x] = (x_x, x_y)
                self.tile_positions[y] = (y_x, y_y)
            elif fact.startswith('(right'):
                _, y, x = fact[1:-1].split()
                y_parts = y.split('_')
                x_parts = x.split('_')
                y_x = int(y_parts[1])
                y_y = int(y_parts[2])
                x_x = int(x_parts[1])
                x_y = int(x_parts[2])
                self.tile_positions[y] = (y_x, y_y)
                self.tile_positions[x] = (x_x, x_y)

        # Extract goal locations and required colors
        self.goal_info = {}
        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'painted':
                tile = parts[1]
                color = parts[2]
                self.goal_info[tile] = color

    def __call__(self, node):
        state = node.state
        current_painted = {}
        robot_info = {}

        # Extract current painted tiles
        for fact in state:
            if fact.startswith('(painted'):
                parts = fact[1:-1].split()
                tile = parts[1]
                color = parts[2]
                current_painted[tile] = color

        # Extract robot positions and colors
        for fact in state:
            if fact.startswith('(robot-at'):
                parts = fact[1:-1].split()
                robot = parts[1]
                pos = parts[2]
                robot_info[robot] = {'pos': pos, 'color': None}
            elif fact.startswith('(robot-has'):
                parts = fact[1:-1].split()
                robot = parts[1]
                color = parts[2]
                robot_info[robot]['color'] = color

        total_cost = 0

        # Process each goal tile
        for tile, required_color in self.goal_info.items():
            if tile in current_painted and current_painted[tile] == required_color:
                continue  # Already correct

            # Get coordinates of the goal tile
            if tile not in self.tile_positions:
                continue  # Tile position not found
            x2, y2 = self.tile_positions[tile]

            min_cost = float('inf')
            for robot, info in robot_info.items():
                current_pos = info['pos']
                current_color = info['color']

                # Skip if robot's position is not in tile_positions
                if current_pos not in self.tile_positions:
                    continue
                x1, y1 = self.tile_positions[current_pos]

                # Calculate Manhattan distance
                distance = abs(x1 - x2) + abs(y1 - y2)
                # Calculate color change cost
                color_cost = 0 if current_color == required_color else 1
                # Total cost for this robot to paint the tile
                cost = distance + color_cost + 1  # +1 for painting action

                if cost < min_cost:
                    min_cost = cost

            if min_cost != float('inf'):
                total_cost += min_cost

        return total_cost
