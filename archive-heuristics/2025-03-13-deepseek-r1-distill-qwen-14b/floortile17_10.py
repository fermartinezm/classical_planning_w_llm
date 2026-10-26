from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class FloortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all required tiles by considering movement costs, painting actions, and color changes.

    # Assumptions:
    - The robot can move up, down, left, right on a grid.
    - Each tile requires exactly one color.
    - The robot can only hold one color at a time.
    - Moving between tiles takes Manhattan distance steps.
    - Painting a tile takes one action.
    - Changing color takes one action.

    # Heuristic Initialization
    - Extract goal conditions to know which tiles need which colors.
    - Extract static facts to determine tile adjacency.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the robot's current position and held color.
    2. For each goal tile, check if it's already painted correctly.
    3. Group the tiles needing painting by their required color.
    4. For each color group, calculate the sum of Manhattan distances from the robot's current position to each tile in the group.
    5. Sum all movement distances, add the number of tiles (paint actions), and add the number of color changes (based on initial color and group sequence).
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract goal conditions: tile to required color mapping
        self.goal_color = {}
        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'painted':
                tile = parts[1]
                color = parts[2]
                self.goal_color[tile] = color

        # Precompute tile coordinates for faster access
        self.tile_coords = {}
        for fact in self.static:
            if fact.startswith('(up ') or fact.startswith('(down ') or fact.startswith('(left ') or fact.startswith('(right '):
                continue
            if fact.startswith('(robot-at ') or fact.startswith('(clear ') or fact.startswith('(painted ') or fact.startswith('(robot-has ') or fact.startswith('(available-color ') or fact.startswith('(free-color '):
                continue
            if fact.startswith('(tile_'):
                tile = fact[1:-1]
                x, y = tile.split('_')[1], tile.split('_')[2]
                self.tile_coords[tile] = (int(x), int(y))

    def __call__(self, node):
        state = node.state

        # Extract robot's current position and color
        robot_pos = None
        robot_color = None
        for fact in state:
            if fact.startswith('(robot-at '):
                parts = fact[1:-1].split()
                robot_pos = parts[2]
            if fact.startswith('(robot-has '):
                parts = fact[1:-1].split()
                robot_color = parts[2]

        if not robot_pos:
            return 0

        # Identify tiles that need painting
        required_tiles = []
        for tile, color in self.goal_color.items():
            painted_correctly = False
            for fact in state:
                if fact.startswith('(painted ' + tile + ' ' + color + ')'):
                    painted_correctly = True
                    break
            if not painted_correctly:
                required_tiles.append((tile, color))

        if not required_tiles:
            return 0

        # Group required tiles by color
        color_groups = {}
        for tile, color in required_tiles:
            if color not in color_groups:
                color_groups[color] = []
            color_groups[color].append(tile)

        # Calculate movement distances
        total_distance = 0
        current_pos = self.tile_coords[robot_pos]
        for color, tiles in color_groups.items():
            for tile in tiles:
                tile_pos = self.tile_coords[tile]
                distance = abs(current_pos[0] - tile_pos[0]) + abs(current_pos[1] - tile_pos[1])
                total_distance += distance
                current_pos = tile_pos  # Move to this tile for the next tile in the same group

        # Calculate color changes
        num_color_changes = 0
        if color_groups:
            first_color = next(iter(color_groups.keys()))
            if first_color != robot_color:
                num_color_changes = len(color_groups)
            else:
                num_color_changes = len(color_groups) - 1

        # Calculate painting actions
        num_paint_actions = len(required_tiles)

        # Total heuristic is movement + painting + color changes
        heuristic_value = total_distance + num_paint_actions + num_color_changes

        return heuristic_value
