from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_coordinates(tile_name):
    """Extract (x, y) coordinates from a tile name like 'tile_0_1'."""
    parts = tile_name.split('_')
    return (int(parts[1]), int(parts[2]))

def manhattan_distance(pos1, pos2):
    """Compute Manhattan distance between two tile positions."""
    x1, y1 = pos1
    x2, y2 = pos2
    return abs(x1 - x2) + abs(y1 - y2)

class FloorTileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the FloorTile domain.

    # Summary
    This heuristic estimates the number of actions needed for each robot to paint all required tiles by considering the minimal steps for movement and painting, including possible color changes.

    # Assumptions:
    - Each robot can move up, down, left, right to adjacent tiles.
    - The Manhattan distance is used to estimate movement steps.
    - Each tile requires exactly one color, specified in the goal.
    - Robots can change colors, with each change costing one action.

    # Heuristic Initialization
    - Extract available colors from static facts.
    - Map each goal tile to its required color.
    - Precompute the coordinates of each tile based on its name.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each robot, determine its current position and color.
    2. For each available color:
       a. Calculate the cost to change color (if needed).
       b. Identify all tiles requiring this color that are not yet painted.
       c. Compute the total movement distance to these tiles.
       d. Sum the movement distance, painting actions, and color change cost.
    3. For each robot, select the color that results in the minimal total actions.
    4. Sum the minimal actions across all robots to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic with necessary information from the task.
        """
        # Extract available colors from static facts
        self.available_colors = set()
        for fact in task.static:
            if fact.startswith('(available-color '):
                color = fact.split()[1]
                self.available_colors.add(color)

        # Map each goal tile to its required color
        self.goal_tiles = {}
        for goal in task.goals:
            if goal.startswith('(painted '):
                parts = goal[1:-1].split()
                tile = parts[0]
                color = parts[1]
                self.goal_tiles[tile] = color

        # Precompute color to goal tiles mapping
        self.color_to_goals = {}
        for tile, color in self.goal_tiles.items():
            if color not in self.color_to_goals:
                self.color_to_goals[color] = []
            self.color_to_goals[color].append(tile)

    def __call__(self, node):
        """
        Compute the heuristic value for the given node.
        """
        state = node.state
        painted_tiles = {fact.split()[1] for fact in state if fact.startswith('(painted ')}

        # Extract robot positions and colors
        robot_positions = {}
        robot_colors = {}
        for fact in state:
            if fact.startswith('(robot-at '):
                parts = fact[1:-1].split()
                robot = parts[1]
                tile = parts[2]
                robot_positions[robot] = tile
            elif fact.startswith('(robot-has '):
                parts = fact[1:-1].split()
                robot = parts[1]
                color = parts[2]
                robot_colors[robot] = color

        total_actions = 0

        for robot in robot_positions:
            current_pos = robot_positions[robot]
            current_color = robot_colors[robot]

            min_actions = float('inf')

            for c in self.available_colors:
                if c not in self.color_to_goals:
                    continue  # No tiles require this color

                # Calculate change cost
                if c == current_color:
                    change_cost = 0
                else:
                    change_cost = 1

                # Get tiles that require color c and are not painted
                tiles = [tile for tile in self.color_to_goals[c] if tile not in painted_tiles]

                if not tiles:
                    continue  # No tiles to paint with this color

                # Calculate total distance
                total_distance = 0
                for tile in tiles:
                    tile_pos = get_coordinates(tile)
                    current_tile_pos = get_coordinates(current_pos)
                    distance = manhattan_distance(current_tile_pos, tile_pos)
                    total_distance += distance

                # Total actions for this color
                total = change_cost + total_distance + len(tiles)

                if total < min_actions:
                    min_actions = total

            if min_actions != float('inf'):
                total_actions += min_actions

        return total_actions
