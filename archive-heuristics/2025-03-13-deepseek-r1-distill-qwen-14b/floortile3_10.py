from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class FloortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all required tiles by considering:
    - The Manhattan distance from the robot's current position to each tile.
    - The number of tiles that need painting.
    - The number of color changes required.

    # Assumptions:
    - The robot can move up, down, left, right.
    - Each paint action requires the robot to be on the tile and have the correct color.
    - Color changes are needed when the robot doesn't have the required color.

    # Heuristic Initialization
    - Extracts the goal conditions and static facts (tile layout) from the task.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all tiles that need to be painted and their required colors.
    2. For each tile not yet painted, calculate the Manhattan distance from the robot's current position.
    3. Sum these distances to estimate the movement cost.
    4. Count the number of tiles needing painting to estimate paint actions.
    5. Determine the number of color changes needed based on the robot's current color and the required colors.
    6. The heuristic value is the sum of movement cost, paint actions, and color changes.
    """

    def __init__(self, task):
        """Initialize the heuristic with goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Build tile_to_pos mapping from static facts
        self.tile_to_pos = {}
        for fact in self.static:
            if fact.startswith("(up ") or fact.startswith("(down ") or fact.startswith("(left ") or fact.startswith("(right "):
                parts = fact[1:-1].split()
                if len(parts) >= 3:
                    tile1 = parts[1]
                    tile2 = parts[2]
                    self.tile_to_pos[tile1] = self.parse_tile_name(tile1)
                    self.tile_to_pos[tile2] = self.parse_tile_name(tile2)

    def parse_tile_name(self, tile_name):
        """Parse a tile name like 'tile_0_1' into (0, 1)."""
        parts = tile_name.split('_')[1:]
        return int(parts[0]), int(parts[1])

    def __call__(self, node):
        """Compute the heuristic value for the given node."""
        state = node.state

        # Extract robot's current position and color
        robot_pos = None
        robot_color = None
        for fact in state:
            if fact.startswith("(robot-at robot1 "):
                robot_pos = fact.split()[2]
            if fact.startswith("(robot-has robot1 "):
                robot_color = fact.split()[2]

        if not robot_pos or not robot_color:
            return 0  # Shouldn't happen in valid state

        # Extract goal conditions: which tiles need to be painted with which colors
        goal_tiles = {}
        for goal in self.goals:
            if goal.startswith("(painted "):
                tile = goal.split()[1]
                color = goal.split()[2]
                goal_tiles[tile] = color

        # Determine which tiles are not painted correctly
        required_tiles = []
        required_colors = set()
        for tile, color in goal_tiles.items():
            # Check if the tile is already painted correctly in the current state
            painted_color = None
            for fact in state:
                if fact.startswith(f"(painted {tile} "):
                    painted_color = fact.split()[2]
                    break
            if painted_color != color:
                required_tiles.append(tile)
                required_colors.add(color)

        if not required_tiles:
            return 0  # Goal already achieved

        # Calculate sum of distances
        sum_distances = 0
        robot_x, robot_y = self.tile_to_pos[robot_pos]
        for tile in required_tiles:
            tile_x, tile_y = self.tile_to_pos[tile]
            distance = abs(tile_x - robot_x) + abs(tile_y - robot_y)
            sum_distances += distance

        # Number of tiles to paint
        num_tiles = len(required_tiles)

        # Number of color changes
        current_color = robot_color
        required_colors = required_colors
        if required_colors:
            if current_color in required_colors:
                color_changes = len(required_colors) - 1
            else:
                color_changes = len(required_colors)
        else:
            color_changes = 0

        # Heuristic is sum of distances, plus number of tiles, plus color changes
        heuristic = sum_distances + num_tiles + color_changes

        return heuristic
