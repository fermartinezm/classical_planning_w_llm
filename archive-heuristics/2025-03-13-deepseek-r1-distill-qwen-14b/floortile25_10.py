from heuristics.heuristic_base import Heuristic

class FloortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all required tiles by considering:
    - The robot's current position and held color.
    - The tiles that need to be painted and their target colors.
    - The minimal movements and color changes required.

    # Assumptions:
    - The robot can move freely between adjacent tiles.
    - The Manhattan distance is used to estimate movement between tiles.
    - Each color change costs 1 action.
    - Each painting action costs 1 action plus the movement cost.

    # Heuristic Initialization
    - Extracts the goal conditions (painted tiles) and static facts from the task.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the robot's current position and held color.
    2. For each tile, determine if it's painted correctly, incorrectly, or not at all.
    3. Group the tiles that need painting by their target color.
    4. For each color group:
       a. If the robot's color is not the target, add 1 for a color change.
       b. For each tile in the group, add the Manhattan distance from the robot's current position to the tile, plus 1 for painting.
       c. Update the robot's position to the tile's position after each movement.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        state = node.state
        cost = 0

        # Extract robot's current position and color
        robot_at = None
        robot_has = None
        for fact in state:
            if fact.startswith('(robot-at'):
                robot_at = fact
            elif fact.startswith('(robot-has'):
                robot_has = fact

        # Parse robot's position
        if robot_at:
            robot_tile = robot_at.split()[2]
            current_robot_coords = self.parse_tile(robot_tile)
        else:
            current_robot_coords = None  # This case should not occur in valid states

        # Parse robot's color
        if robot_has:
            current_robot_color = robot_has.split()[2]
        else:
            current_robot_color = None  # This case should not occur in valid states

        # Collect tiles that need to be painted
        tiles_to_paint = []
        for goal in self.goals:
            if goal.startswith('(painted'):
                tile = goal.split()[1]
                target_color = goal.split()[2]
                # Check if the tile is already painted correctly
                if goal not in state:
                    tiles_to_paint.append((tile, target_color))

        # Group tiles by target color
        color_groups = {}
        for tile, color in tiles_to_paint:
            if color not in color_groups:
                color_groups[color] = []
            color_groups[color].append(tile)

        # Calculate cost for each color group
        for color, tiles in color_groups.items():
            # Check if color change is needed
            if color != current_robot_color:
                cost += 1
                current_robot_color = color

            # Process each tile in the group
            for tile in tiles:
                tile_coords = self.parse_tile(tile)
                # Calculate Manhattan distance
                distance = abs(current_robot_coords[0] - tile_coords[0]) + abs(current_robot_coords[1] - tile_coords[1])
                cost += distance + 1  # distance to move + 1 for painting
                # Update robot's position
                current_robot_coords = tile_coords

        return cost

    def parse_tile(self, name):
        """Parses a tile name (e.g., 'tile_0_1') into (x, y) coordinates."""
        parts = name.split('_')
        return (int(parts[1]), int(parts[2]))
