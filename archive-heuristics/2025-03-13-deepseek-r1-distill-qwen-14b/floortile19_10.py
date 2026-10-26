from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class FloortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all required tiles with the correct colors. It considers the robot's current position, the colors it has, and the distances to each tile that needs painting.

    # Assumptions:
    - The robot can move up, down, left, and right on the grid.
    - Each tile requires exactly one color to be painted.
    - The robot can only paint a tile if it is clear and within reach.

    # Heuristic Initialization
    - Extracts the goal conditions and static facts from the task.
    - Constructs a grid map to translate tile names into coordinates.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the robot's current position and color from the state.
    2. Identify all tiles that need to be painted and their required colors.
    3. Group these tiles by their required color.
    4. For each color group:
       a. If the robot doesn't have the required color, add one action for changing the color.
       b. Calculate the total Manhattan distance from the robot's current position to each tile in the group.
       c. Add the number of tiles in the group (each requires a painting action).
    5. Sum all the actions needed for color changes, movements, and painting to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and building the grid map."""
        self.goals = task.goals
        static_facts = task.static

        # Build grid map to translate tile names into coordinates
        self.grid = {}
        for fact in static_facts:
            if fact.startswith('(up ') or fact.startswith('(down '):
                parts = fact[1:-1].split()
                if parts[0] == 'up' or parts[0] == 'down':
                    child = parts[1]
                    parent = parts[2]
                    x_child, y_child = self.parse_tile_name(child)
                    x_parent, y_parent = self.parse_tile_name(parent)
                    self.grid[child] = (x_child, y_child)
                    self.grid[parent] = (x_parent, y_parent)

    def parse_tile_name(self, tile):
        """Parse a tile name (e.g., 'tile_0_1') into (x, y) coordinates."""
        parts = tile.split('_')
        return int(parts[1]), int(parts[2])

    def __call__(self, node):
        """Compute the heuristic value for the current state."""
        state = node.state

        # Extract robot's current position and color
        robot_pos = None
        robot_color = None
        for fact in state:
            if fact.startswith('(robot-at'):
                parts = fact[1:-1].split()
                _, robot, tile = parts
                robot_pos = self.grid[tile]
            if fact.startswith('(robot-has'):
                parts = fact[1:-1].split()
                _, robot, color = parts
                robot_color = color

        # Extract goal tiles and their required colors
        required_tiles = {}
        for goal in self.goals:
            if goal.startswith('(painted'):
                parts = goal[1:-1].split()
                tile, color = parts[1], parts[2]
                required_tiles[tile] = color

        # Group tiles by their required color
        color_groups = {}
        for tile, color in required_tiles.items():
            # Check if the tile is already painted in the current state
            painted = any(fact.startswith(f'(painted {tile} {color})') for fact in state)
            if not painted:
                if color not in color_groups:
                    color_groups[color] = []
                color_groups[color].append(tile)

        total_actions = 0
        current_pos = robot_pos
        current_color = robot_color

        for color, tiles in color_groups.items():
            if color != current_color:
                total_actions += 1
                current_color = color

            distance_sum = 0
            for tile in tiles:
                tile_pos = self.grid[tile]
                distance = abs(current_pos[0] - tile_pos[0]) + abs(current_pos[1] - tile_pos[1])
                distance_sum += distance

            total_actions += distance_sum
            total_actions += len(tiles)

        return total_actions
