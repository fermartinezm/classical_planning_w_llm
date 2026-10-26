from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class FloorTileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the FloorTile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all required tiles by considering:
    - The distance the robot needs to move to reach each tile
    - The number of color changes required
    - The number of tiles that still need to be painted

    # Assumptions:
    - The robot can move up, down, left, or right in a grid
    - Each tile must be painted exactly once
    - The robot starts with a color that may need to be changed

    # Heuristic Initialization
    - Extracts goal conditions to know which tiles need which colors
    - Maps each tile to its required color
    - Creates a grid representation based on static facts

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all tiles that need painting and their target colors.
    2. For each tile that isn't yet painted correctly:
       a. Calculate the Manhattan distance from the robot's current position
       b. Add the distance to the total cost
    3. Count the number of color changes needed based on the sequence of tiles
    4. Sum the movement costs and color change costs to get the total heuristic value
    """

    def __init__(self, task):
        """Initialize the heuristic with goal information and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Create a mapping from tile to required color
        self.goal_colors = {}
        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'painted':
                tile = parts[1]
                color = parts[2]
                self.goal_colors[tile] = color

        # Create a grid layout based on static facts
        self.up = {}
        self.down = {}
        self.left = {}
        self.right = {}
        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] == 'up':
                y = parts[1]
                x = parts[2]
                self.up[x] = y
                self.down[y] = x
            elif parts[0] == 'left':
                y = parts[1]
                x = parts[2]
                self.left[x] = y
                self.right[y] = x

    def __call__(self, node):
        """Estimate the minimal number of actions needed to reach the goal."""
        state = node.state

        # Helper function to extract tile coordinates
        def get_coords(tile):
            x = tile
            y = None
            # Check if tile is in up relationships (meaning it has a tile below it)
            if x in self.up:
                y = self.up[x]
            # Check if tile is in down relationships (meaning it has a tile above it)
            elif x in self.down:
                y = self.down[x]
            # Check if tile is in left relationships (meaning it has a tile to the right)
            elif x in self.left:
                y = self.left[x]
            # Check if tile is in right relationships (meaning it has a tile to the left)
            elif x in self.right:
                y = self.right[x]
            return (x, y)

        # Get current robot position and color
        robot_pos = None
        current_color = None
        for fact in state:
            if fact.startswith('(robot-at'):
                robot_pos = fact[1:-1].split()[1]
            if fact.startswith('(robot-has'):
                current_color = fact[1:-1].split()[2]

        # If robot position not found, assume it's at the initial position
        if not robot_pos:
            # Extract initial position from problem definition
            # This would be parsed from the problem file, assumed here as 'tile_0_1'
            robot_pos = 'tile_0_1'

        # Calculate Manhattan distance for each tile that needs painting
        total_cost = 0
        visited_colors = set()

        # List of tiles that need painting
        tiles_to_paint = [tile for tile in self.goal_colors.keys()]

        # Process each tile in order
        for tile in tiles_to_paint:
            target_color = self.goal_colors[tile]
            # Check if tile is already painted correctly
            if f'(painted {tile} {target_color})' in state:
                continue

            # Get coordinates for current and target tile
            current_coords = get_coords(robot_pos)
            target_coords = get_coords(tile)

            # Calculate Manhattan distance
            distance = abs(int(current_coords[0].split('_')[1]) - int(target_coords[0].split('_')[1])) + \
                       abs(int(current_coords[1].split('_')[1]) - int(target_coords[1].split('_')[1]))

            # Add movement cost
            total_cost += distance

            # Check if color change is needed
            if current_color != target_color:
                total_cost += 1  # Action to change color
                current_color = target_color

            # Move to the target tile
            robot_pos = tile

        return total_cost
