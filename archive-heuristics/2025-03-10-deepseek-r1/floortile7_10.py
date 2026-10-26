from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict

class floortile7Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all goal tiles with their required colors. It considers the distance robots must move to reach adjacent tiles for painting, the need to change colors, and the actual painting action.

    # Assumptions
    - The grid is connected, and movement between adjacent tiles is possible.
    - Robots can only paint tiles adjacent to their current position.
    - Changing color takes one action.
    - Robots can carry only one color at a time.

    # Heuristic Initialization
    - Extract goal conditions to determine required colors for each tile.
    - Precompute adjacency relationships between tiles from static facts.
    - Parse tile coordinates from their names for Manhattan distance calculations.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each goal tile not yet painted correctly:
        a. Find all adjacent tiles from which it can be painted.
        b. For each robot, calculate the Manhattan distance to each adjacent tile.
        c. Select the minimal distance across all robots and adjacent tiles.
        d. Add the cost for changing color if the robot's current color doesn't match.
        e. Add one for the paint action.
    2. Sum all individual tile costs for the total heuristic value.
    """

    @staticmethod
    def get_parts(fact):
        """Extract components of a PDDL fact."""
        return fact[1:-1].split()

    def __init__(self, task):
        self.goal_painted = {}
        self.adjacent_tiles = defaultdict(list)
        self.tile_coords = {}
        tiles = set()

        # Extract goal painted conditions
        for goal in task.goals:
            parts = self.get_parts(goal)
            if parts[0] == 'painted':
                tile = parts[1]
                color = parts[2]
                self.goal_painted[tile] = color

        # Extract adjacency and tiles from static facts
        for fact in task.static:
            parts = self.get_parts(fact)
            if parts[0] in ['up', 'down', 'left', 'right']:
                y, x = parts[1], parts[2]
                self.adjacent_tiles[y].append(x)
                tiles.add(y)
                tiles.add(x)

        # Parse tile coordinates
        for tile in tiles:
            _, row, col = tile.split('_')
            self.tile_coords[tile] = (int(row), int(col))

    def __call__(self, node):
        state = node.state
        current_painted = {}
        robot_positions = {}
        robot_colors = {}

        # Extract current painted tiles and robot states
        for fact in state:
            parts = self.get_parts(fact)
            if parts[0] == 'painted':
                tile, color = parts[1], parts[2]
                current_painted[tile] = color
            elif parts[0] == 'robot-at':
                robot = parts[1]
                pos = parts[2]
                robot_positions[robot] = pos
            elif parts[0] == 'robot-has':
                robot = parts[1]
                color = parts[2]
                robot_colors[robot] = color

        total_cost = 0

        # Process each goal tile
        for tile, req_color in self.goal_painted.items():
            if current_painted.get(tile) == req_color:
                continue  # Already painted correctly

            min_tile_cost = float('inf')
            adjacent_x = self.adjacent_tiles.get(tile, [])

            if not adjacent_x:
                continue  # No adjacent tiles (invalid state)

            for x in adjacent_x:
                for robot in robot_positions:
                    robot_pos = robot_positions[robot]
                    # Calculate Manhattan distance
                    x_coords = self.tile_coords.get(x, (0, 0))
                    robot_coords = self.tile_coords.get(robot_pos, (0, 0))
                    distance = abs(x_coords[0] - robot_coords[0]) + abs(x_coords[1] - robot_coords[1])
                    # Check color
                    current_color = robot_colors.get(robot, None)
                    color_change = 0 if current_color == req_color else 1
                    # Total cost for this robot and adjacent tile
                    cost = distance + color_change + 1  # +1 for paint
                    if cost < min_tile_cost:
                        min_tile_cost = cost

            if min_tile_cost != float('inf'):
                total_cost += min_tile_cost

        return total_cost
