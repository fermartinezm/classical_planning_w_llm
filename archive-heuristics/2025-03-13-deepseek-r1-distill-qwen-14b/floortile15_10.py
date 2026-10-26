from heuristics.heuristic_base import Heuristic
from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class FloortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all required tiles by considering:
    - The movement cost to the farthest required tile.
    - The number of tiles that need painting.
    - The number of color changes required.

    # Assumptions:
    - The robot can move optimally to the farthest tile and paint others along the way.
    - Each color change is an additional action.
    - The robot starts with a specific color, and may need to change colors to paint tiles of different colors.

    # Heuristic Initialization
    - Extracts goal conditions (which tiles need to be painted with which colors).
    - Parses static facts to determine tile positions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all tiles that need to be painted and their required colors.
    2. For each tile, calculate the Manhattan distance from the robot's current position.
    3. Movement cost is the maximum of these distances, as the robot can move to the farthest tile and paint others en route.
    4. Painting cost is the number of tiles that need painting.
    5. Determine the colors needed for the tiles and calculate the number of color changes required.
    6. Sum movement cost, painting cost, and color change cost to get the heuristic value.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Parse goal tiles and their required colors
        self.goal_tiles = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'painted':
                tile = parts[1]
                color = parts[2]
                self.goal_tiles[tile] = color

    def __call__(self, node):
        state = node.state

        # Find robot's current position
        robot_pos = None
        for fact in state:
            if fact.startswith('(robot-at '):
                parts = get_parts(fact)
                if parts[0] == 'robot-at' and len(parts) >= 3:
                    robot_pos = parts[2]
                    break

        if not robot_pos:
            return float('inf')  # Robot not found, state is invalid

        # Collect required tiles and colors
        required_tiles = []
        required_colors = set()
        for tile, color in self.goal_tiles.items():
            painted_fact = f'(painted {tile} {color})'
            if painted_fact not in state:
                required_tiles.append(tile)
                required_colors.add(color)

        if not required_tiles:
            return 0  # All goals already achieved

        # Calculate movement cost
        movement_distances = []
        for tile in required_tiles:
            # Parse tile coordinates
            tile_parts = tile.split('_')
            if len(tile_parts) < 3:
                continue  # Invalid tile name
            y = int(tile_parts[1])
            x = int(tile_parts[2])  # Assuming tile_0_1 is (0,1), so x is second part

            # Parse robot's position
            robot_parts = robot_pos.split('_')
            if len(robot_parts) < 3:
                continue  # Invalid robot position
            robot_y = int(robot_parts[1])
            robot_x = int(robot_parts[2])

            distance = abs(robot_x - x) + abs(robot_y - y)
            movement_distances.append(distance)

        movement_cost = max(movement_distances) if movement_distances else 0

        painting_cost = len(required_tiles)

        # Determine color change cost
        current_color = None
        for fact in state:
            if fact.startswith('(robot-has '):
                parts = get_parts(fact)
                if parts[0] == 'robot-has' and len(parts) >= 3:
                    current_color = parts[2]
                    break

        if not current_color:
            return float('inf')  # Robot has no color

        num_colors = len(required_colors)
        if num_colors == 0:
            color_change_cost = 0
        else:
            if current_color in required_colors:
                color_change_cost = max(0, num_colors - 1)
            else:
                color_change_cost = num_colors

        heuristic_value = movement_cost + painting_cost + color_change_cost

        return heuristic_value
