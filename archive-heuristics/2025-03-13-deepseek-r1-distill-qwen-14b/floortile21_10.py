from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class FloorTileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the FloorTile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all required tiles by considering:
    - The Manhattan distance from the robot's current position to the farthest unpainted tile.
    - The number of color changes needed if the robot's current color doesn't match the required color.

    # Assumptions:
    - The robot can move optimally between tiles.
    - Color changes are required only when the robot's current color doesn't match the target color.

    # Heuristic Initialization
    - Extracts goal conditions to know which tiles need to be painted and with what colors.
    - Extracts static facts to determine the layout of the tiles and their connections.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if all goal conditions are already met. If so, return 0.
    2. For each goal tile, determine if it is painted. If not, calculate the Manhattan distance from the robot's current position.
    3. Track the farthest tile and its required color.
    4. Sum the Manhattan distances for all unpainted tiles.
    5. If the robot's current color doesn't match the farthest tile's required color, add the cost of changing colors.
    6. If there are multiple color changes needed, add the appropriate number of actions.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract goal locations and required colors
        self.goal_info = {}
        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'painted':
                tile = parts[1]
                color = parts[2]
                self.goal_info[tile] = color

        # Extract static information about tile connections
        self.up = {}
        self.down = {}
        self.left = {}
        self.right = {}
        for fact in self.static:
            if fact.startswith('(up '):
                y, x = fact[4:-1].split()
                self.up[x] = y
                self.down[y] = x
            elif fact.startswith('(left '):
                x, y = fact[6:-1].split()
                self.left[x] = y
                self.right[y] = x

    def __call__(self, node):
        """Estimate the minimum number of actions to reach the goal state."""
        state = node.state

        # Check if all goals are already met
        if all(goal in state for goal in self.goals):
            return 0

        # Extract robot's current position and held color
        robot_pos = None
        robot_color = None
        for fact in state:
            if fact.startswith('(robot-at '):
                robot_pos = fact[10:-1].split()[1]
            elif fact.startswith('(robot-has '):
                robot_color = fact[12:-1].split()[1]

        # If robot position is not found, assume it's at the initial position
        if robot_pos is None:
            # Extract initial position from task's initial state
            for fact in node.task.initial_state:
                if fact.startswith('(robot-at '):
                    robot_pos = fact[10:-1].split()[1]
                    break

        # Calculate the number of unpainted goal tiles
        unpainted = []
        for tile, color in self.goal_info.items():
            if f'(painted {tile} {color})' not in state:
                unpainted.append((tile, color))

        if not unpainted:
            return 0

        # Function to calculate Manhattan distance between two tiles
        def manhattan_distance(a, b):
            x1, y1 = a.split('_')
            x2, y2 = b.split('_')
            return abs(int(x1) - int(x2)) + abs(int(y1) - int(y2))

        # Find the farthest tile from the robot's current position
        max_distance = 0
        farthest_tile = None
        required_color = None
        for tile, color in unpainted:
            distance = manhattan_distance(robot_pos, tile)
            if distance > max_distance:
                max_distance = distance
                farthest_tile = tile
                required_color = color

        # Sum the distances for all unpainted tiles
        total_distance = sum(manhattan_distance(robot_pos, tile) for (tile, _) in unpainted)

        # Add cost for color change if needed
        color_cost = 0
        if robot_color != required_color:
            color_cost = 1

        # Add cost for moving to the farthest tile and back if necessary
        # This is a simplification, assuming each tile requires a visit
        total_cost = total_distance + color_cost + (len(unpainted) - 1)

        return total_cost
