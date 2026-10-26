from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class FloortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all required tiles by considering:
    - The distance the robot needs to move to each unpainted tile.
    - Whether the robot has the required color or needs to change it.
    - The efficiency of the movement and painting actions.

    # Assumptions:
    - The robot can move up, down, left, or right between adjacent tiles.
    - Each painting action requires the robot to be on the tile and have the correct color.
    - The robot can change colors when needed, but this adds an extra action.
    - The shortest path between tiles is used to minimize movement actions.

    # Heuristic Initialization
    - Extracts goal conditions and static facts (tile connections and available colors).
    - Constructs a grid map for efficient distance calculation between tiles.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all tiles that need to be painted and their required colors.
    2. For each tile:
       a. If already painted, no action is needed.
       b. If not painted, calculate the Manhattan distance from the robot's current position to the tile.
       c. Add 1 for the painting action.
       d. If the robot doesn't have the required color, add 1 for the color change action.
    3. Sum the estimated actions for all tiles to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract tile connections to build grid map
        self.grid = {}
        for fact in static_facts:
            if fnmatch(fact, '(up * *)'):
                y, x = fact.split()[1], fact.split()[2]
                self.grid[(x, y)] = 'up'
            elif fnmatch(fact, '(down * *)'):
                y, x = fact.split()[1], fact.split()[2]
                self.grid[(x, y)] = 'down'
            elif fnmatch(fact, '(left * *)'):
                y, x = fact.split()[1], fact.split()[2]
                self.grid[(x, y)] = 'left'
            elif fnmatch(fact, '(right * *)'):
                y, x = fact.split()[1], fact.split()[2]
                self.grid[(x, y)] = 'right'

        # Extract available colors
        self.available_colors = {
            fact.split()[1] for fact in static_facts if fnmatch(fact, '(available-color *)')
        }

    def __call__(self, node):
        """Estimate the minimum cost to reach the goal state."""
        state = node.state

        # Extract current robot position and color
        robot_pos = None
        robot_color = None
        for fact in state:
            if fnmatch(fact, '(robot-at * *)'):
                robot_pos = fact.split()[1], fact.split()[2]
            if fnmatch(fact, '(robot-has * *)'):
                robot_color = fact.split()[2]

        # If robot position is not found, assume it's at the initial position
        if not robot_pos:
            for fact in state:
                if fnmatch(fact, '(robot-at * *)'):
                    robot_pos = fact.split()[1], fact.split()[2]
                    break

        # If robot color is not found, assume it's at the initial color
        if not robot_color:
            for fact in state:
                if fnmatch(fact, '(robot-has * *)'):
                    robot_color = fact.split()[2]
                    break

        # Count required actions
        total_cost = 0

        # For each goal tile, check if it's painted and calculate distance if not
        for goal in self.goals:
            if not fnmatch(goal, '(painted * *)'):
                continue
            tile = goal.split()[1]
            required_color = goal.split()[2]

            # Check if tile is already painted
            if any(fnmatch(fact, f'(painted {tile} *)') for fact in state):
                continue

            # Calculate Manhattan distance from robot's current position to the tile
            x1, y1 = robot_pos
            x2, y2 = tile.split('_')
            distance = abs(int(x1) - int(x2)) + abs(int(y1) - int(y2))

            # Add movement actions
            total_cost += distance

            # Add painting action
            total_cost += 1

            # If robot doesn't have the required color, add color change action
            if robot_color != required_color:
                total_cost += 1

        return total_cost
