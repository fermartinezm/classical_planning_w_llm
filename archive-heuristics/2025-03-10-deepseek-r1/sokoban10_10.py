from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_coords(loc_name):
    """Extract x and y coordinates from a location name formatted as loc_x_y."""
    parts = loc_name.split('_')
    x = int(parts[1])
    y = int(parts[2])
    return (x, y)

class Sokoban10Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to push all boxes to their goal positions. It combines the minimal pushes needed for each box and the robot's distance to the nearest box.

    # Assumptions:
    - The grid is regular, and locations follow the naming convention loc_x_y.
    - Boxes can be pushed in a straight line towards their goals without considering obstacles.
    - The robot can reach the required side of a box to push it in the optimal direction.

    # Heuristic Initialization
    - Extract the goal location for each box from the task's goals.
    - Precompute the coordinates of each goal location based on their names.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box, calculate the Manhattan distance from its current position to its goal. This represents the minimal number of pushes needed.
    2. Calculate the Manhattan distance from the robot's current position to the nearest box. This estimates the steps needed for the robot to reach a box to start pushing.
    3. Sum these values to get the heuristic estimate.
    4. If all boxes are already at their goals, return 0.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal locations for each box."""
        self.goals = {}
        for goal in task.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'at' and len(parts) == 3:
                box = parts[1]
                loc = parts[2]
                self.goals[box] = loc

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        boxes = {}
        robot_loc = None

        # Extract current positions of boxes and robot
        for fact in state:
            if fact.startswith('(at '):
                parts = fact[1:-1].split()
                if parts[0] == 'at' and parts[1] in self.goals:
                    box = parts[1]
                    loc = parts[2]
                    boxes[box] = loc
            elif fact.startswith('(at-robot '):
                parts = fact[1:-1].split()
                robot_loc = parts[1]

        # Check if all boxes are at their goals
        if all(current_loc == self.goals.get(box) for box, current_loc in boxes.items()):
            return 0

        # Calculate sum of box distances to their goals
        total_box_distance = 0
        for box, current_loc in boxes.items():
            goal_loc = self.goals.get(box)
            if not goal_loc:
                continue  # Skip if box has no defined goal (unlikely in valid tasks)
            current_x, current_y = get_coords(current_loc)
            goal_x, goal_y = get_coords(goal_loc)
            total_box_distance += abs(current_x - goal_x) + abs(current_y - goal_y)

        # Calculate robot's distance to the nearest box
        if not robot_loc or not boxes:
            return total_box_distance  # Handle edge cases gracefully

        robot_x, robot_y = get_coords(robot_loc)
        min_robot_distance = min(
            abs(robot_x - get_coords(loc)[0]) + abs(robot_y - get_coords(loc)[1])
            for loc in boxes.values()
        )

        return total_box_distance + min_robot_distance
