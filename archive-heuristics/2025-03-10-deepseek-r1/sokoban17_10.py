from heuristics.heuristic_base import Heuristic

class sokoban17Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to push all boxes to their goal positions. It calculates the Manhattan distance for each box to its goal and the robot's Manhattan distance to the box, then combines them to estimate the total actions.

    # Assumptions
    - The grid is navigable in a Manhattan path (ignoring dynamic obstacles like other boxes).
    - Each push action requires the robot to be adjacent to the box, costing one action.
    - Robot movement to the box's adjacent cell is estimated using Manhattan distance.

    # Heuristic Initialization
    - Extracts goal locations for each box from the task's goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box not at its goal:
        a. Compute Manhattan distance from the box's current location to its goal (number of pushes needed).
        b. Compute Manhattan distance from the robot's current location to the box's current location.
        c. Estimate robot's movement to reach the box's adjacent cell as (robot_manhattan - 1).
        d. Each push requires the robot to move into position after the first push, adding (box_manhattan - 1) moves.
        e. Total actions for the box: (robot_manhattan - 1) + (2 * box_manhattan - 1).
    2. Sum the estimated actions for all boxes to get the heuristic value.
    """

    def __init__(self, task):
        self.goal_locations = {}
        for goal in task.goals:
            if goal.startswith('(at '):
                parts = goal[1:-1].split()
                box = parts[1]
                loc = parts[2]
                self.goal_locations[box] = loc

    def __call__(self, node):
        state = node.state
        robot_loc = None
        boxes = {}

        for fact in state:
            if fact.startswith('(at-robot '):
                parts = fact[1:-1].split()
                robot_loc = parts[1]
            elif fact.startswith('(at '):
                parts = fact[1:-1].split()
                obj = parts[1]
                loc = parts[2]
                boxes[obj] = loc

        if not robot_loc:
            return 0  # Assume goal state if no robot (unlikely)

        total = 0

        for box, current_loc in boxes.items():
            goal_loc = self.goal_locations.get(box)
            if not goal_loc or current_loc == goal_loc:
                continue

            # Parse current and goal coordinates
            current_parts = current_loc.split('_')
            current_x = int(current_parts[1])
            current_y = int(current_parts[2])
            goal_parts = goal_loc.split('_')
            goal_x = int(goal_parts[1])
            goal_y = int(goal_parts[2])
            box_manhattan = abs(current_x - goal_x) + abs(current_y - goal_y)

            # Parse robot's coordinates
            robot_parts = robot_loc.split('_')
            robot_x = int(robot_parts[1])
            robot_y = int(robot_parts[2])
            robot_manhattan = abs(robot_x - current_x) + abs(robot_y - current_y)

            # Calculate contribution
            contribution = (robot_manhattan - 1) + (2 * box_manhattan - 1)
            total += contribution

        return total
