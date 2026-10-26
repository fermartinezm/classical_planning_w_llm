from heuristics.heuristic_base import Heuristic

class SokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to push all boxes to their goal positions. It considers both the distance each box needs to move and the distance the robot needs to travel to reach each box.

    # Assumptions:
    - The robot can push a box if it is adjacent to it.
    - Each box must be pushed to its specific goal location.
    - The heuristic does not account for the exact sequence of pushes, only the total distance required.

    # Heuristic Initialization
    - Extract the goal positions for each box from the task's goals.
    - Store the static facts, which include the adjacency relationships between locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box, determine its current location and its goal location.
    2. Calculate the Manhattan distance from the box's current location to its goal location.
    3. Sum these distances to get the total box movement cost.
    4. Find the robot's current location.
    5. For each box, calculate the Manhattan distance from the robot's current location to the box's current location.
    6. Sum these distances to get the total robot movement cost.
    7. The heuristic value is the sum of the total box movement cost and the total robot movement cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = {}
        for goal in task.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'at' and parts[1].startswith('box'):
                box = parts[1]
                loc = parts[2]
                self.goals[box] = loc

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        current_box_positions = {}
        for fact in state:
            if fact.startswith('(at '):
                parts = fact[1:-1].split()
                if parts[1] == 'box':
                    box = parts[1]
                    loc = parts[2]
                    current_box_positions[box] = loc

        robot_pos = None
        for fact in state:
            if fact.startswith('(at-robot '):
                parts = fact[1:-1].split()
                robot_pos = parts[1]
                break

        if not robot_pos:
            return 0  # No robot position found, assume 0 (shouldn't happen)

        box_distance_sum = 0
        robot_distance_sum = 0

        for box, goal_loc in self.goals.items():
            current_loc = current_box_positions.get(box, None)
            if not current_loc:
                continue  # Box not present, shouldn't happen
            if current_loc == goal_loc:
                continue  # Box is already at goal

            # Calculate Manhattan distance from current_loc to goal_loc
            x1, y1 = self.parse_location(current_loc)
            x2, y2 = self.parse_location(goal_loc)
            box_distance = abs(x1 - x2) + abs(y1 - y2)
            box_distance_sum += box_distance

            # Calculate Manhattan distance from robot_pos to current_loc
            x_r, y_r = self.parse_location(robot_pos)
            robot_distance = abs(x_r - x1) + abs(y_r - y1)
            robot_distance_sum += robot_distance

        total_heuristic = box_distance_sum + robot_distance_sum
        return total_heuristic

    def parse_location(self, loc_str):
        """Parse a location string like 'loc_2_4' into (x, y)."""
        parts = loc_str.split('_')
        x = int(parts[1])
        y = int(parts[2])
        return (x, y)
