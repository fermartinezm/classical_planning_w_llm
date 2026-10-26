from heuristics.heuristic_base import Heuristic

class SokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to move all boxes to their target locations by considering both the robot's movement to each box and the box's movement to the target.

    # Assumptions:
    - Each box must be pushed to its target location.
    - The robot can move freely to reach each box.
    - The Manhattan distance is used to estimate the required movements.

    # Heuristic Initialization
    - Extract the target location for each box from the goal facts.
    - Store the target locations in a dictionary for quick lookup.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box, determine its current location and target location.
    2. If the box is already at the target, no actions are needed.
    3. Calculate the Manhattan distance from the robot's current position to the box's current location.
    4. Calculate the Manhattan distance from the box's current location to its target location.
    5. Sum these distances for all boxes to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting target locations for each box."""
        self.box_targets = {}
        for goal in task.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'at' and parts[1].startswith('box'):
                box = parts[1]
                target = parts[2]
                self.box_targets[box] = target

    def __call__(self, node):
        """Estimate the total number of actions needed to reach the goal state."""
        state = node.state
        current_boxes = {}
        robot_location = None

        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'at-robot':
                robot_location = parts[1]
            elif parts[0] == 'at' and parts[1].startswith('box'):
                box = parts[1]
                loc = parts[2]
                current_boxes[box] = loc

        heuristic = 0

        for box, current_loc in current_boxes.items():
            if box in self.box_targets:
                target_loc = self.box_targets[box]
                if current_loc != target_loc:
                    current = self.parse_location(current_loc)
                    target = self.parse_location(target_loc)
                    box_distance = abs(current[0] - target[0]) + abs(current[1] - target[1])

                    if robot_location is None:
                        robot_current = (0, 0)
                    else:
                        robot_current = self.parse_location(robot_location)
                    robot_distance = abs(robot_current[0] - current[0]) + abs(robot_current[1] - current[1])

                    heuristic += (robot_distance + box_distance)

        return heuristic

    def parse_location(self, loc_str):
        """Parse a location string (e.g., 'loc_2_4') into (row, column) coordinates."""
        parts = loc_str.split('_')[1:]
        return tuple(map(int, parts))
