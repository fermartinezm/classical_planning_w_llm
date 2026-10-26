from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def parse_location(loc_str):
    """Extracts x and y coordinates from a location string like 'loc_x_y'."""
    parts = loc_str.split('_')
    return (int(parts[1]), int(parts[2]))

class sokoban9Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to push all boxes to their goal positions.
    It sums the Manhattan distance of each box to its goal and adds the robot's distance to the nearest box.

    # Assumptions
    - Each box has a unique goal location.
    - The robot can move freely to any adjacent cell if it's clear.
    - The heuristic uses Manhattan distance for efficiency, ignoring obstacles and actual path complexity.

    # Heuristic Initialization
    - Extracts goal locations for each box from the task's goal conditions.
    - Stores these goals in a dictionary mapping each box to its target location.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Extract Current Positions**: For the given state, determine the current location of the robot and each box.
    2. **Check Goal Achievement**: For each box, check if it is already at its goal location.
    3. **Calculate Box Distances**: Sum the Manhattan distances from each box's current position to its goal.
    4. **Robot Proximity**: Compute the Manhattan distance from the robot to the nearest box that is not yet at its goal.
    5. **Combine Values**: The heuristic value is the sum of the box distances and the robot's minimum distance to a box.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal locations for each box."""
        self.goal_locations = {}
        for goal in task.goals:
            if not isinstance(goal, str):
                continue
            if not goal.startswith('(at '):
                continue
            parts = goal[1:-1].split()
            if len(parts) == 3 and parts[0] == 'at' and parts[1].startswith('box'):
                box = parts[1]
                loc = parts[2]
                self.goal_locations[box] = loc

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        current_boxes = {}
        robot_loc = None

        # Extract current positions of boxes and the robot
        for fact in state:
            if not isinstance(fact, str):
                continue
            if fact.startswith('(at '):
                parts = fact[1:-1].split()
                if parts[0] == 'at' and parts[1] in self.goal_locations:
                    current_boxes[parts[1]] = parts[2]
            elif fact.startswith('(at-robot '):
                parts = fact[1:-1].split()
                robot_loc = parts[1]

        if not robot_loc:
            return 0  # Invalid state, assume goal reached

        sum_box_dist = 0
        boxes_not_in_goal = []
        for box, goal_loc in self.goal_locations.items():
            current_loc = current_boxes.get(box)
            if current_loc != goal_loc:
                boxes_not_in_goal.append(box)
                # Calculate Manhattan distance for the box to its goal
                curr_x, curr_y = parse_location(current_loc)
                goal_x, goal_y = parse_location(goal_loc)
                sum_box_dist += abs(curr_x - goal_x) + abs(curr_y - goal_y)

        if not boxes_not_in_goal:
            return 0  # All boxes are at their goals

        # Calculate robot's distance to the nearest box not at goal
        robot_x, robot_y = parse_location(robot_loc)
        min_robot_dist = float('inf')
        for box in boxes_not_in_goal:
            box_loc = current_boxes[box]
            box_x, box_y = parse_location(box_loc)
            dist = abs(robot_x - box_x) + abs(robot_y - box_y)
            if dist < min_robot_dist:
                min_robot_dist = dist

        return sum_box_dist + min_robot_dist
