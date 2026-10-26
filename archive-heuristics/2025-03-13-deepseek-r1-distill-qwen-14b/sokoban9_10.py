from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def parse_location(loc_str):
    """Extracts x and y coordinates from a location string like 'loc_2_3'."""
    parts = loc_str.split('_')
    return int(parts[1]), int(parts[2])

class SokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to move each box to its goal location. It considers the distance the robot must travel to reach each box and the distance each box must travel to its goal.

    # Assumptions:
    - The robot can move to any adjacent clear location.
    - Each box can be pushed one step at a time.
    - The goal is to have all boxes in their respective target locations.

    # Heuristic Initialization
    - Extracts the goal locations for each box from the task's goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot.
    2. For each box, determine its current location and target goal location.
    3. Calculate the Manhattan distance from the robot to the box.
    4. Calculate the Manhattan distance from the box to its goal.
    5. Sum these distances and add one for the push action required for each box.
    6. The total heuristic value is the sum of these values for all boxes.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.goal_locations = {}
        for goal in self.goals:
            predicate, box, loc = goal[1:-1].split()
            self.goal_locations[box] = loc

    def __call__(self, node):
        """Estimate the minimum cost to move all boxes to their goal locations."""
        state = node.state
        robot_location = None
        boxes = {}

        # Extract robot's current location
        for fact in state:
            if fact.startswith('(at-robot '):
                robot_location = fact.split()[2]

        # Extract boxes' current locations
        for fact in state:
            if fact.startswith('(at '):
                parts = fact.split()
                if parts[1] == 'box':
                    box = parts[2]
                    loc = parts[4]
                    boxes[box] = loc

        total_cost = 0

        # Calculate heuristic for each box
        for box, goal in self.goal_locations.items():
            current = boxes.get(box)
            if current != goal:
                # Parse locations
                r_x, r_y = parse_location(robot_location)
                c_x, c_y = parse_location(current)
                g_x, g_y = parse_location(goal)

                # Calculate distances using Manhattan distance
                r_dist = abs(r_x - c_x) + abs(r_y - c_y)
                b_dist = abs(c_x - g_x) + abs(c_y - g_y)

                # Add the cost for this box
                total_cost += r_dist + b_dist + 1  # +1 for the push action

        return total_cost
