from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_2_4)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class SokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to move all boxes to their target locations.

    # Assumptions:
    - The robot can move to adjacent locations if they are clear.
    - A box can only be pushed if the robot is adjacent to it and the target location is clear.
    - The heuristic uses Manhattan distance to estimate the number of moves required.

    # Heuristic Initialization
    - Extract the target location for each box from the goal conditions.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. For each box, determine its current location and target location.
    2. If the box is already at its target location, no actions are needed.
    3. If the box is not at its target location, calculate the Manhattan distance from the robot's current position to the box's current location.
    4. Calculate the Manhattan distance from the box's current location to its target location.
    5. Sum these distances for all boxes that need to be moved to estimate the total number of actions required.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the target location for each box.
        """
        self.goals = task.goals  # Goal conditions.

        # Extract the target location for each box.
        self.box_goals = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.box_goals[box] = location

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state.

        # Extract current locations of boxes and the robot.
        current_locations = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at":
                obj, location = args
                current_locations[obj] = location
            elif predicate == "at-robot":
                current_locations["robot"] = args[0]

        total_cost = 0  # Initialize action cost counter.

        # For each box, calculate the required actions.
        for box, goal_location in self.box_goals.items():
            # Check if the box is already at the goal.
            if box not in current_locations:
                # The box is not present, which should not happen in Sokoban.
                continue
            current_location = current_locations[box]

            if current_location == goal_location:
                continue  # No action needed.

            # Calculate the Manhattan distance from the robot to the box.
            robot_location = current_locations.get("robot", None)
            if robot_location is None:
                # Robot's location is unknown, assume maximum distance.
                continue

            # Split into coordinates (assuming loc_x_y format).
            try:
                x1, y1 = map(int, robot_location.split('_')[1:])
                x2, y2 = map(int, current_location.split('_')[1:])
                distance_robot_to_box = abs(x1 - x2) + abs(y1 - y2)
            except ValueError:
                # Invalid location format, skip distance calculation.
                distance_robot_to_box = 0

            # Calculate the Manhattan distance from the box to its goal.
            try:
                gx, gy = map(int, goal_location.split('_')[1:])
                distance_box_to_goal = abs(x2 - gx) + abs(y2 - gy)
            except ValueError:
                distance_box_to_goal = 0

            total_cost += distance_robot_to_box + distance_box_to_goal

        return total_cost
