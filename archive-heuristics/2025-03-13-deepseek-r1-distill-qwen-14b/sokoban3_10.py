from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at box1 loc_2_3)".
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
    - The robot can move to adjacent clear locations.
    - Boxes can be pushed to adjacent clear locations.
    - The goal is to have all boxes at their respective target locations.

    # Heuristic Initialization
    - Extract the goal locations for each box from the task.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. For each box, determine its current location and target location.
    2. Calculate the Manhattan distance between the box's current location and target location.
    3. Calculate the Manhattan distance between the robot's current location and the box's current location.
    4. Sum the distances from steps 2 and 3, and add 1 for the push action.
    5. Sum these values for all boxes to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each box.
        """
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Static facts

        # Extract goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

        # Extract robot's current location if applicable
        self.robot_location = None
        for fact in task.initial_state:
            if match(fact, "at-robot", "*"):
                self.robot_location = get_parts(fact)[1]
                break

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state

        # Extract current locations of boxes and robot
        current_locations = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at":
                obj, loc = args
                if obj.startswith("box"):
                    current_locations[obj] = loc
            elif predicate == "at-robot":
                self.robot_location = args[0]

        total_cost = 0  # Initialize action cost counter

        # For each box, calculate the required actions
        for box, goal_location in self.goal_locations.items():
            if box not in current_locations:
                continue  # Box is already at the goal

            current_loc = current_locations[box]
            if current_loc == goal_location:
                continue  # Box is already at the target

            # Calculate Manhattan distance between current and goal locations
            def manhattan(a, b):
                x1, y1 = a.split('_')
                x2, y2 = b.split('_')
                return abs(int(x1) - int(x2)) + abs(int(y1) - int(y2))

            box_distance = manhattan(current_loc, goal_location)
            robot_distance = manhattan(self.robot_location, current_loc)

            # Each box requires robot to move to it and push it to the goal
            total_cost += robot_distance + box_distance + 1  # +1 for the push action

        return total_cost
