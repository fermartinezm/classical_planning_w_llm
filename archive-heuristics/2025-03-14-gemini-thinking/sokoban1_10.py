from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state in the Sokoban domain.
    It calculates the sum of Manhattan distances for each box from its current location to its goal location.

    # Assumptions:
    - The heuristic assumes that moving boxes individually towards their goal locations is a good strategy.
    - It does not consider obstacles, deadlocks, or the robot's path.
    - It underestimates the actual cost in cases where boxes need to be moved out of the way first.

    # Heuristic Initialization
    - The heuristic is initialized by parsing the goal conditions from the task definition.
    - It extracts the goal location for each box and stores it in `self.goal_box_locations`.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. For each box that has a goal location specified in the problem definition:
        a. Get the current location of the box from the current state.
        b. Get the goal location of the box from the initialized `self.goal_box_locations`.
        c. Parse the current location and goal location names to extract row and column numbers.
           Location names are assumed to be in the format 'loc_r_c', where r is the row and c is the column.
        d. Calculate the Manhattan distance between the current location and the goal location:
           Manhattan distance = |goal_row - current_row| + |goal_col - current_col|.
        e. Add the calculated Manhattan distance to the heuristic value.
    3. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the sokobanHeuristic by extracting goal box locations.
        """
        self.goal_box_locations = {}
        for goal in task.goals:
            if goal.startswith('(at box'):
                parts = goal[1:-1].split()
                box_name = parts[1]
                location_name = parts[2]
                self.goal_box_locations[box_name] = location_name

    def __call__(self, node):
        """
        Calculate the heuristic value for the given state.
        """
        state = node.state
        heuristic_value = 0
        current_box_locations = {}

        for fact in state:
            if fact.startswith('(at box'):
                parts = fact[1:-1].split()
                box_name = parts[1]
                location_name = parts[2]
                current_box_locations[box_name] = location_name

        for box_name, goal_location_name in self.goal_box_locations.items():
            if box_name in current_box_locations:
                current_location_name = current_box_locations[box_name]

                def parse_location_name(loc_name):
                    parts = loc_name.split('_')
                    return int(parts[1]), int(parts[2])

                try:
                    goal_row, goal_col = parse_location_name(goal_location_name)
                    current_row, current_col = parse_location_name(current_location_name)
                    manhattan_distance = abs(goal_row - current_row) + abs(goal_col - current_col)
                    heuristic_value += manhattan_distance
                except:
                    # Handle cases where location name parsing fails (though unlikely in given format)
                    heuristic_value += 1000 # Assign a large penalty if parsing fails, indicating something is wrong

        return heuristic_value
