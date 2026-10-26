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

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state in the Sokoban domain.
    It calculates the sum of Manhattan distances for each box from its current location to its closest goal location.
    If a box has multiple goal locations specified, it considers the minimum Manhattan distance to any of them.

    # Assumptions:
    - The heuristic assumes a grid-like structure for locations, inferring coordinates from location names like 'loc_r_c'.
    - It assumes that moving a box or the robot to an adjacent location constitutes one step in Manhattan distance.
    - It does not account for obstacles, walls, or deadlocks, thus it is not admissible.

    # Heuristic Initialization
    - The constructor parses the goal conditions from the task to identify the goal locations for each box.
    - It stores the goal locations in a dictionary where keys are box names and values are lists of goal locations for that box (in case of conjunctive goals with multiple possible goal locations for the same box, although typical Sokoban goals specify a unique goal location per box).
    - Static facts are not explicitly used in this simple Manhattan distance heuristic, but could be incorporated for more sophisticated heuristics in the future (e.g., to detect deadlocks or consider maze structure).

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic value to 0.
    2. Extract the current locations of all boxes from the current state. Store them in a dictionary for easy access, e.g., `current_box_locations = {box_name: current_location}`.
    3. For each box that has a goal location specified in the task's goals:
        a. Get the box name and its list of goal locations.
        b. Get the current location of the box from `current_box_locations`.
        c. If the box is not at any of its goal locations:
            i. Calculate the Manhattan distance from the current location to each goal location in the list.
            ii. Find the minimum Manhattan distance among all goal locations for this box.
            iii. Add this minimum Manhattan distance to the total heuristic value.
    4. Return the total heuristic value.

    # Manhattan Distance Calculation:
    - Location names are assumed to be in the format 'loc_r_c', where 'r' and 'c' are row and column numbers.
    - Parse the row and column numbers from both the current location and the goal location.
    - Manhattan distance is calculated as the sum of the absolute differences of row numbers and column numbers: `abs(goal_row - current_row) + abs(goal_col - current_col)`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal box locations."""
        self.goals = task.goals
        self.goal_locations = {} # Store goal locations for each box

        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == 'at':
                box_name, goal_location = args
                if box_name not in self.goal_locations:
                    self.goal_locations[box_name] = []
                self.goal_locations[box_name].append(goal_location)


    def __call__(self, node):
        """Calculate the Manhattan distance heuristic for the given state."""
        state = node.state
        current_box_locations = {}

        for fact in state:
            if match(fact, 'at', '*', '*'):
                _, box_name, location = get_parts(fact)
                current_box_locations[box_name] = location

        heuristic_value = 0
        for box_name, goal_locations in self.goal_locations.items():
            current_location = current_box_locations.get(box_name)
            if current_location: # Only calculate if box is present in the state
                min_distance = float('inf')
                for goal_location in goal_locations:
                    if current_location != goal_location:
                        try:
                            current_loc_parts = current_location.split('_')
                            goal_loc_parts = goal_location.split('_')
                            current_row = int(current_loc_parts[1])
                            current_col = int(current_loc_parts[2])
                            goal_row = int(goal_loc_parts[1])
                            goal_col = int(goal_loc_parts[2])
                            distance = abs(goal_row - current_row) + abs(goal_col - current_col)
                            min_distance = min(min_distance, distance)
                        except:
                            # Fallback if location names are not in 'loc_r_c' format, just count as 1 if not at goal
                            if current_location != goal_location:
                                min_distance = 1 # Assign a distance of 1 if parsing fails and not at goal

                if min_distance != float('inf'): # Add distance only if box is not at goal
                    heuristic_value += min_distance

        return heuristic_value
