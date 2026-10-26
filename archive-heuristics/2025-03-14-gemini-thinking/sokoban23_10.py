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
    A domain-dependent heuristic for the sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state in the Sokoban domain.
    It calculates the sum of Manhattan distances for each box from its current location to its goal location.
    This heuristic is admissible in a relaxed version of the problem where boxes can move freely without considering other boxes or walls.

    # Assumptions:
    - The goal is to move each box to a specific target location.
    - The heuristic only considers the distances of boxes to their goals and ignores obstacles, robot position, and deadlocks.
    - It assumes that each box has a unique goal location specified in the goal description.

    # Heuristic Initialization
    - Parses the goal conditions to identify the goal location for each box.
    - Extracts static facts about adjacent locations, although these are not directly used in the current Manhattan distance heuristic but could be used for more advanced heuristics.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract goal box locations from the task goals. Store them in a dictionary where keys are box names and values are goal locations.
    2. For a given state, iterate through each box that has a goal location.
    3. For each box, find its current location in the state.
    4. Parse the location names to extract row and column numbers (assuming location names are in the format 'loc_r_c').
    5. Calculate the Manhattan distance between the current location and the goal location for each box.
       Manhattan distance is calculated as the sum of the absolute differences in row and column numbers.
    6. Sum up the Manhattan distances for all boxes.
    7. Return the total sum as the heuristic value. This value represents the estimated number of moves needed to reach the goal state.
       A heuristic value of 0 indicates that all boxes are at their goal locations, which is a goal state (or a state that is considered as goal by this heuristic).
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal box locations and static facts."""
        self.goals = task.goals
        self.static = task.static
        self.goal_box_locations = {}

        for goal_fact in self.goals:
            if match(goal_fact, "at", "*", "*"):
                parts = get_parts(goal_fact)
                box_name = parts[1]
                goal_location = parts[2]
                self.goal_box_locations[box_name] = goal_location

    def __call__(self, node):
        """Calculate the Manhattan distance heuristic for the given state."""
        state = node.state
        heuristic_value = 0

        for box_name, goal_location in self.goal_box_locations.items():
            current_location = None
            for fact in state:
                if match(fact, "at", box_name, "*"):
                    current_location = get_parts(fact)[2]
                    break

            if current_location and goal_location:
                try:
                    current_loc_parts = current_location.split('_')
                    goal_loc_parts = goal_location.split('_')

                    current_row = int(current_loc_parts[1])
                    current_col = int(current_loc_parts[2])
                    goal_row = int(goal_loc_parts[1])
                    goal_col = int(goal_loc_parts[2])

                    manhattan_distance = abs(current_row - goal_row) + abs(current_col - goal_col)
                    heuristic_value += manhattan_distance
                except:
                    # Handle cases where location names are not in the expected format.
                    # In such cases, a more robust location parsing might be needed,
                    # or a fallback heuristic could be used. For simplicity, we'll ignore errors for now
                    # and assume location names are correctly formatted.
                    pass

        return heuristic_value
