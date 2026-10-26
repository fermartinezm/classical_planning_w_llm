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
    This heuristic estimates the minimum number of moves required to push all boxes to their goal locations.
    It calculates the sum of Manhattan distances for each box from its current location to its closest goal location.

    # Assumptions:
    - The heuristic assumes that boxes can be moved independently to their goal locations, ignoring potential blockages by other boxes or walls.
    - It only considers the distances of boxes to goal locations and does not account for robot movements explicitly, except implicitly through box movements.
    - It assumes that every box has at least one goal location specified.

    # Heuristic Initialization
    - Extracts the goal locations for each box from the task goals.
    - No static facts are explicitly used in this heuristic calculation, although the layout of the grid (adjacency) is implicitly considered by Manhattan distance.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract Goal Box Locations:
       - Iterate through the goal predicates.
       - Identify predicates of the form '(at box ?location)'.
       - Store the goal location for each box. If a box has multiple goal locations (which is not typical in standard Sokoban but possible in PDDL), consider the closest one in the heuristic calculation. For simplicity, we will assume each box has a single goal location in typical Sokoban problems.

    2. Extract Current Box Locations:
       - Iterate through the current state predicates.
       - Identify predicates of the form '(at box ?location)'.
       - Store the current location for each box.

    3. Calculate Manhattan Distance for Each Box:
       - For each box, retrieve its current location and its goal location.
       - Parse the location names to extract row and column numbers (assuming location names are in the format 'loc_r_c').
       - Calculate the Manhattan distance between the current and goal locations: |goal_row - current_row| + |goal_col - current_col|.

    4. Sum Manhattan Distances:
       - Sum up the Manhattan distances calculated for all boxes.
       - This sum is the estimated heuristic value, representing the minimum number of moves needed to reach the goal state.

    5. Handle Goal States:
       - If the current state is a goal state, the heuristic value is 0. This is implicitly handled as the Manhattan distance will be 0 for all boxes in goal locations.
    """

    def __init__(self, task):
        """
        Initialize the Sokoban heuristic by extracting goal box locations.
        """
        self.goal_box_locations = {}
        self.boxes_in_goal = set()

        for goal in task.goals:
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                if parts[1] not in ['robot', '-robot']: # To avoid issues if goal includes robot position
                    box_name = parts[1]
                    goal_location = parts[2]
                    self.goal_box_locations[box_name] = goal_location


    def __call__(self, node):
        """
        Calculate the heuristic value for the given state.
        """
        state = node.state
        current_box_locations = {}

        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                box_name = parts[1]
                if box_name not in ['robot', '-robot']: # To avoid issues if state includes robot position
                    location = parts[2]
                    current_box_locations[box_name] = location

        total_distance = 0

        for box_name, goal_location in self.goal_box_locations.items():
            if box_name in current_box_locations: # Handle cases where not all boxes from goal are in initial state
                current_location = current_box_locations[box_name]

                try:
                    current_row, current_col = map(int, current_location.split('_')[1:])
                    goal_row, goal_col = map(int, goal_location.split('_')[1:])
                    distance = abs(goal_row - current_row) + abs(goal_col - current_col)
                    total_distance += distance
                except ValueError:
                    # Handle cases where location names are not in 'loc_r_c' format, default to a large distance.
                    total_distance += 100  # Assign a large penalty if location format is unexpected

        return total_distance
