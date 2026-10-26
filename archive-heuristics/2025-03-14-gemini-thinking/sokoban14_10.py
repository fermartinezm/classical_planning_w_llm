from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re

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
    This heuristic estimates the cost to reach the goal state in the Sokoban domain
    by calculating the sum of Manhattan distances for each box from its current location
    to its goal location. It assumes that each move (move or push) has a cost of 1.

    # Assumptions:
    - The heuristic assumes that boxes can be moved independently to their goal locations,
      ignoring potential blockages caused by other boxes or walls.
    - It only considers the distances of boxes to their goals and does not account for
      the robot's movements or the 'clear' predicate directly, except implicitly through
      the possibility of moving boxes.
    - It assumes location names are in the format 'loc_row_column' (e.g., 'loc_2_4'),
      allowing extraction of row and column indices for Manhattan distance calculation.

    # Heuristic Initialization
    - Extracts the goal box locations from the task goals.
    - Stores the goal locations in a dictionary for efficient lookup during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Extract the current locations of all boxes from the current state.
    3. For each box that has a goal location:
        a. Retrieve the current location of the box from the state.
        b. Retrieve the goal location of the box from the pre-processed goal information.
        c. Parse the row and column numbers from both the current and goal location names.
           Location names are assumed to be in the format 'loc_row_column'.
        d. Calculate the Manhattan distance between the current location and the goal location
           using the formula: |goal_row - current_row| + |goal_column - current_column|.
        e. Add the calculated Manhattan distance to the total heuristic value.
    4. Return the total heuristic value as the estimated cost to reach the goal state.
    """

    def __init__(self, task):
        """
        Initialize the sokoban heuristic by extracting goal box locations.
        """
        self.goals = task.goals
        self.goal_box_locations = {}

        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                box_name = parts[1]
                goal_location = parts[2]
                self.goal_box_locations[box_name] = goal_location

    def __call__(self, node):
        """
        Calculate the heuristic value for a given state based on Manhattan distances of boxes to their goal locations.
        """
        state = node.state
        current_box_locations = {}

        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                box_name = parts[1]
                current_location = parts[2]
                current_box_locations[box_name] = current_location

        heuristic_value = 0
        for box_name, goal_location in self.goal_box_locations.items():
            if box_name in current_box_locations:
                current_location = current_box_locations[box_name]

                try:
                    current_row, current_col = map(int, current_location.split('_')[1:])
                    goal_row, goal_col = map(int, goal_location.split('_')[1:])
                    manhattan_distance = abs(goal_row - current_row) + abs(goal_col - current_col)
                    heuristic_value += manhattan_distance
                except ValueError:
                    # Handle cases where location names are not in the expected format.
                    # A simple fallback is to assign a large distance or skip the box.
                    # Here, we skip to avoid crashing, but a better heuristic might handle this more robustly.
                    pass # or heuristic_value += 100  # Assign a large penalty


        return heuristic_value
