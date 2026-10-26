from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the minimum number of moves required to solve a Sokoban puzzle
    by calculating the sum of Manhattan distances for each box from its current location
    to its goal location. It is a simple and fast heuristic that provides a reasonable
    estimate of the remaining work.

    # Assumptions:
    - The heuristic assumes that boxes can be moved independently to their goal locations,
      ignoring potential blockages by other boxes or walls.
    - It does not consider the robot's path or the 'clear' predicate in detail, focusing
      solely on the displacement of boxes.
    - It is a lower bound on the number of moves in an ideal scenario where boxes can be moved
      directly to their goals without interference.

    # Heuristic Initialization
    - The heuristic initialization parses the goal conditions from the task definition
      to identify the goal location for each box.
    - It extracts the goal locations and stores them in a dictionary mapping each box
      to its target location.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract Goal Box Locations:
       - During initialization, parse the goal conditions to identify the target location for each box.
       - Store these goal locations in a dictionary for quick access during heuristic computation.

    2. Identify Current Box Locations:
       - For a given state, iterate through the facts to find the current location of each box.
       - Match facts of the form '(at ?box ?location)' to determine the current location of each box.

    3. Calculate Manhattan Distance for Each Box:
       - For each box that has a goal location:
         - Extract the current location and the goal location.
         - Assuming location names are in the format 'loc_row_col', parse the row and column numbers.
         - Calculate the Manhattan distance between the current and goal locations using the formula:
           |row_goal - row_current| + |col_goal - col_current|.

    4. Sum Manhattan Distances:
       - Sum the Manhattan distances calculated for all boxes.
       - This sum represents the estimated number of moves required to reach the goal state.

    5. Return Heuristic Value:
       - Return the total sum of Manhattan distances as the heuristic value for the given state.
       - A goal state will have a heuristic value of 0 because all boxes will be at their goal locations,
         resulting in a Manhattan distance of 0 for each box.
    """

    def __init__(self, task):
        """Initialize the sokoban heuristic by parsing goal locations for each box."""
        self.goals = task.goals
        self.goal_box_locations = {}

        for goal_fact in self.goals:
            if match(goal_fact, "at", "*", "*"):
                parts = get_parts(goal_fact)
                box_name = parts[1]
                goal_location_name = parts[2]
                self.goal_box_locations[box_name] = goal_location_name

    def __call__(self, node):
        """Calculate the sokoban heuristic value for a given state based on Manhattan distance."""
        state = node.state
        heuristic_value = 0

        current_box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                box_name = parts[1]
                location_name = parts[2]
                current_box_locations[box_name] = location_name

        for box_name, goal_location_name in self.goal_box_locations.items():
            if box_name not in current_box_locations:
                continue # box might not be in the state if problem is malformed, or during search

            current_location_name = current_box_locations[box_name]

            if current_location_name == goal_location_name:
                continue # box already at goal

            try:
                current_row, current_col = map(int, re.findall(r'\d+', current_location_name))
                goal_row, goal_col = map(int, re.findall(r'\d+', goal_location_name))

                manhattan_distance = abs(goal_row - current_row) + abs(goal_col - current_col)
                heuristic_value += manhattan_distance
            except ValueError:
                # Handle cases where location names are not in 'loc_row_col' format if needed.
                # For now, just skip boxes with incorrectly formatted locations for simplicity
                pass

        return heuristic_value
