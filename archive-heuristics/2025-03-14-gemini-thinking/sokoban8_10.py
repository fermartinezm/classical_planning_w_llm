from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re

def get_objects_from_fact(fact_string):
    """
    Extracts objects from a PDDL fact string.
    For example, from '(at box1 loc_3_5)' it extracts ['box1', 'loc_3_5'].
    Ignores the predicate name.
    """
    fact_content = fact_string[1:-1]  # Remove parentheses
    parts = fact_content.split()
    return parts[1:]  # Return objects, skipping the predicate name

def match(fact, *args):
    """
    Utility function to check if a PDDL fact matches a given pattern.
    - `fact`: The fact as a string (e.g., "(at ball1 rooma)").
    - `args`: The pattern to match (e.g., "at", "*", "rooma").
    - Returns `True` if the fact matches the pattern, `False` otherwise.
    """
    parts = fact[1:-1].split()  # Remove parentheses and split into individual elements.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state in the Sokoban domain.
    It calculates the sum of Manhattan distances for each box from its current location to its goal location.
    This heuristic is admissible in a simplified version of Sokoban where boxes can be moved freely without obstacles,
    but in the actual Sokoban domain, it is not admissible as it does not account for walls, other boxes, and the robot's movement constraints.
    However, it provides a reasonable and efficiently computable estimate for guiding greedy best-first search.

    # Assumptions:
    - The heuristic assumes that the primary difficulty in Sokoban is moving boxes to their target locations.
    - It simplifies the problem by ignoring the constraints imposed by walls and other boxes when calculating distances.
    - It assumes that each move or push action has a cost of 1.

    # Heuristic Initialization
    - The heuristic initializes by parsing the goal predicates from the task definition to identify the goal locations for each box.
    - It also extracts the static 'adjacent' facts to understand the grid structure, although Manhattan distance calculation does not directly use adjacency information in this simple version.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract goal box locations: Iterate through the goal predicates and identify the target location for each box. Store this information in a dictionary mapping box names to goal locations.
    2. Extract current box locations: Iterate through the current state and identify the current location of each box. Store this in a dictionary mapping box names to current locations.
    3. Calculate Manhattan distance for each box: For each box that has a goal location specified:
        a. Get the current location and the goal location of the box.
        b. Parse the location names to extract row and column numbers (assuming location names are in the format 'loc_row_column').
        c. Calculate the Manhattan distance between the current and goal locations using the formula: |row_goal - row_current| + |column_goal - column_current|.
    4. Sum up Manhattan distances: Sum the Manhattan distances calculated for all boxes. This sum is the heuristic value for the given state.
    5. Return the total Manhattan distance as the heuristic estimate. If no goal location is specified for a box, it is not considered in the heuristic calculation.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal box locations and static facts."""
        self.goals = task.goals
        self.static = task.static
        self.goal_box_locations = {}

        for goal_fact in self.goals:
            if match(goal_fact, "at", "*", "*"):
                box_name, goal_location = get_objects_from_fact(goal_fact)
                self.goal_box_locations[box_name] = goal_location

    def __call__(self, node):
        """Calculate the heuristic value for a given state."""
        state = node.state
        current_box_locations = {}

        for fact in state:
            if match(fact, "at", "*", "*"):
                box_name, current_location = get_objects_from_fact(fact)
                current_box_locations[box_name] = current_location

        total_manhattan_distance = 0
        for box_name, goal_location in self.goal_box_locations.items():
            if box_name in current_box_locations:
                current_location = current_box_locations[box_name]

                try:
                    current_loc_parts = list(map(int, re.findall(r'\d+', current_location)))
                    goal_loc_parts = list(map(int, re.findall(r'\d+', goal_location)))

                    if len(current_loc_parts) == 2 and len(goal_loc_parts) == 2:
                        row_current, col_current = current_loc_parts
                        row_goal, col_goal = goal_loc_parts
                        manhattan_distance = abs(row_goal - row_current) + abs(col_goal - col_current)
                        total_manhattan_distance += manhattan_distance
                    else:
                        # Fallback for unexpected location format, treat distance as 0 or handle differently
                        pass # or total_manhattan_distance += 0 or raise ValueError
                except:
                    # Handle cases where location names are not in expected format, treat distance as 0
                    pass # or total_manhattan_distance += 0 or raise ValueError

        return total_manhattan_distance
