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
    This heuristic estimates the minimum number of moves required to move each box to its goal location based on Manhattan distance.
    It sums the Manhattan distances of all boxes from their current locations to their respective goal locations.

    # Assumptions
    - The heuristic assumes that boxes can be moved independently of each other and ignores obstacles (other boxes and walls).
    - It considers only the moves needed to push boxes to their goal locations, not the robot's moves to position itself for pushing.
    - Location names are assumed to be in the format 'loc_row_col', where row and col are integers.

    # Heuristic Initialization
    - Extracts the goal locations for each box from the task goals.
    - Parses location names to extract row and column numbers for Manhattan distance calculation.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Extract the goal predicates from the task definition.
    3. For each goal predicate of the form '(at box ?location)':
        a. Identify the box name and its goal location.
        b. Find the current location of the box in the given state.
        c. Parse the row and column numbers from the current location name and the goal location name.
        d. Calculate the Manhattan distance between the current location and the goal location:
           |row_goal - row_current| + |col_goal - col_current|.
        e. Add the Manhattan distance to the total heuristic value.
    4. Return the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal box locations."""
        self.goals = task.goals
        self.goal_box_locations = {}

        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                box_name = parts[1]
                goal_location_name = parts[2]
                self.goal_box_locations[box_name] = goal_location_name

    def __call__(self, node):
        """Estimate the number of moves required to reach the goal state."""
        state = node.state
        heuristic_value = 0

        for box_name, goal_location_name in self.goal_box_locations.items():
            current_location_name = None
            for fact in state:
                if match(fact, "at", box_name, "*"):
                    current_location_name = get_parts(fact)[2]
                    break

            if current_location_name is None:
                # Box location not found in the state, which should not happen in valid Sokoban problems.
                # To avoid errors, assign a very high heuristic value or handle as needed.
                return float('inf')

            if current_location_name == goal_location_name:
                continue # Box already at goal location, no cost for this box.


            def parse_location(location_name):
                """Parses location name 'loc_r_c' and returns (r, c) as integers."""
                match_loc = re.match(r'loc_(\d+)_(\d+)', location_name)
                if match_loc:
                    return int(match_loc.group(1)), int(match_loc.group(2))
                return None, None

            current_row, current_col = parse_location(current_location_name)
            goal_row, goal_col = parse_location(goal_location_name)

            if current_row is not None and goal_row is not None:
                manhattan_distance = abs(goal_row - current_row) + abs(goal_col - current_col)
                heuristic_value += manhattan_distance
            else:
                # If location parsing fails, return a high heuristic value or handle as needed.
                return float('inf')

        return heuristic_value
