from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    This heuristic estimates the number of actions required to solve a Sokoban problem by calculating the sum of Manhattan distances
    for each box from its current location to its closest goal location. It assumes that each box can be moved independently to its goal,
    ignoring potential blockages and robot movements. This provides a lower bound on the number of moves and is efficiently computable.

    # Assumptions:
    - The heuristic assumes that boxes can be moved independently to their goal locations.
    - It does not consider the robot's movements or potential blockages by other boxes or walls.
    - It only considers the Manhattan distance as a measure of distance, ignoring the complexity of navigating the Sokoban grid.

    # Heuristic Initialization
    - Extracts the goal locations for each box from the task goals.
    - No static facts are explicitly used in this simple heuristic, although the grid structure is implicitly used in Manhattan distance calculation.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the goal locations for each box. These are extracted from the `task.goals`.
    2. For the given state, identify the current location of each box.
    3. For each box, calculate the Manhattan distance between its current location and its goal location.
       To calculate Manhattan distance, location names are assumed to be in the format 'loc_row_col'.
       Extract the row and column numbers from the location names and compute the absolute difference in rows and columns.
    4. Sum up the Manhattan distances for all boxes.
    5. The sum of these distances is the heuristic estimate for the given state.
    6. If the state is a goal state (all boxes are at their goal locations), the heuristic value is 0.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal box locations."""
        self.goal_box_locations = {}
        for goal in task.goals:
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                box_name = parts[1]
                location_name = parts[2]
                self.goal_box_locations[box_name] = location_name

    def __call__(self, node):
        """Calculate the heuristic value for a given state."""
        state = node.state
        heuristic_value = 0

        # Check if it is a goal state first
        is_goal_state = True
        for goal in self.goal_box_locations.values():
            goal_fact = f'(at * {goal})'
            found_goal = False
            for fact in state:
                if match(fact, "at", "*", goal):
                    found_goal = True
                    break
            if not found_goal:
                is_goal_state = False
                break
        if is_goal_state:
            return 0

        box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                if parts[1] in self.goal_box_locations: # Only consider boxes that are in goal
                    box_name = parts[1]
                    location_name = parts[2]
                    box_locations[box_name] = location_name

        for box_name, goal_location in self.goal_box_locations.items():
            if box_name in box_locations:
                current_location = box_locations[box_name]

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
                    # Handle cases where location names are not in 'loc_r_c' format, if any.
                    # For simplicity, assume distance is high if parsing fails.
                    heuristic_value += 100 # Assign a large penalty if location parsing fails

        return heuristic_value
