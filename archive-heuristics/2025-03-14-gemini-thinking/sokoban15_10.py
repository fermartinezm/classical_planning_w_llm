from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_objects_from_fact(fact):
    """
    Extract the objects from a PDDL fact string.
    For example, from '(at box1 loc_3_5)' it returns ['box1', 'loc_3_5'].
    """
    return fact[1:-1].split()[1:]

def get_predicate_name(fact):
    """
    Extract the predicate name from a PDDL fact string.
    For example, from '(at box1 loc_3_5)' it returns 'at'.
    """
    return fact[1:-1].split()[0]

def parse_location_name(location_name):
    """
    Parses a location name like 'loc_row_col' and returns (row, col) as integers.
    Assumes location names follow the format 'loc_r_c'.
    Returns None if parsing fails.
    """
    parts = location_name.split('_')
    if len(parts) == 3 and parts[0] == 'loc':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            return None
    return None

def manhattan_distance(loc1_name, loc2_name):
    """
    Calculates the Manhattan distance between two locations given their names
    in the format 'loc_row_col'. Returns infinity if location names are not parseable.
    """
    loc1_coords = parse_location_name(loc1_name)
    loc2_coords = parse_location_name(loc2_name)

    if loc1_coords is None or loc2_coords is None:
        return float('inf')  # Indicate invalid location names

    row1, col1 = loc1_coords
    row2, col2 = loc2_coords
    return abs(row1 - row2) + abs(col1 - col2)


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state in the Sokoban domain
    by calculating the sum of Manhattan distances for each box from its current location
    to its goal location. It assumes that each box needs to be moved to a specific goal location.

    # Assumptions:
    - The goal is defined by specifying the desired location for each box using the 'at' predicate.
    - Location names are in the format 'loc_row_col' to allow Manhattan distance calculation.
    - The heuristic does not consider obstacles, deadlocks, or the robot's path, focusing solely on box movements.

    # Heuristic Initialization
    - Extracts the goal locations for each box from the task's goal conditions.
    - Stores these goal locations in a dictionary for efficient lookup during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is computed as follows:
    1. Initialize the total heuristic value to 0.
    2. Iterate through each goal condition in the task's goal definition.
    3. For each goal condition that is an 'at' predicate (specifying box location):
        a. Extract the box name and the goal location name from the goal condition.
        b. Find the current location of the box in the current state.
        c. Calculate the Manhattan distance between the current location and the goal location.
        d. Add this Manhattan distance to the total heuristic value.
    4. Return the total heuristic value as the estimated cost to reach the goal state.
    """

    def __init__(self, task):
        """
        Initialize the sokoban heuristic by extracting goal box locations.
        """
        self.goal_box_locations = {}
        for goal_fact in task.goals:
            if get_predicate_name(goal_fact) == 'at':
                box_name, goal_location_name = get_objects_from_fact(goal_fact)
                self.goal_box_locations[box_name] = goal_location_name

    def __call__(self, node):
        """
        Calculate the heuristic value for the given state based on Manhattan distances
        of boxes to their goal locations.
        """
        state = node.state
        heuristic_value = 0

        current_box_locations = {}
        for fact in state:
            if get_predicate_name(fact) == 'at':
                box_name, current_location_name = get_objects_from_fact(fact)
                if box_name in self.goal_box_locations: # Only consider boxes that have goal locations
                    current_box_locations[box_name] = current_location_name

        for box_name, goal_location_name in self.goal_box_locations.items():
            if box_name in current_box_locations:
                current_location_name = current_box_locations[box_name]
                distance = manhattan_distance(current_location_name, goal_location_name)
                heuristic_value += distance
            else:
                # Box might not be in the state if the goal is not fully specified for all boxes.
                # In Sokoban, usually goals specify locations for all boxes present in initial state.
                # If a box is not found in current state but is in goal, consider max distance.
                heuristic_value += float('inf') # Or a very large number if you don't want infinity

        return heuristic_value
