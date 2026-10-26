from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_2_3)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_location_name(location_name):
    """
    Parses a location name in the format 'loc_r_c' and returns the row and column as integers.
    Returns None if the format is incorrect.
    """
    match_obj = re.match(r'loc_(\d+)_(\d+)', location_name)
    if match_obj:
        return int(match_obj.group(1)), int(match_obj.group(2))
    return None

def manhattan_distance(loc1_name, loc2_name):
    """
    Calculates the Manhattan distance between two locations given their names in 'loc_r_c' format.
    Returns float('inf') if location names are not in the correct format.
    """
    loc1_coords = parse_location_name(loc1_name)
    loc2_coords = parse_location_name(loc2_name)
    if loc1_coords and loc2_coords:
        r1, c1 = loc1_coords
        r2, c2 = loc2_coords
        return abs(r1 - r2) + abs(c1 - c2)
    return float('inf')

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state in the Sokoban domain.
    It calculates the sum of Manhattan distances for each box from its current location to its goal location.
    This heuristic is based on the intuition that minimizing the total distance boxes need to be moved is a good indicator
    of the number of actions required.

    # Assumptions:
    - The location names are in the format 'loc_r_c', where r and c are row and column indices.
    - The goal is defined by specifying the desired location for each box using 'at' predicates.
    - Moving boxes closer to their goal locations is generally beneficial.

    # Heuristic Initialization
    - Extracts the goal locations for each box from the task goals.
    - No static facts are explicitly used in this heuristic, although the grid structure implied by 'adjacent' predicates
      is implicitly assumed by using Manhattan distance.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Iterate through each goal condition in the task's goal description.
    3. For each goal condition that is an 'at' predicate (specifying a box location):
       a. Extract the box name and the goal location name from the goal condition.
       b. Find the current location of the box in the current state.
       c. Calculate the Manhattan distance between the current location and the goal location using their names.
       d. Add the calculated Manhattan distance to the total heuristic value.
    4. Return the total heuristic value as the estimated cost to reach the goal state.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal box locations."""
        self.goal_box_locations = {}
        for goal in task.goals:
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                box_name = parts[1]
                goal_location = parts[2]
                self.goal_box_locations[box_name] = goal_location

    def __call__(self, node):
        """
        Estimate the cost to reach the goal state from the given state.
        The cost is the sum of Manhattan distances of each box to its goal location.
        """
        state = node.state
        heuristic_value = 0
        current_box_locations = {}

        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                box_name = parts[1]
                location = parts[2]
                current_box_locations[box_name] = location

        for box_name, goal_location in self.goal_box_locations.items():
            if box_name in current_box_locations:
                current_location = current_box_locations[box_name]
                distance = manhattan_distance(current_location, goal_location)
                if distance == float('inf'):
                    # Handle cases where location parsing fails, though it should not happen given the format.
                    # A more robust heuristic might be needed for instances with different location naming.
                    return float('inf') # Or a very large number to penalize such states.
                heuristic_value += distance
            else:
                # Box location not found in the state, which might indicate an invalid state or problem definition.
                # In Sokoban, boxes should always have a location.
                return float('inf') # Indicate this is not a valid state or something is wrong.

        return heuristic_value
