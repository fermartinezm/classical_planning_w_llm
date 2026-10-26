from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re

def get_objects_from_fact(fact_str):
    """
    Extracts objects from a PDDL fact string.
    For example, from '(at box1 loc_3_5)' it returns ['box1', 'loc_3_5'].
    Ignores the predicate name.
    """
    fact_content = fact_str[1:-1]  # Remove parentheses
    parts = fact_content.split()
    return parts[1:]  # Return objects, skipping the predicate

def parse_location_name(location_name):
    """
    Parses a location name like 'loc_1_2' into coordinates (1, 2).
    Returns None if the location name is not in the expected format.
    """
    match = re.match(r'loc_(\d+)_(\d+)', location_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None

def manhattan_distance(loc1_name, loc2_name):
    """
    Calculates the Manhattan distance between two locations given their names
    in the format 'loc_r_c'.
    Returns infinity if location names are invalid.
    """
    loc1_coords = parse_location_name(loc1_name)
    loc2_coords = parse_location_name(loc2_name)

    if loc1_coords and loc2_coords:
        r1, c1 = loc1_coords
        r2, c2 = loc2_coords
        return abs(r1 - r2) + abs(c1 - c2)
    return float('inf')

class SokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state in the Sokoban domain.
    It calculates the sum of Manhattan distances for each box from its current location to its goal location.
    This heuristic is admissible in a simplified version where only box movements are considered, but here,
    we use it as a domain-dependent heuristic for greedy best-first search, so admissibility is not required.

    # Assumptions:
    - The goal is always defined by the 'at' predicate for boxes, specifying the target location for each box.
    - The heuristic focuses on moving boxes to their goal locations and does not explicitly consider robot movements
      or clearing paths, although these are implicitly considered through the Manhattan distance.
    - It assumes a direct path is always possible, ignoring potential deadlocks or complex path planning issues.

    # Heuristic Initialization
    - The constructor parses the goal conditions from the task to identify the goal location for each box.
    - It stores a mapping of each box to its goal location for efficient access during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:
    1. Initialize the total heuristic value to 0.
    2. Iterate through each goal condition in the task's goal definition.
    3. For each goal condition, identify the box and its target goal location.
    4. Find the current location of the box in the given state.
    5. Calculate the Manhattan distance between the box's current location and its goal location.
    6. Add this Manhattan distance to the total heuristic value.
    7. After processing all goal conditions, the total heuristic value is returned as the estimated cost to reach the goal.

    This heuristic essentially estimates the minimum number of moves needed to push each box to its target location,
    assuming each push action moves a box by one Manhattan unit towards its goal. It is a simplification and does not
    account for the complexities of Sokoban, such as path blocking, deadlocks, or the need to move the robot.
    However, it provides a reasonable estimate to guide a greedy search algorithm.
    """

    def __init__(self, task):
        """
        Initialize the Sokoban heuristic by parsing goal conditions to extract box-goal location pairs.
        """
        self.goals = task.goals
        self.box_goal_locations = {}

        for goal_fact in self.goals:
            if goal_fact.startswith('(at box'): # Assuming goal is always of the form (at boxX loc_Y_Z)
                objects = get_objects_from_fact(goal_fact)
                if len(objects) == 2:
                    box_name = objects[0]
                    goal_location_name = objects[1]
                    self.box_goal_locations[box_name] = goal_location_name

    def __call__(self, node):
        """
        Calculate the heuristic value for a given state.
        The heuristic value is the sum of Manhattan distances of each box to its goal location.
        """
        state = node.state
        heuristic_value = 0

        box_current_locations = {}
        for fact in state:
            if fact.startswith('(at box'): # Find current box locations
                objects = get_objects_from_fact(fact)
                if len(objects) == 2:
                    box_name = objects[0]
                    current_location_name = objects[1]
                    box_current_locations[box_name] = current_location_name

        for box_name, goal_location_name in self.box_goal_locations.items():
            if box_name in box_current_locations:
                current_location_name = box_current_locations[box_name]
                distance = manhattan_distance(current_location_name, goal_location_name)
                heuristic_value += distance
            else:
                # Box location not found in state, something is wrong or box is not yet placed.
                # In Sokoban, boxes are always placed initially. Handle gracefully, maybe large value?
                heuristic_value += float('inf') # Indicate high cost if box is missing (should not happen normally)

        return heuristic_value
