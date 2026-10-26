from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # The number of parts must match the number of args for a full match.
    if len(parts) != len(args):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_location(loc_str):
    """
    Parses a location string like 'loc_row_col' into a (row, col) tuple.
    Assumes location names follow the pattern 'loc_R_C' where R and C are integers.
    """
    try:
        parts = loc_str.split('_')
        # Expecting format 'loc_R_C' -> parts = ['loc', 'R', 'C']
        if len(parts) == 3 and parts[0] == 'loc':
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        else:
            # Handle unexpected format
            print(f"Warning: Unexpected location string format '{loc_str}'.")
            return None
    except (ValueError, IndexError):
        # Handle cases where R or C are not integers
        print(f"Warning: Could not parse row/col from location string '{loc_str}'.")
        return None


def manhattan_distance(p1, p2):
    """
    Calculates the Manhattan distance between two (row, col) points.
    Returns infinity if either point is None (indicating a parsing error).
    """
    if p1 is None or p2 is None:
        return float('inf')
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    Estimates the cost as the sum of Manhattan distances from each box to its
    goal location, plus the Manhattan distance from the robot to the closest
    box that is not yet at its goal.

    This heuristic is non-admissible but aims to guide a greedy best-first
    search efficiently by prioritizing states where boxes are closer to goals
    and the robot is closer to a box that needs pushing.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for each box
        from the task's goal conditions.
        """
        # The set of facts that must hold in goal states.
        self.goals = task.goals

        # Store goal locations for each box.
        # Assuming goals are specified as (at <box-name> <location-name>).
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Check if the goal fact is of the form (at ?b ?l)
            if parts[0] == "at" and len(parts) == 3:
                box, location = parts[1], parts[2]
                self.goal_locations[box] = location
            # Other types of goal facts (e.g., robot location) are ignored
            # by this heuristic, as the primary goal is moving boxes.

        # Pre-parse goal locations for efficiency
        self.goal_locations_parsed = {
             box: parse_location(loc_str)
             for box, loc_str in self.goal_locations.items()
        }


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach a goal state from the current state.

        The heuristic value is calculated as:
        Sum of Manhattan distances for all boxes not at their goals
        + Manhattan distance from the robot to the closest box not at its goal.
        """
        state = node.state  # Current world state (frozenset of fact strings).

        # Find robot location and current box locations in the state.
        robot_location_str = None
        box_locations_str = {} # Map box name to current location string

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot" and len(parts) == 2:
                robot_location_str = parts[1]
            elif parts[0] == "at" and len(parts) == 3:
                box, location = parts[1], parts[2]
                box_locations_str[box] = location

        # Parse current locations into (row, col) tuples
        robot_location_parsed = parse_location(robot_location_str)
        box_locations_parsed = {
            box: parse_location(loc_str)
            for box, loc_str in box_locations_str.items()
        }

        total_box_distance = 0
        boxes_not_at_goal = [] # List of box names that are not at their goal locations

        # Iterate through the boxes specified in the goals to check their status
        for box, goal_loc_parsed in self.goal_locations_parsed.items():
            current_loc_parsed = box_locations_parsed.get(box)

            # Check if the box exists in the current state and its location is parseable
            if current_loc_parsed is not None and goal_loc_parsed is not None:
                 # If the box is not at its goal location
                 if current_loc_parsed != goal_loc_parsed:
                    total_box_distance += manhattan_distance(current_loc_parsed, goal_loc_parsed)
                    boxes_not_at_goal.append(box)
            # Note: If a box is in the goal list but not in the state, or its location
            # or the goal location cannot be parsed, it effectively contributes
            # infinity to the total_box_distance via manhattan_distance(None, ...).
            # This makes such states highly undesirable, which is appropriate.


        # If all boxes specified in the goals are at their goal locations, the heuristic is 0.
        if not boxes_not_at_goal:
            return 0

        # Calculate the minimum Manhattan distance from the robot to any box not at its goal.
        min_robot_box_distance = float('inf')

        # Ensure robot location was successfully parsed
        if robot_location_parsed is not None:
            for box in boxes_not_at_goal:
                box_loc_parsed = box_locations_parsed.get(box)
                # Ensure the box's current location was successfully parsed
                if box_loc_parsed is not None:
                    dist = manhattan_distance(robot_location_parsed, box_loc_parsed)
                    min_robot_box_distance = min(min_robot_box_distance, dist)
        # If robot_location_parsed is None, or if none of the locations of
        # boxes_not_at_goal could be parsed, min_robot_box_distance remains inf.

        # The heuristic value is the sum of box-goal distances plus the
        # robot-closest-box distance. If min_robot_box_distance is inf,
        # the heuristic value will also be inf (unless total_box_distance is also inf).
        heuristic_value = total_box_distance + min_robot_box_distance

        return heuristic_value
