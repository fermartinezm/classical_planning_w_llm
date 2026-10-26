from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at-robot loc_6_4)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class SokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to move all boxes to their target locations by considering the Manhattan distances from the robot's current position to each box and from each box's current position to its target.

    # Assumptions:
    - The robot can move to adjacent locations if they are clear.
    - Boxes can only be pushed to adjacent clear locations.
    - The goal is to have each box in a specific target location.

    # Heuristic Initialization
    - Extract the target location for each box from the goal facts.
    - Build an adjacency map from the static facts to determine valid movements.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. Extract the current position of the robot and each box.
    2. For each box, calculate the Manhattan distance from the robot's current position to the box's current position.
    3. Calculate the Manhattan distance from the box's current position to its target position.
    4. Sum these distances for all boxes to estimate the total number of actions needed.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Target locations for each box from the goal facts.
        - Adjacency information from the static facts to build the grid layout.
        """
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Static facts

        # Build adjacency map
        self.adjacent = {}
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                loc1, loc2, dir = get_parts(fact)
                if loc1 not in self.adjacent:
                    self.adjacent[loc1] = []
                self.adjacent[loc1].append(loc2)
                if loc2 not in self.adjacent:
                    self.adjacent[loc2] = []
                self.adjacent[loc2].append(loc1)

        # Extract goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at" and args[0].startswith("box"):
                box, location = args
                self.goal_locations[box] = location

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state

        # Extract robot's current location
        robot_location = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_location = get_parts(fact)[1]
                break

        if robot_location is None:
            return 0  # Robot not found, should not happen in valid state

        # Extract current locations of boxes
        box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                box, location = get_parts(fact)
                if box.startswith("box"):
                    box_locations[box] = location

        total_distance = 0

        # For each box, calculate the distance
        for box, target in self.goal_locations.items():
            current_loc = box_locations.get(box, None)
            if current_loc is None:
                continue  # Box not present, should not happen in valid state

            # Calculate Manhattan distance from robot to box
            # Extracting coordinates from location strings (e.g., loc_2_3 -> (2,3))
            def parse_loc(loc):
                x, y = loc.split('_')[1:]
                return int(x), int(y)

            robot_x, robot_y = parse_loc(robot_location)
            box_x, box_y = parse_loc(current_loc)
            target_x, target_y = parse_loc(target)

            # Distance from robot to box
            distance_robot_to_box = abs(robot_x - box_x) + abs(robot_y - box_y)

            # Distance from box to target
            distance_box_to_target = abs(box_x - target_x) + abs(box_y - target_y)

            total_distance += distance_robot_to_box + distance_box_to_target

        return total_distance
