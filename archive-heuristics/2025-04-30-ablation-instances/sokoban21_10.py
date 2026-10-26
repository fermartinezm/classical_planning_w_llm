import math
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Assumes fact is a string like '(predicate arg1 arg2)'
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         # Handle unexpected format if necessary, though typically planner output is consistent
         raise ValueError(f"Unexpected fact format: {fact}")
    return fact[1:-1].split()

# Helper function to parse Sokoban location strings like 'loc_R_C'
def get_coords(location_string):
    """Parses a location string like 'loc_R_C' into a tuple (R, C)."""
    parts = location_string.split('_')
    if len(parts) == 3 and parts[0] == 'loc':
        try:
            # Assuming R and C are integer coordinates
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # Handle cases where R or C are not integers if necessary
            raise ValueError(f"Invalid coordinate format in location string: {location_string}")
    # Handle unexpected format
    raise ValueError(f"Unexpected location format: {location_string}")

# Helper function to calculate Manhattan distance between two location strings
def manhattan_distance(loc1_str, loc2_str, location_coords):
    """Calculates Manhattan distance between two location strings using pre-parsed coordinates."""
    coords1 = location_coords[loc1_str]
    coords2 = location_coords[loc2_str]
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing two components:
    1. The sum of Manhattan distances for each box that is not yet at its goal location
       to its respective goal location.
    2. The minimum Manhattan distance from the robot's current location to any box
       that is not yet at its goal location.

    This heuristic is non-admissible but aims to guide a greedy best-first search
    by prioritizing states where boxes are closer to their goals and the robot
    is closer to a box that needs to be moved.

    # Assumptions
    - Locations follow the 'loc_R_C' format where R and C are integer coordinates.
    - Manhattan distance on these coordinates is a reasonable approximation of
      grid distance in the Sokoban layout.
    - The primary cost drivers are moving boxes towards goals and positioning the robot.

    # Heuristic Initialization
    - Extracts the goal locations for each box from the task's goal conditions.
    - Collects all unique location strings mentioned in the goals, static facts
      (adjacent predicates), and the initial state.
    - Parses the coordinates (R, C) for each unique location string and stores
      them in a dictionary (`location_coords`) for quick lookup.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot.
    2. Identify the current location of each box relevant to the goal (i.e., boxes
       mentioned in the goal conditions).
    3. Initialize the total heuristic value `h` to 0.
    4. Iterate through each box and its required goal location (as determined during initialization):
       - Find the box's current location in the state.
       - If the box's current location is different from its goal location:
         - Add the Manhattan distance between the box's current location and its
           goal location to `h`.
         - Keep track of the current location of this "misplaced" box.
    5. After checking all goal-relevant boxes, if there are any misplaced boxes:
       - Calculate the Manhattan distance from the robot's current location to
         the current location of *each* misplaced box.
       - Find the minimum of these robot-to-misplaced-box distances.
       - Add this minimum distance to `h`. This component encourages the robot
         to move towards a box that needs attention.
    6. Return the final value of `h`. If no boxes are misplaced, `h` will be 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and location coordinates.

        @param task: The planning task object containing goals, static facts, and initial state.
        """
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state # Needed to collect all possible locations

        # Store goal locations for each box {box_name: goal_location_string}
        self.box_goals = {}
        # Collect all unique location strings from the problem definition
        all_locations = set()

        # Parse goals to find box goal locations and collect locations
        for goal in self.goals:
            parts = get_parts(goal)
            # Assuming goal facts are primarily (at box location)
            if parts[0] == "at" and len(parts) == 3:
                box, location = parts[1], parts[2]
                self.box_goals[box] = location
                all_locations.add(location)
            # Add other goal predicates if necessary, though 'at' is standard for Sokoban goals

        # Parse static facts (adjacent) to collect locations
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "adjacent" and len(parts) == 4:
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                all_locations.add(loc1)
                all_locations.add(loc2)
            # Add other static predicates if they contain locations, e.g., wall locations if explicit

        # Parse initial state facts (at-robot, at box) to collect locations
        for fact in self.initial_state:
             parts = get_parts(fact)
             if parts[0] == "at-robot" and len(parts) == 2:
                 location = parts[1]
                 all_locations.add(location)
             elif parts[0] == "at" and len(parts) == 3: # Assuming (at box location)
                 box, location = parts[1], parts[2]
                 all_locations.add(location)
             # Add other initial state predicates if they contain locations (e.g., clear)
             elif parts[0] == "clear" and len(parts) == 2:
                 location = parts[1]
                 all_locations.add(location)


        # Parse coordinates for all collected unique locations
        self.location_coords = {}
        for loc_str in all_locations:
            try:
                self.location_coords[loc_str] = get_coords(loc_str)
            except ValueError as e:
                 # Handle cases where a location string doesn't match 'loc_R_C' format
                 # This might indicate an issue with the problem file or domain definition
                 print(f"Warning: Could not parse coordinates for location '{loc_str}'. {e}")
                 # Depending on requirements, you might skip this location or raise an error.
                 # Skipping might lead to errors later if this location is encountered.
                 # Raising an error is safer if all locations must follow the format.
                 # For this heuristic, if a location is unparseable, we cannot compute distance.
                 # Let's assume all relevant locations *can* be parsed for now.
                 raise e # Re-raise the error


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach the goal.

        @param node: The search node containing the current state.
        @return: The estimated heuristic value (integer).
        """
        state = node.state  # Current world state as a frozenset of fact strings.

        robot_loc = None
        box_locs = {} # {box_name: location_string}

        # Extract robot and box locations from the current state
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot" and len(parts) == 2:
                robot_loc = parts[1]
            elif parts[0] == "at" and len(parts) == 3:
                box, location = parts[1], parts[2]
                box_locs[box] = location
            # We only need robot and box locations for this heuristic

        # Check if robot_loc was found (should always be the case in a valid state)
        if robot_loc is None:
             # This indicates an invalid state representation, heuristic cannot be computed.
             # Return infinity or a very large value to prune this state.
             return float('inf')

        total_h = 0
        misplaced_box_locations = [] # List of current locations of boxes not at their goal

        # Calculate box-goal distances for misplaced boxes
        for box_name, goal_loc in self.box_goals.items():
            current_loc = box_locs.get(box_name) # Get current location of the box

            # If the box is not found in the state or is not at its goal
            # A box not found might indicate an invalid state or a box that was never in the initial state
            # We assume valid states where all goal boxes are present.
            if current_loc is None:
                 # This box required by the goal is missing from the state. Invalid state?
                 # Return infinity.
                 return float('inf')

            if current_loc != goal_loc:
                # Add box-goal Manhattan distance
                total_h += manhattan_distance(current_loc, goal_loc, self.location_coords)
                misplaced_box_locations.append(current_loc)

        # Calculate robot's contribution: minimum distance to a misplaced box
        if misplaced_box_locations:
            min_robot_dist = float('inf')

            # Find minimum distance from robot to any misplaced box
            for box_loc in misplaced_box_locations:
                 min_robot_dist = min(min_robot_dist, manhattan_distance(robot_loc, box_loc, self.location_coords))

            # Add the minimum robot distance to the total heuristic
            total_h += min_robot_dist

        # The heuristic is 0 if and only if there are no misplaced boxes,
        # which means all goal conditions (at box goal_loc) are met.
        return total_h

