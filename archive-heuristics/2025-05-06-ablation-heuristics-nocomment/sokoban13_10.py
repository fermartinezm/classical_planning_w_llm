from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Removes parentheses from a fact string and splits it into parts."""
    return fact[1:-1].split()

def parse_location(location_str):
    """Parses a location string like 'loc_R_C' into a (row, col) tuple."""
    parts = location_str.split('_')
    # Assuming location names are always in the format loc_row_col
    # Add error handling or validation if necessary, but for typical PDDL
    # this format is standard for grid-based domains.
    try:
        row = int(parts[1])
        col = int(parts[2])
        return (row, col)
    except (ValueError, IndexError):
        # Handle unexpected location format if needed, e.g., return a default or raise error
        # For this problem, assuming valid format based on examples.
        print(f"Warning: Unexpected location format: {location_str}")
        return (0, 0) # Or handle appropriately

def manhattan_distance(loc1_str, loc2_str):
    """Calculates the Manhattan distance between two locations."""
    r1, c1 = parse_location(loc1_str)
    r2, c2 = parse_location(loc2_str)
    return abs(r1 - r2) + abs(c1 - c2)

class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
    This heuristic estimates the cost to reach the goal state by summing the
    Manhattan distances between the current location of each box and its
    designated goal location. It ignores the robot's position and the
    complexities of pushing (e.g., needing space behind the box, robot
    movement costs, obstacles). It is a relaxation that considers only the
    minimum number of grid steps each box needs to move independently.
    This heuristic is non-admissible but efficiently computable and aims to
    guide a greedy best-first search.

    Assumptions:
    - Location names follow the format 'loc_R_C' where R and C are integers
      representing row and column.
    - Each box has a unique, fixed goal location specified in the task goals.
    - The heuristic is non-admissible and designed for greedy best-first search.
    - All boxes specified in the goal are present in the initial state and
      subsequent states with an 'at' predicate.

    Heuristic Initialization:
    In the constructor (__init__), the heuristic parses the task's goal facts
    to create a mapping from each box object (string name) to its target
    goal location string. This mapping is stored in `self.goal_locations`.
    Static facts (like 'adjacent' or 'clear' in the initial state) are not
    used in this specific heuristic calculation.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Get the current state (a frozenset of fact strings) from the input node.
    2.  Initialize a dictionary `current_box_locations` to store the current
        location of each box relevant to the goals.
    3.  Iterate through each fact string in the current state.
    4.  Parse the fact string using `get_parts`.
    5.  If the parsed fact starts with 'at' and has three parts (predicate, object, location),
        check if the object name (second part) is a box that is present in the
        `self.goal_locations` mapping (meaning it's a box we need to move to a goal).
    6.  If it is a relevant box, store its current location string (third part)
        in the `current_box_locations` dictionary, using the box name as the key.
        Ignore the robot's location ('at-robot') and 'clear' facts.
    7.  Initialize a variable `total_distance` to 0.
    8.  Iterate through the `self.goal_locations` dictionary. For each box name
        and its corresponding goal location string:
    9.  Retrieve the box's current location string from the `current_box_locations`
        dictionary. (Based on assumptions, the box should always be found here).
    10. Calculate the Manhattan distance between the current location string and
        the goal location string using the `manhattan_distance` helper function.
    11. Add this calculated distance to `total_distance`.
    12. After iterating through all relevant boxes, return the final `total_distance`
        as the heuristic value for the current state.
    """
    def __init__(self, task):
        # The task object contains initial_state, goals, operators, static facts etc.
        self.goals = task.goals

        # Extract goal locations for each box
        self.goal_locations = {}
        for goal_fact in self.goals:
            # Goal facts are typically like '(at box1 loc_2_4)'
            parts = get_parts(goal_fact)
            # Check if the fact is an 'at' predicate with two arguments (object and location)
            if parts[0] == 'at' and len(parts) == 3:
                box_name = parts[1]
                location_name = parts[2]
                # Assuming objects in goal 'at' predicates are always boxes
                self.goal_locations[box_name] = location_name

    def __call__(self, node):
        state = node.state

        # Find current locations of all boxes that are in the goals
        current_box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            # Check if the fact is an 'at' predicate with two arguments
            if parts[0] == 'at' and len(parts) == 3:
                obj_name = parts[1]
                location_name = parts[2]
                # Check if this object is one of the boxes we need to move to a goal
                if obj_name in self.goal_locations:
                    current_box_locations[obj_name] = location_name
            # We ignore 'at-robot' and 'clear' facts for this heuristic

        total_distance = 0
        # Calculate sum of Manhattan distances for each box to its goal
        # We iterate through goal_locations to ensure we cover all boxes required by the goal
        for box, goal_location in self.goal_locations.items():
            # Ensure the box's current location is found in the state.
            # In a valid Sokoban state, all boxes from the initial state (and thus goals)
            # should have an 'at' predicate.
            if box in current_box_locations:
                current_location = current_box_locations[box]
                total_distance += manhattan_distance(current_location, goal_location)
            else:
                 # This case should ideally not happen in a well-formed Sokoban state,
                 # but as a fallback, we could assume the box is missing or at a default loc.
                 # Given the problem context, it's safer to assume the box is always located.
                 # If it were missing, it might imply it's not reachable, but this heuristic
                 # doesn't handle unreachability. We proceed assuming the box is located.
                 pass # Or handle error/default distance if necessary

        # The total_distance is 0 if and only if all boxes are at their goal locations.
        return total_distance
