from heuristics.heuristic_base import Heuristic

# Helper function to extract parts from a PDDL fact string
def get_parts(fact_str):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact_str[1:-1].split()

# Helper function to parse location string like 'loc_R_C'
def parse_location(location_str):
    """Parses a location string of the format 'loc_R_C' into a tuple (R, C)."""
    try:
        parts = location_str.split('_')
        # Assuming format is always loc_R_C where R and C are integers
        row = int(parts[1])
        col = int(parts[2])
        return row, col
    except (ValueError, IndexError):
        # Handle unexpected location formats gracefully
        # Returning a large value might be appropriate for a heuristic
        # if an invalid location string somehow appears.
        # Using float('inf') as it's standard for unreachable/invalid states in search.
        return float('inf'), float('inf')


# Helper function to calculate Manhattan distance
def manhattan_distance(loc1_str, loc2_str):
    """Calculates the Manhattan distance between two locations given as 'loc_R_C' strings."""
    r1, c1 = parse_location(loc1_str)
    r2, c2 = parse_location(loc2_str)

    # If parsing failed for either location, return infinity distance
    if r1 == float('inf') or r2 == float('inf'):
        return float('inf')

    return abs(r1 - r2) + abs(c1 - c2)


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain based on Manhattan distance.

    Estimates the cost as the sum, over all boxes not at their goal, of:
    - Manhattan distance from the box's current location to its goal location.
    - Manhattan distance from the robot's current location to the box's current location.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for each box.
        """
        self.goals = task.goals

        # Store goal locations for each box.
        self.box_goals = {}
        # Find all 'at' predicates in the goals
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            # Goal facts are typically (at ?obj ?loc)
            if parts[0] == 'at' and len(parts) == 3:
                obj_name, loc_name = parts[1], parts[2]
                # Assuming any object in an (at ?obj ?loc) goal is a box.
                self.box_goals[obj_name] = loc_name

        # Static facts (like 'adjacent') are not explicitly used in this Manhattan distance heuristic,
        # but they define the grid structure implicitly used by parse_location.
        # self.static_facts = task.static

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Find robot's current location
        robot_location = None
        # Find current locations of all objects (boxes)
        current_locations = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot' and len(parts) == 2:
                robot_location = parts[1]
            elif parts[0] == 'at' and len(parts) == 3:
                 obj_name, loc_name = parts[1], parts[2]
                 current_locations[obj_name] = loc_name

        # If robot location is not found, the state is likely invalid or not reachable.
        # Return a large value to discourage exploring this state.
        if robot_location is None:
             return float('inf')

        total_heuristic = 0

        # Iterate through each box that has a goal location defined in the task
        for box_name, goal_location in self.box_goals.items():
            # Find the box's current location in the state
            current_box_location = current_locations.get(box_name)

            # If the box is not found in the current state (shouldn't happen in Sokoban)
            # or is already at its goal, it contributes 0 to the heuristic for this box.
            if current_box_location is None or current_box_location == goal_location:
                continue

            # Calculate Manhattan distance from box to goal
            dist_box_to_goal = manhattan_distance(current_box_location, goal_location)

            # Calculate Manhattan distance from robot to box
            dist_robot_to_box = manhattan_distance(robot_location, current_box_location)

            # If any distance calculation resulted in infinity (due to parsing error),
            # the state is problematic. Return infinity.
            if dist_box_to_goal == float('inf') or dist_robot_to_box == float('inf'):
                 return float('inf')

            # Add the combined distance to the total heuristic
            # This estimates the cost to get the robot to the box and then push the box.
            total_heuristic += dist_box_to_goal + dist_robot_to_box

        return total_heuristic
