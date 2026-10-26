from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# If not, remove this import and the inheritance.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy base class if the actual one is not available
    # This allows the code structure to remain consistent with the examples
    # even if the base class file is not provided in the execution environment.
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError("Heuristic __call__ not implemented")


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or invalid format gracefully
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def parse_location(location_name):
    """
    Parses a location string like 'loc_R_C' into a tuple (R, C).
    Assumes the format is always 'loc_row_col' where row and col are integers.
    """
    parts = location_name.split('_')
    if len(parts) == 3 and parts[0] == 'loc':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # This should not happen with valid PDDL instances following the convention
            # but good practice to handle.
            raise ValueError(f"Invalid row/column format in location name: {location_name}")
    # If the location name doesn't match the expected format, we cannot calculate Manhattan distance.
    # This might indicate a problem with the instance or domain assumption.
    # For a heuristic, we might return a large value or infinity to discourage states
    # with unparseable locations, or raise an error if we strictly assume the format.
    # Let's raise an error as it indicates a mismatch with the heuristic's assumption.
    raise ValueError(f"Unexpected location format: {location_name}")


class sokobanHeuristic(Heuristic): # Inherit from Heuristic
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the Manhattan
    distances of each box from its current location to its goal location.
    It ignores the robot's position, other boxes, and obstacles.

    # Assumptions
    - Goal conditions only specify the final locations of boxes using the `(at ?box ?location)` predicate.
    - Location names follow the format 'loc_R_C' where R and C are integers, allowing Manhattan distance calculation.
    - The heuristic assumes that moving a box one grid unit costs at least 1, which is a lower bound on the actual cost (which involves robot movement and pushing).

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic value to 0.
    2. For each box specified in the goal:
       a. Find the box's current location in the current state by looking for facts like `(at box_name current_location)`.
       b. Get the box's goal location (which was stored during the heuristic's initialization).
       c. If the box is not currently at its goal location:
          i. Parse the current location string (e.g., 'loc_4_4') into its row and column integers (e.g., 4, 4).
          ii. Parse the goal location string (e.g., 'loc_2_4') into its row and column integers (e.g., 2, 4).
          iii. Calculate the Manhattan distance between the current and goal coordinates: `abs(current_row - goal_row) + abs(current_col - goal_col)`.
          iv. Add this calculated distance to the total heuristic value.
    3. Return the final total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for each box.
        """
        # Call the base class constructor if necessary.
        # super().__init__(task) # Assuming Heuristic base doesn't need task

        # Store goal locations for each box.
        # We assume goal facts are of the form (at box_name loc_R_C)
        self.goal_locations = {}
        for goal in task.goals:
            parts = get_parts(goal)
            # Check if the fact is (at ?box ?location) and has 3 parts
            if parts and parts[0] == "at" and len(parts) == 3:
                box_name = parts[1]
                location_name = parts[2]
                self.goal_locations[box_name] = location_name
            # Note: This heuristic ignores other types of goal conditions if any exist.
            # Based on examples, goals are conjunctions of (at box loc).

        # Static facts are not used by this specific heuristic.
        # self.static_facts = task.static

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state  # Current world state (frozenset of facts).

        # Build a dictionary of current box locations from the state.
        # We only care about the boxes that are listed in the goals.
        current_locations = {}
        for fact in state:
            parts = get_parts(fact)
            # Check if the fact is (at ?obj ?location) and has 3 parts
            if parts and parts[0] == "at" and len(parts) == 3:
                obj_name = parts[1]
                location_name = parts[2]
                # If this object is one of the boxes we need to move to a goal
                if obj_name in self.goal_locations:
                    current_locations[obj_name] = location_name

        total_distance = 0  # Initialize heuristic cost.

        # Calculate the sum of Manhattan distances for each box not at its goal.
        for box, goal_location in self.goal_locations.items():
            # Get the box's current location.
            # Use .get() with a default or handle the case where a box might not be found.
            # In a valid Sokoban state, every box should have an 'at' predicate.
            current_location = current_locations.get(box)

            # If the box's current location is known and it's not the goal location
            if current_location is not None and current_location != goal_location:
                try:
                    # Parse current and goal locations into coordinates
                    r1, c1 = parse_location(current_location)
                    r2, c2 = parse_location(goal_location)

                    # Calculate Manhattan distance
                    distance = abs(r1 - r2) + abs(c1 - c2)
                    total_distance += distance
                except ValueError:
                    # If location names don't follow the expected format,
                    # we cannot compute the distance. This indicates an issue
                    # with the problem instance or the heuristic's assumptions.
                    # Returning infinity signifies this state is likely invalid
                    # or cannot be evaluated by this heuristic.
                    # For greedy search, a large finite number is also fine.
                    # Let's return infinity to clearly mark this failure case.
                    return float('inf')

        return total_distance
