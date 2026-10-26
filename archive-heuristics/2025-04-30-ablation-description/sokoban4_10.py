# from heuristics.heuristic_base import Heuristic # Assuming this is provided

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty fact string or malformed facts defensively
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def parse_location(loc_str):
    """Parses a location string like 'loc_R_C' into a tuple (R, C)."""
    parts = loc_str.split('_')
    # Assuming format is always loc_R_C where R and C are integers
    if len(parts) == 3 and parts[0] == 'loc':
        try:
            return (int(parts[1]), int(parts[2]))
        except ValueError:
            # Handle cases where R or C are not integers
            # print(f"Warning: Could not parse location string '{loc_str}' coordinates.")
            return None
    else:
        # Handle cases where location format is not loc_R_C
        # print(f"Warning: Unexpected location format '{loc_str}'.")
        return None # Or some indicator of invalid location


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the Manhattan distance
    for each box from its current location to its goal location.

    # Assumptions
    - The grid is represented by locations named 'loc_R_C' where R and C are integers.
    - Movement costs are uniform (1 per step).
    - The heuristic uses Manhattan distance as an approximation of the minimum number of box movements needed.
    - The heuristic is NOT admissible as it doesn't account for obstacles (walls, other boxes),
      the robot's position, or the need to move other boxes out of the way.
      It is intended for greedy best-first search.

    # Heuristic Initialization
    - Extracts the goal locations for each box from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. Identify the current location of each box from the state.
    3. For each box specified in the goal conditions:
       a. Find the box's current location in the state.
       b. Find the box's goal location (stored during initialization).
       c. If the box is not at its goal location:
          i. Parse the box's current location and goal location into (row, col) coordinates using `parse_location`.
          ii. If parsing is successful for both locations, calculate the Manhattan distance between the box's current and goal locations. Add this distance to the total cost.
          iii. If parsing fails for either location, this location might not be on the grid or the format is unexpected. The heuristic will implicitly ignore this box's contribution if its location cannot be parsed.
    4. Return the total calculated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        super().__init__(task) # Call parent constructor to store goals and static
        self.goal_locations = {}
        # Parse goal conditions to find goal location for each box
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2:
                 obj, location = args
                 # Assuming any 'at' goal predicate refers to a box
                 self.goal_locations[obj] = location

    def __call__(self, node):
        """Estimate the minimum cost to move all boxes to their goal locations."""
        state = node.state  # Current world state.

        total_cost = 0  # Initialize action cost counter.

        # Find current locations of all objects (boxes) that are relevant (in goals)
        current_box_locations = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at" and len(args) == 2:
                 obj, location = args
                 # Only store location if the object is one of the boxes we care about (i.e., in the goals)
                 if obj in self.goal_locations:
                     current_box_locations[obj] = location

        # Calculate heuristic for each box that has a goal
        for box, goal_loc in self.goal_locations.items():
            current_loc = current_box_locations.get(box) # Get current location

            # If box is missing from state or already at goal, cost for this box is 0
            # A box missing from state is likely an error in state representation,
            # but checking current_loc is None handles the .get() default.
            if current_loc is None or current_loc == goal_loc:
                continue

            # Box is not at goal, calculate distance
            current_coords = parse_location(current_loc)
            goal_coords = parse_location(goal_loc)

            # Ensure both locations were parsed successfully
            if current_coords is not None and goal_coords is not None:
                box_distance_to_goal = abs(current_coords[0] - goal_coords[0]) + abs(current_coords[1] - goal_coords[1])
                total_cost += box_distance_to_goal
            # else: If parsing fails, this box's contribution is effectively ignored (cost 0 for this box).
            # This is acceptable for a non-admissible heuristic, assuming valid locations are the norm.


        # The heuristic is 0 if and only if all boxes are at their goal locations.
        # If total_cost is 0, it means the loop finished without adding any cost,
        # which only happens if current_loc == goal_loc for all boxes in self.goal_locations.
        # This correctly identifies goal states.

        return total_cost
