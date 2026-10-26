# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Define the heuristic class
# Assume Heuristic base class is available in the environment and this class will inherit from it.
# The definition below is standalone if the base class is not explicitly imported/needed for syntax.
# If inheritance syntax is required, add 'from heuristics.heuristic_base import Heuristic' and change class definition.
# Based on the example, let's assume inheritance is expected.
from heuristics.heuristic_base import Heuristic

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing up the
    estimated costs for each uncommunicated goal fact. The cost for each
    goal depends on whether the prerequisite data/image has already been
    collected/taken. It assigns fixed costs for collection/imaging and
    communication steps, plus a fixed penalty for necessary navigation,
    ignoring actual distances and resource constraints like store capacity
    or camera calibration state beyond the immediate 'have_*' facts.

    # Assumptions
    - Each uncommunicated goal requires a communication action.
    - If the required data/image is not yet collected/taken, it must be.
    - Collecting soil/rock data requires sampling.
    - Taking an image requires calibration and the take_image action.
    - Each major step (sample, calibrate, take_image, communicate) and
      each transition between major locations (sample/cal/img location
      to comm location, or initial location to sample/cal location)
      incurs a fixed cost penalty (e.g., 1 for an action, 1 for navigation).
    - The heuristic does not consider which specific rover performs the task,
      resource availability (like empty stores or calibrated cameras *before*
      starting the process), or actual travel distances. It only checks if
      the 'have_*' prerequisite fact exists for *any* rover.
    - It assumes that if a `communicated_*` goal exists for a waypoint/objective/mode,
      the necessary physical sample/objective visibility exists in the problem
      definition.

    # Heuristic Initialization
    - Stores the set of goal facts from the task. Static facts are not
      explicitly processed in this simple heuristic version.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic cost `h` to 0.
    2. Extract sets of currently achieved `have_soil_analysis`, `have_rock_analysis`,
       `have_image`, `communicated_soil_data`, `communicated_rock_data`, and
       `communicated_image_data` facts from the current state for efficient lookup.
    3. Iterate through each goal fact specified in the task's goals.
    4. For each goal fact string:
       - Parse the goal string to identify the predicate and arguments.
       - If the predicate is `communicated_soil_data` for waypoint `W`:
         - If `(communicated_soil_data W)` is not present in the state:
           - Check if `(have_soil_analysis R W)` is present for *any* rover `R` in the state.
           - If YES (sample collected): Add 2 to `h` (estimated cost for communicate + navigate).
           - If NO (sample not collected): Add 4 to `h` (estimated cost for sample + communicate + 2 navigates).
       - If the predicate is `communicated_rock_data` for waypoint `W`:
         - If `(communicated_rock_data W)` is not present in the state:
           - Check if `(have_rock_analysis R W)` is present for *any* rover `R` in the state.
           - If YES (sample collected): Add 2 to `h` (estimated cost for communicate + navigate).
           - If NO (sample not collected): Add 4 to `h` (estimated cost for sample + communicate + 2 navigates).
       - If the predicate is `communicated_image_data` for objective `O` and mode `M`:
         - If `(communicated_image_data O M)` is not present in the state:
           - Check if `(have_image R O M)` is present for *any* rover `R` in the state.
           - If YES (image taken): Add 2 to `h` (estimated cost for communicate + navigate).
           - If NO (image not taken): Add 6 to `h` (estimated cost for calibrate + take_image + communicate + 3 navigates).
    5. Return the total heuristic cost `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by storing the goal conditions.
        """
        self.goals = task.goals
        # Static facts are not used in this simple heuristic version.

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state from the current state.
        """
        state = node.state

        # Check if the goal is already reached
        if self.goals <= state:
            return 0

        h = 0

        # Extract relevant facts from the current state for quick lookup
        # Using sets of parsed fact tuples for efficient checking
        have_soil_in_state_tuples = {tuple(get_parts(fact)) for fact in state if get_parts(fact)[0] == 'have_soil_analysis'}
        have_rock_in_state_tuples = {tuple(get_parts(fact)) for fact in state if get_parts(fact)[0] == 'have_rock_analysis'}
        have_image_in_state_tuples = {tuple(get_parts(fact)) for fact in state if get_parts(fact)[0] == 'have_image'}
        communicated_soil_in_state_waypoints = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == 'communicated_soil_data'}
        communicated_rock_in_state_waypoints = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == 'communicated_rock_data'}
        communicated_image_in_state_obj_modes = {(get_parts(fact)[1], get_parts(fact)[2]) for fact in state if get_parts(fact)[0] == 'communicated_image_data'}


        # Iterate through each goal fact
        for goal_fact_str in self.goals:
            goal_parts = get_parts(goal_fact_str)
            predicate = goal_parts[0]

            if predicate == 'communicated_soil_data':
                waypoint = goal_parts[1]
                # Check if this specific soil data goal is achieved
                if waypoint not in communicated_soil_in_state_waypoints:
                    # Goal not achieved, estimate cost
                    # Check if sample is collected by any rover for this waypoint
                    sample_collected = any(fact_parts[2] == waypoint for fact_parts in have_soil_in_state_tuples)
                    if sample_collected:
                        # Need to communicate (1) + navigate to comm (1)
                        h += 2
                    else:
                        # Need to sample (1) + communicate (1) + navigate to sample (1) + navigate to comm (1)
                        h += 4

            elif predicate == 'communicated_rock_data':
                waypoint = goal_parts[1]
                 # Check if this specific rock data goal is achieved
                if waypoint not in communicated_rock_in_state_waypoints:
                    # Goal not achieved, estimate cost
                    # Check if sample is collected by any rover for this waypoint
                    sample_collected = any(fact_parts[2] == waypoint for fact_parts in have_rock_in_state_tuples)
                    if sample_collected:
                        # Need to communicate (1) + navigate to comm (1)
                        h += 2
                    else:
                        # Need to sample (1) + communicate (1) + navigate to sample (1) + navigate to comm (1)
                        h += 4

            elif predicate == 'communicated_image_data':
                objective = goal_parts[1]
                mode = goal_parts[2]
                 # Check if this specific image data goal is achieved
                if (objective, mode) not in communicated_image_in_state_obj_modes:
                    # Goal not achieved, estimate cost
                    # Check if image is taken by any rover for this objective and mode
                    image_taken = any(fact_parts[2] == objective and fact_parts[3] == mode for fact_parts in have_image_in_state_tuples)
                    if image_taken:
                        # Need to communicate (1) + navigate to comm (1)
                        h += 2
                    else:
                        # Need to calibrate (1) + take_image (1) + communicate (1) + navigate to cal (1) + navigate to img (1) + navigate to comm (1)
                        h += 6
            # Add other goal types if necessary, but the examples only show these three.

        return h
