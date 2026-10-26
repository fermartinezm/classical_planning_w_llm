import re
from heuristics.heuristic_base import Heuristic
# No other imports needed for this implementation

# Helper function to parse PDDL facts
def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Example: "(at rover1 waypoint1)" -> ["at", "rover1", "waypoint1"]
    Returns an empty list if the fact is malformed or empty.
    """
    fact = fact.strip()
    # Check for basic structure: non-empty, starts with '(', ends with ')'
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    # Split the content within parentheses by spaces
    return fact[1:-1].split()

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Rovers domain, designed for use
    with Greedy Best-First Search.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    from the current state. It operates by summing the estimated costs for achieving
    each unsatisfied goal predicate independently. The cost estimation for each goal
    is refined by checking if key intermediate steps (like having sampled data or
    having taken an image) are already completed in the current state. The heuristic
    is designed to be computationally efficient and informative for guiding the search,
    but it is not admissible.

    # Assumptions
    - The primary goals involve communicating data back to the lander:
      `(communicated_soil_data ?w)`, `(communicated_rock_data ?w)`, or
      `(communicated_image_data ?o ?m)`. Other goal types are ignored by this heuristic.
    - The cost estimation uses fixed values representing typical action sequences
      needed for each goal type (navigation, sampling/imaging, calibration, communication).
    - Navigation cost between any two relevant locations (e.g., current location to
      sample site, sample site to communication range) is simplified to a cost of 1
      per required navigation segment.
    - Resource contention (e.g., multiple goals needing the same rover, store capacity limits)
      and precise resource allocation are not modeled; costs for unsatisfied goals are summed.
    - The existence of necessary static capabilities (e.g., a rover equipped for a task,
      a camera supporting a mode, visibility between waypoints) is implicitly assumed
      when estimating costs. For instance, checking if `(calibrated ?i ?r)` exists assumes
      such a camera *could* potentially be used for an imaging goal if needed.
    - The heuristic prioritizes computational speed over absolute accuracy, making it
      suitable for greedy search strategies.

    # Heuristic Initialization
    - The constructor (`__init__`) stores the goal predicates defined in the planning task.
    - It pre-parses these goal predicates into a more accessible format (predicate name
      and arguments tuple) to speed up processing during heuristic evaluation.
    - It does not perform extensive analysis or storage of static facts (like visibility graph,
      equipment lists) to maintain initialization and evaluation efficiency.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total estimated cost `h` to 0.
    2. Parse the current `state` (represented as a frozenset of fact strings) to efficiently
       extract key information relevant to goal achievement:
       - Identify which soil data points `?w` have been communicated (`communicated_soil` set).
       - Identify which rock data points `?w` have been communicated (`communicated_rock` set).
       - Identify which image data points `(?o, ?m)` have been communicated (`communicated_image` set).
       - Identify which soil analyses `?w` have already been performed by *any* rover (`have_soil` set).
       - Identify which rock analyses `?w` have already been performed by *any* rover (`have_rock` set).
       - Identify which images `(?o, ?m)` have already been taken by *any* rover (`have_image` set).
       - Determine if *any* camera is currently calibrated (`is_calibrated` boolean flag).
    3. Iterate through the list of pre-parsed `goals` stored during initialization.
    4. For each `goal` (consisting of `goal_pred` and `goal_args`):
       - Check if this specific goal is already satisfied by looking it up in the corresponding
         `communicated_` set based on `goal_pred` and `goal_args`.
       - If the goal is **not satisfied**:
         - **Case 1: Soil Goal `(communicated_soil_data ?w)`:**
           - If the analysis for `?w` already exists (`?w` is in `have_soil` set): Add 2 to `h`.
             (Estimate: Navigate_to_comm(1) + Communicate(1)).
           - Else (analysis needed): Add 5 to `h`.
             (Estimate: Nav_to_sample(1) + Sample(1) + Nav_to_comm(1) + Communicate(1) + Buffer/Drop(1)).
         - **Case 2: Rock Goal `(communicated_rock_data ?w)`:**
           - If the analysis for `?w` already exists (`?w` is in `have_rock` set): Add 2 to `h`.
             (Estimate: Nav_to_comm(1) + Communicate(1)).
           - Else (analysis needed): Add 5 to `h`.
             (Estimate: Nav_to_sample(1) + Sample(1) + Nav_to_comm(1) + Communicate(1) + Buffer/Drop(1)).
         - **Case 3: Image Goal `(communicated_image_data ?o ?m)`:**
           - Let `goal_key = (?o, ?m)`.
           - If the image `goal_key` already exists (`goal_key` is in `have_image` set): Add 2 to `h`.
             (Estimate: Nav_to_comm(1) + Communicate(1)).
           - Else if *any* camera is currently calibrated (`is_calibrated` is True): Add 4 to `h`.
             (Estimate: Nav_to_image(1) + Take_Image(1) + Nav_to_comm(1) + Communicate(1)). This optimistically
             assumes the calibrated camera is suitable for this specific image task.
           - Else (no image exists, and no camera is currently calibrated): Add 6 to `h`.
             (Estimate: Nav_to_calibrate(1) + Calibrate(1) + Nav_to_image(1) + Take_Image(1) + Nav_to_comm(1) + Communicate(1)).
    5. Return the final summed heuristic value `h`. This value is 0 if and only if all goals
       handled by the heuristic are satisfied in the state.
    """

    def __init__(self, task):
        """
        Initializes the heuristic. Stores and pre-parses goal conditions
        relevant to communication tasks.
        """
        self.goals = task.goals
        # Pre-parse goals into (predicate, args_tuple) for faster access
        self.parsed_goals = []
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            # Ensure the goal fact is valid and is a communication goal
            if parts:
                predicate = parts[0]
                args = tuple(parts[1:])
                if predicate == "communicated_soil_data" and len(args) == 1:
                     self.parsed_goals.append((predicate, args))
                elif predicate == "communicated_rock_data" and len(args) == 1:
                     self.parsed_goals.append((predicate, args))
                elif predicate == "communicated_image_data" and len(args) == 2:
                     self.parsed_goals.append((predicate, args))
                # Silently ignore other goal types if present

    def __call__(self, node):
        """
        Computes the heuristic value for the given state node. It estimates the
        remaining cost by summing fixed estimates for each unsatisfied communication goal,
        adjusting the estimate based on whether intermediate results (analysis, images)
        are already present in the state.
        """
        state = node.state
        heuristic_value = 0

        # --- Parse current state for relevant facts ---
        # Sets store achieved goals/facts for O(1) lookup on average.
        communicated_soil = set()
        communicated_rock = set()
        communicated_image = set() # Stores tuples (objective, mode)

        # Sets store intermediate achievements (we only care if they exist for the target)
        have_soil = set() # Stores waypoints ?w where (have_soil_analysis ?r ?w) exists
        have_rock = set() # Stores waypoints ?w where (have_rock_analysis ?r ?w) exists
        have_image = set() # Stores tuples (objective, mode) where (have_image ?r ?o ?m) exists

        is_calibrated = False # Flag: True if at least one (calibrated ?i ?r) fact exists

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip potential malformed facts

            predicate = parts[0]
            args = parts[1:]

            # Check for communicated goals
            if predicate == "communicated_soil_data" and len(args) == 1:
                communicated_soil.add(args[0]) # Add waypoint
            elif predicate == "communicated_rock_data" and len(args) == 1:
                communicated_rock.add(args[0]) # Add waypoint
            elif predicate == "communicated_image_data" and len(args) == 2:
                communicated_image.add(tuple(args)) # Add (objective, mode) tuple

            # Check for intermediate achievements
            elif predicate == "have_soil_analysis" and len(args) == 2:
                have_soil.add(args[1]) # Add waypoint ?w
            elif predicate == "have_rock_analysis" and len(args) == 2:
                have_rock.add(args[1]) # Add waypoint ?w
            elif predicate == "have_image" and len(args) == 3:
                have_image.add((args[1], args[2])) # Add tuple (objective, mode)
            elif predicate == "calibrated" and len(args) == 2:
                # If we find any calibrated camera, set the flag
                is_calibrated = True

        # --- Estimate cost for each unsatisfied goal ---
        for goal_pred, goal_args in self.parsed_goals:

            if goal_pred == "communicated_soil_data":
                waypoint = goal_args[0]
                if waypoint not in communicated_soil:
                    # Goal not satisfied. Check if analysis exists.
                    if waypoint in have_soil:
                        # Analysis exists, estimate cost to communicate
                        heuristic_value += 2 # Estimate: NavComm(1) + Comm(1)
                    else:
                        # Analysis does not exist, estimate cost for full sequence
                        heuristic_value += 5 # Estimate: NavS(1)+Sample(1)+NavC(1)+Comm(1)+Drop?(1)

            elif goal_pred == "communicated_rock_data":
                waypoint = goal_args[0]
                if waypoint not in communicated_rock:
                    # Goal not satisfied. Check if analysis exists.
                    if waypoint in have_rock:
                        # Analysis exists, estimate cost to communicate
                        heuristic_value += 2 # Estimate: NavComm(1) + Comm(1)
                    else:
                        # Analysis does not exist, estimate cost for full sequence
                        heuristic_value += 5 # Estimate: NavS(1)+Sample(1)+NavC(1)+Comm(1)+Drop?(1)

            elif goal_pred == "communicated_image_data":
                goal_key = goal_args # This is the (objective, mode) tuple
                if goal_key not in communicated_image:
                    # Goal not satisfied. Check if image exists.
                    if goal_key in have_image:
                        # Image exists, estimate cost to communicate
                        heuristic_value += 2 # Estimate: NavComm(1) + Comm(1)
                    elif is_calibrated:
                        # Image doesn't exist, but *a* camera is calibrated.
                        # Optimistically assume it can be used. Estimate cost to take image + communicate.
                        heuristic_value += 4 # Estimate: NavIm(1)+TakeIm(1)+NavComm(1)+Comm(1)
                    else:
                        # Image doesn't exist, and no camera is calibrated.
                        # Estimate cost for full sequence: calibrate + take image + communicate.
                        heuristic_value += 6 # Estimate: NavCal(1)+Cal(1)+NavIm(1)+TakeIm(1)+NavComm(1)+Comm(1)

        # The final sum estimates the total remaining actions.
        return heuristic_value
