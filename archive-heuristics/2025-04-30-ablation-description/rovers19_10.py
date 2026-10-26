from fnmatch import fnmatch

# Assuming Heuristic base class exists and is imported elsewhere
# from heuristics.heuristic_base import Heuristic

# Helper functions (copied from examples)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic: # Inherit from Heuristic if available
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of major actions required to satisfy
    each unachieved communication goal. It counts 1 for communication,
    plus additional costs for sampling/imaging and necessary preconditions
    like dropping samples or calibrating cameras. Navigation costs are ignored.

    # Assumptions
    - Each unachieved communication goal requires at least one communication action.
    - Achieving a soil/rock communication goal requires sampling the data first,
      unless the sample is already collected. Sampling might require dropping
      a previous sample if the store is full.
    - Achieving an image communication goal requires taking the image first,
      unless the image is already taken. Taking an image requires a calibrated
      camera, which might require a calibration action.
    - Navigation between waypoints is assumed to have zero cost for heuristic
      estimation purposes.
    - The necessary static conditions (equipment, visible_from, calibration_target,
      initial sample locations) exist in the problem to make the goals achievable.

    # Heuristic Initialization
    - Extracts static information about rover capabilities (equipped_for_soil_analysis,
      equipped_for_rock_analysis, equipped_for_imaging), store ownership,
      camera properties (on_board, supports, calibration_target). This information
      is used to determine which rovers/cameras are relevant and whether
      sampling/imaging is fundamentally possible for a given goal type.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is the sum of estimated costs for each goal fact
    that is not present in the current state.

    For each goal `g` in `task.goals`:
    1.  If `g` is already in the current `state`, the cost for this goal is 0.
    2.  If `g` is `(communicated_soil_data ?w)`:
        -   Add 1 to the cost (for the `communicate_soil_data` action).
        -   Check if `(have_soil_analysis ?r ?w)` is true for *any* rover `?r` in the current `state`.
        -   If *no* rover has the soil analysis for `?w`:
            -   Add 1 to the cost (for the `sample_soil` action).
            -   Check if *any* rover equipped for soil analysis has a `full` store in the current `state`. If yes, add 1 to the cost (for a `drop` action needed before sampling). This is a simplification; ideally, we'd check the store of the specific rover that will do the sampling.
    3.  If `g` is `(communicated_rock_data ?w)`:
        -   Add 1 to the cost (for the `communicate_rock_data` action).
        -   Check if `(have_rock_analysis ?r ?w)` is true for *any* rover `?r` in the current `state`.
        -   If *no* rover has the rock analysis for `?w`:
            -   Add 1 to the cost (for the `sample_rock` action).
            -   Check if *any* rover equipped for rock analysis has a `full` store in the current `state`. If yes, add 1 to the cost (for a `drop` action).
    4.  If `g` is `(communicated_image_data ?o ?m)`:
        -   Add 1 to the cost (for the `communicate_image_data` action).
        -   Check if `(have_image ?r ?o ?m)` is true for *any* rover `?r` in the current `state`.
        -   If *no* rover has the image `(?o ?m)`:
            -   Add 1 to the cost (for the `take_image` action).
            -   Check if there exists *any* imaging-equipped rover `?r_img` with a camera `?i` that `supports ?m` (static facts) such that `(calibrated ?i ?r_img)` is true in the current `state`.
            -   If *no* such suitable camera is currently calibrated, add 1 to the cost (for a `calibrate` action).

    The total heuristic value is the sum of costs calculated for each unachieved goal.
    A goal state is one where all goals are achieved, resulting in a heuristic of 0.
    For solvable states, the heuristic counts a finite number of required actions.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static_facts = task.static # Store static facts for reference

        # Extract static information into useful structures
        self.equipped_soil = {get_parts(fact)[1] for fact in self.static_facts if match(fact, "equipped_for_soil_analysis", "*")}
        self.equipped_rock = {get_parts(fact)[1] for fact in self.static_facts if match(fact, "equipped_for_rock_analysis", "*")}
        self.equipped_imaging = {get_parts(fact)[1] for fact in self.static_facts if match(fact, "equipped_for_imaging", "*")}
        # Mapping store to rover and rover to store
        self.store_of_rover = {get_parts(fact)[2]: get_parts(fact)[1] for fact in self.static_facts if match(fact, "store_of", "*", "*")} # rover -> store
        self.rover_of_store = {get_parts(fact)[1]: get_parts(fact)[2] for fact in self.static_facts if match(fact, "store_of", "*", "*")} # store -> rover
        # Mapping camera and mode to check support
        self.camera_supports_mode = {(get_parts(fact)[1], get_parts(fact)[2]) for fact in self.static_facts if match(fact, "supports", "*", "*")} # (camera, mode)
        # Mapping camera to rover to check on_board
        self.camera_on_board = {(get_parts(fact)[1], get_parts(fact)[2]) for fact in self.static_facts if match(fact, "on_board", "*", "*")} # (camera, rover)

    def __call__(self, node):
        """Estimate the minimum cost to achieve all unachieved goals."""
        state = node.state
        h = 0

        # Check if the current state is a goal state
        if self.goals <= state:
            return 0

        for goal in self.goals:
            # If the goal is already achieved, it costs 0 for this goal
            if goal in state:
                continue

            # Parse the goal fact
            parts = get_parts(goal)
            if not parts: # Skip malformed facts
                continue

            comm_type = parts[0]

            if comm_type == "communicated_soil_data":
                if len(parts) != 2: continue # Malformed goal
                waypoint = parts[1]
                # Cost for communication action
                h += 1

                # Check if soil analysis is already done by any rover
                soil_analysis_done = any(match(fact, "have_soil_analysis", "*", waypoint) for fact in state)

                if not soil_analysis_done:
                    # Need to sample
                    h += 1 # Cost for sample_soil action

                    # Check if a drop is needed before sampling.
                    # A drop is needed if *any* soil-equipped rover has a full store.
                    # This is a simplification.
                    needs_drop = False
                    for rover in self.equipped_soil:
                         store = self.store_of_rover.get(rover)
                         if store and f"(full {store})" in state:
                              needs_drop = True
                              break
                    if needs_drop:
                         h += 1 # Cost for drop action

            elif comm_type == "communicated_rock_data":
                if len(parts) != 2: continue # Malformed goal
                waypoint = parts[1]
                # Cost for communication action
                h += 1

                # Check if rock analysis is already done by any rover
                rock_analysis_done = any(match(fact, "have_rock_analysis", "*", waypoint) for fact in state)

                if not rock_analysis_done:
                    # Need to sample
                    h += 1 # Cost for sample_rock action

                    # Check if a drop is needed before sampling.
                    # A drop is needed if *any* rock-equipped rover has a full store.
                    # This is a simplification.
                    needs_drop = False
                    for rover in self.equipped_rock:
                         store = self.store_of_rover.get(rover)
                         if store and f"(full {store})" in state:
                              needs_drop = True
                              break
                    if needs_drop:
                         h += 1 # Cost for drop action

            elif comm_type == "communicated_image_data":
                if len(parts) != 3: continue # Malformed goal
                objective = parts[1]
                mode = parts[2]
                # Cost for communication action
                h += 1

                # Check if image is already taken by any rover
                image_taken = any(match(fact, "have_image", "*", objective, mode) for fact in state)

                if not image_taken:
                    # Need to take image
                    h += 1 # Cost for take_image action

                    # Check if calibration is needed.
                    # Calibration is needed if *no* suitable camera is currently calibrated.
                    # A suitable camera is one on board an imaging-equipped rover that supports the required mode.
                    needs_calibration = True # Assume needed unless proven otherwise
                    for rover in self.equipped_imaging:
                         # Find cameras on board this rover supporting this mode
                         suitable_cameras_on_rover = [
                             camera for camera, r in self.camera_on_board if r == rover and (camera, mode) in self.camera_supports_mode
                         ]
                         for camera in suitable_cameras_on_rover:
                              if f"(calibrated {camera} {rover})" in state:
                                   needs_calibration = False
                                   break # Found a calibrated camera
                         if not needs_calibration:
                              break # Found a suitable calibrated rover/camera

                    if needs_calibration:
                         h += 1 # Cost for calibrate action
            # else: # Ignore other potential goal types if any

        return h
