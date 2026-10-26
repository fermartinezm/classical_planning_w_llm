from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match PDDL facts with patterns
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goal
    conditions by summing up the estimated costs for each unachieved goal fact.
    It breaks down the cost for each goal type (soil, rock, image) based on
    whether the required data (sample analysis or image) has already been
    obtained by any rover, and for images, whether a suitable camera is calibrated.
    Navigation costs are simplified.

    # Assumptions
    - Each required navigation step (to sample/image location, to calibration
      target, to lander-visible location) costs 1 action.
    - Sampling, dropping, calibrating, taking image, and communicating each cost 1 action.
    - Resource constraints (which specific rover, store, or camera is used) are
      simplified; the heuristic assumes a suitable resource is available if one
      exists according to static facts.
    - If a sample is no longer at its original waypoint, it has been successfully
      sampled by some rover, and the data exists (even if the `have_X_analysis`
      fact is not explicitly in the state, which might indicate a state representation
      issue or a simplification in the heuristic's view).

    # Heuristic Initialization
    - Extracts goal conditions.
    - Extracts static facts to identify:
        - Lander location(s) and lander-visible waypoints.
        - Rovers equipped for different tasks.
        - Camera capabilities (on which rover, which modes supported, calibration target).
        - Possible (camera, rover, mode) combinations based on static facts that are equipped for imaging.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize heuristic value `h` to 0.
    2. Identify all goal facts that are not present in the current state.
    3. For each unachieved goal fact `G`:
        a. If `G` is `(communicated_soil_data ?w)`:
            - Check if `(have_soil_analysis ?r ?w)` exists in the state for *any* rover `?r`.
            - If yes: Add 2 to `h` (estimated cost for navigation to lander-visible spot + communication).
            - If no:
                - Check if `(at_soil_sample ?w)` exists in the state.
                - If yes: Add 4 to `h` (estimated cost for navigation to `?w` + sample + navigation to lander-visible spot + communication).
                - If no: Add 2 to `h` (estimated cost for navigation to lander-visible spot + communication, assuming the sample was taken by some rover and the data exists, but hasn't been communicated yet).
        b. If `G` is `(communicated_rock_data ?w)`: (Same logic as soil)
            - Check if `(have_rock_analysis ?r ?w)` exists in the state for *any* rover `?r`.
            - If yes: Add 2 to `h` (estimated cost for navigation to lander-visible spot + communication).
            - If no:
                - Check if `(at_rock_sample ?w)` exists in the state.
                - If yes: Add 4 to `h` (estimated cost for navigation to `?w` + sample + navigation to lander-visible spot + communication).
                - If no: Add 2 to `h` (estimated cost for navigation to lander-visible spot + communication, assuming the sample was taken by some rover and the data exists, but hasn't been communicated yet).
        c. If `G` is `(communicated_image_data ?o ?m)`:
            - Check if `(have_image ?r ?o ?m)` exists in the state for *any* rover `?r`.
            - If yes: Add 2 to `h` (estimated cost for navigation to lander-visible spot + communication).
            - If no:
                - Base cost for taking image and communicating: 1 (nav to image spot) + 1 (take image) + 1 (nav to lander) + 1 (communicate) = 4.
                - Check if calibration is needed for *any* suitable camera/rover: Determine if there is *any* rover `?r` and camera `?i` (based on pre-calculated `self.possible_camera_rover_modes` for mode `?m`) such that `(calibrated ?i ?r)` is true in the current state.
                - If at least one such calibrated camera exists: Calibration cost is 0. Total cost for this goal = 4.
                - If no such calibrated camera exists: Calibration is needed. Add 2 to the cost (estimated cost for navigation to calibration target + calibrate). Total cost for this goal = 4 + 2 = 6.
            - Add the calculated cost (4 or 6) to `h`.
    4. Return `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals  # Goal conditions (frozenset of strings).
        static_facts = task.static  # Static facts (frozenset of strings).

        # Pre-process static facts for efficient lookup

        # Identify lander locations and lander-visible waypoints
        self.lander_locations = {get_parts(fact)[2] for fact in static_facts if match(fact, "at_lander", "*", "*")}
        self.lander_visible_waypoints = set()
        for fact in static_facts:
            if match(fact, "visible", "*", "*"):
                wp1, wp2 = get_parts(fact)[1:]
                if wp2 in self.lander_locations:
                    self.lander_visible_waypoints.add(wp1)

        # Store camera capabilities and calibration targets
        # {camera: {'rover': rover, 'modes': {mode}, 'cal_target': target}}
        self.camera_info = {}
        for fact in static_facts:
            if match(fact, "on_board", "*", "*"):
                camera, rover = get_parts(fact)[1:]
                if camera not in self.camera_info:
                    self.camera_info[camera] = {'rover': rover, 'modes': set(), 'cal_target': None}
                self.camera_info[camera]['rover'] = rover # Should only be one rover per camera

            elif match(fact, "supports", "*", "*"):
                camera, mode = get_parts(fact)[1:]
                if camera not in self.camera_info:
                     self.camera_info[camera] = {'rover': None, 'modes': set(), 'cal_target': None}
                self.camera_info[camera]['modes'].add(mode)

            elif match(fact, "calibration_target", "*", "*"):
                camera, target = get_parts(fact)[1:]
                if camera not in self.camera_info:
                     self.camera_info[camera] = {'rover': None, 'modes': set(), 'cal_target': None}
                self.camera_info[camera]['cal_target'] = target

        # Pre-calculate possible (camera, rover, mode) tuples based on static facts
        # Filter for rovers equipped for imaging
        equipped_imaging_rovers = {get_parts(fact)[1] for fact in static_facts if match(fact, "equipped_for_imaging", "*")}

        self.possible_camera_rover_modes = set()
        for camera, info in self.camera_info.items():
            rover = info.get('rover') # Use .get for safety if on_board fact is missing
            if rover and rover in equipped_imaging_rovers:
                 for mode in info['modes']:
                     self.possible_camera_rover_modes.add((camera, rover, mode))


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state (frozenset of strings).
        h = 0

        # Helper to check for existence of a fact pattern in the state
        def state_has_fact_pattern(state, *args):
            return any(match(fact, *args) for fact in state)

        # Helper to check if a specific fact string is in the state
        def state_has_fact(state, fact_str):
             return fact_str in state

        # Iterate through all goal facts
        for goal in self.goals:
            # If the goal is already achieved, continue
            if state_has_fact(state, goal):
                continue

            # Parse the goal fact
            parts = get_parts(goal)
            predicate = parts[0]

            if predicate == "communicated_soil_data":
                waypoint = parts[1]
                # Check if soil analysis is already done by any rover
                if state_has_fact_pattern(state, "have_soil_analysis", "*", waypoint):
                    # Need to communicate: nav to lander-visible + communicate
                    h += 2
                else:
                    # Need to sample and communicate
                    if state_has_fact(state, f"(at_soil_sample {waypoint})"):
                        # Need nav to sample spot + sample + nav to lander + communicate
                        h += 4
                    else:
                        # Sample was taken, but not communicated. Need nav to lander + communicate.
                        h += 2

            elif predicate == "communicated_rock_data":
                waypoint = parts[1]
                # Check if rock analysis is already done by any rover
                if state_has_fact_pattern(state, "have_rock_analysis", "*", waypoint):
                    # Need to communicate: nav to lander-visible + communicate
                    h += 2
                else:
                    # Need to sample and communicate
                    if state_has_fact(state, f"(at_rock_sample {waypoint})"):
                        # Need nav to sample spot + sample + nav to lander + communicate
                        h += 4
                    else:
                        # Sample was taken, but not communicated. Need nav to lander + communicate.
                        h += 2

            elif predicate == "communicated_image_data":
                objective, mode = parts[1:]
                # Check if image is already taken by any rover
                if state_has_fact_pattern(state, "have_image", "*", objective, mode):
                    # Need to communicate: nav to lander-visible + communicate
                    h += 2
                else:
                    # Need to take image and communicate
                    # Base cost: nav to image spot + take image + nav to lander + communicate
                    base_cost = 4

                    # Check if calibration is needed for *any* suitable camera/rover
                    calibration_needed = True
                    # Iterate through possible camera/rover pairs that can take this image
                    for camera, rover, supported_mode in self.possible_camera_rover_modes:
                        if supported_mode == mode:
                            # Check if this specific camera on this specific rover is calibrated
                            if state_has_fact(state, f"(calibrated {camera} {rover})"):
                                calibration_needed = False
                                break # Found a calibrated camera, calibration is not needed for this goal

                    if calibration_needed:
                        # Add cost for calibration: nav to cal spot + calibrate
                        h += base_cost + 2
                    else:
                        # No calibration needed (a suitable camera is already calibrated)
                        h += base_cost

        return h
