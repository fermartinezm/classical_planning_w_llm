from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts in the fact is at least the number of arguments in the pattern
    if len(parts) < len(args):
         return False
    # Check if each part matches the corresponding argument pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to satisfy all
    goal conditions. It breaks down each goal (communicating soil, rock, or
    image data) into necessary stages (sampling/imaging, calibrating,
    communicating) and assigns a fixed cost to each stage or action,
    ignoring complex interactions like resource contention or precise
    navigation costs.

    # Assumptions
    - Navigation between any two required waypoints costs a fixed amount (implicitly 1 in the stage costs).
    - A suitable rover with the necessary equipment exists for each task.
    - Dropping a sample from a full store costs a fixed amount (implicitly 1 if needed before sampling, but ignored in the simplified cost).
    - Calibration is needed for an image goal unless a relevant camera on a capable rover is already calibrated.
    - Goals can be pursued somewhat independently.

    # Heuristic Initialization
    - Extract static information: lander location, rover capabilities, camera details (on board, supports mode, calibration target), store ownership. This information is used to determine which rovers/cameras are relevant for different tasks and check preconditions based on static facts.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is the sum of estimated costs for each unsatisfied goal.

    For each goal `g` in the task goals:
    1.  If `g` is already true in the current state, the cost for this goal is 0.
    2.  If `g` is `(communicated_soil_data ?w)`:
        - Check if `(have_soil_analysis ?r ?w)` is true for any rover `?r` in the current state.
        - If yes (sample exists): Estimated cost is 2 (navigate to comm point + communicate).
        - If no (sample needed): Estimated cost is 4 (navigate to sample + sample + navigate to comm point + communicate).
    3.  If `g` is `(communicated_rock_data ?w)`:
        - Similar logic to soil data.
        - Estimated cost is 2 if sample exists, 4 if sample needed.
    4.  If `g` is `(communicated_image_data ?o ?m)`:
        - Check if `(have_image ?r ?o ?m)` is true for any rover `?r` in the current state.
        - If yes (image exists): Estimated cost is 2 (navigate to comm point + communicate).
        - If no (image needed):
            - Check if calibration is needed: Find if there is *any* imaging-equipped rover `?r` with a camera `?i` supporting mode `?m` such that `(calibrated ?i ?r)` is true in the current state.
            - If calibration is needed (no such calibrated camera exists): Estimated cost is 6 (nav cal + calibrate + nav img + take img + nav comm + communicate).
            - If calibration is not needed (a relevant camera is already calibrated): Estimated cost is 4 (nav img + take img + nav comm + communicate).

        The total heuristic is the sum of these estimated costs for all unsatisfied goals.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract static information
        self.lander_location = None
        self.rover_capabilities = {} # rover -> set of capabilities (soil, rock, imaging)
        self.rover_cameras = {}      # rover -> set of cameras
        self.camera_modes = {}       # camera -> set of modes
        self.camera_caltarget = {}   # camera -> calibration_target objective
        self.rover_store = {}        # rover -> store

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'at_lander':
                self.lander_location = parts[2] # (at_lander ?l ?y)
            elif parts[0] == 'equipped_for_soil_analysis':
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, set()).add('soil_analysis')
            elif parts[0] == 'equipped_for_rock_analysis':
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, set()).add('rock_analysis')
            elif parts[0] == 'equipped_for_imaging':
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, set()).add('imaging')
            elif parts[0] == 'on_board':
                camera, rover = parts[1], parts[2]
                self.rover_cameras.setdefault(rover, set()).add(camera)
            elif parts[0] == 'supports':
                camera, mode = parts[1], parts[2]
                self.camera_modes.setdefault(camera, set()).add(mode)
            elif parts[0] == 'calibration_target':
                camera, target = parts[1], parts[2]
                self.camera_caltarget[camera] = target
            elif parts[0] == 'store_of':
                store, rover = parts[1], parts[2]
                self.rover_store[rover] = store

            # Pre-compute relevant camera/rover pairs for imaging goals
            # This helps quickly check if a camera supporting a mode exists on an imaging rover
            self.imaging_capable_pairs = set() # (rover, camera, mode)
            for rover, caps in self.rover_capabilities.items():
                if 'imaging' in caps:
                    for camera in self.rover_cameras.get(rover, set()):
                        for mode in self.camera_modes.get(camera, set()):
                             self.imaging_capable_pairs.add((rover, camera, mode))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0

        # Helper sets for quick lookup in the state
        state_have_soil = {fact for fact in state if match(fact, 'have_soil_analysis', '*', '*')}
        state_have_rock = {fact for fact in state if match(fact, 'have_rock_analysis', '*', '*')}
        state_have_image = {fact for fact in state if match(fact, 'have_image', '*', '*', '*')}
        state_calibrated = {fact for fact in state if match(fact, 'calibrated', '*', '*')}

        for goal in self.goals:
            if goal in state:
                continue # Goal already satisfied

            parts = get_parts(goal)
            predicate = parts[0]

            if predicate == 'communicated_soil_data':
                waypoint = parts[1]
                # Check if sample is already collected
                sample_collected = any(match(fact, 'have_soil_analysis', '*', waypoint) for fact in state_have_soil)

                if sample_collected:
                    # Need to navigate to comm point + communicate (2 actions)
                    h += 2
                else:
                    # Need to navigate to sample + sample + navigate to comm point + communicate (4 actions)
                    h += 4

            elif predicate == 'communicated_rock_data':
                waypoint = parts[1]
                # Check if sample is already collected
                sample_collected = any(match(fact, 'have_rock_analysis', '*', waypoint) for fact in state_have_rock)

                if sample_collected:
                    # Need to navigate to comm point + communicate (2 actions)
                    h += 2
                else:
                    # Need to navigate to sample + sample + navigate to comm point + communicate (4 actions)
                    h += 4

            elif predicate == 'communicated_image_data':
                objective, mode = parts[1], parts[2]
                # Check if image is already taken
                image_taken = any(match(fact, 'have_image', '*', objective, mode) for fact in state_have_image)

                if image_taken:
                    # Need to navigate to comm point + communicate (2 actions)
                    h += 2
                else:
                    # Need to navigate to imaging point + take image + navigate to comm point + communicate (4 actions)
                    base_cost = 4

                    # Check if calibration is needed
                    # Find if any imaging-capable (rover, camera, mode) pair is currently calibrated
                    calibration_needed = True
                    # Iterate through all imaging-capable (rover, camera, mode) combinations
                    for r_cap, c, m_support in self.imaging_capable_pairs:
                        # If this combination supports the mode required by the goal
                        if m_support == mode:
                            # Check if this specific camera on this specific rover is calibrated in the current state
                            if f'(calibrated {c} {r_cap})' in state_calibrated:
                                calibration_needed = False
                                break # Found a calibrated camera/rover pair for this mode

                    if calibration_needed:
                        # Need to navigate to calibration point + calibrate (2 actions)
                        h += base_cost + 2
                    else:
                        # Calibration not needed (already calibrated)
                        h += base_cost

            # Note: This heuristic sums costs for each goal independently.
            # It does not account for shared costs (e.g., navigating to a waypoint
            # that serves multiple purposes or using the same rover/store).
            # This is acceptable for a non-admissible heuristic.

        return h
