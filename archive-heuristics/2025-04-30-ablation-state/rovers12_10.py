from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class for standalone testing if needed
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state # Need initial state for sample locations

    def __call__(self, node):
        raise NotImplementedError

# Helper functions to parse PDDL facts
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
    # Ensure we have enough parts to match against args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, args_part) for part, args_part in zip(parts, args))

# Helper function to extract objects by type from a collection of facts
def get_objects_by_type(facts):
    objects = {}
    # Common predicates that list objects and their types
    type_predicates = {
        'at': {0: 'rover', 1: 'waypoint'},
        'at_lander': {0: 'lander', 1: 'waypoint'},
        'can_traverse': {0: 'rover', 1: 'waypoint', 2: 'waypoint'},
        'equipped_for_soil_analysis': {0: 'rover'},
        'equipped_for_rock_analysis': {0: 'rover'},
        'equipped_for_imaging': {0: 'rover'},
        'empty': {0: 'store'},
        'have_rock_analysis': {0: 'rover', 1: 'waypoint'},
        'have_soil_analysis': {0: 'rover', 1: 'waypoint'},
        'full': {0: 'store'},
        'calibrated': {0: 'camera', 1: 'rover'},
        'supports': {0: 'camera', 1: 'mode'},
        'visible': {0: 'waypoint', 1: 'waypoint'},
        'have_image': {0: 'rover', 1: 'objective', 2: 'mode'},
        'communicated_soil_data': {0: 'waypoint'},
        'communicated_rock_data': {0: 'waypoint'},
        'communicated_image_data': {0: 'objective', 1: 'mode'},
        'at_soil_sample': {0: 'waypoint'},
        'at_rock_sample': {0: 'waypoint'},
        'visible_from': {0: 'objective', 1: 'waypoint'},
        'store_of': {0: 'store', 1: 'rover'},
        'calibration_target': {0: 'camera', 1: 'objective'},
        'on_board': {0: 'camera', 1: 'rover'},
    }

    for fact in facts:
        parts = get_parts(fact)
        if not parts: continue
        predicate = parts[0]
        if predicate in type_predicates:
            param_types = type_predicates[predicate]
            for i, obj_type in param_types.items():
                if i < len(parts):
                    obj_name = parts[i]
                    if obj_type not in objects:
                        objects[obj_type] = set()
                    objects[obj_type].add(obj_name)

    # Convert sets to sorted lists for consistent ordering
    return {k: sorted(list(v)) for k, v in objects.items()}


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the required number of actions to reach the goal
    by summing up the estimated costs for each unsatisfied goal predicate.
    The cost for each goal is estimated based on the necessary actions
    (sample/image, calibrate, communicate) and a simplified navigation cost
    (a fixed cost of 1 per required navigation stage, ignoring actual distances).

    # Assumptions
    - The heuristic is non-admissible and designed for greedy best-first search.
    - Samples (soil/rock) are only available at locations specified in the initial state
      using `at_soil_sample` or `at_rock_sample` predicates, and are consumed upon sampling.
    - All necessary equipment, cameras, calibration targets, and visible locations
      exist in the instance to make the goals reachable.
    - Navigation cost between any two required points (sample/cal/img/comm) is a fixed cost of 1 per leg.
    - Dropping a sample from a full store costs 1 action and is needed if an equipped rover
      with a full store is required to take a sample.

    # Heuristic Initialization
    - Parses objects (rovers, waypoints, cameras, etc.) from the task facts.
    - Extracts static facts about rover capabilities, stores, cameras, calibration targets,
      visibility between waypoints, and visibility from objectives/targets.
    - Stores initial locations of soil and rock samples.
    - Identifies lander location(s) and communication waypoints (visible from lander).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic calculates the sum of costs for each goal predicate
    that is not yet satisfied in the state:

    1.  Initialize total heuristic cost `h = 0`.
    2.  Identify unsatisfied goal predicates from `self.goals` that are not in the current `state`.
    3.  For each unsatisfied goal predicate:
        -   **If the goal is `(communicated_soil_data ?w)`:**
            -   Add 1 to `h` for the `communicate_soil_data` action.
            -   Check if any rover currently has `(have_soil_analysis ?r ?w)`.
            -   If no rover has the sample:
                -   Check if `(at_soil_sample ?w)` was true in the initial state. If yes:
                    -   Add 1 to `h` for the `sample_soil` action.
                    -   Check if any soil-equipped rover has a full store. If yes:
                        -   Add 1 to `h` for the `drop` action.
                    -   Add 2 to `h` for simplified navigation (one leg to reach waypoint `w`, one leg to reach a communication waypoint).
                -   (If sample was not initially available, this goal might be impossible; heuristic assumes solvable instance and proceeds).
            -   If some rover already has the sample:
                -   Add 1 to `h` for simplified navigation (one leg to reach a communication waypoint).
        -   **If the goal is `(communicated_rock_data ?w)`:**
            -   Similar logic as for soil data, checking for rock samples and rock-equipped rovers.
        -   **If the goal is `(communicated_image_data ?o ?m)`:**
            -   Add 1 to `h` for the `communicate_image_data` action.
            -   Check if any rover currently has `(have_image ?r ?o ?m)` (considering cameras and modes).
            -   If no rover has the image:
                -   Add 1 to `h` for the `take_image` action.
                -   Check if any suitable camera (on an imaging-equipped rover, supporting mode `m`) is currently calibrated.
                -   If no suitable camera is calibrated:
                    -   Add 1 to `h` for the `calibrate` action.
                    -   Add 3 to `h` for simplified navigation (one leg to calibration point, one leg to imaging point, one leg to communication point).
                -   If a suitable camera is calibrated:
                    -   Add 2 to `h` for simplified navigation (one leg to imaging point, one leg to communication point).
            -   If some rover already has the image:
                -   Add 1 to `h` for simplified navigation (one leg to reach a communication waypoint).
    4.  Return the total calculated cost `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        # Inherit goals, static facts, and initial state from the base task
        super().__init__(task)

        # --- Precomputation ---

        # 1. Parse objects by type
        all_facts = set(self.initial_state) | set(self.static) | set(self.goals)
        objects = get_objects_by_type(all_facts)
        self.rovers = objects.get('rover', [])
        self.waypoints = objects.get('waypoint', [])
        self.cameras = objects.get('camera', [])
        self.stores = objects.get('store', [])
        self.objectives = objects.get('objective', [])
        self.modes = objects.get('mode', [])
        self.landers = objects.get('lander', [])

        # 2. Extract static facts into useful data structures
        self.lander_locations = {get_parts(fact)[1] for fact in self.static if match(fact, "at_lander", "*", "*")}

        self.rover_info = {} # {rover_name: {'equipped_soil': bool, 'equipped_rock': bool, 'equipped_imaging': bool, 'store': store_name, 'cameras': [camera_names]}}
        for rover in self.rovers:
            self.rover_info[rover] = {
                'equipped_soil': f'(equipped_for_soil_analysis {rover})' in self.static,
                'equipped_rock': f'(equipped_for_rock_analysis {rover})' in self.static,
                'equipped_imaging': f'(equipped_for_imaging {rover})' in self.static,
                'store': None, # Will fill this next
                'cameras': [] # Will fill this next
            }
        for fact in self.static:
            if match(fact, "store_of", "*", "*"):
                store, rover = get_parts(fact)[1:3]
                if rover in self.rover_info:
                    self.rover_info[rover]['store'] = store
            elif match(fact, "on_board", "*", "*"):
                camera, rover = get_parts(fact)[1:3]
                if rover in self.rover_info:
                    self.rover_info[rover]['cameras'].append(camera)

        self.camera_info = {} # {camera_name: {'rover': rover_name, 'supported_modes': [mode_names], 'cal_target': objective_name}}
        for camera in self.cameras:
             self.camera_info[camera] = {
                 'rover': None, # Will fill this next
                 'supported_modes': [], # Will fill this next
                 'cal_target': None # Will fill this next
             }
        for fact in self.static:
            if match(fact, "on_board", "*", "*"):
                camera, rover = get_parts(fact)[1:3]
                if camera in self.camera_info:
                    self.camera_info[camera]['rover'] = rover
            elif match(fact, "supports", "*", "*"):
                camera, mode = get_parts(fact)[1:3]
                if camera in self.camera_info:
                    self.camera_info[camera]['supported_modes'].append(mode)
            elif match(fact, "calibration_target", "*", "*"):
                camera, target = get_parts(fact)[1:3]
                if camera in self.camera_info:
                    self.camera_info[camera]['cal_target'] = target

        self.visible_wps = set() # Pairs (wp1, wp2) where wp2 is visible from wp1
        for fact in self.static:
            if match(fact, "visible", "*", "*"):
                wp1, wp2 = get_parts(fact)[1:3]
                self.visible_wps.add((wp1, wp2))

        self.visible_from_obj = {} # {objective_name: [waypoint_names]}
        for fact in self.static:
            if match(fact, "visible_from", "*", "*"):
                obj, wp = get_parts(fact)[1:3]
                if obj not in self.visible_from_obj:
                    self.visible_from_obj[obj] = []
                self.visible_from_obj[obj].append(wp)

        # Identify communication waypoints (visible from any lander location)
        self.comm_wps = set()
        for lander_loc in self.lander_locations:
             for wp1, wp2 in self.visible_wps:
                 if wp2 == lander_loc:
                     self.comm_wps.add(wp1)
                 if wp1 == lander_loc: # visible is often symmetric, but check both
                     self.comm_wps.add(wp2)

        # Identify calibration waypoints (visible from calibration target for a camera)
        self.cal_wps = {} # {camera_name: [waypoint_names]}
        for camera, info in self.camera_info.items():
            cal_target = info.get('cal_target')
            if cal_target and cal_target in self.visible_from_obj:
                self.cal_wps[camera] = self.visible_from_obj[cal_target]
            else:
                 self.cal_wps[camera] = [] # No calibration target or no visible_from fact for target

        # Identify imaging waypoints (visible from objective)
        self.img_wps = self.visible_from_obj # Same structure

        # 3. Store initial sample locations
        self.initial_soil_samples = {get_parts(fact)[1] for fact in self.initial_state if match(fact, "at_soil_sample", "*")}
        self.initial_rock_samples = {get_parts(fact)[1] for fact in self.initial_state if match(fact, "at_rock_sample", "*")}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of facts)

        # Check if goal is reached
        if self.goals <= state:
            return 0

        h = 0 # Initialize heuristic cost

        # Get current state facts relevant to heuristic calculation
        current_soil_samples_held = {(get_parts(fact)[1], get_parts(fact)[2]) for fact in state if match(fact, "have_soil_analysis", "*", "*")}
        current_rock_samples_held = {(get_parts(fact)[1], get_parts(fact)[2]) for fact in state if match(fact, "have_rock_analysis", "*", "*")}
        current_images_held = {(get_parts(fact)[1], get_parts(fact)[2], get_parts(fact)[3]) for fact in state if match(fact, "have_image", "*", "*", "*")}


        # Identify unsatisfied goals
        pending_soil_wps = set()
        pending_rock_wps = set()
        pending_image_oms = set() # (objective, mode)

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            predicate = parts[0]

            if predicate == "communicated_soil_data":
                wp = parts[1]
                if f'(communicated_soil_data {wp})' not in state:
                    pending_soil_wps.add(wp)
            elif predicate == "communicated_rock_data":
                wp = parts[1]
                if f'(communicated_rock_data {wp})' not in state:
                    pending_rock_wps.add(wp)
            elif predicate == "communicated_image_data":
                obj, mode = parts[1:3]
                if f'(communicated_image_data {obj} {mode})' not in state:
                    pending_image_oms.add((obj, mode))

        # Calculate cost for each pending goal
        for wp in pending_soil_wps:
            h += 1 # communicate_soil_data action

            # Check if sample is already collected by any rover
            sample_collected = any((r, wp) in current_soil_samples_held for r in self.rovers)

            if not sample_collected:
                # Need to sample
                h += 1 # sample_soil action

                # Check if sample was initially available at this waypoint
                sample_initially_present = wp in self.initial_soil_samples

                if sample_initially_present:
                    # Check if any equipped soil rover has a full store
                    needs_drop = False
                    for rover in self.rovers:
                        if self.rover_info.get(rover, {}).get('equipped_soil', False):
                            store = self.rover_info[rover].get('store')
                            if store and f'(full {store})' in state:
                                needs_drop = True
                                break
                    if needs_drop:
                        h += 1 # drop action

                    # Simplified navigation cost: to sample point, then to comm point
                    h += 2
                # Else: sample not initially present, and not collected -> goal likely impossible via sampling path.
                # The heuristic adds action costs but not nav costs if sample wasn't initially there.
                # This assumes if sample wasn't initially present, the goal is unreachable via sampling.
                # If sample_collected is also False, this goal might be impossible.
                # For simplicity, we only add nav cost if sample was initially present.
                # If sample wasn't initially there, but goal requires it, something is wrong with the instance or domain.
                # Let's assume if sample_initially_present is False, the goal is unreachable via sampling path.
                # If sample_collected is also False, this goal might be impossible.
                # For simplicity, we only add nav cost if sample was initially present.

            else:
                # Sample is collected, just need to communicate
                # Simplified navigation cost: to comm point
                h += 1

        for wp in pending_rock_wps:
            h += 1 # communicate_rock_data action

            # Check if sample is already collected by any rover
            sample_collected = any((r, wp) in current_rock_samples_held for r in self.rovers)

            if not sample_collected:
                # Need to sample
                h += 1 # sample_rock action

                # Check if sample was initially available at this waypoint
                sample_initially_present = wp in self.initial_rock_samples

                if sample_initially_present:
                     # Check if any equipped rock rover has a full store
                    needs_drop = False
                    for rover in self.rovers:
                        if self.rover_info.get(rover, {}).get('equipped_rock', False):
                            store = self.rover_info[rover].get('store')
                            if store and f'(full {store})' in state:
                                needs_drop = True
                                break
                    if needs_drop:
                        h += 1 # drop action

                    # Simplified navigation cost: to sample point, then to comm point
                    h += 2
                # Else: sample not initially present, assume impossible via sampling path.

            else:
                # Sample is collected, just need to communicate
                # Simplified navigation cost: to comm point
                h += 1

        for (obj, mode) in pending_image_oms:
            h += 1 # communicate_image_data action

            # Check if image is already taken by any rover
            image_taken = any((r, obj, mode) in current_images_held for r in self.rovers)

            if not image_taken:
                # Need to take image
                h += 1 # take_image action

                # Check if calibration is needed
                needs_calibrate = True
                suitable_cameras_calibrated = False
                for rover in self.rovers:
                    if self.rover_info.get(rover, {}).get('equipped_imaging', False):
                        for camera in self.rover_info[rover].get('cameras', []):
                            if mode in self.camera_info.get(camera, {}).get('supported_modes', []):
                                # Found a suitable camera
                                if f'(calibrated {camera} {rover})' in state:
                                    suitable_cameras_calibrated = True
                                    break # Found a calibrated suitable camera
                        if suitable_cameras_calibrated:
                            break # Found a rover with a calibrated suitable camera

                if suitable_cameras_calibrated:
                    needs_calibrate = False

                if needs_calibrate:
                    h += 1 # calibrate action
                    # Simplified navigation cost: to cal point, then to img point, then to comm point
                    h += 3
                else:
                    # Simplified navigation cost: to img point, then to comm point
                    h += 2
            else:
                # Image is taken, just need to communicate
                # Simplified navigation cost: to comm point
                h += 1

        return h
