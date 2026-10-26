from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle cases like "(at rover1 waypoint1)" or "(communicated_image_data objective1 high_res)"
    # Remove outer parentheses and split by space.
    return fact[1:-1].split()

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all uncommunicated goals
    (soil data, rock data, images). It counts abstract steps needed for each goal, including
    sampling/imaging, communication, and simplified navigation/preparation steps.

    # Assumptions
    - Each unachieved goal requires a sequence of steps: obtain the data/image, then communicate it.
    - Obtaining data/image may require navigation, calibration (for images), and managing the store (for samples).
    - Communication requires navigation to a waypoint visible from the lander.
    - Navigation cost is simplified to 1 if the required agent is not already at a suitable location for the next step.
    - Store management (dropping) is simplified to 1 action if a store is full and a sample is needed.
    - Calibration is simplified to 1 action if an image is needed and the camera is not calibrated.
    - The heuristic assumes the problem is solvable by checking initial conditions and capabilities.

    # Heuristic Initialization
    - Extracts static information such as rover capabilities, store ownership, camera properties,
      waypoint visibility, objective visibility, calibration targets, and initial sample locations.
    - Identifies the lander location and potential communication waypoints.
    - Stores the set of goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic iterates through each goal condition that is not yet satisfied in the current state.
    For each unachieved goal, it estimates the remaining cost based on the type of goal:

    1.  **For `(communicated_soil_data ?w)`:**
        *   If the goal is already true, cost is 0 for this goal.
        *   If the goal is false:
            *   Check if the goal is impossible (no soil sample at `?w` initially, no equipped soil rover, or no communication waypoints). If impossible, return infinity.
            *   Add 1 to cost (for the `sample_soil` action).
            *   Check if any soil-equipped rover already has `(have_soil_analysis ?r ?w)`. If yes, this sample step is already done, subtract the previously added 1.
            *   If the sample is still needed (no rover has it):
                *   Check if any soil-equipped rover is currently at waypoint `?w`. If not, add 1 (for navigation to `?w`).
                *   Check if any soil-equipped rover has an empty store. If not, add 1 (for the `drop` action).
            *   Add 1 to cost (for the `communicate_soil_data` action).
            *   Check if any soil-equipped rover (or any rover that could potentially get the sample) is currently at a waypoint visible from the lander. If not, add 1 (for navigation to a communication waypoint).

    2.  **For `(communicated_rock_data ?w)`:**
        *   Similar logic to `communicated_soil_data`, but checking for rock samples, rock equipment, and `have_rock_analysis`.

    3.  **For `(communicated_image_data ?o ?m)`:**
        *   If the goal is already true, cost is 0 for this goal.
        *   If the goal is false:
            *   Check if the goal is impossible (no imaging rover, no camera supporting mode `?m`, objective `?o` not visible from any waypoint, or no communication waypoints). If impossible, return infinity.
            *   Add 1 to cost (for the `take_image` action).
            *   Check if any suitable rover/camera combination already has `(have_image ?r ?o ?m)`. If yes, this image step is already done, subtract the previously added 1.
            *   If the image is still needed (no rover has it):
                *   Find rovers equipped for imaging with cameras supporting mode `?m`.
                *   Check if any suitable camera on a suitable rover is calibrated. If not, add 1 (for the `calibrate` action).
                *   If calibration is needed: Check if any suitable rover is at a waypoint visible from the camera's calibration target objective. If not, add 1 (for navigation to a calibration waypoint).
                *   Check if any suitable rover is at a waypoint from which objective `?o` is visible. If not, add 1 (for navigation to an imaging waypoint).
            *   Add 1 to cost (for the `communicate_image_data` action).
            *   Check if any suitable rover (or any rover that could potentially get the image) is currently at a waypoint visible from the lander. If not, add 1 (for navigation to a communication waypoint).

    The total heuristic value is the sum of the estimated costs for all unachieved goals.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals

        # Extract static information
        self.lander_location = None
        self.rover_capabilities = {} # {rover: {cap1, cap2, ...}}
        self.rover_stores = {} # {rover: store}
        self.camera_modes = {} # {camera: {mode1, mode2, ...}}
        self.visible_waypoints = set() # {(w1, w2), ...}
        self.visible_from_objective = {} # {objective: {wp1, wp2, ...}}
        self.calibration_targets = {} # {camera: objective}
        self.rover_cameras = {} # {rover: {cam1, cam2, ...}}
        self.initial_soil_samples = set() # {wp1, wp2, ...}
        self.initial_rock_samples = set() # {wp1, wp2, ...}

        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts

            predicate = parts[0]
            if predicate == 'at_lander':
                # Assuming one lander
                if len(parts) > 2: self.lander_location = parts[2]
            elif predicate == 'equipped_for_soil_analysis':
                if len(parts) > 1:
                    rover = parts[1]
                    self.rover_capabilities.setdefault(rover, set()).add('soil')
            elif predicate == 'equipped_for_rock_analysis':
                if len(parts) > 1:
                    rover = parts[1]
                    self.rover_capabilities.setdefault(rover, set()).add('rock')
            elif predicate == 'equipped_for_imaging':
                if len(parts) > 1:
                    rover = parts[1]
                    self.rover_capabilities.setdefault(rover, set()).add('imaging')
            elif predicate == 'store_of':
                if len(parts) > 2:
                    store, rover = parts[1], parts[2]
                    self.rover_stores[rover] = store
            elif predicate == 'supports':
                if len(parts) > 2:
                    camera, mode = parts[1], parts[2]
                    self.camera_modes.setdefault(camera, set()).add(mode)
            elif predicate == 'visible':
                if len(parts) > 2:
                    w1, w2 = parts[1], parts[2]
                    self.visible_waypoints.add((w1, w2))
            elif predicate == 'visible_from':
                if len(parts) > 2:
                    objective, waypoint = parts[1], parts[2]
                    self.visible_from_objective.setdefault(objective, set()).add(waypoint)
            elif predicate == 'calibration_target':
                if len(parts) > 2:
                    camera, objective = parts[1], parts[2]
                    self.calibration_targets[camera] = objective
            elif predicate == 'on_board':
                if len(parts) > 2:
                    camera, rover = parts[1], parts[2]
                    self.rover_cameras.setdefault(rover, set()).add(camera)

        # Extract initial sample locations from initial state
        for fact in task.initial_state:
             parts = get_parts(fact)
             if not parts: continue
             predicate = parts[0]
             if predicate == 'at_soil_sample':
                 if len(parts) > 1: self.initial_soil_samples.add(parts[1])
             elif predicate == 'at_rock_sample':
                 if len(parts) > 1: self.initial_rock_samples.add(parts[1])

        # Determine communication waypoints (visible from lander)
        self.communication_waypoints = set()
        if self.lander_location:
             # A waypoint x is a communication waypoint if (visible x lander_location) is true
             self.communication_waypoints = {w1 for w1, w2 in self.visible_waypoints if w2 == self.lander_location}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Helper to check if a specific fact exists in the state
        def state_has(predicate, *args):
            fact_str = "(" + predicate + (" " + " ".join(args) if args else "") + ")"
            return fact_str in state

        # Iterate through goals
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts: continue # Should not happen for valid goals

            predicate = parts[0]

            if predicate == 'communicated_soil_data':
                waypoint_w = parts[1]

                # Check impossibility first
                can_sample_at_w_initially = waypoint_w in self.initial_soil_samples
                equipped_soil_rovers = {r for r, caps in self.rover_capabilities.items() if 'soil' in caps}
                if not can_sample_at_w_initially or not equipped_soil_rovers or not self.communication_waypoints:
                     return float('inf') # Impossible goal

                # Check if sample exists in state for any equipped rover
                has_sample_in_state = any(state_has('have_soil_analysis', r, waypoint_w) for r in equipped_soil_rovers)

                if not has_sample_in_state:
                    total_cost += 1 # Need sample action

                    # Check if any equipped rover is at w
                    any_equipped_at_w = any(state_has('at', r, waypoint_w) for r in equipped_soil_rovers)
                    if not any_equipped_at_w:
                        total_cost += 1 # Need navigation to w

                    # Check if any equipped rover has an empty store
                    # Need to check if *any* equipped rover has an empty store, as they might share tasks or stores.
                    # The store_of predicate links store to rover, so we check if the store belonging to an equipped rover is empty.
                    any_equipped_store_empty = any(state_has('empty', self.rover_stores[r]) for r in equipped_soil_rovers if r in self.rover_stores)
                    if not any_equipped_store_empty:
                        total_cost += 1 # Need drop action

                total_cost += 1 # Need communication action

                # Check if any equipped rover (that could potentially have the sample) is at a comms point
                any_equipped_at_comms = any(state_has('at', r, comm_wp)
                                            for r in equipped_soil_rovers
                                            for comm_wp in self.communication_waypoints)
                if not any_equipped_at_comms:
                     total_cost += 1 # Need navigation to comms point


            elif predicate == 'communicated_rock_data':
                waypoint_w = parts[1]

                # Check impossibility first
                can_sample_at_w_initially = waypoint_w in self.initial_rock_samples
                equipped_rock_rovers = {r for r, caps in self.rover_capabilities.items() if 'rock' in caps}
                if not can_sample_at_w_initially or not equipped_rock_rovers or not self.communication_waypoints:
                     return float('inf') # Impossible goal

                # Check if sample exists in state for any equipped rover
                has_sample_in_state = any(state_has('have_rock_analysis', r, waypoint_w) for r in equipped_rock_rovers)

                if not has_sample_in_state:
                    total_cost += 1 # Need sample action

                    # Check if any equipped rover is at w
                    any_equipped_at_w = any(state_has('at', r, waypoint_w) for r in equipped_rock_rovers)
                    if not any_equipped_at_w:
                        total_cost += 1 # Need navigation to w

                    # Check if any equipped rover has an empty store
                    any_equipped_store_empty = any(state_has('empty', self.rover_stores[r]) for r in equipped_rock_rovers if r in self.rover_stores)
                    if not any_equipped_store_empty:
                        total_cost += 1 # Need drop action

                total_cost += 1 # Need communication action

                # Check if any equipped rover (that could potentially have the sample) is at a comms point
                any_equipped_at_comms = any(state_has('at', r, comm_wp)
                                            for r in equipped_rock_rovers
                                            for comm_wp in self.communication_waypoints)
                if not any_equipped_at_comms:
                     total_cost += 1 # Need navigation to comms point


            elif predicate == 'communicated_image_data':
                objective_o = parts[1]
                mode_m = parts[2]

                # Find suitable rovers and cameras first for impossibility check and subsequent checks
                suitable_rovers_cameras = [] # List of (rover, camera) tuples
                for rover, caps in self.rover_capabilities.items():
                    if 'imaging' in caps and rover in self.rover_cameras:
                        for camera in self.rover_cameras[rover]:
                            if camera in self.camera_modes and mode_m in self.camera_modes[camera]:
                                 suitable_rovers_cameras.append((rover, camera))

                objective_visible_from_anywhere = objective_o in self.visible_from_objective and len(self.visible_from_objective[objective_o]) > 0

                if not suitable_rovers_cameras or not objective_visible_from_anywhere or not self.communication_waypoints:
                     return float('inf') # Impossible goal

                # Check if image exists in state for any suitable rover
                has_image_in_state = any(state_has('have_image', r, objective_o, mode_m) for r, c in suitable_rovers_cameras)

                if not has_image_in_state:
                    total_cost += 1 # Need take_image action

                    # Check if any suitable camera is calibrated
                    any_suitable_calibrated = any(state_has('calibrated', c, r) for r, c in suitable_rovers_cameras)
                    if not any_suitable_calibrated:
                        total_cost += 1 # Need calibrate action

                        # Check if any suitable rover is at a calibration target waypoint
                        any_suitable_at_cal_wp = False
                        for r, c in suitable_rovers_cameras:
                            if c in self.calibration_targets: # Camera must have a target
                                cal_target = self.calibration_targets[c] # Get the target objective
                                if cal_target in self.visible_from_objective: # Target must be visible from somewhere
                                    cal_waypoints = self.visible_from_objective[cal_target] # Get visible waypoints
                                    if any(state_has('at', r, cal_wp) for cal_wp in cal_waypoints):
                                        any_suitable_at_cal_wp = True
                                        break # Found one suitable rover at a cal waypoint

                        if not any_suitable_at_cal_wp:
                             total_cost += 1 # Need navigation to calibration waypoint

                    # Check if any suitable rover is at an imaging waypoint for the objective
                    imaging_waypoints = self.visible_from_objective.get(objective_o, set()) # Use .get for safety
                    any_suitable_at_img_wp = any(state_has('at', r, img_wp)
                                                 for r, c in suitable_rovers_cameras
                                                 for img_wp in imaging_waypoints)
                    if not any_suitable_at_img_wp:
                         total_cost += 1 # Need navigation to imaging waypoint


                total_cost += 1 # Need communication action

                # Check if any suitable rover (that could potentially have the image) is at a comms point
                suitable_rovers = {r for r, c in suitable_rovers_cameras}
                any_suitable_at_comms = any(state_has('at', r, comm_wp)
                                            for r in suitable_rovers
                                            for comm_wp in self.communication_waypoints)
                if not any_suitable_at_comms:
                     total_cost += 1 # Need navigation to comms point

        return total_cost
