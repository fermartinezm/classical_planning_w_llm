from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions for parsing PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and not empty
    if not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove leading/trailing parentheses and split by whitespace
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding argument pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the minimum number of actions required to satisfy
    the unsatisfied goal conditions. It uses a relaxed plan approach, counting
    actions needed for collection (sampling, imaging, calibration, dropping)
    and navigation to relevant locations (sample/image/calibration waypoint,
    communication waypoint). It assumes a simplified cost of 1 for each
    necessary action or navigation step and uses relaxations such as assuming
    any capable rover/camera can perform a task and any rover can move to a
    communication point.

    # Assumptions
    - Each unsatisfied goal requires a final communication action.
    - Collecting a sample (soil/rock) requires being at the sample location,
      having an empty store, being equipped, and performing the sample action.
    - Taking an image requires being at the image location, being equipped,
      having a camera on board that supports the mode, and the camera being
      calibrated.
    - Calibrating a camera requires being at a calibration location visible
      from the camera's target, being equipped, and having the camera on board.
    - Moving between waypoints costs 1 action (simplified, ignoring path distance).
    - Dropping a sample costs 1 action.
    - Calibration costs 1 action.
    - Sampling costs 1 action.
    - Taking an image costs 1 action.
    - Communication costs 1 action.
    - The heuristic assumes that if a goal exists for a sample/image, the
      sample is initially present at the waypoint, or the objective is visible
      from some waypoint, and suitable rovers/cameras exist. It checks for
      the *current* availability of samples at waypoints.
    - It uses a strong relaxation: if a prerequisite (like having a sample)
      is not met, it adds the cost assuming *any* suitable rover can perform
      the collection/imaging task and *any* rover can move to the required
      waypoint for collection/imaging/communication.

    # Heuristic Initialization
    - Extracts static information from the task:
        - Lander location.
        - Rover capabilities (soil, rock, imaging).
        - Rover-store mapping.
        - Rover-camera mapping.
        - Camera-mode support mapping.
        - Camera-calibration target mapping.
        - Waypoints visible from the lander location (communication points).
        - Waypoints from which objectives are visible (image points).
        - Waypoints suitable for camera calibration.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize heuristic value `h` to 0.
    2. Extract relevant dynamic information from the current state (rover locations,
       store status, collected samples/images, calibrated cameras, current sample locations).
    3. Iterate through each goal predicate defined in the task.
    4. For each goal predicate:
       - If the goal predicate is already true in the current state, add 0 to `h`.
       - If the goal predicate is not true:
         - Add 1 to `h` for the final communication action required for this goal.
         - Determine the type of goal (soil, rock, or image communication).
         - If it's a soil or rock communication goal for waypoint `?w`:
           - Check if the corresponding sample (`have_soil_analysis ?r ?w` or
             `have_rock_analysis ?r ?w`) is held by any rover.
           - If the sample is NOT held:
             - Check if the sample is currently available at waypoint `?w`.
             - If the sample IS available:
               - Add 1 for the `sample_soil` or `sample_rock` action.
               - Check if any equipped rover has a full store. If yes, add 1 for a `drop` action.
               - Check if any equipped rover is NOT at waypoint `?w`. If yes, add 1 for a `navigate` action to the sample location.
           - Check if any rover is NOT at a communication waypoint (visible from the lander). If yes, add 1 for a `navigate` action to a communication location.
         - If it's an image communication goal for objective `?o` and mode `?m`:
           - Check if the image (`have_image ?r ?o ?m`) is held by any rover.
           - If the image is NOT held:
             - Find suitable imaging rovers/cameras (equipped for imaging, camera on board, supports mode).
             - Check if the objective `?o` is visible from any waypoint.
             - If suitable rovers/cameras exist AND the objective is visible:
               - Add 1 for the `take_image` action.
               - Check if calibration is needed for any suitable camera/rover combination (i.e., not currently calibrated). If yes:
                 - Check if calibration is possible (calibration target visible from any waypoint for any suitable camera). If yes:
                   - Add 1 for the `calibrate` action.
                   - Check if any suitable rover is NOT at a calibration waypoint. If yes, add 1 for a `navigate` action to a calibration location.
               - Check if any suitable rover is NOT at an image waypoint (visible from `?o`). If yes, add 1 for a `navigate` action to an image location.
           - Check if any rover is NOT at a communication waypoint (visible from the lander). If yes, add 1 for a `navigate` action to a communication location.
    5. Return the total calculated value of `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract static information
        self.lander_waypoint = None
        self.rover_capabilities = {} # {rover_name: set of capabilities}
        self.rover_stores = {}       # {rover_name: store_name}
        self.rover_cameras = {}      # {rover_name: list of camera_names}
        self.camera_modes = {}       # {camera_name: set of mode_names}
        self.camera_calibration_target = {} # {camera_name: objective_name}
        self.comms_waypoints = set() # set of waypoint_names visible from lander
        self.objective_image_waypoints = {} # {objective_name: set of waypoint_names}
        self.calibration_waypoints = {} # {camera_name: set of waypoint_names}

        lander_waypoint_name = None
        visible_facts = []
        visible_from_facts = []

        for fact in self.static:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            predicate = parts[0]
            if predicate == "at_lander":
                # (at_lander ?x - lander ?y - waypoint)
                if len(parts) == 3:
                    lander_waypoint_name = parts[2]
            elif predicate.startswith("equipped_for_"):
                # (equipped_for_soil_analysis ?r - rover) etc.
                if len(parts) == 2:
                    rover = parts[1]
                    capability = predicate[len("equipped_for_"):] # e.g., "soil_analysis"
                    self.rover_capabilities.setdefault(rover, set()).add(capability)
            elif predicate == "store_of":
                # (store_of ?s - store ?r - rover)
                if len(parts) == 3:
                    store, rover = parts[1], parts[2]
                    self.rover_stores[rover] = store
            elif predicate == "on_board":
                # (on_board ?i - camera ?r - rover)
                if len(parts) == 3:
                    camera, rover = parts[1], parts[2]
                    self.rover_cameras.setdefault(rover, []).append(camera)
            elif predicate == "supports":
                # (supports ?c - camera ?m - mode)
                if len(parts) == 3:
                    camera, mode = parts[1], parts[2]
                    self.camera_modes.setdefault(camera, set()).add(mode)
            elif predicate == "calibration_target":
                # (calibration_target ?i - camera ?t - objective)
                if len(parts) == 3:
                    camera, objective = parts[1], parts[2]
                    self.camera_calibration_target[camera] = objective
            elif predicate == "visible":
                # (visible ?w1 - waypoint ?w2 - waypoint)
                if len(parts) == 3:
                    visible_facts.append(fact)
            elif predicate == "visible_from":
                # (visible_from ?o - objective ?w - waypoint)
                if len(parts) == 3:
                    visible_from_facts.append(fact)

        self.lander_waypoint = lander_waypoint_name

        # Determine communication waypoints (visible from lander waypoint)
        if self.lander_waypoint:
            # The communicate action requires (visible ?x ?y) where ?x is rover loc and ?y is lander loc
            # So we need waypoints ?x such that (visible ?x lander_waypoint) is true.
            self.comms_waypoints = {get_parts(fact)[1] for fact in visible_facts if match(fact, "*", self.lander_waypoint)}

        # Determine objective image waypoints
        for fact in visible_from_facts:
            # (visible_from ?o - objective ?w - waypoint)
            _, objective, waypoint = get_parts(fact)
            self.objective_image_waypoints.setdefault(objective, set()).add(waypoint)

        # Determine calibration waypoints for each camera
        for camera, target_objective in self.camera_calibration_target.items():
            # Calibration requires (visible_from ?t ?w) where ?t is the target
            if target_objective in self.objective_image_waypoints:
                 self.calibration_waypoints[camera] = self.objective_image_waypoints[target_objective]
            else:
                 self.calibration_waypoints[camera] = set() # No visible waypoints for target


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if the goal is already reached
        if self.goals <= state:
            return 0

        # Extract dynamic information from the current state
        current_rover_locations = {} # {rover_name: waypoint_name}
        current_store_status = {}    # {store_name: 'empty' or 'full'}
        current_have_soil = set()    # {(rover_name, waypoint_name)}
        current_have_rock = set()    # {(rover_name, waypoint_name)}
        current_have_image = set()   # {(rover_name, objective_name, mode_name)}
        current_calibrated_cameras = set() # {(camera_name, rover_name)}
        current_soil_sample_locations = set() # {waypoint_name}
        current_rock_sample_locations = set() # {waypoint_name}


        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "at":
                # (at ?x - rover ?y - waypoint)
                if len(parts) == 3: # Ensure it's a rover at a waypoint
                     current_rover_locations[parts[1]] = parts[2]
            elif predicate == "empty":
                # (empty ?s - store)
                if len(parts) == 2:
                    current_store_status[parts[1]] = 'empty'
            elif predicate == "full":
                # (full ?s - store)
                if len(parts) == 2:
                    current_store_status[parts[1]] = 'full'
            elif predicate == "have_soil_analysis":
                # (have_soil_analysis ?r - rover ?w - waypoint)
                if len(parts) == 3:
                    current_have_soil.add((parts[1], parts[2]))
            elif predicate == "have_rock_analysis":
                # (have_rock_analysis ?r - rover ?w - waypoint)
                if len(parts) == 3:
                    current_have_rock.add((parts[1], parts[2]))
            elif predicate == "have_image":
                # (have_image ?r - rover ?o - objective ?m - mode)
                if len(parts) == 4:
                    current_have_image.add((parts[1], parts[2], parts[3]))
            elif predicate == "calibrated":
                # (calibrated ?c - camera ?r - rover)
                if len(parts) == 3:
                    current_calibrated_cameras.add((parts[1], parts[2]))
            elif predicate == "at_soil_sample":
                # (at_soil_sample ?w - waypoint)
                if len(parts) == 2:
                    current_soil_sample_locations.add(parts[1])
            elif predicate == "at_rock_sample":
                # (at_rock_sample ?w - waypoint)
                if len(parts) == 2:
                    current_rock_sample_locations.add(parts[1])


        total_cost = 0  # Initialize action cost counter.

        # Iterate through each goal condition
        for goal in self.goals:
            # If the goal is already satisfied, it contributes 0 to the heuristic
            if goal in state:
                continue

            # Goal is not satisfied, estimate cost
            # Add cost for the final communication action
            total_cost += 1

            predicate, *args = get_parts(goal)

            if predicate == "communicated_soil_data":
                waypoint = args[0]
                # Check if the sample has been collected by any rover
                sample_collected = any((r, waypoint) in current_have_soil for r in self.rover_capabilities)

                if not sample_collected:
                    # Need to collect the sample
                    # Check if the sample is available at the waypoint in the current state
                    if waypoint in current_soil_sample_locations:
                        total_cost += 1 # Cost for sample_soil action

                        # Check if any soil-equipped rover needs to drop a full store
                        needs_drop = any(self.rover_stores.get(r) and current_store_status.get(self.rover_stores.get(r)) == 'full'
                                         for r, caps in self.rover_capabilities.items() if 'soil_analysis' in caps)
                        if needs_drop:
                             total_cost += 1 # Cost for drop action

                        # Check if any soil-equipped rover needs to navigate to the sample location
                        # Only add cost if there is at least one soil-equipped rover
                        soil_rovers = [r for r, caps in self.rover_capabilities.items() if 'soil_analysis' in caps]
                        if soil_rovers:
                            needs_move_to_sample = any(current_rover_locations.get(r) != waypoint
                                                       for r in soil_rovers)
                            if needs_move_to_sample:
                                 total_cost += 1 # Cost for navigate to sample location
                    # else: sample not available at waypoint, cannot collect it in this relaxed view, don't add collection costs

                # Check if any rover needs to navigate to a communication location
                # This is needed whether the sample is collected or not, as communication is the next step
                # Only add cost if there is at least one rover and at least one comms waypoint
                if current_rover_locations and self.comms_waypoints:
                    needs_move_to_comms = any(loc not in self.comms_waypoints
                                              for loc in current_rover_locations.values())
                    if needs_move_to_comms:
                         total_cost += 1 # Cost for navigate to comms location


            elif predicate == "communicated_rock_data":
                waypoint = args[0]
                # Check if the sample has been collected by any rover
                sample_collected = any((r, waypoint) in current_have_rock for r in self.rover_capabilities)

                if not sample_collected:
                    # Need to collect the sample
                    # Check if the sample is available at the waypoint in the current state
                    if waypoint in current_rock_sample_locations:
                        total_cost += 1 # Cost for sample_rock action

                        # Check if any rock-equipped rover needs to drop a full store
                        needs_drop = any(self.rover_stores.get(r) and current_store_status.get(self.rover_stores.get(r)) == 'full'
                                         for r, caps in self.rover_capabilities.items() if 'rock_analysis' in caps)
                        if needs_drop:
                             total_cost += 1 # Cost for drop action

                        # Check if any rock-equipped rover needs to navigate to the sample location
                        # Only add cost if there is at least one rock-equipped rover
                        rock_rovers = [r for r, caps in self.rover_capabilities.items() if 'rock_analysis' in caps]
                        if rock_rovers:
                            needs_move_to_sample = any(current_rover_locations.get(r) != waypoint
                                                       for r in rock_rovers)
                            if needs_move_to_sample:
                                 total_cost += 1 # Cost for navigate to sample location
                    # else: sample not available at waypoint, don't add collection costs

                # Check if any rover needs to navigate to a communication location
                # Only add cost if there is at least one rover and at least one comms waypoint
                if current_rover_locations and self.comms_waypoints:
                    needs_move_to_comms = any(loc not in self.comms_waypoints
                                              for loc in current_rover_locations.values())
                    if needs_move_to_comms:
                         total_cost += 1 # Cost for navigate to comms location


            elif predicate == "communicated_image_data":
                objective, mode = args
                # Check if the image has been taken by any rover
                image_taken = any((r, objective, mode) in current_have_image for r in self.rover_capabilities)

                if not image_taken:
                    # Need to take the image
                    # Find any imaging rover with a camera supporting the mode
                    suitable_imaging_rovers = [r for r, caps in self.rover_capabilities.items()
                                               if 'imaging' in caps and any(mode in self.camera_modes.get(cam, set())
                                                                            for cam in self.rover_cameras.get(r, []))]

                    # Check if the objective is visible from any waypoint
                    img_waypoints = self.objective_image_waypoints.get(objective, set())
                    image_possible = bool(img_waypoints)

                    if suitable_imaging_rovers and image_possible:
                        total_cost += 1 # Cost for take_image action

                        # Check if calibration is needed for any suitable camera/rover combination
                        needs_calibration = any((cam, r) not in current_calibrated_cameras
                                                for r in suitable_imaging_rovers
                                                for cam in self.rover_cameras.get(r, [])
                                                if mode in self.camera_modes.get(cam, set()))

                        if needs_calibration:
                            # Find calibration waypoints for any suitable camera
                            all_cal_waypoints = set().union(*(self.calibration_waypoints.get(cam, set())
                                                              for r in suitable_imaging_rovers
                                                              for cam in self.rover_cameras.get(r, [])
                                                              if mode in self.camera_modes.get(cam, set())))
                            # Check if calibration is possible from any visible waypoint
                            calibration_possible = bool(all_cal_waypoints)

                            if calibration_possible:
                                total_cost += 1 # Cost for calibrate action
                                # Check if any suitable rover needs to navigate to a calibration location
                                needs_move_to_calibrate = any(current_rover_locations.get(r) not in all_cal_waypoints
                                                              for r in suitable_imaging_rovers)
                                if needs_move_to_calibrate:
                                    total_cost += 1 # Cost for navigate to calibration location
                            # else: calibration not possible, don't add calibrate/move costs

                        # Check if any suitable rover needs to navigate to an image location
                        needs_move_to_image = any(current_rover_locations.get(r) not in img_waypoints
                                                  for r in suitable_imaging_rovers)
                        if needs_move_to_image:
                             total_cost += 1 # Cost for navigate to image location
                    # else: image not possible (no suitable rover/camera or no visible waypoint), don't add image/calibrate/move costs

                # Check if any rover needs to navigate to a communication location
                # Only add cost if there is at least one rover and at least one comms waypoint
                if current_rover_locations and self.comms_waypoints:
                    needs_move_to_comms = any(loc not in self.comms_waypoints
                                              for loc in current_rover_locations.values())
                    if needs_move_to_comms:
                         total_cost += 1 # Cost for navigate to comms location

        return total_cost
