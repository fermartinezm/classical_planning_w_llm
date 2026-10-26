from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal conditions.
    It is an additive heuristic that sums the estimated costs for each unachieved goal.
    The cost for each goal is estimated based on the sequence of high-level actions
    (sample/take_image, calibrate, communicate) and necessary navigation steps,
    assuming the use of the first available suitable rover/camera and waypoint.

    # Assumptions
    - The cost of any navigation action between two waypoints is 1.
    - The cost of sample, drop, calibrate, take_image, and communicate actions is 1.
    - For a given goal, any suitable equipped rover and camera (if needed) can be used. The heuristic picks the first one found.
    - For imaging goals, any waypoint visible from the objective can be used for taking the image. The heuristic picks the first one found.
    - For calibration goals, any waypoint visible from the calibration target can be used. The heuristic picks the first one found.
    - The lander location is static.
    - Waypoint visibility is static.
    - Rover equipment, store ownership, camera ownership, camera modes, and calibration targets are static.
    - If a sample (soil/rock) is not at its waypoint and no rover has the analysis, the goal is unreachable.
    - If an objective is not visible from any waypoint, or a calibration target is not visible from any waypoint, or a camera has no calibration target, the image goal is unreachable.

    # Heuristic Initialization
    - Extracts static facts from the task, including:
        - Lander location.
        - Waypoint visibility pairs (used to find communication points).
        - Rover capabilities (soil, rock, imaging).
        - Store ownership by rovers.
        - Camera ownership by rovers.
        - Camera supported modes.
        - Camera calibration targets.
        - Waypoints visible from objectives.
        - Waypoints visible from calibration targets.
        - Pre-calculates the set of waypoints visible from the lander (communication points).

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic is the sum of estimated costs for each goal not yet satisfied in the current state.
    For each unachieved goal:

    1.  **Identify the goal type:** `communicated_soil_data`, `communicated_rock_data`, or `communicated_image_data`.
    2.  **Initialize goal cost:** Start with 0.
    3.  **Find a suitable rover/camera:** Identify the first available rover (and camera, if needed) that is equipped/configured to potentially achieve this goal. Prioritize rovers that have already completed an intermediate step (e.g., have the analysis/image). If no suitable rover/camera exists, the goal is considered unreachable from this state (add a large cost).
    4.  **Estimate cost based on required intermediate steps and navigation:**
        - **Soil/Rock Goal (W):**
            - If the chosen rover already has `have_soil_analysis R W` (or `have_rock_analysis R W`):
                - Add 1 for `communicate`.
                - Add 1 for `navigate` if the rover is not currently at a communication waypoint.
            - Else (need to sample):
                - Add 1 for `sample`.
                - Check if the sample is at W. If not, goal is unreachable (add large cost).
                - Check if the chosen rover has a full store. If so, add 1 for `drop`.
                - Add 1 for `navigate` from the rover's current location to W (sample location).
                - Add 1 for `communicate`.
                - Add 1 for `navigate` from W (where sampling happened) to a communication waypoint.
        - **Image Goal (O, M):**
            - If the chosen rover/camera pair already has `have_image R O M`:
                - Add 1 for `communicate`.
                - Add 1 for `navigate` if the rover is not currently at a communication waypoint.
            - Else (need to take image):
                - Add 1 for `take_image`.
                - Find a waypoint P visible from O. If none, goal is unreachable (add large cost).
                - If the chosen camera is not calibrated with the rover:
                    - Add 1 for `calibrate`.
                    - Find a calibration target T for the camera and a waypoint W visible from T. If none, goal is unreachable (add large cost).
                    - Add 1 for `navigate` from the rover's current location to W (calibration location).
                    - Add 1 for `navigate` from W to P (image location).
                - Else (already calibrated):
                    - Add 1 for `navigate` from the rover's current location to P (image location).
                - Add 1 for `communicate`.
                - Add 1 for `navigate` from P to a communication waypoint.
    5.  **Sum goal costs:** Add the estimated cost for the current goal to the total heuristic value.
    6.  **Handle unreachable goals:** If any goal is determined to be unreachable, the total heuristic is set to a large value.
    7.  **Return total cost:** Return the sum of costs for all unachieved goals, or 0 if all goals are met, or the large value if any goal is unreachable.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract static information
        self.lander_location = {} # {lander: waypoint}
        self.visible_waypoints = set() # {(wp1, wp2)}
        self.equipped_for_soil = set() # {rover}
        self.equipped_for_rock = set() # {rover}
        self.equipped_for_imaging = set() # {rover}
        self.rover_to_store = {} # {rover: store}
        self.rover_to_camera = {} # {rover: set {camera}}
        self.camera_to_modes = {} # {camera: set {mode}}
        self.camera_to_caltarget = {} # {camera: objective}
        self.objective_to_waypoints = {} # {objective: set {waypoint}}
        self.caltarget_to_waypoints = {} # {objective: set {waypoint}} # Calibration targets are objectives

        for fact in static_facts:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "at_lander":
                lander, waypoint = parts[1:]
                self.lander_location[lander] = waypoint
            elif predicate == "visible":
                wp1, wp2 = parts[1:]
                self.visible_waypoints.add((wp1, wp2))
            elif predicate == "equipped_for_soil_analysis":
                self.equipped_for_soil.add(parts[1])
            elif predicate == "equipped_for_rock_analysis":
                self.equipped_for_rock.add(parts[1])
            elif predicate == "equipped_for_imaging":
                self.equipped_for_imaging.add(parts[1])
            elif predicate == "store_of":
                store, rover = parts[1:]
                self.rover_to_store[rover] = store # Assuming one store per rover
            elif predicate == "on_board":
                camera, rover = parts[1:]
                self.rover_to_camera.setdefault(rover, set()).add(camera)
            elif predicate == "supports":
                camera, mode = parts[1:]
                self.camera_to_modes.setdefault(camera, set()).add(mode)
            elif predicate == "calibration_target":
                camera, objective = parts[1:]
                self.camera_to_caltarget[camera] = objective
            elif predicate == "visible_from":
                objective, waypoint = parts[1:]
                # visible_from applies to both objectives (for imaging) and calibration targets (which are objectives)
                self.objective_to_waypoints.setdefault(objective, set()).add(waypoint)
                self.caltarget_to_waypoints.setdefault(objective, set()).add(waypoint) # Calibration targets are objectives

        # Pre-calculate communication waypoints for the lander
        self.comm_waypoints = set()
        # Assuming there is at least one lander and its location is in static facts
        if self.lander_location:
            lander_wp = list(self.lander_location.values())[0]
            for wp1, wp2 in self.visible_waypoints:
                 if wp2 == lander_wp:
                     self.comm_waypoints.add(wp1)
                 if wp1 == lander_wp: # Visibility is often symmetric, but check both
                     self.comm_waypoints.add(wp2)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0
        large_cost = 1000 # Use a large value for unreachable goals

        # Extract dynamic information from the current state
        rover_locations = {} # {rover: waypoint}
        soil_samples_at = set() # {waypoint}
        rock_samples_at = set() # {waypoint}
        stores_empty = set() # {store}
        stores_full = set() # {store}
        soil_analyses_had = set() # {(rover, waypoint)}
        rock_analyses_had = set() # {(rover, waypoint)}
        images_had = set() # {(rover, objective, mode)}
        cameras_calibrated = set() # {(camera, rover)}
        soil_data_communicated = set() # {waypoint}
        rock_data_communicated = set() # {waypoint}
        image_data_communicated = set() # {(objective, mode)}

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "at":
                obj, loc = parts[1:]
                # Only track rover locations, ignore lander location which is static
                # Check if obj is a rover by seeing if it has a store (simple heuristic)
                if obj in self.rover_to_store:
                     rover_locations[obj] = loc
            elif predicate == "at_soil_sample":
                soil_samples_at.add(parts[1])
            elif predicate == "at_rock_sample":
                rock_samples_at.add(parts[1])
            elif predicate == "empty":
                stores_empty.add(parts[1])
            elif predicate == "full":
                stores_full.add(parts[1])
            elif predicate == "have_soil_analysis":
                rover, waypoint = parts[1:]
                soil_analyses_had.add((rover, waypoint))
            elif predicate == "have_rock_analysis":
                rover, waypoint = parts[1:]
                rock_analyses_had.add((rover, waypoint))
            elif predicate == "have_image":
                rover, objective, mode = parts[1:]
                images_had.add((rover, objective, mode))
            elif predicate == "calibrated":
                camera, rover = parts[1:]
                cameras_calibrated.add((camera, rover))
            elif predicate == "communicated_soil_data":
                soil_data_communicated.add(parts[1])
            elif predicate == "communicated_rock_data":
                rock_data_communicated.add(parts[1])
            elif predicate == "communicated_image_data":
                objective, mode = parts[1:]
                image_data_communicated.add((objective, mode))

        # Iterate through goals and sum costs for unachieved ones
        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]

            if predicate == "communicated_soil_data":
                waypoint = parts[1]
                if waypoint in soil_data_communicated:
                    continue # Goal already achieved

                h_g = 0
                best_rover = None

                # Find a rover that *could* achieve this goal (equipped for soil)
                # Prioritize rovers that already have the analysis
                for R in self.equipped_for_soil:
                     if (R, waypoint) in soil_analyses_had:
                         best_rover = R
                         break
                # If no rover has the analysis, just pick the first equipped one
                if best_rover is None and self.equipped_for_soil:
                     best_rover = list(self.equipped_for_soil)[0]

                if best_rover is None:
                     total_cost += large_cost # No equipped rover, goal unreachable
                     continue # Move to next goal

                current_loc = rover_locations.get(best_rover)
                if current_loc is None: # Should not happen for a valid rover
                     total_cost += large_cost
                     continue

                if (best_rover, waypoint) in soil_analyses_had: # Have analysis
                    # Sequence: Communicate (at Comm)
                    # Actions: communicate (1)
                    h_g += 1
                    # Navs: nav(current, Comm)
                    if current_loc not in self.comm_waypoints:
                        h_g += 1
                else: # Need to sample
                    # Sequence: Sample (at W) -> Communicate (at Comm)
                    # Actions: sample (1), communicate (1) [+ drop (1)]
                    h_g += 1 # sample
                    h_g += 1 # communicate

                    if waypoint not in soil_samples_at:
                        total_cost += large_cost # Sample is gone, goal unreachable
                        continue # Move to next goal

                    # Check if drop is needed for the chosen rover
                    store = self.rover_to_store.get(best_rover)
                    if store and store in stores_full:
                         h_g += 1 # drop action

                    # Navs: nav(current, W), nav(W, Comm)
                    if current_loc != waypoint:
                        h_g += 1 # nav to sample loc
                    # Rover is at W after sampling
                    if waypoint not in self.comm_waypoints:
                         h_g += 1 # nav from sample loc to comm point

                total_cost += h_g

            elif predicate == "communicated_rock_data":
                waypoint = parts[1]
                if waypoint in rock_data_communicated:
                    continue # Goal already achieved

                h_g = 0
                best_rover = None

                # Find a rover that *could* achieve this goal (equipped for rock)
                # Prioritize rovers that already have the analysis
                for R in self.equipped_for_rock:
                     if (R, waypoint) in rock_analyses_had:
                         best_rover = R
                         break
                # If no rover has the analysis, just pick the first equipped one
                if best_rover is None and self.equipped_for_rock:
                     best_rover = list(self.equipped_for_rock)[0]

                if best_rover is None:
                     total_cost += large_cost # No equipped rover, goal unreachable
                     continue # Move to next goal

                current_loc = rover_locations.get(best_rover)
                if current_loc is None: # Should not happen for a valid rover
                     total_cost += large_cost
                     continue

                if (best_rover, waypoint) in rock_analyses_had: # Have analysis
                    # Sequence: Communicate (at Comm)
                    # Actions: communicate (1)
                    h_g += 1
                    # Navs: nav(current, Comm)
                    if current_loc not in self.comm_waypoints:
                        h_g += 1
                else: # Need to sample
                    # Sequence: Sample (at W) -> Communicate (at Comm)
                    # Actions: sample (1), communicate (1) [+ drop (1)]
                    h_g += 1 # sample
                    h_g += 1 # communicate

                    if waypoint not in rock_samples_at:
                        total_cost += large_cost # Sample is gone, goal unreachable
                        continue # Move to next goal

                    # Check if drop is needed for the chosen rover
                    store = self.rover_to_store.get(best_rover)
                    if store and store in stores_full:
                         h_g += 1 # drop action

                    # Navs: nav(current, W), nav(W, Comm)
                    if current_loc != waypoint:
                        h_g += 1 # nav to sample loc
                    # Rover is at W after sampling
                    if waypoint not in self.comm_waypoints:
                         h_g += 1 # nav from sample loc to comm point

                total_cost += h_g

            elif predicate == "communicated_image_data":
                objective, mode = parts[1:]
                if (objective, mode) in image_data_communicated:
                    continue # Goal already achieved

                h_g = 0
                best_rover = None
                best_camera = None

                # Find a rover/camera that *could* achieve this goal (equipped for imaging + on_board + supports mode)
                # Prioritize pairs that already have the image
                for R in self.equipped_for_imaging:
                    for I in self.rover_to_camera.get(R, set()):
                        if mode in self.camera_to_modes.get(I, set()):
                            if (R, objective, mode) in images_had:
                                best_rover = R
                                best_camera = I
                                break
                    if best_rover: break # Found a pair with the image

                # If no pair has the image, find any pair that can take the image
                if best_rover is None:
                     for R in self.equipped_for_imaging:
                         for I in self.rover_to_camera.get(R, set()):
                             if mode in self.camera_to_modes.get(I, set()):
                                 best_rover = R
                                 best_camera = I
                                 break
                         if best_rover: break # Found a pair that can take the image

                if best_rover is None:
                     total_cost += large_cost # No suitable rover/camera, goal unreachable
                     continue # Move to next goal

                current_loc = rover_locations.get(best_rover)
                if current_loc is None: # Should not happen for a valid rover
                     total_cost += large_cost
                     continue

                if (best_rover, objective, mode) in images_had: # Have image
                    # Sequence: Communicate (at Comm)
                    # Actions: communicate (1)
                    h_g += 1
                    # Navs: nav(current, Comm)
                    if current_loc not in self.comm_waypoints:
                        h_g += 1
                else: # Need to take image
                    # Sequence: [Calibrate (at W)] -> TakeImage (at P) -> Communicate (at Comm)
                    # Actions: take_image (1), communicate (1) [+ calibrate (1)]
                    h_g += 1 # take_image
                    h_g += 1 # communicate

                    # Find waypoint P visible from O
                    image_waypoint_options = self.objective_to_waypoints.get(objective, set())
                    if not image_waypoint_options:
                        total_cost += large_cost # Objective not visible from anywhere, goal unreachable
                        continue # Move to next goal
                    image_waypoint = list(image_waypoint_options)[0] # Pick one

                    cal_needed = (best_camera, best_rover) not in cameras_calibrated

                    if cal_needed: # Need to calibrate
                        h_g += 1 # calibrate action
                        # Find calibration target T for I and waypoint W visible from T
                        cal_target = self.camera_to_caltarget.get(best_camera)
                        if not cal_target:
                            total_cost += large_cost # Camera has no calibration target, goal unreachable
                            continue # Move to next goal
                        cal_waypoint_options = self.caltarget_to_waypoints.get(cal_target, set())
                        if not cal_waypoint_options:
                            total_cost += large_cost # Calibration target not visible from anywhere, goal unreachable
                            continue # Move to next goal
                        cal_waypoint = list(cal_waypoint_options)[0] # Pick one

                        # Navs: nav(current, W), nav(W, P), nav(P, Comm)
                        if current_loc != cal_waypoint:
                            h_g += 1 # nav(current, W)
                        # Rover is at W after calibrating. Needs to go to P.
                        loc_after_cal = cal_waypoint
                        if loc_after_cal != image_waypoint:
                            h_g += 1 # nav(W, P)
                        # Rover is at P after imaging. Needs to go to Comm.
                        loc_after_image = image_waypoint
                        if loc_after_image not in self.comm_waypoints:
                            h_g += 1 # nav(P, Comm)

                    else: # Already calibrated
                        # Navs: nav(current, P), nav(P, Comm)
                        if current_loc != image_waypoint:
                            h_g += 1 # nav(current, P)
                        # Rover is at P after imaging. Needs to go to Comm.
                        loc_after_image = image_waypoint
                        if loc_after_image not in self.comm_waypoints:
                            h_g += 1 # nav(P, Comm)

                total_cost += h_g

        # If total_cost is non-zero and less than large_cost, it's the estimated cost.
        # If total_cost is 0, all goals were achieved.
        # If total_cost is >= large_cost, at least one goal is unreachable.
        if total_cost >= large_cost:
             return large_cost
        return total_cost
