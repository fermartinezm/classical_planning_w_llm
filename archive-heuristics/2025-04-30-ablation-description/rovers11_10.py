from fnmatch import fnmatch
# Assuming heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not provided
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#
#     def __call__(self, node):
#         raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by whitespace
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of pattern arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing up the estimated costs
    for each individual goal condition that is not yet satisfied. The estimated cost
    for an unsatisfied goal is based on the required steps (collect/image, communicate)
    and penalties for being in the wrong location or state (navigation, dropping sample, calibration).

    # Assumptions
    - Each goal (communicating soil data, rock data, or image data) can be pursued somewhat independently.
    - The cost of navigation between any two relevant waypoints is a fixed value (e.g., 1).
    - The cost of sampling, dropping, calibrating, taking an image, and communicating is a fixed value (e.g., 1).
    - Assumes that if a goal requires a specific equipment type (soil, rock, imaging) or a camera with a specific mode/target, at least one such equipped rover/camera exists in the problem instance if the goal is solvable.
    - Ignores negative interactions between actions (e.g., a store becoming full preventing another sample, or a camera becoming uncalibrated after taking an image).

    # Heuristic Initialization
    The constructor extracts static information from the task, which does not change during planning:
    - The location of the lander (`lander_location`).
    - The set of waypoints visible from the lander's location (`communication_points`).
    - Sets of rovers equipped for soil analysis (`equipped_soil`), rock analysis (`equipped_rock`), and imaging (`equipped_imaging`).
    - Mapping from stores to their owning rovers (`store_owner`).
    - Mapping from rovers to the cameras they have on board (`rover_cameras`).
    - Mapping from cameras to the modes they support (`camera_modes`).
    - Mapping from cameras to their calibration targets (`camera_targets`).
    - Mapping from objectives to the waypoints from which they are visible (`objective_viewpoints`).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is calculated as follows:
    1. Initialize `total_cost` to 0.
    2. Parse the current state to extract dynamic information:
       - Current location of each rover.
       - Status (empty/full) of each store.
       - Waypoints with uncollected soil/rock samples.
       - Data collected by each rover (soil, rock, image).
       - Data already communicated.
       - Calibrated status of cameras.
    3. Iterate through each goal fact defined in the task.
    4. For each goal fact:
       - If the goal fact is already true in the current state, add 0 cost for this goal.
       - If the goal fact is not true:
         - Add a base cost of 1 for the final communication action required for this goal.
         - Determine the type of data needed (soil, rock, or image) and its parameters (waypoint for soil/rock, objective/mode for image).
         - Check if the required data/sample/image is already collected by any rover in the current state.
           - If YES (data is collected by some rover `r_h`):
             - Check if `r_h` is currently at a waypoint from which communication is possible (`communication_points`). If not, add 1 to `total_cost` (estimating navigation cost).
           - If NO (data is not collected):
             - Add 1 to `total_cost` for the collection/imaging action (sample_soil, sample_rock, or take_image).
             - Identify a suitable rover/camera combination based on static equipment facts. (The heuristic assumes at least one such combination exists if the goal is solvable).
             - Check if the suitable rover is at the required location for collection/imaging (waypoint for sample, viewpoint for image). If not, add 1 to `total_cost` (estimating navigation cost).
             - If the goal is sampling (soil/rock): Check if the suitable rover's store is full. If yes, add 1 to `total_cost` (estimating drop action cost).
             - If the goal is imaging: Check if the suitable camera is calibrated. If not, add 1 to `total_cost` (estimating calibrate action cost). (This simplified heuristic does not add extra navigation cost specifically for reaching a calibration target viewpoint if it differs from the objective viewpoint).
    5. Return the final `total_cost`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals

        # --- Extract Static Information ---
        self.lander_location = None
        self.communication_points = set()
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        self.store_owner = {} # store -> rover
        self.rover_cameras = {} # rover -> set of cameras
        self.camera_modes = {} # camera -> set of modes
        self.camera_targets = {} # camera -> objective (calibration target)
        self.objective_viewpoints = {} # objective -> set of waypoints

        # First pass to find lander location
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == 'at_lander' and len(parts) == 3:
                # (at_lander ?l ?y)
                self.lander_location = parts[2]
                break # Assuming only one lander location

        # Second pass to find communication points (visible from lander location)
        if self.lander_location:
             for fact in task.static:
                parts = get_parts(fact)
                # (visible ?w1 ?w2)
                if parts and parts[0] == 'visible' and len(parts) == 3:
                    w1, w2 = parts[1], parts[2]
                    if w2 == self.lander_location:
                        self.communication_points.add(w1)

        # Third pass for other static facts
        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts

            predicate = parts[0]
            if predicate == 'equipped_for_soil_analysis' and len(parts) == 2:
                self.equipped_soil.add(parts[1])
            elif predicate == 'equipped_for_rock_analysis' and len(parts) == 2:
                self.equipped_rock.add(parts[1])
            elif predicate == 'equipped_for_imaging' and len(parts) == 2:
                self.equipped_imaging.add(parts[1])
            elif predicate == 'store_of' and len(parts) == 3:
                # (store_of ?s ?r)
                self.store_owner[parts[1]] = parts[2]
            elif predicate == 'on_board' and len(parts) == 3:
                # (on_board ?i ?r)
                camera, rover = parts[1], parts[2]
                self.rover_cameras.setdefault(rover, set()).add(camera)
            elif predicate == 'supports' and len(parts) == 3:
                # (supports ?c ?m)
                camera, mode = parts[1], parts[2]
                self.camera_modes.setdefault(camera, set()).add(mode)
            elif predicate == 'calibration_target' and len(parts) == 3:
                # (calibration_target ?i ?t)
                camera, target = parts[1], parts[2]
                self.camera_targets[camera] = target
            elif predicate == 'visible_from' and len(parts) == 3:
                # (visible_from ?o ?w)
                objective, waypoint = parts[1], parts[2]
                self.objective_viewpoints.setdefault(objective, set()).add(waypoint)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # --- Parse Dynamic State Information ---
        rover_locations = {} # rover -> waypoint
        store_status = {} # store -> 'empty' or 'full'
        # soil_samples_at = set() # waypoint - Not needed for heuristic calculation
        # rock_samples_at = set() # waypoint - Not needed for heuristic calculation
        rover_soil_data = {} # rover -> set of waypoints
        rover_rock_data = {} # rover -> set of waypoints
        rover_image_data = {} # rover -> set of (objective, mode) tuples
        communicated_soil = set() # waypoint
        communicated_rock = set() # waypoint
        communicated_image = set() # (objective, mode) tuple
        calibrated_cameras = set() # (camera, rover) tuple

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'at' and len(parts) == 3:
                # (at ?x - rover ?y - waypoint)
                # Assuming first arg is rover based on domain
                rover_locations[parts[1]] = parts[2]
            elif predicate == 'empty' and len(parts) == 2:
                 # (empty ?s - store)
                 store_status[parts[1]] = 'empty'
            elif predicate == 'full' and len(parts) == 2:
                 # (full ?s - store)
                 store_status[parts[1]] = 'full'
            # elif predicate == 'at_soil_sample' and len(parts) == 2:
            #      soil_samples_at.add(parts[1])
            # elif predicate == 'at_rock_sample' and len(parts) == 2:
            #      rock_samples_at.add(parts[1])
            elif predicate == 'have_soil_analysis' and len(parts) == 3:
                 # (have_soil_analysis ?r - rover ?w - waypoint)
                 rover, waypoint = parts[1], parts[2]
                 rover_soil_data.setdefault(rover, set()).add(waypoint)
            elif predicate == 'have_rock_analysis' and len(parts) == 3:
                 # (have_rock_analysis ?r - rover ?w - waypoint)
                 rover, waypoint = parts[1], parts[2]
                 rover_rock_data.setdefault(rover, set()).add(waypoint)
            elif predicate == 'have_image' and len(parts) == 4:
                 # (have_image ?r - rover ?o - objective ?m - mode)
                 rover, objective, mode = parts[1], parts[2], parts[3]
                 rover_image_data.setdefault(rover, set()).add((objective, mode))
            elif predicate == 'communicated_soil_data' and len(parts) == 2:
                 # (communicated_soil_data ?w - waypoint)
                 communicated_soil.add(parts[1])
            elif predicate == 'communicated_rock_data' and len(parts) == 2:
                 # (communicated_rock_data ?w - waypoint)
                 communicated_rock.add(parts[1])
            elif predicate == 'communicated_image_data' and len(parts) == 3:
                 # (communicated_image_data ?o - objective ?m - mode)
                 communicated_image.add((parts[1], parts[2]))
            elif predicate == 'calibrated' and len(parts) == 3:
                 # (calibrated ?c - camera ?r - rover)
                 calibrated_cameras.add((parts[1], parts[2]))

        # --- Calculate Cost based on Unmet Goals ---
        for goal_fact_str in self.goals:
            goal_parts = get_parts(goal_fact_str)
            if not goal_parts: continue # Skip malformed goals

            goal_predicate = goal_parts[0]

            if goal_predicate == 'communicated_soil_data' and len(goal_parts) == 2:
                waypoint = goal_parts[1]
                if waypoint not in communicated_soil:
                    total_cost += 1 # Cost for the final communicate action

                    # Check if sample is collected by any rover
                    has_sample = any(waypoint in data for data in rover_soil_data.values())

                    if has_sample:
                        # Find *a* rover that has the sample
                        rover_with_sample = None
                        for r, data in rover_soil_data.items():
                            if waypoint in data:
                                rover_with_sample = r
                                break
                        # Check if rover is at a communication point
                        current_rover_loc = rover_locations.get(rover_with_sample)
                        if current_rover_loc is None or current_rover_loc not in self.communication_points:
                            total_cost += 1 # Cost for navigation to communication point
                    else:
                        total_cost += 1 # Cost for sample action

                        # Check if sampling is possible (is there an equipped rover?)
                        if self.equipped_soil:
                            # Find *a* soil-equipped rover (doesn't matter which one for this heuristic)
                            r_s = next(iter(self.equipped_soil))
                            # Check if rover is at the sample location
                            if rover_locations.get(r_s) != waypoint:
                                total_cost += 1 # Cost for navigation to sample location
                            # Check if rover's store is full
                            rover_store = None
                            for s, owner in self.store_owner.items():
                                if owner == r_s:
                                    rover_store = s
                                    break
                            if rover_store is not None and store_status.get(rover_store) == 'full':
                                total_cost += 1 # Cost for drop action

            elif goal_predicate == 'communicated_rock_data' and len(goal_parts) == 2:
                waypoint = goal_parts[1]
                if waypoint not in communicated_rock:
                    total_cost += 1 # Cost for the final communicate action

                    # Check if sample is collected by any rover
                    has_sample = any(waypoint in data for data in rover_rock_data.values())

                    if has_sample:
                        # Find *a* rover that has the sample
                        rover_with_sample = None
                        for r, data in rover_rock_data.items():
                            if waypoint in data:
                                rover_with_sample = r
                                break
                        # Check if rover is at a communication point
                        current_rover_loc = rover_locations.get(rover_with_sample)
                        if current_rover_loc is None or current_rover_loc not in self.communication_points:
                            total_cost += 1 # Cost for navigation to communication point
                    else:
                        total_cost += 1 # Cost for sample action

                        # Check if sampling is possible (is there an equipped rover?)
                        if self.equipped_rock:
                            # Find *a* rock-equipped rover
                            r_r = next(iter(self.equipped_rock))
                            # Check if rover is at the sample location
                            if rover_locations.get(r_r) != waypoint:
                                total_cost += 1 # Cost for navigation to sample location
                            # Check if rover's store is full
                            rover_store = None
                            for s, owner in self.store_owner.items():
                                if owner == r_r:
                                    rover_store = s
                                    break
                            if rover_store is not None and store_status.get(rover_store) == 'full':
                                total_cost += 1 # Cost for drop action

            elif goal_predicate == 'communicated_image_data' and len(goal_parts) == 3:
                objective, mode = goal_parts[1], goal_parts[2]
                if (objective, mode) not in communicated_image:
                    total_cost += 1 # Cost for the final communicate action

                    # Check if image is taken by any rover
                    has_image = any((objective, mode) in data for data in rover_image_data.values())

                    if has_image:
                        # Find *a* rover that has the image
                        rover_with_image = None
                        for r, data in rover_image_data.items():
                            if (objective, mode) in data:
                                rover_with_image = r
                                break
                        # Check if rover is at a communication point
                        current_rover_loc = rover_locations.get(rover_with_image)
                        if current_rover_loc is None or current_rover_loc not in self.communication_points:
                            total_cost += 1 # Cost for navigation to communication point
                    else:
                        total_cost += 1 # Cost for take_image action

                        # Check if imaging is possible (is there an equipped rover with a supporting camera?)
                        suitable_rover_camera = None
                        for r in self.equipped_imaging:
                            for cam in self.rover_cameras.get(r, set()):
                                if mode in self.camera_modes.get(cam, set()):
                                    suitable_rover_camera = (r, cam)
                                    break
                            if suitable_rover_camera:
                                break

                        if suitable_rover_camera:
                            r_i, cam_i = suitable_rover_camera
                            # Check if rover is at a viewpoint for the objective
                            current_rover_loc = rover_locations.get(r_i)
                            viewpoints = self.objective_viewpoints.get(objective, set())
                            if current_rover_loc is None or current_rover_loc not in viewpoints:
                                total_cost += 1 # Cost for navigation to viewpoint
                            # Check if camera is calibrated
                            if (cam_i, r_i) not in calibrated_cameras:
                                total_cost += 1 # Cost for calibrate action
                                # Note: This simplified heuristic doesn't check if the current location
                                # is also a viewpoint for the calibration target.

        return total_cost
