from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(at rover1 waypoint2)" -> ["at", "rover1", "waypoint2"]
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of arguments, unless args has wildcards
    if len(parts) != len(args) and '*' not in args:
         return False
    # Use zip to handle cases where parts might be longer than args (e.g., extra args in fact)
    # or args might have trailing wildcards. all() handles the shorter iterable ending.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to reach a goal state
    by summing the estimated costs for each unsatisfied goal. The cost for a goal
    is estimated based on the sequence of actions needed (sampling/imaging,
    calibrating, communicating) and includes a navigation cost (1 action) if a
    suitable rover is not already at a required location for a specific step.
    It also accounts for the need to drop samples if stores are full and the
    calibration state of cameras for imaging goals.

    # Assumptions
    - Each action (navigate, sample, drop, calibrate, take_image, communicate) costs 1.
    - Navigation to any required location (sample site, image site, calibration site,
      communication site) costs 1 action if no suitable rover is currently at *any*
      of the possible locations for that step. This is a simplification and ignores
      actual path distance.
    - Resource conflicts (e.g., multiple rovers needing the same store or camera)
      and optimal task assignment to specific rovers are ignored. The heuristic
      checks if *any* suitable rover/resource exists in the required state/location.
    - The heuristic is non-admissible but aims to be informative to guide a greedy search.

    # Heuristic Initialization
    The heuristic precomputes static information from the task definition to
    efficiently determine required locations and rover/camera capabilities.
    This includes:
    - Locations of landers.
    - Waypoints from which communication is possible (visible from lander locations).
    - Which rovers are equipped for soil analysis, rock analysis, and imaging.
    - The mapping of rovers to their stores.
    - The mapping of rovers to cameras on board.
    - The mapping of cameras to the modes they support.
    - The mapping of cameras to their calibration targets.
    - The mapping of objectives to waypoints from which they are visible.
    - The mapping of calibration targets to waypoints from which they are visible.
    - The set of objectives and modes that appear in the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is computed as follows:

    1. Initialize the total heuristic cost `h` to 0.
    2. Iterate through each goal condition specified in the task.
    3. For each goal `(communicated_soil_data W)`:
       - If the goal is already true in the current state, add 0 to `h`.
       - Otherwise, add 1 to `h` for the `communicate_soil_data` action.
       - Check if any equipped soil rover currently has `(have_soil_analysis R W)`.
         - If yes: Check if *any* equipped soil rover that has this data is currently at a communication waypoint. If not, add 1 to `h` for navigation to a communication point.
         - If no: Add 1 to `h` for the `sample_soil` action. Check if *any* equipped soil rover is at waypoint W. If not, add 1 to `h` for navigation to W. Check if *all* equipped soil rovers have full stores. If yes (and there are equipped rovers), add 1 to `h` for a `drop` action. Check if *any* equipped soil rover is at a communication waypoint. If not, add 1 to `h` for navigation to a communication point.
    4. For each goal `(communicated_rock_data W)`:
       - Similar logic as for soil goals, checking for `(have_rock_analysis R W)` and using equipped rock rovers.
    5. For each goal `(communicated_image_data O M)`:
       - If the goal is already true in the current state, add 0 to `h`.
       - Otherwise, add 1 to `h` for the `communicate_image_data` action.
       - Check if any equipped imaging rover currently has `(have_image R O M)`.
         - If yes: Check if *any* equipped imaging rover that has this image is currently at a communication waypoint. If not, add 1 to `h` for navigation to a communication point.
         - If no: Add 1 to `h` for the `take_image` action. Identify suitable imaging rovers/cameras for objective O and mode M. Check if *any* suitable camera on *any* suitable rover is already calibrated; if not, add 1 to `h` for the `calibrate` action and check if *any* suitable rover is at a calibration waypoint. If not, add 1 to `h` for navigation to a calibration waypoint. Check if *any* suitable rover is at a waypoint where O is visible; if not, add 1 to `h` for navigation to an image waypoint. Check if *any* suitable rover is at a communication waypoint; if not, add 1 to `h` for navigation to a communication point.
    6. The total value of `h` is the heuristic estimate.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Precompute static information
        self.lander_locations = {get_parts(fact)[2] for fact in static_facts if match(fact, "at_lander", "*", "*")}
        self.communication_waypoints = {
            get_parts(fact)[1] for fact in static_facts
            if match(fact, "visible", "*", "*") and get_parts(fact)[2] in self.lander_locations
        }

        self.equipped_soil_rovers = {get_parts(fact)[1] for fact in static_facts if match(fact, "equipped_for_soil_analysis", "*")}
        self.equipped_rock_rovers = {get_parts(fact)[1] for fact in static_facts if match(fact, "equipped_for_rock_analysis", "*")}
        self.equipped_imaging_rovers = {get_parts(fact)[1] for fact in static_facts if match(fact, "equipped_for_imaging", "*")}

        self.rover_stores = {get_parts(fact)[2]: get_parts(fact)[1] for fact in static_facts if match(fact, "store_of", "*", "*")}
        self.rover_cameras = {}
        for fact in static_facts:
            if match(fact, "on_board", "*", "*"):
                camera, rover = get_parts(fact)[1:3]
                self.rover_cameras.setdefault(rover, set()).add(camera)

        self.camera_modes = {}
        for fact in static_facts:
            if match(fact, "supports", "*", "*"):
                camera, mode = get_parts(fact)[1:3]
                self.camera_modes.setdefault(camera, set()).add(mode)

        self.camera_calibration_target = {get_parts(fact)[1]: get_parts(fact)[2] for fact in static_facts if match(fact, "calibration_target", "*", "*")}

        # Extract objectives and modes from goals
        self.goal_objectives_modes = {(get_parts(g)[1], get_parts(g)[2]) for g in self.goals if match(g, "communicated_image_data", "*", "*")}
        goal_objectives = {obj for obj, mode in self.goal_objectives_modes}

        # Identify cameras and targets relevant to goals
        relevant_cameras = set()
        for rover in self.equipped_imaging_rovers:
            for camera in self.rover_cameras.get(rover, set()):
                if any(mode in self.camera_modes.get(camera, set()) for obj, mode in self.goal_objectives_modes):
                    relevant_cameras.add(camera)

        relevant_calibration_targets = {self.camera_calibration_target[i] for i in relevant_cameras if i in self.camera_calibration_target}


        self.objective_image_waypoints = {}
        self.target_calibration_waypoints = {}

        for fact in static_facts:
            if match(fact, "visible_from", "*", "*"):
                obj_or_target, waypoint = get_parts(fact)[1:3]
                if obj_or_target in goal_objectives:
                     self.objective_image_waypoints.setdefault(obj_or_target, set()).add(waypoint)
                if obj_or_target in relevant_calibration_targets:
                     self.target_calibration_waypoints.setdefault(obj_or_target, set()).add(waypoint)


    def get_suitable_imaging_rovers_cameras(self, objective, mode):
        """Helper to find (rover, camera) pairs suitable for a specific objective and mode."""
        suitable_pairs = []
        for rover in self.equipped_imaging_rovers:
            for camera in self.rover_cameras.get(rover, set()):
                if mode in self.camera_modes.get(camera, set()):
                    suitable_pairs.append((rover, camera))
        return suitable_pairs

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0

        # Process Soil Goals
        soil_goals = {get_parts(g)[1] for g in self.goals if match(g, "communicated_soil_data", "*")}
        for waypoint in soil_goals:
            if f'(communicated_soil_data {waypoint})' in state:
                continue

            h += 1 # communicate_soil_data action

            # Check if soil data is already collected by *any* equipped rover
            has_data = any(f'(have_soil_analysis {r} {waypoint})' in state for r in self.equipped_soil_rovers)

            if has_data:
                # Need to navigate to communication point if no rover with data is there
                # Check if *any* rover that *has this specific data* is at a communication point
                at_comm = any(
                    f'(at {r} {wp})' in state
                    for r in self.equipped_soil_rovers if f'(have_soil_analysis {r} {waypoint})' in state
                    for wp in self.communication_waypoints
                )
                if not at_comm:
                    h += 1 # navigate to communication point
            else:
                # Need to sample soil
                h += 1 # sample_soil action

                # Need to navigate to sample waypoint if no equipped rover is there
                at_sample = any(f'(at {r} {waypoint})' in state for r in self.equipped_soil_rovers)
                if not at_sample:
                    h += 1 # navigate to sample waypoint

                # Need to drop if all equipped rovers have full stores
                all_equipped_full = (
                    len(self.equipped_soil_rovers) > 0 and
                    all(f'(full {self.rover_stores[r]})' in state for r in self.equipped_soil_rovers if r in self.rover_stores)
                )
                if all_equipped_full:
                    h += 1 # drop action

                # Need to navigate to communication point if no equipped rover is there
                # Check if *any* equipped rover is at a communication point
                at_comm = any(
                    f'(at {r} {wp})' in state
                    for r in self.equipped_soil_rovers
                    for wp in self.communication_waypoints
                )
                if not at_comm:
                    h += 1 # navigate to communication point

        # Process Rock Goals
        rock_goals = {get_parts(g)[1] for g in self.goals if match(g, "communicated_rock_data", "*")}
        for waypoint in rock_goals:
            if f'(communicated_rock_data {waypoint})' in state:
                continue

            h += 1 # communicate_rock_data action

            # Check if rock data is already collected by *any* equipped rover
            has_data = any(f'(have_rock_analysis {r} {waypoint})' in state for r in self.equipped_rock_rovers)

            if has_data:
                # Need to navigate to communication point if no rover with data is there
                 at_comm = any(
                    f'(at {r} {wp})' in state
                    for r in self.equipped_rock_rovers if f'(have_rock_analysis {r} {waypoint})' in state
                    for wp in self.communication_waypoints
                )
                 if not at_comm:
                    h += 1 # navigate to communication point
            else:
                # Need to sample rock
                h += 1 # sample_rock action

                # Need to navigate to sample waypoint if no equipped rover is there
                at_sample = any(f'(at {r} {waypoint})' in state for r in self.equipped_rock_rovers)
                if not at_sample:
                    h += 1 # navigate to sample waypoint

                # Need to drop if all equipped rovers have full stores
                all_equipped_full = (
                    len(self.equipped_rock_rovers) > 0 and
                    all(f'(full {self.rover_stores[r]})' in state for r in self.equipped_rock_rovers if r in self.rover_stores)
                )
                if all_equipped_full:
                    h += 1 # drop action

                # Need to navigate to communication point if no equipped rover is there
                # Check if *any* equipped rover is at a communication point
                at_comm = any(
                    f'(at {r} {wp})' in state
                    for r in self.equipped_rock_rovers
                    for wp in self.communication_waypoints
                )
                if not at_comm:
                    h += 1 # navigate to communication point

        # Process Image Goals
        # Use the precomputed set self.goal_objectives_modes
        for objective, mode in self.goal_objectives_modes:
            if f'(communicated_image_data {objective} {mode})' in state:
                continue

            h += 1 # communicate_image_data action

            # Check if image is already taken by *any* equipped rover with a suitable camera
            has_image = any(f'(have_image {r} {objective} {mode})' in state for r in self.equipped_imaging_rovers) # Simplified check

            if has_image:
                 # Need to navigate to communication point if no rover with image is there
                 # Check if *any* rover that *has this specific image* is at a communication point
                 at_comm = any(
                    f'(at {r} {wp})' in state
                    for r in self.equipped_imaging_rovers if f'(have_image {r} {objective} {mode})' in state
                    for wp in self.communication_waypoints
                )
                 if not at_comm:
                    h += 1 # navigate to communication point
            else:
                # Need to take image
                h += 1 # take_image action

                suitable_rovers_cameras = self.get_suitable_imaging_rovers_cameras(objective, mode)

                # Need to calibrate if no suitable camera is calibrated
                any_suitable_calibrated = any(f'(calibrated {i} {r})' in state for r, i in suitable_rovers_cameras)
                if not any_suitable_calibrated:
                    h += 1 # calibrate action
                    # Need to navigate to calibration waypoint if no suitable rover is there
                    at_calibration_wp = False
                    for r, i in suitable_rovers_cameras:
                        target = self.camera_calibration_target.get(i)
                        if target:
                            for cal_wp in self.target_calibration_waypoints.get(target, []):
                                # Check if *this specific rover* is at the calibration waypoint
                                if f'(at {r} {cal_wp})' in state:
                                    at_calibration_wp = True
                                    break # Found rover r at a cal point for camera i
                        if at_calibration_wp: break # Found a suitable rover/camera pair at a cal point
                    if not at_calibration_wp:
                        h += 1 # navigate to calibration waypoint


                # Need to navigate to image waypoint if no suitable rover is there
                # Check if *any* suitable rover is at *any* image waypoint for this objective
                at_image_wp = any(
                    f'(at {r} {wp})' in state
                    for r, i in suitable_rovers_cameras
                    for wp in self.objective_image_waypoints.get(objective, [])
                )
                if not at_image_wp:
                    h += 1 # navigate to image waypoint

                # Need to navigate to communication point if no suitable rover is there
                # Check if *any* suitable rover is at a communication point
                at_comm = any(
                    f'(at {r} {wp})' in state
                    for r, i in suitable_rovers_cameras
                    for wp in self.communication_waypoints
                )
                if not at_comm:
                    h += 1 # navigate to communication point

        return h
