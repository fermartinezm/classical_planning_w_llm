from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully, though PDDL facts are structured.
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

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to satisfy the goal
    conditions, which involve communicating soil, rock, and image data. It
    breaks down each unachieved communication goal into necessary sub-goals
    (acquiring data, navigating) and sums the estimated actions for each.

    # Assumptions
    - Each unachieved communication goal contributes independently to the heuristic cost.
    - Acquiring data (sampling, imaging) and communicating data require navigation if the relevant rover is not already at a suitable location.
    - Calibration is needed before taking an image if the camera is not currently calibrated.
    - Store capacity is ignored (assumed a store is available or can be emptied).
    - The heuristic does not model resource contention (multiple goals needing the same rover/camera/store).
    - Navigation cost is a fixed cost of 1 action if the rover is not at *any* suitable location for the next step of a task, otherwise 0. This is a simplification of actual pathfinding.
    - A rover is considered capable of a task (sampling, imaging, communicating) if it has the required equipment or already possesses the required data/image.

    # Heuristic Initialization
    The constructor extracts static information from the task, including:
    - The location of the lander.
    - Waypoints visible from the lander (communicable locations).
    - Waypoints visible from objectives (potential image locations).
    - Waypoints visible from calibration targets (potential calibration locations).
    - Which rovers have which equipment.
    - Which cameras are on which rovers and support which modes.
    - Which objectives are calibration targets for which cameras.
    - The set of goal predicates related to communication.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is the sum of costs for each goal
    predicate that is not yet satisfied in the state.

    For each unachieved goal `G`:
    1.  Add 1 to the cost for the final `communicate_X_data` action.
    2.  Determine if the required data (`have_soil_analysis`, `have_rock_analysis`, or `have_image`) is present in the current state for *any* rover.
    3.  If the data is *not* present:
        a.  Add 1 to the cost for the data acquisition action (`sample_soil`, `sample_rock`, or `take_image`).
        b.  If the acquisition is `take_image`:
            i.  Check if *any* camera on a suitable rover (equipped for imaging, supports mode) is currently calibrated.
            ii. If NO camera is calibrated: Add 1 to the cost for the `calibrate` action.
            iii. If `calibrate` was needed: Check if a suitable rover is currently at *any* waypoint visible from *any* calibration target for a suitable camera. If NO, add 1 for navigation to a calibration location.
            iv. Check if a suitable rover is currently at *any* waypoint visible from the objective `o` for this image goal. If NO, add 1 for navigation to an image location.
        c.  If the acquisition is `sample_soil` or `sample_rock`:
            i.  Check if a suitable rover (equipped for soil/rock) is currently at the sample waypoint `w`. If NO, add 1 for navigation to the sample location.
    4.  Check if a rover capable of communicating this data (either possessing the data or equipped to acquire it) is currently at *any* waypoint visible from the lander. If NO, add 1 for navigation to a communicable location.

    The total heuristic is the sum of these costs over all unachieved goals.
    A heuristic value of 0 is returned if and only if all goals are satisfied.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract all objects by type (approximate, based on common PDDL structure)
        self.all_rovers = {get_parts(f)[1] for f in static_facts if match(f, "equipped_for_soil_analysis", "*") or match(f, "equipped_for_rock_analysis", "*") or match(f, "equipped_for_imaging", "*")}
        self.all_waypoints = set()
        for fact in static_facts:
             parts = get_parts(fact)
             if parts and parts[0] in ["visible", "at_lander", "visible_from", "can_traverse"]:
                 self.all_waypoints.update(parts[1:])

        self.all_cameras = {get_parts(f)[1] for f in static_facts if match(f, "on_board", "*", "*")}
        self.all_objectives = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] in ["visible_from", "calibration_target"]:
                 self.all_objectives.update(parts[1:])

        self.all_landers = {get_parts(f)[1] for f in static_facts if match(f, "at_lander", "*", "*")}


        # Precompute static information
        self.lander_location = None
        for fact in static_facts:
            if match(fact, "at_lander", "*", "*"):
                self.lander_location = get_parts(fact)[2]
                break # Assuming only one lander for simplicity

        self.communicable_waypoints = set()
        if self.lander_location:
             self.communicable_waypoints = {
                 get_parts(fact)[1] for fact in static_facts
                 if match(fact, "visible", "*", self.lander_location)
             }

        self.objective_visible_waypoints = {}
        for obj in self.all_objectives:
            self.objective_visible_waypoints[obj] = {
                get_parts(fact)[2] for fact in static_facts
                if match(fact, "visible_from", obj, "*")
            }

        self.camera_calibration_target = {}
        for fact in static_facts:
            if match(fact, "calibration_target", "*", "*"):
                camera, target = get_parts(fact)[1:3]
                self.camera_calibration_target[camera] = target

        self.calibration_target_visible_waypoints = set()
        for target in self.camera_calibration_target.values():
             self.calibration_target_visible_waypoints.update(
                 self.objective_visible_waypoints.get(target, set())
             )

        self.rover_equipment = {}
        for rover in self.all_rovers:
            self.rover_equipment[rover] = set()
            if f'(equipped_for_soil_analysis {rover})' in static_facts:
                self.rover_equipment[rover].add('soil')
            if f'(equipped_for_rock_analysis {rover})' in static_facts:
                self.rover_equipment[rover].add('rock')
            if f'(equipped_for_imaging {rover})' in static_facts:
                self.rover_equipment[rover].add('imaging')

        self.rover_cameras = {}
        for rover in self.all_rovers:
            self.rover_cameras[rover] = {
                get_parts(fact)[1] for fact in static_facts
                if match(fact, "on_board", "*", rover)
            }

        self.camera_modes = {}
        for camera in self.all_cameras:
            self.camera_modes[camera] = {
                get_parts(fact)[2] for fact in static_facts
                if match(fact, "supports", camera, "*")
            }

        # Extract goal requirements (soil, rock, image data needed)
        self.goal_data_needed = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "communicated_soil_data":
                if len(parts) > 1: self.goal_data_needed.add(('soil', parts[1]))
            elif parts and parts[0] == "communicated_rock_data":
                 if len(parts) > 1: self.goal_data_needed.add(('rock', parts[1]))
            elif parts and parts[0] == "communicated_image_data":
                 if len(parts) > 2: self.goal_data_needed.add(('image', parts[1], parts[2]))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Get current rover locations
        rover_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                rover, waypoint = get_parts(fact)[1:3]
                rover_locations[rover] = waypoint

        # Get current data/image facts
        have_soil = {(get_parts(f)[2], get_parts(f)[1]) for f in state if match(f, "have_soil_analysis", "*", "*")} # (waypoint, rover)
        have_rock = {(get_parts(f)[2], get_parts(f)[1]) for f in state if match(f, "have_rock_analysis", "*", "*")} # (waypoint, rover)
        have_image = {(get_parts(f)[2], get_parts(f)[3], get_parts(f)[1]) for f in state if match(f, "have_image", "*", "*", "*")} # (objective, mode, rover)

        # Get current calibration facts
        calibrated_cameras = {(get_parts(f)[1], get_parts(f)[2]) for f in state if match(f, "calibrated", "*", "*")} # (camera, rover)

        # Check each unachieved goal
        for goal_item in self.goal_data_needed:
            goal_type = goal_item[0]

            if goal_type == 'soil':
                waypoint = goal_item[1]
                goal_predicate = f'(communicated_soil_data {waypoint})'

                if goal_predicate not in state:
                    total_cost += 1 # Cost for communicate

                    # Check if data is acquired
                    data_acquired = any(w == waypoint for w, r in have_soil)

                    if not data_acquired:
                        total_cost += 1 # Cost for sample
                        # Check if any equipped rover is at the sample location
                        equipped_rovers = {r for r in self.all_rovers if 'soil' in self.rover_equipment.get(r, set())}
                        if equipped_rovers: # Only check location if a capable rover exists
                            at_sample_loc = any(rover_locations.get(r) == waypoint for r in equipped_rovers)
                            if not at_sample_loc:
                                total_cost += 1 # Cost for navigate to sample location

                    # Check if any rover that could communicate is at a communicable location
                    # A rover can communicate soil data if it has the data OR is equipped to get it.
                    capable_rovers = {r for r in self.all_rovers if (waypoint, r) in have_soil or 'soil' in self.rover_equipment.get(r, set())}
                    if capable_rovers: # Only check location if a capable rover exists
                        at_comm_loc = any(
                            rover_locations.get(r) in self.communicable_waypoints
                            for r in capable_rovers
                        )
                        if not at_comm_loc:
                             total_cost += 1 # Cost for navigate to communicable location

            elif goal_type == 'rock':
                waypoint = goal_item[1]
                goal_predicate = f'(communicated_rock_data {waypoint})'

                if goal_predicate not in state:
                    total_cost += 1 # Cost for communicate

                    # Check if data is acquired
                    data_acquired = any(w == waypoint for w, r in have_rock)

                    if not data_acquired:
                        total_cost += 1 # Cost for sample
                        # Check if any equipped rover is at the sample location
                        equipped_rovers = {r for r in self.all_rovers if 'rock' in self.rover_equipment.get(r, set())}
                        if equipped_rovers: # Only check location if a capable rover exists
                            at_sample_loc = any(rover_locations.get(r) == waypoint for r in equipped_rovers)
                            if not at_sample_loc:
                                total_cost += 1 # Cost for navigate to sample location

                    # Check if any rover that could communicate is at a communicable location
                    # A rover can communicate rock data if it has the data OR is equipped to get it.
                    capable_rovers = {r for r in self.all_rovers if (waypoint, r) in have_rock or 'rock' in self.rover_equipment.get(r, set())}
                    if capable_rovers: # Only check location if a capable rover exists
                        at_comm_loc = any(
                            rover_locations.get(r) in self.communicable_waypoints
                            for r in capable_rovers
                        )
                        if not at_comm_loc:
                             total_cost += 1 # Cost for navigate to communicable location

            elif goal_type == 'image':
                objective, mode = goal_item[1:3]
                goal_predicate = f'(communicated_image_data {objective} {mode})'

                if goal_predicate not in state:
                    total_cost += 1 # Cost for communicate

                    # Check if image is acquired
                    image_acquired = any(o == objective and m == mode for o, m, r in have_image)

                    if not image_acquired:
                        total_cost += 1 # Cost for take_image

                        # Find suitable rover/camera combinations for this image goal
                        suitable_rover_cameras = []
                        for r in self.all_rovers:
                            if 'imaging' in self.rover_equipment.get(r, set()):
                                for i in self.rover_cameras.get(r, set()):
                                    if mode in self.camera_modes.get(i, set()):
                                        suitable_rover_cameras.append((r, i))

                        if suitable_rover_cameras: # Only check state/location if a suitable combo exists
                            # Check if any suitable camera is calibrated
                            is_calibrated = any((i, r) in calibrated_cameras for r, i in suitable_rover_cameras)

                            if not is_calibrated:
                                total_cost += 1 # Cost for calibrate
                                # Check if any suitable rover is at a calibration location
                                if self.calibration_target_visible_waypoints: # Only check location if cal targets exist
                                    at_cal_loc = any(
                                        rover_locations.get(r) in self.calibration_target_visible_waypoints
                                        for r, i in suitable_rover_cameras
                                    )
                                    if not at_cal_loc:
                                        total_cost += 1 # Cost for navigate to calibration location

                            # Check if any suitable rover is at an image location for this objective
                            image_waypoints = self.objective_visible_waypoints.get(objective, set())
                            if image_waypoints: # Only check location if image waypoints exist
                                at_image_loc = any(
                                    rover_locations.get(r) in image_waypoints
                                    for r, i in suitable_rover_cameras
                                )
                                if not at_image_loc:
                                    total_cost += 1 # Cost for navigate to image location

                    # Check if any rover that could communicate is at a communicable location
                    # A rover can communicate image data if it has the image OR is equipped to get it.
                    capable_rovers = {r for r in self.all_rovers if (objective, mode, r) in have_image or 'imaging' in self.rover_equipment.get(r, set())}
                    if capable_rovers: # Only check location if a capable rover exists
                        at_comm_loc = any(
                            rover_locations.get(r) in self.communicable_waypoints
                            for r in capable_rovers
                        )
                        if not at_comm_loc:
                             total_cost += 1 # Cost for navigate to communicable location

        return total_cost
