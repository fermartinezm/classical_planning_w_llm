from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions
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
    This heuristic estimates the number of actions required to achieve all goal
    conditions. It sums up the estimated cost for each unachieved goal fact.
    The estimated cost for a goal fact is based on the sequence of actions
    and intermediate conditions needed to produce that fact, assuming the
    easiest path (e.g., any available rover/resource can be used, navigation
    costs a fixed amount if needed).

    # Assumptions
    - All required static conditions (e.g., rover equipment, camera capabilities,
      visible_from, can_traverse, at_lander) are met in the initial state or static facts.
    - Soil and rock samples exist at the specified waypoints if they are goals.
    - The lander is at a fixed location.
    - Any rover can navigate between any two waypoints if a path exists (we
      don't compute actual paths, just assume navigation is possible and costs 1
      action if the rover is not already at a suitable location).
    - Any rover equipped for a task can perform it if the conditions are met.
    - Any store on an equipped rover can be used.
    - Any camera on an equipped rover supporting the mode can be used.

    # Heuristic Initialization
    - Extract goal conditions.
    - Extract static facts:
        - Lander location.
        - Waypoints visible from the lander.
        - Rovers equipped for soil, rock, imaging.
        - Stores for each rover.
        - Cameras on each rover, modes supported, calibration targets.
        - Waypoints visible from objectives (for imaging).
        - Waypoints visible from calibration targets (for calibration).
        - Identify all objects of relevant types (rover, waypoint, store, camera, mode, objective, lander).

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is the sum of estimated costs for each goal fact that
    is not yet true in the current state.

    For each goal fact `g` not in the current state:
        If `g` is `(communicated_soil_data W)`:
            Add 1 (for the communicate action).
            If `(have_soil_analysis R W)` is not true for *any* rover R equipped for soil:
                Add 1 (for the sample_soil action).
                If `(at R W)` is not true for *any* rover R equipped for soil:
                    Add 1 (for navigation to W).
                If `(empty S)` is not true for *any* store S of *any* rover equipped for soil:
                    Add 1 (for drop action).
            If `(at R X)` is not true for *any* rover R and *any* waypoint X visible from the lander:
                Add 1 (for navigation to X).

        If `g` is `(communicated_rock_data W)`:
            Add 1 (for the communicate action).
            If `(have_rock_analysis R W)` is not true for *any* rover R equipped for rock:
                Add 1 (for the sample_rock action).
                If `(at R W)` is not true for *any* rover R equipped for rock:
                    Add 1 (for navigation to W).
                If `(empty S)` is not true for *any* store S of *any* rover equipped for rock:
                    Add 1 (for drop action).
            If `(at R X)` is not true for *any* rover R and *any* waypoint X visible from the lander:
                Add 1 (for navigation to X).

        If `g` is `(communicated_image_data O M)`:
            Add 1 (for the communicate action).
            If `(have_image R O M)` is not true for *any* rover R equipped for imaging:
                Add 1 (for the take_image action).
                If `(at R P)` is not true for *any* rover R equipped for imaging and *any* waypoint P visible from O:
                    Add 1 (for navigation to P).
                If `(calibrated I R)` is not true for *any* camera I on an equipped imaging rover R supporting M:
                    Add 1 (for the calibrate action).
                    If `(at R W)` is not true for *any* rover R equipped for imaging and *any* waypoint W visible from *any* calibration target T for *any* camera I on R supporting M:
                        Add 1 (for navigation to W).
            If `(at R X)` is not true for *any* rover R and *any* waypoint X visible from the lander:
                Add 1 (for navigation to X).

    The total heuristic value is the sum of these costs for all unachieved goals.
    A goal state has a heuristic value of 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Need initial state to find all objects

        # Extract static information
        self.lander_location = None
        self.waypoints_visible_from_lander = set()
        self.equipped_rovers = {'soil': set(), 'rock': set(), 'imaging': set()}
        self.rover_stores = {} # rover -> store
        self.rover_cameras = {} # rover -> set of cameras
        self.camera_modes = {} # camera -> set of modes
        self.camera_calibration_targets = {} # camera -> set of objectives
        self.objective_visible_waypoints = {} # objective -> set of waypoints
        self.calibration_target_visible_waypoints = {} # objective (target) -> set of waypoints

        # Identify all objects of each type from initial state and static facts
        all_objects = set()
        for fact in initial_state | static_facts:
             parts = get_parts(fact)
             all_objects.update(parts[1:]) # Add all arguments as potential objects

        self.all_rovers = {obj for obj in all_objects if obj.startswith('rover')}
        self.all_waypoints = {obj for obj in all_objects if obj.startswith('waypoint')}
        self.all_stores = {obj for obj in all_objects if obj.startswith('store')}
        self.all_cameras = {obj for obj in all_objects if obj.startswith('camera')}
        self.all_modes = {obj for obj in all_objects if obj.startswith('mode')} # Modes are objects in PDDL typing
        self.all_objectives = {obj for obj in all_objects if obj.startswith('objective')}
        self.all_landers = {obj for obj in all_objects if obj.startswith('lander')}


        # Populate static data structures
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'at_lander':
                # Assuming only one lander and its location is static
                if self.lander_location is None:
                     self.lander_location = parts[2]
            elif parts[0] == 'visible':
                 # Store visibility in both directions for easy lookup
                 wp1, wp2 = parts[1], parts[2]
                 # Only add to visible_from_lander if lander location is known
                 if self.lander_location:
                     if wp1 == self.lander_location:
                         self.waypoints_visible_from_lander.add(wp2)
                     if wp2 == self.lander_location:
                         self.waypoints_visible_from_lander.add(wp1)
            elif parts[0] == 'equipped_for_soil_analysis':
                self.equipped_rovers['soil'].add(parts[1])
            elif parts[0] == 'equipped_for_rock_analysis':
                self.equipped_rovers['rock'].add(parts[1])
            elif parts[0] == 'equipped_for_imaging':
                self.equipped_rovers['imaging'].add(parts[1])
            elif parts[0] == 'store_of':
                self.rover_stores[parts[2]] = parts[1] # rover -> store
            elif parts[0] == 'on_board':
                rover, camera = parts[2], parts[1]
                self.rover_cameras.setdefault(rover, set()).add(camera)
            elif parts[0] == 'supports':
                camera, mode = parts[1], parts[2]
                self.camera_modes.setdefault(camera, set()).add(mode)
            elif parts[0] == 'calibration_target':
                camera, objective = parts[1], parts[2]
                self.camera_calibration_targets.setdefault(camera, set()).add(objective)
            elif parts[0] == 'visible_from':
                objective, waypoint = parts[1], parts[2]
                # Check if the objective is actually defined as an objective or calibration target
                is_objective = objective in self.all_objectives
                is_cal_target = False
                for targets in self.camera_calibration_targets.values():
                    if objective in targets:
                        is_cal_target = True
                        break

                if is_objective:
                    self.objective_visible_waypoints.setdefault(objective, set()).add(waypoint)
                if is_cal_target:
                     self.calibration_target_visible_waypoints.setdefault(objective, set()).add(waypoint)

        # Ensure lander location was found (might be only in initial state)
        if self.lander_location is None and self.all_landers:
             # Attempt to find lander location from initial state if not in static
             lander_obj = list(self.all_landers)[0] # Assuming one lander
             for fact in initial_state:
                 if match(fact, 'at_lander', lander_obj, '*'):
                     self.lander_location = get_parts(fact)[2]
                     # Re-populate visible_from_lander if lander location found late
                     for fact_static in static_facts:
                         if match(fact_static, 'visible', '*', '*'):
                             wp1, wp2 = get_parts(fact_static)[1], get_parts(fact_static)[2]
                             if wp1 == self.lander_location:
                                 self.waypoints_visible_from_lander.add(wp2)
                             if wp2 == self.lander_location:
                                 self.waypoints_visible_from_lander.add(wp1)
                     break


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0

        # Check each goal condition
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            predicate = parts[0]

            if predicate == 'communicated_soil_data':
                waypoint_w = parts[1]
                h += 1 # Cost for the communicate action

                # Check if any equipped soil rover has the soil analysis
                has_sample = False
                for rover in self.equipped_rovers['soil']:
                    if f'(have_soil_analysis {rover} {waypoint_w})' in state:
                        has_sample = True
                        break

                if not has_sample:
                    h += 1 # Cost for the sample_soil action

                    # Check if any equipped soil rover is at the sample location W
                    at_sample_location = False
                    for rover in self.equipped_rovers['soil']:
                         if f'(at {rover} {waypoint_w})' in state:
                             at_sample_location = True
                             break
                    if not at_sample_location:
                        h += 1 # Cost for navigation to W

                    # Check if any equipped soil rover has an empty store
                    has_empty_store = False
                    for rover in self.equipped_rovers['soil']:
                        store = self.rover_stores.get(rover)
                        if store and f'(empty {store})' in state:
                            has_empty_store = True
                            break
                    if not has_empty_store:
                        h += 1 # Cost for drop action

                # Check if any rover is at a lander-visible waypoint for communication
                at_lander_visible_waypoint = False
                if self.waypoints_visible_from_lander: # Only check if lander location and visible waypoints are known
                    for rover in self.all_rovers: # Any rover can communicate if at the right spot with the data
                         current_rover_location = None
                         for fact in state:
                             if match(fact, 'at', rover, '*'):
                                 current_rover_location = get_parts(fact)[2]
                                 break
                         if current_rover_location and current_rover_location in self.waypoints_visible_from_lander:
                             at_lander_visible_waypoint = True
                             break

                    if not at_lander_visible_waypoint:
                         h += 1 # Cost for navigation to lander-visible waypoint


            elif predicate == 'communicated_rock_data':
                waypoint_w = parts[1]
                h += 1 # Cost for the communicate action

                # Check if any equipped rock rover has the rock analysis
                has_sample = False
                for rover in self.equipped_rovers['rock']:
                    if f'(have_rock_analysis {rover} {waypoint_w})' in state:
                        has_sample = True
                        break

                if not has_sample:
                    h += 1 # Cost for the sample_rock action

                    # Check if any equipped rock rover is at the sample location W
                    at_sample_location = False
                    for rover in self.equipped_rovers['rock']:
                         if f'(at {rover} {waypoint_w})' in state:
                             at_sample_location = True
                             break
                    if not at_sample_location:
                        h += 1 # Cost for navigation to W

                    # Check if any equipped rock rover has an empty store
                    has_empty_store = False
                    for rover in self.equipped_rovers['rock']:
                        store = self.rover_stores.get(rover)
                        if store and f'(empty {store})' in state:
                            has_empty_store = True
                            break
                    if not has_empty_store:
                        h += 1 # Cost for drop action

                # Check if any rover is at a lander-visible waypoint for communication
                at_lander_visible_waypoint = False
                if self.waypoints_visible_from_lander: # Only check if lander location and visible waypoints are known
                    for rover in self.all_rovers: # Any rover can communicate if at the right spot with the data
                         current_rover_location = None
                         for fact in state:
                             if match(fact, 'at', rover, '*'):
                                 current_rover_location = get_parts(fact)[2]
                                 break
                         if current_rover_location and current_rover_location in self.waypoints_visible_from_lander:
                             at_lander_visible_waypoint = True
                             break

                    if not at_lander_visible_waypoint:
                         h += 1 # Cost for navigation to lander-visible waypoint


            elif predicate == 'communicated_image_data':
                objective_o = parts[1]
                mode_m = parts[2]
                h += 1 # Cost for the communicate action

                # Check if any equipped imaging rover has the image
                has_image = False
                for rover in self.equipped_rovers['imaging']:
                    if f'(have_image {rover} {objective_o} {mode_m})' in state:
                        has_image = True
                        break

                if not has_image:
                    h += 1 # Cost for the take_image action

                    # Check if any equipped imaging rover is at a waypoint visible from the objective
                    at_image_location = False
                    suitable_waypoints = self.objective_visible_waypoints.get(objective_o, set())
                    if suitable_waypoints:
                        for rover in self.equipped_rovers['imaging']:
                            if any(f'(at {rover} {wp})' in state for wp in suitable_waypoints):
                                at_image_location = True
                                break
                    if not at_image_location:
                        h += 1 # Cost for navigation to P

                    # Check if any camera on an equipped imaging rover supporting the mode is calibrated
                    is_calibrated = False
                    for rover in self.equipped_rovers['imaging']:
                        for camera in self.rover_cameras.get(rover, set()):
                            if mode_m in self.camera_modes.get(camera, set()):
                                if f'(calibrated {camera} {rover})' in state:
                                    is_calibrated = True
                                    break
                        if is_calibrated: break

                    if not is_calibrated:
                        h += 1 # Cost for the calibrate action

                        # Check if any equipped imaging rover is at a waypoint visible from a calibration target for a suitable camera
                        at_calibration_location = False
                        suitable_cal_waypoints = set()
                        for rover in self.equipped_rovers['imaging']:
                             for camera in self.rover_cameras.get(rover, set()):
                                 if mode_m in self.camera_modes.get(camera, set()):
                                     for target in self.camera_calibration_targets.get(camera, set()):
                                         suitable_cal_waypoints.update(self.calibration_target_visible_waypoints.get(target, set()))

                        if suitable_cal_waypoints:
                            for rover in self.equipped_rovers['imaging']:
                                if any(f'(at {rover} {wp})' in state for wp in suitable_cal_waypoints):
                                    at_calibration_location = True
                                    break
                        if not at_calibration_location:
                            h += 1 # Cost for navigation to W


                # Check if any rover is at a lander-visible waypoint for communication
                at_lander_visible_waypoint = False
                if self.waypoints_visible_from_lander: # Only check if lander location and visible waypoints are known
                    for rover in self.all_rovers: # Any rover can communicate if at the right spot with the data
                         current_rover_location = None
                         for fact in state:
                             if match(fact, 'at', rover, '*'):
                                 current_rover_location = get_parts(fact)[2]
                                 break
                         if current_rover_location and current_rover_location in self.waypoints_visible_from_lander:
                             at_lander_visible_waypoint = True
                             break

                    if not at_lander_visible_waypoint:
                         h += 1 # Cost for navigation to lander-visible waypoint


        return h
