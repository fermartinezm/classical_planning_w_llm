from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming this base class exists

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
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
    conditions, which involve communicating collected data (soil, rock, images)
    from specific waypoints or for specific objectives/modes. It counts the
    missing communication actions, data collection actions (sampling or imaging),
    and necessary supporting actions like calibration and navigation, aggregating
    needs across all goals.

    # Assumptions
    - All goal conditions are of the form `(communicated_X_data ...)`.
    - The heuristic provides a lower bound on the number of *types* of actions
      needed for unsatisfied goals, but does not guarantee admissibility.
    - Navigation cost is simplified to 1 if *any* relevant navigation is needed
      for a group of goals (e.g., getting data, communicating).
    - Resource constraints (like store capacity, camera calibration state) are
      considered at a high level (e.g., is *any* relevant camera calibrated?).
    - The heuristic assumes the problem is solvable (e.g., required samples exist,
      rovers have necessary equipment, locations are reachable).

    # Heuristic Initialization
    The constructor extracts static information from the task:
    - The lander's location.
    - Waypoints visible from the lander.
    - Rover capabilities (soil, rock, imaging).
    - Store details (which store belongs to which rover).
    - Camera details (on which rover, supported modes, calibration target).
    - Waypoints from which objectives and calibration targets are visible.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:

    1.  Initialize heuristic value `h = 0`.
    2.  Identify all goal facts that are not yet true in the current state.
    3.  Identify which data types (soil from waypoint W, rock from waypoint W,
        image of objective O in mode M) are required by the unsatisfied goals
        and are not yet collected (`have_soil_analysis`, `have_rock_analysis`,
        `have_image` facts are missing).
    4.  Count the number of unsatisfied communication goals. Add this count to `h`.
        (Each goal requires one final communication action).
    5.  Count the number of unique soil samples required but not collected. Add this count to `h`.
        (Each requires one `sample_soil` action).
    6.  Count the number of unique rock samples required but not collected. Add this count to `h`.
        (Each requires one `sample_rock` action).
    7.  Count the number of unique image data required but not collected. Add this count to `h`.
        (Each requires one `take_image` action).
    8.  Check if calibration is needed for any required, uncollected image data.
        Calibration is needed if there is at least one required image `(o, m)`
        for which no relevant camera `i` on an imaging-equipped rover `r`
        (`on_board i r`, `supports i m`) is currently calibrated (`calibrated i r`).
        If calibration is needed, add 1 to `h` (for a `calibrate` action).
    9.  Check if navigation is needed to collect any required, uncollected image data.
        Navigation is needed if there is at least one required image `(o, m)`
        such that no imaging-equipped rover is currently at a waypoint visible
        from objective `o`. If navigation is needed, add 1 to `h`.
    10. If calibration was deemed necessary in step 8, check if navigation is
        needed to reach a calibration target location. Navigation is needed if
        there is at least one required image `(o, m)` needing calibration,
        and for all relevant cameras `i` with calibration target `t`, no
        imaging-equipped rover is currently at a waypoint visible from `t`.
        If navigation is needed for calibration, add 1 to `h`.
    11. Check if navigation is needed to reach a communication location for any
        unsatisfied communication goal. Navigation is needed if there is at least
        one unsatisfied communication goal (soil, rock, or image) and no rover
        is currently at a waypoint visible from the lander's location. If
        navigation is needed, add 1 to `h`.
    12. Return `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals

        # --- Extract Static Information ---
        self.lander_location = None
        self.lander_visible_waypoints = set()
        self.rover_capabilities = {} # {rover: {soil, rock, imaging}}
        self.store_of = {} # {store: rover}
        self.camera_info = {} # {camera: (rover, {modes}, cal_target)}
        self.objective_visible_from = {} # {objective: {waypoints}}
        self.caltarget_visible_from = {} # {cal_target: {waypoints}} # Note: cal_target is an objective

        # Intermediate storage for camera info
        camera_supports = {} # {camera: {modes}}
        camera_cal_targets = {} # {camera: cal_target}
        camera_on_board = {} # {camera: rover}

        # First pass to get lander location and basic static facts
        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "at_lander":
                # (at_lander ?x - lander ?y - waypoint)
                if len(parts) == 3:
                    lander, waypoint = parts[1], parts[2]
                    self.lander_location = waypoint
            elif predicate.startswith("equipped_for_"):
                # (equipped_for_soil_analysis ?r - rover) etc.
                if len(parts) == 2:
                    capability = predicate.split('_')[2] # soil, rock, imaging
                    rover = parts[1]
                    self.rover_capabilities.setdefault(rover, set()).add(capability)
            elif predicate == "store_of":
                # (store_of ?s - store ?r - rover)
                if len(parts) == 3:
                    store, rover = parts[1], parts[2]
                    self.store_of[store] = rover
            elif predicate == "supports":
                # (supports ?c - camera ?m - mode)
                if len(parts) == 3:
                    camera, mode = parts[1], parts[2]
                    camera_supports.setdefault(camera, set()).add(mode)
            elif predicate == "visible_from":
                # (visible_from ?o - objective ?w - waypoint)
                if len(parts) == 3:
                    obj, waypoint = parts[1], parts[2]
                    self.objective_visible_from.setdefault(obj, set()).add(waypoint)
                    # Also store for calibration targets (which are objectives)
                    self.caltarget_visible_from.setdefault(obj, set()).add(waypoint)
            elif predicate == "calibration_target":
                # (calibration_target ?i - camera ?t - objective)
                if len(parts) == 3:
                    camera, cal_target = parts[1], parts[2]
                    camera_cal_targets[camera] = cal_target
            elif predicate == "on_board":
                # (on_board ?i - camera ?r - rover)
                if len(parts) == 3:
                    camera, rover = parts[1], parts[2]
                    camera_on_board[camera] = rover

        # Process camera info
        for camera, rover in camera_on_board.items():
            modes = camera_supports.get(camera, set())
            cal_target = camera_cal_targets.get(camera)
            self.camera_info[camera] = (rover, modes, cal_target)

        # Process lander visibility now that lander_location is known
        if self.lander_location:
             for fact in task.static:
                 parts = get_parts(fact)
                 if not parts: continue
                 predicate = parts[0]
                 if predicate == "visible":
                     if len(parts) == 3:
                         wp1, wp2 = parts[1], parts[2]
                         if wp2 == self.lander_location:
                              self.lander_visible_waypoints.add(wp1)
             # Add lander location itself as a communication point
             self.lander_visible_waypoints.add(self.lander_location)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0

        # --- Extract Dynamic Information from State ---
        rover_locations = {} # {rover: waypoint}
        have_soil = set() # {waypoint}
        have_rock = set() # {waypoint}
        have_image = set() # {(objective, mode)}
        calibrated_cameras = set() # {(camera, rover)}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "at":
                # (at ?x - rover ?y - waypoint)
                if len(parts) == 3:
                    obj, loc = parts[1], parts[2]
                    # Assuming only rovers are 'at' waypoints in dynamic state
                    # Check if obj is a rover based on capabilities found in static
                    if obj in self.rover_capabilities:
                        rover_locations[obj] = loc
            elif predicate == "have_soil_analysis":
                # (have_soil_analysis ?r - rover ?w - waypoint)
                if len(parts) == 3:
                    have_soil.add(parts[2]) # Add waypoint
            elif predicate == "have_rock_analysis":
                # (have_rock_analysis ?r - rover ?w - waypoint)
                 if len(parts) == 3:
                    have_rock.add(parts[2]) # Add waypoint
            elif predicate == "have_image":
                # (have_image ?r - rover ?o - objective ?m - mode)
                 if len(parts) == 4:
                    have_image.add((parts[2], parts[3])) # Add (objective, mode)
            elif predicate == "calibrated":
                # (calibrated ?c - camera ?r - rover)
                 if len(parts) == 3:
                    calibrated_cameras.add((parts[1], parts[2])) # Add (camera, rover)

        # --- Compute Heuristic ---

        missing_comm_soil = set()
        missing_comm_rock = set()
        missing_comm_image = set()

        for goal in self.goals:
            if goal in state:
                continue # Goal already satisfied

            parts = get_parts(goal)
            if not parts: continue

            predicate = parts[0]
            if predicate == "communicated_soil_data":
                if len(parts) == 2: # Basic check for expected format
                    missing_comm_soil.add(parts[1]) # waypoint
            elif predicate == "communicated_rock_data":
                 if len(parts) == 2: # Basic check for expected format
                    missing_comm_rock.add(parts[1]) # waypoint
            elif predicate == "communicated_image_data":
                 if len(parts) == 3: # Basic check for expected format
                    missing_comm_image.add((parts[1], parts[2])) # (objective, mode)

        # 1. Cost for communication actions
        h += len(missing_comm_soil)
        h += len(missing_comm_rock)
        h += len(missing_comm_image)

        # 2. Cost for data collection actions (if data is missing)
        needed_soil_data = {w for w in missing_comm_soil if w not in have_soil}
        needed_rock_data = {w for w in missing_comm_rock if w not in have_rock}
        needed_image_data = {om for om in missing_comm_image if om not in have_image}

        h += len(needed_soil_data) # sample_soil action per needed sample
        h += len(needed_rock_data) # sample_rock action per needed sample
        h += len(needed_image_data) # take_image action per needed image

        # 3. Cost for supporting actions for image data (calibration, navigation)
        calibration_needed = False
        nav_image_loc_needed = False
        nav_cal_loc_needed = False

        if needed_image_data:
            # Check if calibration is needed for any missing image
            # Find relevant cameras/rovers for the needed images
            relevant_camera_rovers = set() # {(camera, rover)}
            for obj, mode in needed_image_data:
                # Find cameras that support this mode and are on imaging rovers
                for camera, (rover, modes, cal_target) in self.camera_info.items():
                    if 'imaging' in self.rover_capabilities.get(rover, set()) and mode in modes:
                         relevant_camera_rovers.add((camera, rover))

            if relevant_camera_rovers:
                 # Is *any* of these relevant cameras calibrated?
                 if not any(cr in calibrated_cameras for cr in relevant_camera_rovers):
                     calibration_needed = True
                     h += 1 # calibrate action

                 # Check if navigation to image location is needed
                 # Is any imaging rover at *any* waypoint visible from *any* needed objective?
                 imaging_rovers = {r for r, caps in self.rover_capabilities.items() if 'imaging' in caps}
                 if imaging_rovers:
                     at_suitable_image_loc = False
                     for obj, mode in needed_image_data:
                         suitable_wps = self.objective_visible_from.get(obj, set())
                         # Check if any imaging rover is at any of these suitable waypoints
                         if any(rover_locations.get(r) in suitable_wps for r in imaging_rovers):
                             at_suitable_image_loc = True
                             break # Found one, navigation might not be needed for this type of goal
                     if not at_suitable_image_loc:
                         nav_image_loc_needed = True
                         h += 1 # navigate action for image location

                 # Check if navigation to calibration location is needed (only if calibration is needed)
                 if calibration_needed:
                     at_suitable_cal_loc = False
                     # We need to calibrate *a* relevant camera. Check if any imaging rover
                     # is at a suitable location for *any* calibration target relevant to the needed images.
                     cal_targets_relevant_to_needed_images = set()
                     for camera, rover in relevant_camera_rovers:
                          cal_target = self.camera_info.get(camera, (None, None, None))[2]
                          if cal_target:
                              cal_targets_relevant_to_needed_images.add(cal_target)

                     if cal_targets_relevant_to_needed_images:
                         for cal_target in cal_targets_relevant_to_needed_images:
                             suitable_wps = self.caltarget_visible_from.get(cal_target, set())
                             # Check if any imaging rover is at any of these suitable waypoints
                             if any(rover_locations.get(r) in suitable_wps for r in imaging_rovers):
                                 at_suitable_cal_loc = True
                                 break # Found one, navigation might not be needed for calibration
                         if not at_suitable_cal_loc:
                             nav_cal_loc_needed = True
                             h += 1 # navigate action for calibration location


        # 4. Cost for navigation to lander location (if communication is needed)
        if missing_comm_soil or missing_comm_rock or missing_comm_image:
            # Is any rover at a waypoint visible from the lander?
            at_lander_visible_loc = False
            for rover, loc in rover_locations.items():
                if loc in self.lander_visible_waypoints:
                    at_lander_visible_loc = True
                    break
            if not at_lander_visible_loc:
                h += 1 # navigate action for communication location

        # The heuristic value is 0 if and only if all goals are satisfied.
        # If h > 0, it means at least one goal is missing or a step towards it is missing.
        # The heuristic is finite for solvable states.

        return h
