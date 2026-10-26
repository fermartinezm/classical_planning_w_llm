from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Check if fact is a string and not empty before processing
    if not isinstance(fact, str) or len(fact) < 2:
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

    Estimates the number of actions needed to achieve uncommunicated goals.
    It counts the number of unachieved goals (for communication actions),
    the number of distinct data items (soil, rock, image) that need to be
    acquired, and adds costs for necessary preconditions like navigation
    to sample/image/calibration locations and dropping samples, based on
    whether any capable rover is not yet in the required state/location.
    It aims to avoid overcounting shared costs like navigation to a
    communication waypoint or calibrating a camera needed for multiple images.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        self.lander_waypoint = None
        self.visible_to_lander_waypoints = set()
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        self.store_owner = {} # store -> rover
        self.camera_on_board = {} # camera -> rover
        self.camera_supports = {} # camera -> set(modes)
        self.camera_cal_target = {} # camera -> objective
        self.visible_from_obj = {} # objective -> set(waypoints)
        self.visible_from_waypoint = {} # waypoint -> set(waypoints)
        self.visible_to_waypoint = {} # waypoint -> set(waypoints)


        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            predicate = parts[0]
            args = parts[1:]

            if predicate == 'at_lander':
                # Assuming only one lander
                if len(args) == 2: self.lander_waypoint = args[1]
            elif predicate == 'visible':
                if len(args) == 2:
                    w1, w2 = args
                    self.visible_from_waypoint.setdefault(w1, set()).add(w2)
                    self.visible_to_waypoint.setdefault(w2, set()).add(w1)
            elif predicate == 'equipped_for_soil_analysis':
                if len(args) == 1: self.equipped_soil.add(args[0])
            elif predicate == 'equipped_for_rock_analysis':
                if len(args) == 1: self.equipped_rock.add(args[0])
            elif predicate == 'equipped_for_imaging':
                if len(args) == 1: self.equipped_imaging.add(args[0])
            elif predicate == 'store_of':
                if len(args) == 2: self.store_owner[args[0]] = args[1]
            elif predicate == 'on_board':
                if len(args) == 2: self.camera_on_board[args[0]] = args[1]
            elif predicate == 'supports':
                if len(args) == 2:
                    camera, mode = args
                    self.camera_supports.setdefault(camera, set()).add(mode)
            elif predicate == 'calibration_target':
                 if len(args) == 2: self.camera_cal_target[args[0]] = args[1]
            elif predicate == 'visible_from':
                if len(args) == 2:
                    objective, waypoint = args
                    self.visible_from_obj.setdefault(objective, set()).add(waypoint)

        # Pre-calculate waypoints visible *to* the lander
        if self.lander_waypoint:
             self.visible_to_lander_waypoints = self.visible_to_waypoint.get(self.lander_waypoint, set())


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Identify current rover locations and store states
        rover_locations = {} # rover -> waypoint
        full_stores = set() # set of full stores
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            args = parts[1:]

            if predicate == 'at' and len(args) == 2 and args[0].startswith('rover'):
                 rover_locations[args[0]] = args[1]
            elif predicate == 'full' and len(args) == 1:
                 full_stores.add(args[0])

        # Track required data items and conditions across all goals
        needed_soil_waypoints = set()
        needed_rock_waypoints = set()
        needed_images = set() # set of (objective, mode) tuples
        needs_comm_waypoint = False # Flag if any goal requires communication

        # Check each goal fact
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            # Goal is not achieved
            total_cost += 1 # Count the final communication action

            needs_comm_waypoint = True # At least one goal needs communication

            goal_parts = get_parts(goal)
            predicate = goal_parts[0]
            args = goal_parts[1:]

            if predicate == 'communicated_soil_data':
                waypoint = args[0]
                # Check if (have_soil_analysis ?r waypoint) is true for any rover
                data_acquired = any(f'(have_soil_analysis {r} {waypoint})' in state for r in self.equipped_soil)
                if not data_acquired:
                    needed_soil_waypoints.add(waypoint)

            elif predicate == 'communicated_rock_data':
                waypoint = args[0]
                # Check if (have_rock_analysis ?r waypoint) is true for any rover
                data_acquired = any(f'(have_rock_analysis {r} {waypoint})' in state for r in self.equipped_rock)
                if not data_acquired:
                    needed_rock_waypoints.add(waypoint)

            elif predicate == 'communicated_image_data':
                objective, mode = args
                 # Check if (have_image ?r objective mode) is true for any rover
                data_acquired = any(f'(have_image {r} {objective} {mode})' in state for r in self.equipped_imaging)
                if not data_acquired:
                    needed_images.add((objective, mode))

        # --- Estimate costs for acquiring needed data ---

        # Soil Sampling
        total_cost += len(needed_soil_waypoints) # sample_soil actions
        if needed_soil_waypoints and self.equipped_soil:
            # Navigation cost for sampling
            # Check if any equipped rover is at *any* needed soil waypoint
            at_any_needed_soil_waypoint = any(rover_locations.get(r) in needed_soil_waypoints for r in self.equipped_soil)
            if not at_any_needed_soil_waypoint:
                 total_cost += 1 # navigate to a sample waypoint

        # Rock Sampling
        total_cost += len(needed_rock_waypoints) # sample_rock actions
        if needed_rock_waypoints and self.equipped_rock:
            # Navigation cost for sampling
            # Check if any equipped rover is at *any* needed rock waypoint
            at_any_needed_rock_waypoint = any(rover_locations.get(r) in needed_rock_waypoints for r in self.equipped_rock)
            if not at_any_needed_rock_waypoint:
                 total_cost += 1 # navigate to a sample waypoint

        # Drop cost for sampling (applies if *any* soil or rock sample is needed by an equipped rover with a full store)
        needs_drop = False
        if (needed_soil_waypoints and self.equipped_soil) or (needed_rock_waypoints and self.equipped_rock):
             # Check if any equipped rover (soil or rock) has a full store
             relevant_rovers = self.equipped_soil | self.equipped_rock
             if any(self.store_owner.get(s) in relevant_rovers and s in full_stores for s in self.store_owner):
                 needs_drop = True
        if needs_drop:
             total_cost += 1 # drop


        # Imaging
        total_cost += len(needed_images) # take_image actions
        needed_calibrations = set() # Store (camera, rover) pairs that need calibration for *any* needed image

        if needed_images and self.equipped_imaging:
            suitable_rovers_cameras_for_any_image = [(r, i) for i, r in self.camera_on_board.items() if r in self.equipped_imaging and any(mode in self.camera_supports.get(i, set()) for obj, mode in needed_images)]

            if suitable_rovers_cameras_for_any_image:
                # Navigation cost for imaging
                # Check if any suitable rover is at *any* waypoint visible from *any* needed objective
                at_any_visible_waypoint = False
                for obj, mode in needed_images:
                    visible_waypoints_o = self.visible_from_obj.get(obj, set())
                    if visible_waypoints_o:
                        if any(rover_locations.get(r) in visible_waypoints_o for r, i in suitable_rovers_cameras_for_any_image):
                            at_any_visible_waypoint = True
                            break # Found one, no need to check others
                if not at_any_visible_waypoint:
                    total_cost += 1 # navigate to an image waypoint

                # Calibration cost
                # Check which specific (rover, camera) pairs need calibration for the needed images
                for obj, mode in needed_images:
                     suitable_pairs_for_this_image = [(r, i) for r, i in suitable_rovers_cameras_for_any_image if mode in self.camera_supports.get(i, set())]
                     for r, i in suitable_pairs_for_this_image:
                         if f'(calibrated {i} {r})' not in state:
                             needed_calibrations.add((r, i))

        total_cost += len(needed_calibrations) # calibrate actions

        # Navigation cost for calibration
        if needed_calibrations:
            # Check if any rover needing calibration is at a waypoint visible from its calibration target
            at_any_cal_waypoint = False
            for r, i in needed_calibrations:
                cal_target = self.camera_cal_target.get(i)
                if cal_target:
                    visible_waypoints_t = self.visible_from_obj.get(cal_target, set())
                    if visible_waypoints_t:
                        if (rover_locations.get(r) in visible_waypoints_t):
                            at_any_cal_waypoint = True
                            break # Found one, no need to check others
            if not at_any_cal_waypoint:
                total_cost += 1 # navigate to a calibration waypoint


        # --- Estimate cost for communication location ---
        if needs_comm_waypoint and self.visible_to_lander_waypoints:
            # Check if *any* rover is at a communication waypoint.
            # This is a simplification. Ideally, it should be a rover that *has* or *will get* the data.
            # But tracking that is complex. Checking if *any* rover is there is a reasonable proxy.
            all_rovers_in_state = {r for r in rover_locations}
            at_comm_waypoint = any(loc in self.visible_to_lander_waypoints for loc in rover_locations.values())
            if not at_comm_waypoint:
                total_cost += 1 # navigate to communication waypoint


        return total_cost
