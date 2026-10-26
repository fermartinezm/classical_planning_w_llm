from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact_parts, *args):
    """
    Check if PDDL fact parts match a given pattern.

    - `fact_parts`: The components of a fact as a list of strings (e.g., ['at', 'rover1', 'waypoint1']).
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact parts match the pattern, else `False`.
    """
    if len(fact_parts) != len(args):
        return False
    return all(fnmatch(f, t) for f, t in zip(fact_parts, args))

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Estimates the number of actions required to achieve the goal conditions.
    The heuristic counts the necessary non-navigation actions (sample,
    communicate, calibrate, take_image, drop) and adds a navigation cost
    (estimated as 1) for each distinct location type a rover needs to reach
    to complete the remaining steps for a specific goal.

    Assumptions:
    - Solvable instances have necessary equipped rovers, cameras, etc.
    - Navigation between any two relevant waypoints costs at least 1 action.
    - The heuristic doesn't track specific rover assignments optimally but
      checks for the *existence* of a suitable rover state or adds navigation
      cost if a relevant rover isn't at a required location type.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals

        # Extract static information
        self.lander_location = None
        self.comm_waypoints = set()
        self.equipped_for_soil = set()
        self.equipped_for_rock = set()
        self.equipped_for_imaging = set()
        self.rover_stores = {} # rover -> store
        self.camera_details = {} # camera -> {on_board: rover, calibration_target: objective, supports: {mode}}
        self.objective_visibility = {} # objective -> {waypoint}
        self.all_rovers = set() # Keep track of all rovers mentioned in static facts

        for fact_str in task.static:
            parts = get_parts(fact_str)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'at_lander' and len(parts) == 3:
                self.lander_location = parts[2]
            elif predicate == 'equipped_for_soil_analysis' and len(parts) == 2:
                self.equipped_for_soil.add(parts[1])
                self.all_rovers.add(parts[1])
            elif predicate == 'equipped_for_rock_analysis' and len(parts) == 2:
                self.equipped_for_rock.add(parts[1])
                self.all_rovers.add(parts[1])
            elif predicate == 'equipped_for_imaging' and len(parts) == 2:
                self.equipped_for_imaging.add(parts[1])
                self.all_rovers.add(parts[1])
            elif predicate == 'store_of' and len(parts) == 3:
                store, rover = parts[1], parts[2]
                self.rover_stores[rover] = store
                self.all_rovers.add(rover)
            elif predicate == 'on_board' and len(parts) == 3:
                camera, rover = parts[1], parts[2]
                if camera not in self.camera_details:
                    self.camera_details[camera] = {'on_board': None, 'calibration_target': None, 'supports': set()}
                self.camera_details[camera]['on_board'] = rover
                self.all_rovers.add(rover)
            elif predicate == 'calibration_target' and len(parts) == 3:
                camera, target = parts[1], parts[2]
                if camera not in self.camera_details:
                    self.camera_details[camera] = {'on_board': None, 'calibration_target': None, 'supports': set()}
                self.camera_details[camera]['calibration_target'] = target
            elif predicate == 'supports' and len(parts) == 3:
                camera, mode = parts[1], parts[2]
                if camera not in self.camera_details:
                    self.camera_details[camera] = {'on_board': None, 'calibration_target': None, 'supports': set()}
                self.camera_details[camera]['supports'].add(mode)
            elif predicate == 'visible_from' and len(parts) == 3:
                objective, waypoint = parts[1], parts[2]
                if objective not in self.objective_visibility:
                    self.objective_visibility[objective] = set()
                self.objective_visibility[objective].add(waypoint)

        # Calculate comm waypoints after finding lander location
        if self.lander_location:
             for fact_str in task.static:
                 parts = get_parts(fact_str)
                 # Check visibility from any waypoint to the lander location
                 if match(parts, 'visible', '*', self.lander_location):
                     self.comm_waypoints.add(parts[1])
                 # Assuming visible is symmetric based on domain examples,
                 # we also check visibility from the lander location to any waypoint
                 # if match(parts, 'visible', self.lander_location, '*'):
                 #     self.comm_waypoints.add(parts[2])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Pre-process state for faster lookups
        state_predicates = {}
        for fact_str in state:
            parts = get_parts(fact_str)
            if not parts: continue # Skip malformed facts
            pred = parts[0]
            args = tuple(parts[1:])
            if pred not in state_predicates:
                state_predicates[pred] = []
            state_predicates[pred].append(args)

        def find_matching_fact_args(predicate, *args):
             """Finds all argument tuples matching the pattern for a given predicate in the state."""
             if predicate not in state_predicates:
                 return []
             target_args = tuple(args)
             results = []
             for fact_args in state_predicates.get(predicate, []): # Use .get for safety
                  if len(fact_args) == len(target_args) and all(fnmatch(f, t) for f, t in zip(fact_args, target_args)):
                      results.append(fact_args)
             return results

        def is_fact_in_state(predicate, *args):
            """Checks if at least one fact matching the pattern exists in the state."""
            return len(find_matching_fact_args(predicate, *args)) > 0

        def find_rover_at_waypoint(waypoint):
            """Finds the first rover located at the given waypoint."""
            for rover, w in find_matching_fact_args('at', '*', '*'):
                if w == waypoint:
                    return rover
            return None

        def find_rover_with_sample(sample_type, waypoint):
             """Finds the first rover with the specified sample."""
             pred = f'have_{sample_type}_analysis'
             for rover, w in find_matching_fact_args(pred, '*', '*'):
                 if w == waypoint:
                     return rover
             return None

        def find_rover_with_image(objective, mode):
             """Finds the first rover with the specified image."""
             for rover, o, m in find_matching_fact_args('have_image', '*', '*', '*'):
                 if o == objective and m == mode:
                     return rover
             return None

        def is_rover_at_waypoint_type(rover, waypoint_type_set):
            """Checks if the given rover is at any waypoint in the provided set."""
            if not rover or not waypoint_type_set: return False
            for w in waypoint_type_set:
                if is_fact_in_state('at', rover, w):
                    return True
            return False

        def is_rover_store_full(rover):
            """Checks if the given rover's store is full."""
            store = self.rover_stores.get(rover)
            if store:
                return is_fact_in_state('full', store)
            return False # Assume not full if no store info or rover doesn't exist

        def is_camera_calibrated(camera, rover):
            """Checks if the given camera on the given rover is calibrated."""
            if not camera or not rover: return False
            return is_fact_in_state('calibrated', camera, rover)

        # --- Heuristic Calculation Logic ---

        for goal_str in self.goals:
            if goal_str in state:
                continue # Goal already achieved

            parts = get_parts(goal_str)
            if not parts: continue # Skip malformed goals

            predicate = parts[0]

            if predicate == 'communicated_soil_data' and len(parts) == 2:
                waypoint = parts[1]
                total_cost += 1 # communicate action

                rover_with_sample = find_rover_with_sample('soil', waypoint)

                if rover_with_sample:
                    # Have sample, need to communicate
                    if not is_rover_at_waypoint_type(rover_with_sample, self.comm_waypoints):
                        total_cost += 1 # navigate to comm location
                else:
                    # Need to sample and communicate
                    total_cost += 1 # sample action
                    # Find an equipped rover that *could* sample and is currently located
                    equipped_rover = next((r for r in self.equipped_for_soil if is_fact_in_state('at', r, '*')), None)
                    if equipped_rover:
                        # Need rover at sample location
                        if not is_fact_in_state('at', equipped_rover, waypoint):
                             total_cost += 1 # navigate to sample location
                        # Need empty store at sample location
                        if is_rover_store_full(equipped_rover):
                             total_cost += 1 # drop action
                        # After sampling (conceptually at waypoint), need to go to comm
                        if not is_rover_at_waypoint_type(equipped_rover, self.comm_waypoints):
                             total_cost += 1 # navigate from sample location to comm location
                    else:
                         # Fallback if no equipped rover is located (should not happen in solvable)
                         # Add costs assuming a rover needs to be moved into position
                         total_cost += 1 # navigate to sample location
                         total_cost += 1 # drop (worst case)
                         total_cost += 1 # navigate to comm location


            elif predicate == 'communicated_rock_data' and len(parts) == 2:
                waypoint = parts[1]
                total_cost += 1 # communicate action

                rover_with_sample = find_rover_with_sample('rock', waypoint)

                if rover_with_sample:
                    # Have sample, need to communicate
                    if not is_rover_at_waypoint_type(rover_with_sample, self.comm_waypoints):
                        total_cost += 1 # navigate to comm location
                else:
                    # Need to sample and communicate
                    total_cost += 1 # sample action
                    # Find an equipped rover that *could* sample and is currently located
                    equipped_rover = next((r for r in self.equipped_for_rock if is_fact_in_state('at', r, '*')), None)
                    if equipped_rover:
                        # Need rover at sample location
                        if not is_fact_in_state('at', equipped_rover, waypoint):
                             total_cost += 1 # navigate to sample location
                        # Need empty store at sample location
                        if is_rover_store_full(equipped_rover):
                             total_cost += 1 # drop action
                        # After sampling (conceptually at waypoint), need to go to comm
                        if not is_rover_at_waypoint_type(equipped_rover, self.comm_waypoints):
                             total_cost += 1 # navigate from sample location to comm location
                    else:
                         # Fallback if no equipped rover is located
                         total_cost += 1 # navigate to sample location
                         total_cost += 1 # drop
                         total_cost += 1 # navigate to comm location


            elif predicate == 'communicated_image_data' and len(parts) == 3:
                objective, mode = parts[1], parts[2]
                total_cost += 1 # communicate action

                rover_with_image = find_rover_with_image(objective, mode)

                if rover_with_image:
                    # Have image, need to communicate
                    if not is_rover_at_waypoint_type(rover_with_image, self.comm_waypoints):
                        total_cost += 1 # navigate to comm location
                else:
                    # Need to take image and communicate
                    total_cost += 1 # take_image action
                    # Find an equipped rover with a camera supporting the mode that is currently located
                    suitable_rover = None
                    suitable_camera = None
                    for camera, details in self.camera_details.items():
                        rover = details.get('on_board')
                        if rover and rover in self.equipped_for_imaging and mode in details.get('supports', set()) and is_fact_in_state('at', rover, '*'):
                             # Found a suitable rover and camera that is located
                             suitable_rover = rover
                             suitable_camera = camera
                             break # Use the first one found

                    if suitable_rover and suitable_camera:
                        is_calibrated = is_camera_calibrated(suitable_camera, suitable_rover)

                        if not is_calibrated:
                            total_cost += 1 # calibrate action
                            # Need rover at calibration waypoint
                            cal_target = self.camera_details[suitable_camera].get('calibration_target')
                            cal_wps = self.objective_visibility.get(cal_target, set()) if cal_target else set()
                            if not is_rover_at_waypoint_type(suitable_rover, cal_wps):
                                 total_cost += 1 # navigate to calibration waypoint

                        # Need rover at image waypoint
                        img_wps = self.objective_visibility.get(objective, set())
                        if not is_rover_at_waypoint_type(suitable_rover, img_wps):
                             total_cost += 1 # navigate to image waypoint

                        # After taking image (conceptually at image waypoint), need to go to comm
                        if not is_rover_at_waypoint_type(suitable_rover, self.comm_waypoints):
                             total_cost += 1 # navigate to comm location
                    else:
                        # Fallback if no suitable equipped and located rover with camera found
                        # Add max costs for this phase
                        total_cost += 1 # calibrate
                        total_cost += 1 # navigate to calibration waypoint
                        total_cost += 1 # navigate to image waypoint
                        total_cost += 1 # navigate to comm location


        return total_cost
