# Need to import Heuristic base class. Assuming it's available in the environment.
# from heuristics.heuristic_base import Heuristic

# Need fnmatch for pattern matching
import fnmatch
import collections

# Assuming Heuristic base class is defined elsewhere and available
# If not, a minimal dummy class would be needed for standalone testing,
# but the problem description implies it's part of a larger framework.
# class Heuristic:
#     def __init__(self, task):
#         pass
#     def __call__(self, node):
#         raise NotImplementedError

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential whitespace issues and ensure robust splitting
    # Check if fact is a string and has expected format before slicing
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        # Handle unexpected fact format, maybe return empty list or raise error
        # For this problem, assuming valid PDDL fact strings.
        return []
    return fact.strip()[1:-1].split()

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
    return all(fnmatch.fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal
    conditions. It counts the necessary high-level steps for each unachieved
    goal fact, such as sampling, imaging, calibrating, communicating, and
    moving to required locations. It does not consider the optimal sequencing
    of actions or resource constraints beyond the need for an empty store
    before sampling and a calibrated camera before imaging.

    # Assumptions
    - The heuristic assumes that all necessary resources (rovers with equipment,
      cameras, stores, landers, sample locations, viewpoints) exist to solve
      the problem. It does not check for unsolvability due to missing resources.
    - It assumes that if a sample was initially present but is no longer in
      the state, the corresponding data exists on some rover (i.e., it was sampled).
    - Movement cost is a fixed value (1) per required location change for
      sampling/imaging and communication, regardless of distance or traversability.
    - Drop action cost (1) is added if *any* rover equipped for the required
      sampling task has a full store, simplifying the check.
    - Calibration cost (1) is added if the specific camera/rover combination
      needed for an image goal is not currently calibrated.

    # Heuristic Initialization
    The heuristic extracts the following information from the task's static facts:
    - The set of goal facts.
    - The initial locations of soil and rock samples.
    - Which rovers are equipped for soil analysis, rock analysis, and imaging.
    - The mapping from rovers to their stores.
    - The locations of landers.
    - The set of waypoints visible from any lander location (communication points).
    - The mapping from objectives to waypoints from which they are visible (viewpoints).
    - Information about cameras: which rover they are on, their calibration target, and supported modes.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic iterates through each goal fact and adds an estimated cost
    if the goal fact is not yet achieved in the current state.

    For each unachieved goal `(communicated_soil_data ?w)`:
    1. Add 1 for the `communicate_soil_data` action.
    2. Add 1 for the `navigate` action required to reach a communication point.
    3. Check if `(have_soil_analysis ?r ?w)` is true for any rover `?r`.
    4. If not (meaning the soil hasn't been analyzed yet):
       a. Check if `(at_soil_sample ?w)` was initially true and is currently true.
       b. If yes (sample is present and needs sampling):
          i. Add 1 for the `sample_soil` action.
          ii. Add 1 for the `navigate` action required to reach waypoint `?w`.
          iii. Check if *any* soil-equipped rover has a full store. If yes, add 1 for a `drop` action.
       c. If no (sample is gone or wasn't initially there): Assume data exists if sample was initially there but is gone; otherwise, goal might be impossible via sampling (heuristic doesn't handle this explicitly, assumes solvability).

    For each unachieved goal `(communicated_rock_data ?w)`:
    - Follow the same logic as for soil data, substituting rock-specific predicates and actions.

    For each unachieved goal `(communicated_image_data ?o ?m)`:
    1. Add 1 for the `communicate_image_data` action.
    2. Add 1 for the `navigate` action required to reach a communication point.
    3. Check if `(have_image ?r ?o ?m)` is true for any rover `?r`.
    4. If not (meaning the image hasn't been taken yet):
       a. Find an imaging-equipped rover `?r` with a camera `?i` that supports mode `?m` and is on board `?r`. (Assumes such a combination exists).
       b. Add 1 for the `take_image` action.
       c. Add 1 for the `navigate` action required to reach a viewpoint `?p` for objective `?o`.
       d. Check if `(calibrated ?i ?r)` is true for the chosen camera `?i` and rover `?r`.
       e. If not calibrated:
          i. Add 1 for the `calibrate` action.
          ii. Add 1 for the `navigate` action required to reach a viewpoint `?w` for the calibration target `?t` of camera `?i`.

    The total heuristic value is the sum of costs for all unachieved goal facts.
    A state is a goal state if and only if all goal facts are present, resulting in a heuristic of 0.
    For solvable non-goal states, the heuristic will be a finite positive integer.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals  # Set of goal facts as strings

        # Extract static information
        static_facts = task.static

        self.initial_soil_samples = {get_parts(fact)[1] for fact in static_facts if match(fact, "at_soil_sample", "*")}
        self.initial_rock_samples = {get_parts(fact)[1] for fact in static_facts if match(fact, "at_rock_sample", "*")}

        self.equipped_for_soil_analysis = {get_parts(fact)[1] for fact in static_facts if match(fact, "equipped_for_soil_analysis", "*")}
        self.equipped_for_rock_analysis = {get_parts(fact)[1] for fact in static_facts if match(fact, "equipped_for_rock_analysis", "*")}
        self.equipped_for_imaging = {get_parts(fact)[1] for fact in static_facts if match(fact, "equipped_for_imaging", "*")}

        self.rover_to_store = {get_parts(fact)[2]: get_parts(fact)[1] for fact in static_facts if match(fact, "store_of", "*", "*")}

        self.lander_locations = {get_parts(fact)[2] for fact in static_facts if match(fact, "at_lander", "*", "*")}

        # Precompute communication points (waypoints visible from any lander)
        self.communication_points = set()
        visible_facts = {fact for fact in static_facts if match(fact, "visible", "*", "*")}
        for lander_loc in self.lander_locations:
             for fact in visible_facts:
                 wp1, wp2 = get_parts(fact)[1:]
                 # (visible ?w1 ?w2) means ?w2 is visible from ?w1
                 # Communication needs (visible ?x ?y) where ?y is lander loc
                 # So rover needs to be at ?x such that ?y is visible from ?x
                 if wp2 == lander_loc:
                     self.communication_points.add(wp1)
        # Note: Lander location itself might be a communication point if visible from itself,
        # but the 'visible' predicate usually connects distinct waypoints.
        # The definition of communication_points seems correct based on the action precondition.


        self.objective_viewpoints = collections.defaultdict(set)
        for fact in static_facts:
            if match(fact, "visible_from", "*", "*"):
                obj, wp = get_parts(fact)[1:]
                self.objective_viewpoints[obj].add(wp)

        self.camera_info = {} # camera -> { 'rover': rover, 'calibration_target': objective, 'modes': {mode, ...} }
        # Collect all camera-related facts first
        camera_on_board = {} # camera -> rover
        camera_cal_target = {} # camera -> objective
        camera_supports = collections.defaultdict(set) # camera -> {mode, ...}

        for fact in static_facts:
            if match(fact, "on_board", "*", "*"):
                camera, rover = get_parts(fact)[1:]
                camera_on_board[camera] = rover
            elif match(fact, "calibration_target", "*", "*"):
                 camera, target = get_parts(fact)[1:]
                 camera_cal_target[camera] = target
            elif match(fact, "supports", "*", "*"):
                 camera, mode = get_parts(fact)[1:]
                 camera_supports[camera].add(mode)

        # Populate camera_info for cameras that are on board a rover
        all_cameras = set(camera_on_board.keys()) | set(camera_cal_target.keys()) | set(camera_supports.keys())
        for camera in all_cameras:
             rover = camera_on_board.get(camera)
             # Only include cameras on board a rover AND that rover is equipped for imaging
             if rover and rover in self.equipped_for_imaging:
                 self.camera_info[camera] = {
                     'rover': rover,
                     'calibration_target': camera_cal_target.get(camera),
                     'modes': camera_supports.get(camera, set())
                 }


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state as a frozenset of strings

        total_cost = 0

        # Check each goal fact
        for goal_fact in self.goals:
            if goal_fact in state:
                continue # Goal already achieved

            # Parse the goal fact
            parts = get_parts(goal_fact)
            predicate = parts[0]

            if predicate == "communicated_soil_data":
                waypoint = parts[1]

                # Cost for communication
                total_cost += 1 # communicate_soil_data action
                # Add cost for moving to a communication point.
                # This is a simplification; ideally, check if any rover with the data is already at one.
                # But we don't track which rover has the data easily here.
                # Just add a fixed cost if communication is needed.
                total_cost += 1 # move to communication point

                # Check if soil analysis data exists on any rover
                has_soil_data = False
                for fact in state:
                    if match(fact, "have_soil_analysis", "*", waypoint):
                        has_soil_data = True
                        break

                # If data does not exist, need to sample
                if not has_soil_data:
                    # Check if sample was initially present and is still there
                    if waypoint in self.initial_soil_samples and f'(at_soil_sample {waypoint})' in state:
                        total_cost += 1 # sample_soil action
                        total_cost += 1 # move to waypoint for sampling

                        # Check if any soil-equipped rover has a full store
                        need_drop = False
                        for rover in self.equipped_for_soil_analysis:
                            store = self.rover_to_store.get(rover)
                            if store and f'(full {store})' in state:
                                need_drop = True
                                break
                        if need_drop:
                            total_cost += 1 # drop action
                    # Note: If sample is gone but data doesn't exist, this heuristic assumes solvability
                    # and might underestimate or be inaccurate. We assume if sample is gone, data exists.


            elif predicate == "communicated_rock_data":
                waypoint = parts[1]

                # Cost for communication
                total_cost += 1 # communicate_rock_data action
                total_cost += 1 # move to communication point

                # Check if rock analysis data exists on any rover
                has_rock_data = False
                for fact in state:
                    if match(fact, "have_rock_analysis", "*", waypoint):
                        has_rock_data = True
                        break

                # If data does not exist, need to sample
                if not has_rock_data:
                    # Check if sample was initially present and is still there
                    if waypoint in self.initial_rock_samples and f'(at_rock_sample {waypoint})' in state:
                        total_cost += 1 # sample_rock action
                        total_cost += 1 # move to waypoint for sampling

                        # Check if any rock-equipped rover has a full store
                        need_drop = False
                        for rover in self.equipped_for_rock_analysis:
                            store = self.rover_to_store.get(rover)
                            if store and f'(full {store})' in state:
                                need_drop = True
                                break
                        if need_drop:
                            total_cost += 1 # drop action


            elif predicate == "communicated_image_data":
                objective = parts[1]
                mode = parts[2]

                # Cost for communication
                total_cost += 1 # communicate_image_data action
                total_cost += 1 # move to communication point

                # Check if image data exists on any rover
                has_image_data = False
                for fact in state:
                    if match(fact, "have_image", "*", objective, mode):
                        has_image_data = True
                        break

                # If data does not exist, need to take image
                if not has_image_data:
                    # Find a suitable camera/rover combination that can take this image
                    # This heuristic simplifies by assuming one exists and calculating cost for it.
                    # A more complex heuristic might find the *best* one (e.g., already calibrated, closest).
                    suitable_camera_rover = None
                    suitable_camera = None # Store camera object name
                    for camera, info in self.camera_info.items():
                        rover = info.get('rover')
                        supported_modes = info.get('modes', set())
                        # Check if camera supports the mode
                        if mode in supported_modes:
                             suitable_camera_rover = (camera, rover)
                             suitable_camera = camera
                             break # Found a suitable one

                    if suitable_camera_rover: # Assume one is found in solvable problems
                        camera, rover = suitable_camera_rover

                        total_cost += 1 # take_image action
                        # Add cost for moving to a viewpoint for the objective
                        # This is a simplification; ideally, check if the rover is already at one.
                        total_cost += 1 # move to viewpoint for objective

                        # Check if the camera is calibrated
                        if f'(calibrated {camera} {rover})' not in state:
                            total_cost += 1 # calibrate action
                            # Add cost for moving to calibration target viewpoint
                            # This is a simplification; ideally, check if the rover is already at one.
                            cal_target = self.camera_info[camera].get('calibration_target')
                            if cal_target: # Assume calibration target exists
                                total_cost += 1 # move to viewpoint for calibration target
                    # Note: If no suitable camera/rover exists, heuristic doesn't handle unsolvability explicitly.


        return total_cost
