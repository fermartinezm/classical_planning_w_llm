from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Helper function to parse facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match facts
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args): # Strict match on number of arguments
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class roversHeuristic: # Inherit from Heuristic if available
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing up the estimated costs
    for each unachieved goal condition. The cost for each goal is estimated based on
    the minimum number of actions (sample/take_image, calibrate, communicate, drop)
    and a simplified movement cost (1 per significant location change needed).

    # Assumptions
    - Each action (sample, take_image, calibrate, communicate, drop) costs 1.
    - Movement between relevant waypoints (sample location, image location,
      calibration location, communication location) costs 1 per segment if a move is needed.
    - If a sample (soil/rock) was initially present at a waypoint but is no longer
      in the state and no rover has it, it is assumed to have been sampled and
      is waiting for communication by some rover.
    - For image goals, it is assumed that suitable rovers, cameras, image waypoints,
      calibration targets, and calibration waypoints exist if the goal is solvable.
    - Calibration is assumed to be needed before taking an image if the image goal
      is not yet achieved and no rover has the image.

    # Heuristic Initialization
    The constructor extracts static information from the task definition:
    - Which rovers are equipped for soil, rock, and imaging analysis.
    - The mapping from stores to rovers.
    - The pairs of waypoints that are mutually visible (used for communication points).
    - The lander's location.
    - The mapping from cameras to their calibration targets.
    - The mapping from cameras to the rovers they are on board.
    - The mapping from cameras to the modes they support.
    - The mapping from objectives to the waypoints from which they are visible.
    - The initial locations of soil and rock samples.
    - The set of goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is the sum of costs for each goal predicate that is not yet satisfied in the current state.

    For each unachieved `(communicated_soil_data ?w)` goal:
    1. Check if any rover currently has the soil analysis for waypoint `?w` (`(have_soil_analysis ?r ?w)`).
       - If yes: Add 1 (for the `communicate_soil_data` action). Find the rover `?r` that has it. Find its current location. Check if the rover's current location is visible from the lander's location. If not, add 1 (for the necessary `navigate` action to a communication point).
       - If no, check if a soil sample is still present at waypoint `?w` (`(at_soil_sample ?w)`).
         - If yes: Add 1 (for `sample_soil`) + 1 (for `communicate_soil_data`). Find a soil-equipped rover. Check if its store is full; if so, add 1 (for `drop`). Find the rover's current location. Add 1 if the rover is not at waypoint `?w` (move to sample). Check if waypoint `?w` is visible from the lander's location; if not, add 1 (move from sample point to a communication point).
         - If no (sample was initially there but is gone, and no rover has it): Assume it was sampled and needs communication. Add 1 (for `communicate_soil_data`). Find a soil-equipped rover (as a proxy for the one that sampled). Find its current location. Check if the rover's current location is visible from the lander's location; if not, add 1 (move to a communication point).

    For each unachieved `(communicated_rock_data ?w)` goal:
    - Follow the same logic as for soil data, replacing soil with rock and checking for rock-equipped rovers and rock samples.

    For each unachieved `(communicated_image_data ?o ?m)` goal:
    1. Check if any rover currently has the image for objective `?o` in mode `?m` (`(have_image ?r ?o ?m)`).
       - If yes: Add 1 (for `communicate_image_data`). Find the rover `?r` that has it. Find its current location. Check if the rover's current location is visible from the lander's location. If not, add 1 (for the necessary `navigate` action to a communication point).
       - If no: Add 1 (for `calibrate`) + 1 (for `take_image`) + 1 (for `communicate_image_data`).
         - Add move costs: Find *any* imaging-equipped rover `?r_img` with a suitable camera `?i` (on board, supports mode). Find *any* waypoint `?p` visible from `?o`. Find the calibration target `?t` for `?i` and *any* waypoint `?w` visible from `?t`. Find the rover `?r_img`'s current location.
         - Add 1 if the rover's current location is not `?w` (move to calibration point).
         - Add 1 if `?w` is not `?p` (move from calibration to image point).
         - Check if `?p` is visible from the lander's location. If not, add 1 (move from image point to a communication point).

    The total heuristic is the sum of costs calculated for each unachieved goal.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract static information
        self.soil_rovers = set()
        self.rock_rovers = set()
        self.imaging_rovers = set()
        self.store_to_rover = {}
        self.visible_wps = set()
        self.lander_location = None
        self.calibration_targets = {} # camera -> objective (target)
        self.on_board = {} # camera -> rover
        self.supports = {} # camera -> set of modes
        self.visible_from = {} # objective -> set of waypoints
        self.initial_soil_samples = set()
        self.initial_rock_samples = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'equipped_for_soil_analysis':
                self.soil_rovers.add(parts[1])
            elif parts[0] == 'equipped_for_rock_analysis':
                self.rock_rovers.add(parts[1])
            elif parts[0] == 'equipped_for_imaging':
                self.imaging_rovers.add(parts[1])
            elif parts[0] == 'store_of':
                self.store_to_rover[parts[1]] = parts[2]
            elif parts[0] == 'visible':
                self.visible_wps.add((parts[1], parts[2]))
            elif parts[0] == 'at_lander':
                self.lander_location = parts[2]
            elif parts[0] == 'calibration_target':
                self.calibration_targets[parts[1]] = parts[2]
            elif parts[0] == 'on_board':
                self.on_board[parts[1]] = parts[2]
            elif parts[0] == 'supports':
                camera, mode = parts[1], parts[2]
                if camera not in self.supports:
                    self.supports[camera] = set()
                self.supports[camera].add(mode)
            elif parts[0] == 'visible_from':
                objective, waypoint = parts[1], parts[2]
                if objective not in self.visible_from:
                    self.visible_from[objective] = set()
                self.visible_from[objective].add(waypoint)
            elif parts[0] == 'at_soil_sample':
                 self.initial_soil_samples.add(parts[1])
            elif parts[0] == 'at_rock_sample':
                 self.initial_rock_samples.add(parts[1])

        # Pre-calculate communication waypoints (visible from lander)
        self.communication_wps = set()
        if self.lander_location:
             for wp1, wp2 in self.visible_wps:
                 if wp1 == self.lander_location:
                     self.communication_wps.add(wp2)
                 if wp2 == self.lander_location:
                     self.communication_wps.add(wp1)


    def __call__(self, node):
        """Estimate the minimum cost to reach the goal state."""
        state = node.state
        h = 0

        # Extract current state information
        current_rover_locations = {}
        current_store_status = {} # store -> 'empty' or 'full'
        current_have_soil = set() # (rover, waypoint)
        current_have_rock = set() # (rover, waypoint)
        current_have_image = set() # (rover, objective, mode)
        current_calibrated = set() # (camera, rover)
        current_at_soil_sample = set() # waypoint
        current_at_rock_sample = set() # waypoint
        current_communicated_soil = set() # waypoint
        current_communicated_rock = set() # waypoint
        current_communicated_image = set() # (objective, mode)

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1] in self.soil_rovers.union(self.rock_rovers).union(self.imaging_rovers): # Assuming 'at' only applies to rovers here
                 current_rover_locations[parts[1]] = parts[2]
            elif parts[0] == 'empty' and parts[1] in self.store_to_rover:
                 current_store_status[parts[1]] = 'empty'
            elif parts[0] == 'full' and parts[1] in self.store_to_rover:
                 current_store_status[parts[1]] = 'full'
            elif parts[0] == 'have_soil_analysis':
                 current_have_soil.add((parts[1], parts[2]))
            elif parts[0] == 'have_rock_analysis':
                 current_have_rock.add((parts[1], parts[2]))
            elif parts[0] == 'have_image':
                 current_have_image.add((parts[1], parts[2], parts[3]))
            elif parts[0] == 'calibrated':
                 current_calibrated.add((parts[1], parts[2]))
            elif parts[0] == 'at_soil_sample':
                 current_at_soil_sample.add(parts[1])
            elif parts[0] == 'at_rock_sample':
                 current_at_rock_sample.add(parts[1])
            elif parts[0] == 'communicated_soil_data':
                 current_communicated_soil.add(parts[1])
            elif parts[0] == 'communicated_rock_data':
                 current_communicated_rock.add(parts[1])
            elif parts[0] == 'communicated_image_data':
                 current_communicated_image.add((parts[1], parts[2]))

        # Calculate cost for each unachieved goal
        for goal in self.goals:
            goal_parts = get_parts(goal)
            goal_predicate = goal_parts[0]

            if goal_predicate == 'communicated_soil_data':
                waypoint = goal_parts[1]
                if waypoint in current_communicated_soil:
                    continue # Goal achieved

                # Check if sample is already collected by a rover
                rover_with_sample = None
                for r, w in current_have_soil:
                    if w == waypoint:
                        rover_with_sample = r
                        break

                if rover_with_sample:
                    # Need to communicate
                    h += 1 # communicate action
                    rover_loc = current_rover_locations.get(rover_with_sample)
                    if rover_loc and rover_loc not in self.communication_wps:
                         h += 1 # move to communication point
                elif waypoint in current_at_soil_sample:
                    # Need to sample and communicate
                    h += 1 # sample action
                    h += 1 # communicate action

                    # Find a suitable rover (soil-equipped with empty store)
                    suitable_rover = None
                    for r in self.soil_rovers:
                        store = next((s for s, rv in self.store_to_rover.items() if rv == r), None)
                        if store and current_store_status.get(store) == 'empty':
                            suitable_rover = r
                            break

                    if suitable_rover is None:
                        # No rover with empty store, need to drop first?
                        # Find any soil-equipped rover
                        any_soil_rover = next(iter(self.soil_rovers), None)
                        if any_soil_rover:
                             store = next((s for s, rv in self.store_to_rover.items() if rv == any_soil_rover), None)
                             if store and current_store_status.get(store) == 'full':
                                 h += 1 # drop action cost
                             suitable_rover = any_soil_rover # Use this rover for move cost estimation
                        # else: problem likely unsolvable, heuristic might be inaccurate

                    if suitable_rover:
                        rover_loc = current_rover_locations.get(suitable_rover)
                        if rover_loc and rover_loc != waypoint:
                            h += 1 # move to sample point
                        if waypoint not in self.communication_wps:
                            h += 1 # move from sample point to communication point
                    # else: problem likely unsolvable, heuristic might be inaccurate
                else:
                    # Sample was initially at waypoint but is gone, and no rover has it.
                    # Assume it was sampled and needs communication by some rover.
                    h += 1 # communicate action
                    # Estimate move cost: Find any soil-equipped rover and its location
                    any_soil_rover = next(iter(self.soil_rovers), None)
                    if any_soil_rover:
                        rover_loc = current_rover_locations.get(any_soil_rover)
                        if rover_loc and rover_loc not in self.communication_wps:
                            h += 1 # move to communication point
                    # else: problem likely unsolvable, heuristic might be inaccurate


            elif goal_predicate == 'communicated_rock_data':
                waypoint = goal_parts[1]
                if waypoint in current_communicated_rock:
                    continue # Goal achieved

                # Check if sample is already collected by a rover
                rover_with_sample = None
                for r, w in current_have_rock:
                    if w == waypoint:
                        rover_with_sample = r
                        break

                if rover_with_sample:
                    # Need to communicate
                    h += 1 # communicate action
                    rover_loc = current_rover_locations.get(rover_with_sample)
                    if rover_loc and rover_loc not in self.communication_wps:
                         h += 1 # move to communication point
                elif waypoint in current_at_rock_sample:
                    # Need to sample and communicate
                    h += 1 # sample action
                    h += 1 # communicate action

                    # Find a suitable rover (rock-equipped with empty store)
                    suitable_rover = None
                    for r in self.rock_rovers:
                        store = next((s for s, rv in self.store_to_rover.items() if rv == r), None)
                        if store and current_store_status.get(store) == 'empty':
                            suitable_rover = r
                            break

                    if suitable_rover is None:
                        # No rover with empty store, need to drop first?
                        any_rock_rover = next(iter(self.rock_rovers), None)
                        if any_rock_rover:
                             store = next((s for s, rv in self.store_to_rover.items() if rv == any_rock_rover), None)
                             if store and current_store_status.get(store) == 'full':
                                 h += 1 # drop action cost
                             suitable_rover = any_rock_rover # Use this rover for move cost estimation
                        # else: problem likely unsolvable

                    if suitable_rover:
                        rover_loc = current_rover_locations.get(suitable_rover)
                        if rover_loc and rover_loc != waypoint:
                            h += 1 # move to sample point
                        if waypoint not in self.communication_wps:
                            h += 1 # move from sample point to communication point
                    # else: problem likely unsolvable
                else:
                    # Sample was initially at waypoint but is gone, and no rover has it.
                    # Assume it was sampled and needs communication by some rover.
                    h += 1 # communicate action
                    # Estimate move cost: Find any rock-equipped rover and its location
                    any_rock_rover = next(iter(self.rock_rovers), None)
                    if any_rock_rover:
                        rover_loc = current_rover_locations.get(any_rock_rover)
                        if rover_loc and rover_loc not in self.communication_wps:
                            h += 1 # move to communication point
                    # else: problem likely unsolvable


            elif goal_predicate == 'communicated_image_data':
                objective, mode = goal_parts[1], goal_parts[2]
                if (objective, mode) in current_communicated_image:
                    continue # Goal achieved

                # Check if image is already taken by a rover
                rover_with_image = None
                for r, o, m in current_have_image:
                    if o == objective and m == mode:
                        rover_with_image = r
                        break

                if rover_with_image:
                    # Need to communicate
                    h += 1 # communicate action
                    rover_loc = current_rover_locations.get(rover_with_image)
                    if rover_loc and rover_loc not in self.communication_wps:
                         h += 1 # move to communication point
                else:
                    # Need to take image and communicate
                    h += 1 # calibrate action
                    h += 1 # take_image action
                    h += 1 # communicate action

                    # Estimate move costs
                    # Need to visit calibration waypoint, image waypoint, communication waypoint
                    # Find *a* suitable rover/camera pair to estimate moves
                    # Find *an* image waypoint for this objective
                    # Find *a* calibration waypoint for *a* camera that supports the mode

                    # Find *any* imaging-equipped rover
                    img_rover = next(iter(self.imaging_rovers), None)
                    if img_rover:
                        # Find *any* camera on board that supports the mode
                        img_camera = None
                        for camera, rover in self.on_board.items():
                            if rover == img_rover and mode in self.supports.get(camera, set()):
                                img_camera = camera
                                break

                        image_wp = next(iter(self.visible_from.get(objective, set())), None)
                        calib_wp = None
                        if img_camera and img_camera in self.calibration_targets:
                             calib_target = self.calibration_targets[img_camera]
                             calib_wp = next(iter(self.visible_from.get(calib_target, set())), None)

                        # If any required waypoint is missing, heuristic might be inaccurate for unsolvable problems
                        # but for solvable ones, they should exist.
                        if image_wp and calib_wp:
                            rover_loc = current_rover_locations.get(img_rover)
                            if rover_loc:
                                move_cost = 0
                                current_move_loc = rover_loc

                                # Move to calibration waypoint
                                if current_move_loc != calib_wp:
                                    move_cost += 1
                                    current_move_loc = calib_wp

                                # Move to image waypoint
                                if current_move_loc != image_wp:
                                    move_cost += 1
                                    current_move_loc = image_wp

                                # Move to communication waypoint
                                if image_wp not in self.communication_wps: # Check if image waypoint is a comm point
                                    move_cost += 1 # Need to move from image_wp to a comm_wp

                                h += move_cost
                            # else: problem likely unsolvable
                        # else: problem likely unsolvable
                    # else: problem likely unsolvable


        return h
