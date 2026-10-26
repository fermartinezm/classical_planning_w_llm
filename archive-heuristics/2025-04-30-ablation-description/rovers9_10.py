from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming this is available

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match_fact(fact, *args):
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

# Define the heuristic class
class roversHeuristic(Heuristic): # Inherit from Heuristic
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the cost to reach the goal by counting the number
    of unsatisfied goal conditions and adding estimated costs for necessary
    intermediate steps (sampling, imaging, calibrating, navigating, dropping)
    for each unsatisfied goal. It simplifies navigation cost to 1 if the rover
    is not at the required location for the next step towards a specific goal.

    # Assumptions
    - The heuristic assumes that a suitable rover/camera/waypoint combination
      exists for each required task (sampling, imaging, calibrating) in solvable
      problems. It picks one such combination arbitrarily if multiple exist.
    - Navigation cost is a constant 1 if the rover is not currently at the
      required waypoint for the next step of a goal.
    - The heuristic does not consider optimal resource allocation (e.g., which
      rover is best suited for a task, sharing stores, etc.).
    - The heuristic does not consider pathfinding distance, only whether the
      rover is *at* the required location.
    - The heuristic counts missing state facts or conditions as a proxy for action costs.

    # Heuristic Initialization
    - Extracts static information from the task, including:
        - Rover capabilities (soil, rock, imaging).
        - Store ownership.
        - Camera properties (on-board rover, supported modes, calibration target).
        - Objective visibility from waypoints.
        - Lander location and waypoints visible from the lander.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is the sum of estimated costs for each unsatisfied goal.
    For each goal `G` in the task's goals:
    1. If `G` is already true in the current state, its cost contribution is 0.
    2. If `G` is `(communicated_soil_data W)`:
       - Add 1 (missing `communicated_soil_data` fact).
       - Check if `(have_soil_analysis R W)` is true for *any* rover `R`.
       - If not:
         - Add 1 (missing `have_soil_analysis` fact).
         - Find a soil-equipped rover `R_s`. If `R_s` exists:
           - If `(at R_s W)` is not true, add 1 (missing `at R_s W` fact).
           - Find the store `S` for `R_s`. If `(full S)` is true, add 1 (store is not empty).
       - Else (`(have_soil_analysis R W)` IS true for some `R`):
         - Find *a* rover `R_data` that has `(have_soil_analysis R_data W)`.
         - If `R_data` exists and is not at a communication waypoint, add 1 (missing `at R_data X` fact where X is a comm waypoint).
    3. If `G` is `(communicated_rock_data W)`:
       - Add 1 (missing `communicated_rock_data` fact).
       - Check if `(have_rock_analysis R W)` is true for *any* rover `R`.
       - If not:
         - Add 1 (missing `have_rock_analysis` fact).
         - Find a rock-equipped rover `R_r`. If `R_r` exists:
           - If `(at R_r W)` is not true, add 1 (missing `at R_r W` fact).
           - Find the store `S` for `R_r`. If `(full S)` is true, add 1 (store is not empty).
       - Else (`(have_rock_analysis R W)` IS true for some `R`):
         - Find *a* rover `R_data` that has `(have_rock_analysis R_data W)`.
         - If `R_data` exists and is not at a communication waypoint, add 1 (missing `at R_data X` fact where X is a comm waypoint).
    4. If `G` is `(communicated_image_data O M)`:
       - Add 1 (missing `communicated_image_data` fact).
       - Check if `(have_image R O M)` is true for *any* rover `R`.
       - If not:
         - Add 1 (missing `have_image` fact).
         - Find *a* suitable (rover `R_i`, camera `I`, image waypoint `P`) combination where `R_i` is imaging-equipped, `I` is on `R_i` supporting mode `M`, and `P` is visible from `O`. If found:
           - If `(at R_i P)` is not true, add 1 (missing `at R_i P` fact).
           - Check if `(calibrated I R_i)` is true.
           - If not:
             - Add 1 (missing `calibrated` fact).
             - Find *a* suitable (calibration target `T`, calibration waypoint `W`) setup for camera `I`. If found:
               - If `(at R_i W)` is not true, add 1 (missing `at R_i W` fact).
       - Else (`(have_image R O M)` IS true for some `R`):
         - Find *a* rover `R_data` that has `(have_image R_data O M)`.
         - If `R_data` exists and is not at a communication waypoint, add 1 (missing `at R_data X` fact where X is a comm waypoint).

    The total heuristic value is the sum of costs for all unsatisfied goals.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract static information
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        self.store_to_rover = {}
        self.rover_to_store = {}
        self.camera_to_rover = {}
        self.rover_to_camera = {} # rover -> set of cameras
        self.camera_supports = {} # camera -> set of modes
        self.camera_cal_target = {} # camera -> objective
        self.objective_visible_from = {} # objective -> set of waypoints
        self.lander_location = None
        # self.visible_waypoints = set() # set of (w1, w2) tuples - not strictly needed for this heuristic

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "equipped_for_soil_analysis":
                self.equipped_soil.add(parts[1])
            elif predicate == "equipped_for_rock_analysis":
                self.equipped_rock.add(parts[1])
            elif predicate == "equipped_for_imaging":
                self.equipped_imaging.add(parts[1])
            elif predicate == "store_of":
                store, rover = parts[1], parts[2]
                self.store_to_rover[store] = rover
                self.rover_to_store[rover] = store
            elif predicate == "on_board":
                camera, rover = parts[1], parts[2]
                self.camera_to_rover[camera] = rover
                self.rover_to_camera.setdefault(rover, set()).add(camera)
            elif predicate == "supports":
                camera, mode = parts[1], parts[2]
                self.camera_supports.setdefault(camera, set()).add(mode)
            elif predicate == "calibration_target":
                camera, objective = parts[1], parts[2]
                self.camera_cal_target[camera] = objective
            elif predicate == "visible_from":
                objective, waypoint = parts[1], parts[2]
                self.objective_visible_from.setdefault(objective, set()).add(waypoint)
            elif predicate == "at_lander":
                # Assuming only one lander
                self.lander_location = parts[2]
            # elif predicate == "visible":
            #      self.visible_waypoints.add((parts[1], parts[2]))

        # Pre-calculate waypoints visible from the lander for communication
        self.comm_waypoints = set()
        if self.lander_location:
             for fact in static_facts:
                 # (visible ?x ?y) where ?y is lander_location
                 if match_fact(fact, "visible", "*", self.lander_location):
                     self.comm_waypoints.add(get_parts(fact)[1])
             # Also the lander location itself is a communication point
             self.comm_waypoints.add(self.lander_location)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Helper to find rover location
        def get_rover_location(rover, current_state):
            for fact in current_state:
                if match_fact(fact, "at", rover, "*"):
                    return get_parts(fact)[2]
            return None # Rover location not found (shouldn't happen in valid states)

        # Helper to check if a rover has a full store
        def has_full_store(rover, current_state):
            store = self.rover_to_store.get(rover)
            if store:
                return match_fact(f"(full {store})", "full", store) in current_state
            return False # Rover has no store? (Shouldn't happen)

        # Helper to check if a 'have' predicate is true for any rover
        def have_analysis(pred_name, waypoint, current_state):
            for fact in current_state:
                if match_fact(fact, pred_name, "*", waypoint):
                    return True
            return False

        # Helper to find a rover that has a specific analysis data
        def find_rover_with_analysis(pred_name, waypoint, current_state):
             for fact in current_state:
                 if match_fact(fact, pred_name, "*", waypoint):
                     return get_parts(fact)[1]
             return None

        # Helper to check if a 'have_image' predicate is true for any rover
        def have_image(objective, mode, current_state):
             for fact in current_state:
                 if match_fact(fact, "have_image", "*", objective, mode):
                     return True
             return False

        # Helper to find a rover that has a specific image data
        def find_rover_with_image(objective, mode, current_state):
             for fact in current_state:
                 if match_fact(fact, "have_image", "*", objective, mode):
                     return get_parts(fact)[1]
             return None


        # Helper to check if a camera is calibrated for a rover
        def is_calibrated(camera, rover, current_state):
            return match_fact(f"(calibrated {camera} {rover})", "calibrated", camera, rover) in current_state

        # Helper to check if a rover is at a communication waypoint
        def is_at_comm_waypoint(rover, current_state):
            rover_loc = get_rover_location(rover, current_state)
            return rover_loc in self.comm_waypoints

        # Helper to find a suitable imaging setup (R, I, P) for (O, M)
        # Returns (rover, camera, image_waypoint) or None
        def find_imaging_setup(objective, mode):
             for rover in self.equipped_imaging:
                 cameras_on_r = self.rover_to_camera.get(rover, set())
                 for camera in cameras_on_r:
                     supported_modes = self.camera_supports.get(camera, set())
                     if mode in supported_modes:
                         visible_wps = self.objective_visible_from.get(objective, set())
                         if visible_wps:
                             # Found R, I, and potential Ps. Pick one P.
                             image_waypoint = next(iter(visible_wps))
                             return (rover, camera, image_waypoint)
             return None # No suitable setup found

        # Helper to find a suitable calibration setup (T, W) for (I, R)
        # Returns (cal_target, cal_waypoint) or None
        def find_calibration_setup(camera, rover):
            cal_target = self.camera_cal_target.get(camera)
            if cal_target:
                cal_wps = self.objective_visible_from.get(cal_target, set())
                if cal_wps:
                    # Found T and potential Ws. Pick one W.
                    cal_waypoint = next(iter(cal_wps))
                    return (cal_target, cal_waypoint)
            return None # No suitable setup found


        # Iterate through each goal
        for goal in self.goals:
            if goal in state:
                continue # Goal already satisfied

            parts = get_parts(goal)
            predicate = parts[0]

            if predicate == "communicated_soil_data":
                waypoint = parts[1]
                total_cost += 1 # missing communicated_soil_data fact

                has_data = have_analysis("have_soil_analysis", waypoint, state)
                if not has_data:
                    total_cost += 1 # missing have_soil_analysis fact

                    # Find a soil-equipped rover (assume one exists in solvable problems)
                    suitable_rover = next(iter(self.equipped_soil), None)

                    if suitable_rover:
                        # Check if rover is at the sample location
                        rover_loc = get_rover_location(suitable_rover, state)
                        if rover_loc != waypoint:
                            total_cost += 1 # missing at R_s W fact

                        # Check if rover's store is full
                        if has_full_store(suitable_rover, state):
                            total_cost += 1 # store is not empty (full)

                else: # Data exists, need to communicate
                    data_rover = find_rover_with_analysis("have_soil_analysis", waypoint, state)
                    if data_rover and not is_at_comm_waypoint(data_rover, state):
                        total_cost += 1 # missing at R_data X fact (where X is comm waypoint)


            elif predicate == "communicated_rock_data":
                waypoint = parts[1]
                total_cost += 1 # missing communicated_rock_data fact

                has_data = have_analysis("have_rock_analysis", waypoint, state)
                if not has_data:
                    total_cost += 1 # missing have_rock_analysis fact

                    # Find a rock-equipped rover (assume one exists)
                    suitable_rover = next(iter(self.equipped_rock), None)

                    if suitable_rover:
                        # Check if rover is at the sample location
                        rover_loc = get_rover_location(suitable_rover, state)
                        if rover_loc != waypoint:
                            total_cost += 1 # missing at R_r W fact

                        # Check if rover's store is full
                        if has_full_store(suitable_rover, state):
                            total_cost += 1 # store is not empty (full)

                else: # Data exists, need to communicate
                    data_rover = find_rover_with_analysis("have_rock_analysis", waypoint, state)
                    if data_rover and not is_at_comm_waypoint(data_rover, state):
                        total_cost += 1 # missing at R_data X fact (where X is comm waypoint)


            elif predicate == "communicated_image_data":
                objective, mode = parts[1], parts[2]
                total_cost += 1 # missing communicated_image_data fact

                has_data = have_image(objective, mode, state)
                if not has_data:
                    total_cost += 1 # missing have_image fact

                    # Find a suitable imaging setup (R, I, P)
                    imaging_setup = find_imaging_setup(objective, mode)

                    if imaging_setup:
                        suitable_rover, suitable_camera, image_waypoint = imaging_setup

                        # Check if rover is at the image location
                        rover_loc = get_rover_location(suitable_rover, state)
                        if rover_loc != image_waypoint:
                            total_cost += 1 # missing at R_i P fact

                        # Check if camera is calibrated
                        if not is_calibrated(suitable_camera, suitable_rover, state):
                            total_cost += 1 # missing calibrated fact

                            # Find a suitable calibration setup (T, W) for this camera/rover
                            calibration_setup = find_calibration_setup(suitable_camera, suitable_rover)

                            if calibration_setup:
                                cal_target, cal_waypoint = calibration_setup

                                # Check if rover is at the calibration location
                                if rover_loc != cal_waypoint:
                                    total_cost += 1 # missing at R_i W fact
                            # Else: No cal target/waypoint found (shouldn't happen in solvable problems)
                    # Else: No imaging setup found (shouldn't happen in solvable problems)

                else: # Data exists, need to communicate
                    data_rover = find_rover_with_image(objective, mode, state)
                    if data_rover and not is_at_comm_waypoint(data_rover, state):
                        total_cost += 1 # missing at R_data X fact (where X is comm waypoint)


        return total_cost
