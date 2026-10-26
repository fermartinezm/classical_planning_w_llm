import itertools
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
# Assuming task representation is available as described in the problem description
# from planning_task import Task, Operator # Adjust import if necessary

def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting the string.
    Example: "(at rover1 waypoint1)" -> ["at", "rover1", "waypoint1"]
    """
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern. Wildcards (*) are allowed.
    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern elements (e.g., "at", "rover*", "*").
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by summing the estimated costs for achieving each unsatisfied goal predicate.
    It focuses on the core actions (sample, calibrate, take_image, communicate, drop)
    needed for each goal type, ignoring navigation costs and complex resource allocation
    for efficiency. It considers the current state to determine if prerequisites
    (like having a sample/image or a calibrated camera) are met and adds costs
    for preparatory actions (drop, calibrate) if potentially needed.

    # Assumptions
    - The heuristic assumes that if a goal requires a sample or an image, the
      necessary resources (e.g., `at_soil_sample`, `visible_from`) exist or
      will become available.
    - Navigation costs are ignored to keep the heuristic computation fast.
    - The cost estimation for each goal is done independently, potentially
      overcounting actions that could satisfy multiple goals (e.g., one
      calibration might enable multiple images).
    - The 'drop' action cost is added if *any* capable rover currently has a full store.
    - The 'calibrate' action cost is added if *no* capable rover/camera is currently
      calibrated for the required image mode, but calibration is possible.

    # Heuristic Initialization
    The constructor (`__init__`) preprocesses the static information from the task
    to build data structures for efficient lookup during heuristic evaluation.
    This includes:
    - Identifying rovers, cameras, objectives, modes, waypoints.
    - Mapping rovers to their equipment, stores, and onboard cameras.
    - Mapping cameras to supported modes and calibration targets.
    - Determining which objectives (used for calibration) are visible from which waypoints.
    - Storing the goal predicates.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total estimated cost `h` to 0.
    2. Iterate through each goal predicate `g` defined in the task's goals.
    3. Check if the current state `s` satisfies the goal `g`.
    4. If `g` is *not* satisfied in `s`:
        a. **Identify Goal Type:** Determine if `g` is `communicated_soil_data`,
           `communicated_rock_data`, or `communicated_image_data`.
        b. **Estimate Cost for `communicated_soil_data(w)`:**
           - Check if `(have_soil_analysis ?r w)` exists in `s` for any rover `r`.
           - If yes: Add 1 (for `communicate`).
           - If no: Add 1 (for `sample_soil`) + 1 (for `communicate`).
             Check if *any* rover `r` equipped for soil analysis has a full store
             (`(full (store_of ?s r))`). If yes, add 1 (for `drop`).
             Total cost added: 1 or 2 or 3.
        c. **Estimate Cost for `communicated_rock_data(w)`:**
           - Similar logic to soil data, using `have_rock_analysis`,
             `equipped_for_rock_analysis`, and `sample_rock`.
             Total cost added: 1 or 2 or 3.
        d. **Estimate Cost for `communicated_image_data(o, m)`:**
           - Check if `(have_image ?r o m)` exists in `s` for any rover `r`.
           - If yes: Add 1 (for `communicate`).
           - If no: Add 1 (for `take_image`) + 1 (for `communicate`).
             Check if calibration is potentially needed: Iterate through all rovers `r`
             equipped for imaging and their cameras `c` that support mode `m`.
             If *none* of these cameras are currently `(calibrated c r)` in `s`,
             *but* at least one such camera *can* be calibrated (i.e., its
             `calibration_target` exists and is visible from somewhere),
             add 1 (for `calibrate`).
             Total cost added: 1 or 2 or 3.
    5. Return the total accumulated cost `h`. This value is 0 if and only if
       all goals are satisfied.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static information."""
        self.goals = task.goals
        static_facts = task.static

        # --- Extract object types ---
        self.rovers = set()
        self.cameras = set()
        self.objectives = set()
        self.modes = set()
        self.waypoints = set()
        self.stores = set()
        # Note: We infer objects from predicates, which is typical in PDDL parsers.
        # A more robust way would be to parse the :objects section if available.

        # --- Process static facts ---
        self.rover_equipment = {} # rover -> {'soil': bool, 'rock': bool, 'image': bool}
        self.rover_stores = {}    # rover -> store_name
        self.rover_cameras = {}   # rover -> set(camera_names)
        self.camera_modes = {}    # camera -> {mode: bool}
        self.camera_calibration_target = {} # camera -> objective_name
        self.calibration_visibility = {} # objective -> set(waypoints)

        # Temporary set to store objectives used for calibration
        calibration_targets_objectives = set()

        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]

            # Infer objects and populate basic structures
            if pred == "store_of": # (store_of ?s - store ?r - rover)
                store, rover = parts[1], parts[2]
                self.stores.add(store)
                self.rovers.add(rover)
                self.rover_stores[rover] = store
                if rover not in self.rover_equipment: self.rover_equipment[rover] = {'soil': False, 'rock': False, 'image': False}
                if rover not in self.rover_cameras: self.rover_cameras[rover] = set()
            elif pred == "on_board": # (on_board ?i - camera ?r - rover)
                camera, rover = parts[1], parts[2]
                self.cameras.add(camera)
                self.rovers.add(rover)
                if rover not in self.rover_cameras: self.rover_cameras[rover] = set()
                self.rover_cameras[rover].add(camera)
                if camera not in self.camera_modes: self.camera_modes[camera] = {}
            elif pred == "supports": # (supports ?c - camera ?m - mode)
                camera, mode = parts[1], parts[2]
                self.cameras.add(camera)
                self.modes.add(mode)
                if camera not in self.camera_modes: self.camera_modes[camera] = {}
                self.camera_modes[camera][mode] = True
            elif pred == "calibration_target": # (calibration_target ?i - camera ?o - objective)
                camera, objective = parts[1], parts[2]
                self.cameras.add(camera)
                self.objectives.add(objective)
                self.camera_calibration_target[camera] = objective
                calibration_targets_objectives.add(objective)
            elif pred == "equipped_for_soil_analysis":
                rover = parts[1]
                self.rovers.add(rover)
                if rover not in self.rover_equipment: self.rover_equipment[rover] = {'soil': False, 'rock': False, 'image': False}
                self.rover_equipment[rover]['soil'] = True
            elif pred == "equipped_for_rock_analysis":
                rover = parts[1]
                self.rovers.add(rover)
                if rover not in self.rover_equipment: self.rover_equipment[rover] = {'soil': False, 'rock': False, 'image': False}
                self.rover_equipment[rover]['rock'] = True
            elif pred == "equipped_for_imaging":
                rover = parts[1]
                self.rovers.add(rover)
                if rover not in self.rover_equipment: self.rover_equipment[rover] = {'soil': False, 'rock': False, 'image': False}
                self.rover_equipment[rover]['image'] = True
            elif pred == "visible_from": # (visible_from ?o - objective ?w - waypoint)
                 objective, waypoint = parts[1], parts[2]
                 self.objectives.add(objective)
                 self.waypoints.add(waypoint)
                 # Check if this objective is used for calibration
                 if objective in calibration_targets_objectives:
                     if objective not in self.calibration_visibility:
                         self.calibration_visibility[objective] = set()
                     self.calibration_visibility[objective].add(waypoint)
            elif pred in ["at", "at_lander", "can_traverse", "visible", "at_soil_sample", "at_rock_sample"]:
                 # Infer waypoints and potentially other objects if not seen before
                 for obj in parts[1:]:
                     # Basic type inference based on predicate structure
                     if pred in ["at", "can_traverse"] and obj in self.rovers: continue
                     if pred == "at_lander": continue # Lander type known
                     if pred == "visible_from" and obj in self.objectives: continue
                     # Assume others might be waypoints if not already typed
                     if obj not in self.rovers and obj not in self.cameras and \
                        obj not in self.objectives and obj not in self.modes and \
                        obj not in self.stores:
                         self.waypoints.add(obj)


    def __call__(self, node):
        """Estimate the cost to reach a goal state from the given state node."""
        state = node.state
        cost = 0

        for goal in self.goals:
            if goal not in state:
                parts = get_parts(goal)
                pred = parts[0]

                if pred == "communicated_soil_data":
                    waypoint = parts[1]
                    # Check if analysis is already done
                    has_analysis = any(match(fact, "have_soil_analysis", "*", waypoint) for fact in state)
                    if has_analysis:
                        cost += 1  # Only communication needed
                    else:
                        cost += 1  # sample_soil
                        cost += 1  # communicate_soil_data
                        # Check if drop might be needed
                        any_rover_needs_drop = False
                        for r in self.rovers:
                            # Check if rover is equipped and has a store
                            if self.rover_equipment.get(r, {}).get('soil', False) and r in self.rover_stores:
                                store = self.rover_stores[r]
                                if f"(full {store})" in state:
                                    any_rover_needs_drop = True
                                    break
                        if any_rover_needs_drop:
                            cost += 1 # drop (potential)

                elif pred == "communicated_rock_data":
                    waypoint = parts[1]
                    # Check if analysis is already done
                    has_analysis = any(match(fact, "have_rock_analysis", "*", waypoint) for fact in state)
                    if has_analysis:
                        cost += 1  # Only communication needed
                    else:
                        cost += 1  # sample_rock
                        cost += 1  # communicate_rock_data
                        # Check if drop might be needed
                        any_rover_needs_drop = False
                        for r in self.rovers:
                             # Check if rover is equipped and has a store
                            if self.rover_equipment.get(r, {}).get('rock', False) and r in self.rover_stores:
                                store = self.rover_stores[r]
                                if f"(full {store})" in state:
                                    any_rover_needs_drop = True
                                    break
                        if any_rover_needs_drop:
                            cost += 1 # drop (potential)

                elif pred == "communicated_image_data":
                    objective, mode = parts[1], parts[2]
                    # Check if image is already taken
                    has_image = any(match(fact, "have_image", "*", objective, mode) for fact in state)
                    if has_image:
                        cost += 1  # Only communication needed
                    else:
                        cost += 1  # take_image
                        cost += 1  # communicate_image_data
                        # Check if calibration might be needed
                        calibration_needed = True # Assume needed unless proven otherwise
                        possible_to_calibrate_if_needed = False # Can *any* suitable camera be calibrated?

                        for r in self.rovers:
                            # Check if rover is equipped for imaging
                            if self.rover_equipment.get(r, {}).get('image', False):
                                # Check cameras on this rover
                                for cam in self.rover_cameras.get(r, set()):
                                    # Check if camera supports the mode
                                    if self.camera_modes.get(cam, {}).get(mode, False):
                                        # Found a capable rover/camera combination
                                        # Is it calibrated?
                                        if f"(calibrated {cam} {r})" in state:
                                            calibration_needed = False # Found one calibrated, no extra cost needed
                                            break # Stop checking cameras for this rover

                                        # If not calibrated, check if calibration is possible for *this* camera
                                        cal_target = self.camera_calibration_target.get(cam)
                                        if cal_target and self.calibration_visibility.get(cal_target):
                                            possible_to_calibrate_if_needed = True
                                            # Continue checking other cameras/rovers, maybe one is already calibrated

                                if not calibration_needed:
                                    break # Stop checking rovers

                        # Add calibration cost only if no suitable camera was found calibrated,
                        # AND at least one suitable camera *could* be calibrated.
                        if calibration_needed and possible_to_calibrate_if_needed:
                            cost += 1 # calibrate (potential)

        return cost

