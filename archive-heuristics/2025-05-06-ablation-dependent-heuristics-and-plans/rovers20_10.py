from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and handle potential whitespace
    if not isinstance(fact, str):
        return [] # Or raise an error, depending on expected input
    fact = fact.strip()
    if fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    # Handle facts without parentheses if necessary, though PDDL facts usually have them
    return fact.split()

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
    This heuristic estimates the number of actions required to reach a goal state
    by summing up the estimated costs for each unachieved goal condition.
    The cost for each goal is estimated based on the necessary steps (sampling/imaging,
    calibrating, communicating) and a simplified navigation cost based on the
    types of locations that need to be visited.

    # Assumptions
    - The heuristic assumes that necessary equipment (soil/rock/imaging), cameras,
      stores, calibration targets, and visible locations exist if required by a goal
      in a solvable instance.
    - Navigation between any two required locations is simplified to a fixed cost
      (1) added for each *type* of location visited (sample, image, calibrate, communicate)
      if that type of visit is required for any unachieved goal.
    - Store capacity is effectively 1 (either empty or full) for the purpose of drop cost estimation.
    - Taking an image uncalibrates the camera. Recalibration is needed for subsequent images
      taken with the same camera if no other calibrated camera is available. The heuristic
      simplifies this by adding a calibration cost if *all* suitable cameras for a needed
      image are currently uncalibrated.

    # Heuristic Initialization
    - Extracts static facts such as lander locations, waypoint visibility, sample locations,
      objective visibility, camera properties (on-board, supported modes, calibration target),
      rover capabilities, and store information.
    - Precomputes useful mappings like communication points (waypoints visible from landers).
    - Stores rover capabilities, cameras on board, camera supported modes, and rover stores.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is calculated as the sum of estimated action costs and navigation costs
    for all unachieved goal conditions.

    1.  **Identify Unachieved Goals:** Determine which goal facts from `task.goals` are not present in the current `state`. If none, heuristic is 0.

    2.  **Categorize Pending Tasks:** For each unachieved goal, determine the immediate missing precondition facts that represent required actions:
        -   `communicated_soil_data ?w`: Is `(have_soil_analysis ?r ?w)` true for any rover `?r`? If not, sampling is needed (`S_to_sample`). Communication is always needed if the goal is unachieved (`S_needed`).
        -   `communicated_rock_data ?w`: Similar logic for rock analysis (`R_to_sample`, `R_needed`).
        -   `communicated_image_data ?o ?m`: Is `(have_image ?r ?o ?m)` true for any rover `?r`? If not, taking the image is needed (`I_to_image`). Communication is always needed (`I_needed`). If taking the image is needed, is calibration required? Calibration is needed (`I_to_cal_goals`) if *all* suitable cameras for this `(o, m)` pair (considering rover capabilities and cameras on board) are currently uncalibrated.

    3.  **Calculate Action Costs:** Sum the number of required actions based on the categorized tasks:
        -   `sample_soil`: +1 for each waypoint in `S_to_sample`.
        -   `sample_rock`: +1 for each waypoint in `R_to_sample`.
        -   `take_image`: +1 for each `(o, m)` pair in `I_to_image`.
        -   `calibrate`: +1 for each `(o, m)` pair in `I_to_cal_goals`.
        -   `communicate_soil_data`: +1 for each waypoint in `S_needed`.
        -   `communicate_rock_data`: +1 for each waypoint in `R_needed`.
        -   `communicate_image_data`: +1 for each `(o, m)` pair in `I_needed`.

    4.  **Calculate Drop Cost:** If any soil or rock sample needs to be taken (`|S_to_sample| + |R_to_sample| > 0`) AND there exists at least one rover equipped for soil/rock analysis whose store is currently full (checked against the current state), add 1 for a `drop` action. This is a simplified estimate for needing to empty a store before sampling.

    5.  **Calculate Navigation Cost:** Add a fixed cost (1) for each *type* of location that needs to be visited to achieve the pending tasks:
        -   Add 1 if any sample needs to be taken (`|S_to_sample| + |R_to_sample| > 0`).
        -   Add 1 if any image needs to be taken (`|I_to_image| > 0`).
        -   Add 1 if any calibration is needed (`|I_to_cal_goals| > 0`).
        -   Add 1 if any communication is needed (`|S_needed| + |R_needed| + |I_needed| > 0`).

    6.  **Sum Costs:** The total heuristic value is the sum of action costs, drop cost, and navigation cost. If all goals are achieved, the heuristic is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts and precomputing data structures.
        """
        self.goals = task.goals
        static_facts = task.static

        # Precompute static information
        self.lander_locations = {get_parts(fact)[2] for fact in static_facts if match(fact, "at_lander", "*", "*")}

        # Build visibility graph to find communication points
        visibility_map = {}
        for fact in static_facts:
            if match(fact, "visible", "*", "*"):
                wp1, wp2 = get_parts(fact)[1], get_parts(fact)[2]
                visibility_map.setdefault(wp1, set()).add(wp2)
                visibility_map.setdefault(wp2, set()).add(wp1) # Visibility is symmetric

        self.comm_points = set()
        for lander_loc in self.lander_locations:
             # Any waypoint visible from a lander location is a communication point
             if lander_loc in visibility_map:
                 self.comm_points.update(visibility_map[lander_loc])
             # The lander location itself is a communication point if a rover can be there
             # and communicate from there. The action requires (visible ?x ?y) where ?y is lander loc.
             # So ?x must be visible from ?y. The set of ?x visible from any lander loc ?y are the comm points.
             # Including the lander location itself if visible from itself is redundant.
             # Let's stick to waypoints visible from lander locations as per action definition.

        self.soil_sample_locations = {get_parts(fact)[1] for fact in static_facts if match(fact, "at_soil_sample", "*")}
        self.rock_sample_locations = {get_parts(fact)[1] for fact in static_facts if match(fact, "at_rock_sample", "*")}

        self.objective_image_points = {}
        for fact in static_facts:
            if match(fact, "visible_from", "*", "*"):
                obj, wp = get_parts(fact)[1], get_parts(fact)[2]
                self.objective_image_points.setdefault(obj, set()).add(wp)

        self.camera_cal_target = {}
        for fact in static_facts:
             if match(fact, "calibration_target", "*", "*"):
                 cam, target = get_parts(fact)[1], get_parts(fact)[2]
                 self.camera_cal_target[cam] = target

        self.cal_target_cal_points = {}
        for fact in static_facts:
             if match(fact, "visible_from", "*", "*"):
                 obj = get_parts(fact)[1]
                 # Check if this objective is a calibration target for any camera
                 if obj in self.camera_cal_target.values():
                     target, wp = get_parts(fact)[1], get_parts(fact)[2]
                     self.cal_target_cal_points.setdefault(target, set()).add(wp)

        self.rover_cameras = {}
        self.all_rovers = set()
        for fact in static_facts:
             if match(fact, "on_board", "*", "*"):
                 cam, rover = get_parts(fact)[1], get_parts(fact)[2]
                 self.rover_cameras.setdefault(rover, set()).add(cam)
                 self.all_rovers.add(rover)

        self.camera_supported_modes = {}
        for fact in static_facts:
             if match(fact, "supports", "*", "*"):
                 cam, mode = get_parts(fact)[1], get_parts(fact)[2]
                 self.camera_supported_modes.setdefault(cam, set()).add(mode)

        self.rover_capabilities = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ["equipped_for_soil_analysis", "equipped_for_rock_analysis", "equipped_for_imaging"]:
                cap, rover = parts[0], parts[1]
                self.rover_capabilities.setdefault(rover, set()).add(cap)
                self.all_rovers.add(rover) # Ensure rover is added
            elif parts[0] == "store_of":
                 store, rover = parts[1], parts[2]
                 self.rover_capabilities.setdefault(rover, set()) # Ensure rover is added even if no capability listed
                 self.all_rovers.add(rover)


        self.rover_store = {}
        for fact in static_facts:
             if match(fact, "store_of", "*", "*"):
                 store, rover = get_parts(fact)[1], get_parts(fact)[2]
                 self.rover_store[rover] = store

        # Precompute suitable (camera, rover) pairs for imaging a given mode
        # This doesn't depend on the objective, only on rover equipment and camera support
        self.suitable_imaging_pairs_by_mode = {} # {mode: set((camera, rover))}
        for rover in self.all_rovers:
            if "equipped_for_imaging" in self.rover_capabilities.get(rover, set()):
                for camera in self.rover_cameras.get(rover, set()):
                    for mode in self.camera_supported_modes.get(camera, set()):
                        self.suitable_imaging_pairs_by_mode.setdefault(mode, set()).add((camera, rover))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        goals = self.goals

        # 1. Identify Unachieved Goals
        unachieved_goals = goals - state

        if not unachieved_goals:
            return 0 # Goal reached

        # Initialize costs
        action_cost = 0
        drop_cost = 0
        nav_cost = 0

        # Sets to track what needs sampling, imaging, communicating
        S_needed = set() # waypoints for soil data communication
        R_needed = set() # waypoints for rock data communication
        I_needed = set() # (objective, mode) for image data communication

        # Sets to track what needs sampling/imaging from scratch (haven't got the data yet)
        S_to_sample = set() # waypoints for soil sampling
        R_to_sample = set() # waypoints for rock sampling
        I_to_image = set() # (objective, mode) for taking image

        # Set to track image goals that need calibration because *all* suitable cameras are uncalibrated
        I_to_cal_goals = set() # (objective, mode) tuples

        # Track if any sample is needed for drop cost calculation
        any_sample_needed = False

        # Precompute current 'have' and 'calibrated' facts for quick lookup
        have_soil_facts = {tuple(get_parts(fact)[1:]) for fact in state if match(fact, "have_soil_analysis", "*", "*")}
        have_rock_facts = {tuple(get_parts(fact)[1:]) for fact in state if match(fact, "have_rock_analysis", "*", "*")}
        have_image_facts = {tuple(get_parts(fact)[1:]) for fact in state if match(fact, "have_image", "*", "*", "*")}
        calibrated_facts = {tuple(get_parts(fact)[1:]) for fact in state if match(fact, "calibrated", "*", "*")}
        full_stores = {get_parts(fact)[1] for fact in state if match(fact, "full", "*")}

        # 2. Categorize Pending Tasks & Calculate Action Costs (partial)
        for goal in unachieved_goals:
            parts = get_parts(goal)
            predicate = parts[0]

            if predicate == "communicated_soil_data":
                waypoint = parts[1]
                S_needed.add(waypoint)
                # Check if have_soil_analysis is true for any rover
                if not any((r, waypoint) in have_soil_facts for r in self.all_rovers):
                    S_to_sample.add(waypoint)
                    any_sample_needed = True

            elif predicate == "communicated_rock_data":
                waypoint = parts[1]
                R_needed.add(waypoint)
                # Check if have_rock_analysis is true for any rover
                if not any((r, waypoint) in have_rock_facts for r in self.all_rovers):
                    R_to_sample.add(waypoint)
                    any_sample_needed = True

            elif predicate == "communicated_image_data":
                objective, mode = parts[1], parts[2]
                I_needed.add((objective, mode))
                # Check if have_image is true for any rover
                if not any((r, objective, mode) in have_image_facts for r in self.all_rovers):
                    I_to_image.add((objective, mode))

        # Calculate action costs for sampling, imaging, communicating
        action_cost += len(S_to_sample) # sample_soil
        action_cost += len(R_to_sample) # sample_rock
        action_cost += len(I_to_image)   # take_image
        action_cost += len(S_needed)     # communicate_soil_data
        action_cost += len(R_needed)     # communicate_rock_data
        action_cost += len(I_needed)     # communicate_image_data

        # Calculate calibration cost
        # For each image goal that needs taking, check if calibration is required
        for (objective, mode) in I_to_image:
            # Find all suitable (camera, rover) pairs for this mode
            suitable_ir_pairs = self.suitable_imaging_pairs_by_mode.get(mode, set())

            # If no suitable pair exists, problem is likely unsolvable, return infinity?
            # Assuming solvable instances, this set is not empty if the goal exists.

            # Check if *all* suitable pairs are uncalibrated in the current state
            all_uncalibrated = all((cam, r) not in calibrated_facts for (cam, r) in suitable_ir_pairs)

            if all_uncalibrated:
                 I_to_cal_goals.add((objective, mode))

        action_cost += len(I_to_cal_goals) # calibrate actions

        # 4. Calculate Drop Cost
        if any_sample_needed:
            # Check if any rover equipped for soil/rock has a full store
            for rover in self.all_rovers:
                if ("equipped_for_soil_analysis" in self.rover_capabilities.get(rover, set()) or
                    "equipped_for_rock_analysis" in self.rover_capabilities.get(rover, set())):
                    store = self.rover_store.get(rover)
                    if store and store in full_stores:
                        drop_cost = 1 # Add 1 if at least one such rover needs to drop
                        break # Only need to find one instance

        # 5. Calculate Navigation Cost (Location Types)
        if len(S_to_sample) + len(R_to_sample) > 0:
            nav_cost += 1 # Need to visit sample waypoints
        if len(I_to_image) > 0:
            nav_cost += 1 # Need to visit image waypoints
        if len(I_to_cal_goals) > 0:
            nav_cost += 1 # Need to visit calibration waypoints
        if len(S_needed) + len(R_needed) + len(I_needed) > 0:
            nav_cost += 1 # Need to visit communication waypoints

        # 6. Sum Costs
        total_cost = action_cost + drop_cost + nav_cost

        return total_cost

