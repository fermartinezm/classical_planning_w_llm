from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected fact format, maybe return None or raise error
        # For this domain, facts are expected to be strings like '(predicate arg1 arg2)'
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
    This heuristic estimates the number of actions required to achieve all
    goal conditions. It does this by identifying the sequence of steps needed
    for each unachieved goal (e.g., sample -> analyze -> communicate for soil data,
    calibrate -> take_image -> communicate for image data) and summing the
    estimated cost for the missing steps in each chain. Navigation costs and
    resource costs (like needing an empty store) are added when the prerequisites
    for the next step in a goal chain are not met at the required location or state.

    # Assumptions
    - Each missing action in a goal chain (sample, analyze, communicate, calibrate, take_image) costs 1.
    - Navigation to a required location (sample/analysis/image waypoint, lander location) costs 1 if no suitable rover is already there.
    - Ensuring a rover's store is empty for sampling costs 1 if no suitable rover has an empty store.
    - The heuristic ignores negative effects of actions (e.g., consuming calibration).
    - The heuristic considers if *any* suitable rover/camera can fulfill a requirement, not necessarily a specific one across the whole plan.
    - Admissibility is not required; the focus is on guiding a greedy best-first search effectively.

    # Heuristic Initialization
    The constructor extracts static information from the task, which does not change
    during planning. This includes:
    - The lander's location.
    - Which rovers are equipped for soil analysis, rock analysis, and imaging.
    - The mapping from rovers to their stores.
    - Which cameras are on board which rovers.
    - Which modes each camera supports.
    - Which objective each camera calibrates for.
    - Which waypoints each objective is visible from.
    - Which waypoints have soil or rock samples.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:
    1. Initialize the total heuristic cost to 0.
    2. Extract relevant dynamic information from the current state for quick lookup
       (e.g., current rover locations, which samples/analyses/images are held,
       which cameras are calibrated, which stores are empty).
    3. Iterate through each goal condition defined in the task.
    4. If a goal condition is already true in the current state, it contributes 0 to the heuristic.
    5. If a goal condition is not true, estimate the cost to achieve it based on the missing steps in its required causal chain:

       - **For `(communicated_soil_data ?w)`:**
         - Add 1 for the final `communicate_soil_data` action.
         - Check if `(have_soil_analysis ?r ?w)` is true for any equipped rover `r`. If not:
           - Add 1 for the `analyze_soil_sample` action.
           - Check if `(have_soil_sample ?r ?w)` is true for any equipped rover `r`. If not:
             - Add 1 for the `take_soil_sample` action.
             - Check if any soil-equipped rover is at waypoint `?w`. If not, add 1 (navigate to `?w`).
             - Check if any soil-equipped rover has an empty store. If not, add 1 (empty store).
           - If `(have_soil_sample ?r ?w)` is true for some rover `r`, check if that rover is at waypoint `?w` (needed for analysis). If not, add 1 (navigate to `?w`).
         - If `(have_soil_analysis ?r ?w)` is true for some rover `r`, check if that rover is at the lander location (needed for communication). If not, add 1 (navigate to lander location).

       - **For `(communicated_rock_data ?w)`:** (Similar logic as soil data, using rock-specific predicates and equipment)
         - Add 1 for the final `communicate_rock_data` action.
         - Check if `(have_rock_analysis ?r ?w)` is true for any equipped rover `r`. If not:
           - Add 1 for the `analyze_rock_sample` action.
           - Check if `(have_rock_sample ?r ?w)` is true for any equipped rover `r`. If not:
             - Add 1 for the `take_rock_sample` action.
             - Check if any rock-equipped rover is at waypoint `?w`. If not, add 1 (navigate to `?w`).
             - Check if any rock-equipped rover has an empty store. If not, add 1 (empty store).
           - If `(have_rock_sample ?r ?w)` is true for some rover `r`, check if that rover is at waypoint `?w` (needed for analysis). If not, add 1 (navigate to `?w`).
         - If `(have_rock_analysis ?r ?w)` is true for some rover `r`, check if that rover is at the lander location (needed for communication). If not, add 1 (navigate to lander location).

       - **For `(communicated_image_data ?o ?m)`:**
         - Add 1 for the final `communicate_image_data` action.
         - Check if `(have_image ?r ?o ?m)` is true for any equipped rover `r`. If not:
           - Add 1 for the `take_image` action.
           - Check if a suitable camera (on an imaging rover, supporting mode `?m`, calibrating for objective `?o`) is calibrated. If not:
             - Add 1 for the `calibrate` action.
             - Check if an imaging rover with a suitable camera is at a waypoint visible from objective `?o`. If not, add 1 (navigate to a visible waypoint).
           - If a suitable camera is calibrated, check if an imaging rover with that calibrated camera is at a waypoint visible from objective `?o` (needed for taking the image). If not, add 1 (navigate to a visible waypoint).
         - If `(have_image ?r ?o ?m)` is true for some rover `r`, check if that rover is at the lander location (needed for communication). If not, add 1 (navigate to lander location).

    6. The total accumulated cost is the heuristic value for the state. This value is 0 if and only if all goals are already achieved.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts.

        @param task: The planning task object containing initial state, goals, and static facts.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Extract static information into useful data structures
        self.lander_location = None
        self.equipped_rovers = {'soil': set(), 'rock': set(), 'imaging': set()}
        self.rover_stores = {}  # rover -> store
        self.cameras_on_board = {}  # camera -> rover
        self.camera_supports = {}  # camera -> set of modes
        self.calibration_targets = {}  # camera -> objective
        self.visible_from = {}  # objective -> set of waypoints
        self.at_sample_locations = {'soil': set(), 'rock': set()} # Not strictly needed for this heuristic logic, but good to have

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'at_lander':
                self.lander_location = parts[2]
            elif predicate == 'equipped_for_soil_analysis':
                self.equipped_rovers['soil'].add(parts[1])
            elif predicate == 'equipped_for_rock_analysis':
                self.equipped_rovers['rock'].add(parts[1])
            elif predicate == 'equipped_for_imaging':
                self.equipped_rovers['imaging'].add(parts[1])
            elif predicate == 'store_of':
                # Fact is (store_of <store> <rover>)
                self.rover_stores[parts[2]] = parts[1]
            elif predicate == 'on_board':
                # Fact is (on_board <camera> <rover>)
                self.cameras_on_board[parts[1]] = parts[2]
            elif predicate == 'supports':
                # Fact is (supports <camera> <mode>)
                camera, mode = parts[1], parts[2]
                if camera not in self.camera_supports:
                    self.camera_supports[camera] = set()
                self.camera_supports[camera].add(mode)
            elif predicate == 'calibration_target':
                # Fact is (calibration_target <camera> <objective>)
                self.calibration_targets[parts[1]] = parts[2]
            elif predicate == 'visible_from':
                # Fact is (visible_from <objective> <waypoint>)
                objective, waypoint = parts[1], parts[2]
                if objective not in self.visible_from:
                    self.visible_from[objective] = set()
                self.visible_from[objective].add(waypoint)
            elif predicate == 'at_soil_sample':
                self.at_sample_locations['soil'].add(parts[1])
            elif predicate == 'at_rock_sample':
                self.at_sample_locations['rock'].add(parts[1])

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach a goal state.

        @param node: The search node containing the current state.
        @return: The estimated heuristic cost (non-negative integer).
        """
        state = node.state  # Current world state as a frozenset of strings.
        total_cost = 0  # Initialize action cost counter.

        # Pre-calculate current states for quick lookup
        current_rover_locations = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "at", "rover*", "*")}
        current_have_soil_sample = {(get_parts(fact)[1], get_parts(fact)[2]) for fact in state if match(fact, "have_soil_sample", "*", "*")}  # (rover, waypoint)
        current_have_rock_sample = {(get_parts(fact)[1], get_parts(fact)[2]) for fact in state if match(fact, "have_rock_sample", "*", "*")}  # (rover, waypoint)
        current_have_soil_analysis = {(get_parts(fact)[1], get_parts(fact)[2]) for fact in state if match(fact, "have_soil_analysis", "*", "*")}  # (rover, waypoint)
        current_have_rock_analysis = {(get_parts(fact)[1], get_parts(fact)[2]) for fact in state if match(fact, "have_rock_analysis", "*", "*")}  # (rover, waypoint)
        current_calibrated_cameras = {(get_parts(fact)[1], get_parts(fact)[2]) for fact in state if match(fact, "calibrated", "*", "*")}  # (camera, rover)
        current_have_image = {(get_parts(fact)[1], get_parts(fact)[2], get_parts(fact)[3]) for fact in state if match(fact, "have_image", "*", "*", "*")}  # (rover, objective, mode)
        current_empty_stores = {get_parts(fact)[1] for fact in state if match(fact, "empty", "*")}

        for goal in self.goals:
            # If the goal is already achieved, it contributes 0 cost.
            if goal in state:
                continue

            # Parse the goal predicate and arguments
            parts = get_parts(goal)
            if not parts: continue # Skip malformed goals

            predicate = parts[0]

            if predicate == "communicated_soil_data":
                w = parts[1]
                total_cost += 1  # Cost for the final communicate action

                # Check if analysis is done for this waypoint by *any* equipped rover
                analysis_done = any((r, w) in current_have_soil_analysis for r in self.equipped_rovers['soil'])

                if not analysis_done:
                    total_cost += 1  # Cost for the analyze action

                    # Check if sample is taken for this waypoint by *any* equipped rover
                    sample_taken = any((r, w) in current_have_soil_sample for r in self.equipped_rovers['soil'])

                    if not sample_taken:
                        total_cost += 1  # Cost for the sample action

                        # Need an equipped rover at sample location w
                        rover_at_w = any(current_rover_locations.get(r) == w for r in self.equipped_rovers['soil'])
                        if not rover_at_w:
                            total_cost += 1  # Cost for navigate to w

                        # Need an empty store on an equipped rover for sampling
                        store_empty = any(self.rover_stores.get(r) in current_empty_stores for r in self.equipped_rovers['soil'])
                        if not store_empty:
                            total_cost += 1  # Cost for emptying store (simplified)

                    # Navigation cost for Analyze step: Need equipped rover with sample at sample location w
                    # Add navigation cost only if sample exists but the rover is not at the location for analysis
                    rover_with_sample_at_w = any((r, w) in current_have_soil_sample and current_rover_locations.get(r) == w for r in self.equipped_rovers['soil'])
                    if sample_taken and not rover_with_sample_at_w:
                        total_cost += 1  # Cost for navigate to w

                # Navigation cost for Communicate step: Need rover with analysis at lander location
                # Add navigation cost only if analysis exists but the rover is not at the lander location for communication
                rover_with_analysis_at_lander = any((r, w) in current_have_soil_analysis and current_rover_locations.get(r) == self.lander_location for r in self.equipped_rovers['soil'])
                if analysis_done and not rover_with_analysis_at_lander:
                    total_cost += 1  # Cost for navigate to lander

            elif predicate == "communicated_rock_data":
                w = parts[1]
                total_cost += 1  # Communicate action

                # Check if analysis is done for this waypoint by *any* equipped rover
                analysis_done = any((r, w) in current_have_rock_analysis for r in self.equipped_rovers['rock'])

                if not analysis_done:
                    total_cost += 1  # Analyze action

                    # Check if sample is taken for this waypoint by *any* equipped rover
                    sample_taken = any((r, w) in current_have_rock_sample for r in self.equipped_rovers['rock'])

                    if not sample_taken:
                        total_cost += 1  # Sample action

                        # Need an equipped rover at sample location w
                        rover_at_w = any(current_rover_locations.get(r) == w for r in self.equipped_rovers['rock'])
                        if not rover_at_w:
                            total_cost += 1  # Navigate to w

                        # Need an empty store on an equipped rover for sampling
                        store_empty = any(self.rover_stores.get(r) in current_empty_stores for r in self.equipped_rovers['rock'])
                        if not store_empty:
                            total_cost += 1  # Empty store (simplified)

                    # Navigation cost for Analyze step: Need equipped rover with sample at sample location w
                    rover_with_sample_at_w = any((r, w) in current_have_rock_sample and current_rover_locations.get(r) == w for r in self.equipped_rovers['rock'])
                    if sample_taken and not rover_with_sample_at_w:
                        total_cost += 1  # Navigate to w

                # Navigation cost for Communicate step: Need rover with analysis at lander location
                rover_with_analysis_at_lander = any((r, w) in current_have_rock_analysis and current_rover_locations.get(r) == self.lander_location for r in self.equipped_rovers['rock'])
                if analysis_done and not rover_with_analysis_at_lander:
                    total_cost += 1  # Navigate to lander

            elif predicate == "communicated_image_data":
                o, m = parts[1], parts[2]
                total_cost += 1  # Communicate action

                # Check if image is taken for this objective and mode by *any* equipped rover
                image_taken = any((r, o, m) in current_have_image for r in self.equipped_rovers['imaging'])

                if not image_taken:
                    total_cost += 1  # Take image action

                    # Check if a suitable camera on an imaging rover is calibrated for this objective and mode
                    # Suitable camera c on rover r: on_board c r, equipped_for_imaging r, supports c m, calibration_target c o
                    suitable_calibrated_camera_on_imaging_rover = any(
                        (c, r) in current_calibrated_cameras and
                        r in self.equipped_rovers['imaging'] and
                        self.cameras_on_board.get(c) == r and
                        m in self.camera_supports.get(c, set()) and
                        self.calibration_targets.get(c) == o
                        for c in self.cameras_on_board # Iterate through all cameras
                    )

                    if not suitable_calibrated_camera_on_imaging_rover:
                        total_cost += 1  # Calibrate action

                        # Need imaging rover with suitable camera at a waypoint visible from objective o
                        # A waypoint w is suitable if visible_from o w
                        suitable_waypoints = self.visible_from.get(o, set())
                        # Find if any imaging rover has a camera that supports mode m and calibrates for objective o
                        imaging_rovers_with_suitable_camera = {
                            r for r in self.equipped_rovers['imaging']
                            if any(self.cameras_on_board.get(c) == r and m in self.camera_supports.get(c, set()) and self.calibration_targets.get(c) == o for c in self.cameras_on_board)
                        }
                        rover_at_suitable_waypoint = any(
                            current_rover_locations.get(r) in suitable_waypoints
                            for r in imaging_rovers_with_suitable_camera
                        )

                        if not rover_at_suitable_waypoint:
                            total_cost += 1  # Navigate to suitable waypoint

                    # Navigation cost for Take Image step: Need calibrated imaging rover with suitable camera at a waypoint visible from objective o
                    # Add navigation cost only if calibrated camera exists but the rover is not at the location for taking image
                    suitable_waypoints = self.visible_from.get(o, set())
                    calibrated_rover_at_suitable_waypoint = any(
                        current_rover_locations.get(r) in suitable_waypoints and
                        r in self.equipped_rovers['imaging'] and
                        any((c, r) in current_calibrated_cameras and self.cameras_on_board.get(c) == r and m in self.camera_supports.get(c, set()) and self.calibration_targets.get(c) == o for c in self.cameras_on_board)
                        for r in self.equipped_rovers['imaging']
                    )
                    if suitable_calibrated_camera_on_imaging_rover and not calibrated_rover_at_suitable_waypoint:
                        total_cost += 1  # Navigate to suitable waypoint

                # Navigation cost for Communicate step: Need rover with image at lander location
                # Add navigation cost only if image exists but the rover is not at the lander location for communication
                rover_with_image_at_lander = any((r, o, m) in current_have_image and current_rover_locations.get(r) == self.lander_location for r in self.equipped_rovers['imaging'])
                if image_taken and not rover_with_image_at_lander:
                    total_cost += 1  # Navigate to lander

        return total_cost

