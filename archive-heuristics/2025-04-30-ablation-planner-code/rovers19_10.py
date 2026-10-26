from fnmatch import fnmatch

# Assume this base class is provided by the planning framework
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
        # task object is assumed to have attributes like .goals (frozenset of strings)
        # and .static (frozenset of strings for static facts)
        # and .init (frozenset of strings for initial state facts)

    def __call__(self, node):
        # node object is assumed to have attribute .state (frozenset of strings)
        raise NotImplementedError

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or non-fact format gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
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
    This heuristic estimates the number of actions required to achieve all goal
    conditions. It does this by summing up the estimated costs for each
    unachieved goal predicate independently. The cost for each goal is estimated
    based on the sequence of actions needed to achieve it (e.g., navigate,
    sample/image, communicate), assuming a fixed cost for each step and
    ignoring complex interactions or optimal pathfinding.

    # Assumptions
    - Navigation between any two required waypoints is possible and costs a fixed amount (e.g., 1 action).
    - Resource constraints (like store capacity or camera availability/calibration state for multiple image goals) are simplified or ignored when estimating costs for individual goals.
    - For sampling goals, having an empty store available for the equipped rover costs a fixed amount (e.g., 1 action if a drop is needed).
    - For imaging goals, a suitable equipped rover and camera exist for each required image type if the goal is present in the task.
    - The lander location is static and known.

    # Heuristic Initialization
    - Extract the set of goal predicates.
    - Parse static facts to build lookup structures for:
        - Lander location.
        - Rover capabilities (soil, rock, imaging).
        - Store ownership.
        - Camera information (on-board rover, supported modes, calibration target).
        - Objective visibility from waypoints (not strictly used in this simple version, but parsed).
        - Calibration target objective for each camera.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is the sum of estimated costs for each goal predicate
    that is not currently satisfied in the state.

    For each unachieved goal `g`:

    1.  If `g` is `(communicated_soil_data ?w)`:
        -   Add 1 for the `communicate_soil_data` action.
        -   Add 1 for navigating the rover to a communication point (visible from lander).
        -   Check if `(have_soil_analysis ?r ?w)` exists for *any* rover `r` in the state.
        -   If not, the soil sample needs to be collected:
            -   Add 1 for the `sample_soil` action.
            -   Add 1 for navigating the rover to waypoint `?w`.
            -   Add 1 for preparing the store (e.g., dropping if full).

    2.  If `g` is `(communicated_rock_data ?w)`:
        -   Add 1 for the `communicate_rock_data` action.
        -   Add 1 for navigating the rover to a communication point (visible from lander).
        -   Check if `(have_rock_analysis ?r ?w)` exists for *any* rover `r` in the state.
        -   If not, the rock sample needs to be collected:
            -   Add 1 for the `sample_rock` action.
            -   Add 1 for navigating the rover to waypoint `?w`.
            -   Add 1 for preparing the store (e.g., dropping if full).

    3.  If `g` is `(communicated_image_data ?o ?m)`:
        -   Add 1 for the `communicate_image_data` action.
        -   Add 1 for navigating the rover to a communication point (visible from lander).
        -   Check if `(have_image ?r ?o ?m)` exists for *any* rover `r` in the state.
        -   If not, the image needs to be taken:
            -   Add 1 for the `take_image` action.
            -   Add 1 for navigating the rover to an imaging point (visible from objective `?o`).
            -   Find if *any* suitable camera `?i` on a rover `?r` exists that supports mode `?m` and is equipped for imaging (using static facts).
            -   Check if *any* such suitable camera `?i` on its rover `?r` is calibrated in the state.
            -   If no suitable camera is calibrated, calibration is needed for one of them:
                -   Add 1 for the `calibrate` action.
                -   Add 1 for navigating the rover to a calibration point (visible from the calibration target of the chosen camera).

    The total heuristic value is the sum of these estimated costs for all
    unachieved goal predicates. If all goals are achieved, the heuristic is 0.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals

        # Preprocess static facts for quick lookups
        self.lander_location = {}
        self.rover_capabilities = {} # {rover: {soil: bool, rock: bool, imaging: bool}}
        self.store_owner = {} # {store: rover}
        self.camera_info = {} # {camera: {rover: r, modes: {m}, cal_target: t}}
        self.objective_visible_from = {} # {objective: {waypoint}} # Not strictly used, but parsed
        self.cal_target_objective = {} # {camera: objective}

        # Collect all rovers mentioned in static facts first
        all_rovers = set()
        for fact in task.static:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] in ['equipped_for_soil_analysis', 'equipped_for_rock_analysis', 'equipped_for_imaging']:
                 all_rovers.add(parts[1])
             elif parts[0] == 'on_board':
                 all_rovers.add(parts[2])
             elif parts[0] == 'store_of':
                 all_rovers.add(parts[2])

        # Initialize capabilities for all known rovers
        for rover in all_rovers:
             self.rover_capabilities[rover] = {'soil': False, 'rock': False, 'imaging': False}

        # Process static facts
        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or invalid facts

            predicate = parts[0]
            if predicate == 'at_lander':
                lander, waypoint = parts[1], parts[2]
                self.lander_location[lander] = waypoint
            elif predicate == 'equipped_for_soil_analysis':
                 rover = parts[1]
                 self.rover_capabilities[rover]['soil'] = True
            elif predicate == 'equipped_for_rock_analysis':
                 rover = parts[1]
                 self.rover_capabilities[rover]['rock'] = True
            elif predicate == 'equipped_for_imaging':
                 rover = parts[1]
                 self.rover_capabilities[rover]['imaging'] = True
            elif predicate == 'store_of':
                store, rover = parts[1], parts[2]
                self.store_owner[store] = rover
            elif predicate == 'on_board':
                camera, rover = parts[1], parts[2]
                if camera not in self.camera_info:
                    self.camera_info[camera] = {'rover': None, 'modes': set(), 'cal_target': None}
                self.camera_info[camera]['rover'] = rover
            elif predicate == 'supports':
                camera, mode = parts[1], parts[2]
                if camera not in self.camera_info:
                    self.camera_info[camera] = {'rover': None, 'modes': set(), 'cal_target': None}
                self.camera_info[camera]['modes'].add(mode)
            elif predicate == 'calibration_target':
                camera, objective = parts[1], parts[2]
                if camera not in self.camera_info:
                    self.camera_info[camera] = {'rover': None, 'modes': set(), 'cal_target': None}
                self.camera_info[camera]['cal_target'] = objective
                self.cal_target_objective[camera] = objective # Also store camera -> cal_target mapping
            elif predicate == 'visible_from':
                objective, waypoint = parts[1], parts[2]
                if objective not in self.objective_visible_from:
                    self.objective_visible_from[objective] = set()
                self.objective_visible_from[objective].add(waypoint)

        # We don't strictly need visible_waypoints or can_traverse for this simplified heuristic,
        # as navigation cost is a fixed +1 per leg.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0

        # Check if the current state is a goal state
        # Use issubset for frozensets
        if self.goals.issubset(state):
             return 0

        # Find existing analyses and images and calibrated cameras across *all* rovers
        all_known_rovers = set(self.rover_capabilities.keys()) # Use keys from preprocessed static info

        have_soil = {(parts[1], parts[2]) for fact in state if match(fact, "have_soil_analysis", "*", "*")} # (rover, waypoint)
        have_rock = {(parts[1], parts[2]) for fact in state if match(fact, "have_rock_analysis", "*", "*")} # (rover, waypoint)
        have_image = {(parts[1], parts[2], parts[3]) for fact in state if match(fact, "have_image", "*", "*", "*")} # (rover, objective, mode)
        calibrated_cameras = {(parts[1], parts[2]) for fact in state if match(fact, "calibrated", "*", "*")} # (camera, rover)

        for goal_fact_str in self.goals:
            if goal_fact_str in state:
                continue # Goal already achieved

            parts = get_parts(goal_fact_str)
            if not parts: continue # Should not happen for valid goals

            predicate = parts[0]

            if predicate == 'communicated_soil_data':
                waypoint = parts[1]
                # Cost: communicate (1) + nav to lander comm point (1)
                cost_for_goal = 2

                # Check if soil analysis is needed (if *any* rover has it)
                soil_analysis_exists = any((rover, waypoint) in have_soil for rover in all_known_rovers)

                if not soil_analysis_exists:
                    # Cost: sample (1) + nav to sample point (1) + store prep (1)
                    cost_for_goal += 3

                h += cost_for_goal

            elif predicate == 'communicated_rock_data':
                waypoint = parts[1]
                # Cost: communicate (1) + nav to lander comm point (1)
                cost_for_goal = 2

                # Check if rock analysis is needed (if *any* rover has it)
                rock_analysis_exists = any((rover, waypoint) in have_rock for rover in all_known_rovers)

                if not rock_analysis_exists:
                    # Cost: sample (1) + nav to sample point (1) + store prep (1)
                    cost_for_goal += 3

                h += cost_for_goal

            elif predicate == 'communicated_image_data':
                objective, mode = parts[1], parts[2]
                # Cost: communicate (1) + nav to lander comm point (1)
                cost_for_goal = 2

                # Check if image is needed (if *any* rover has it)
                image_exists = any((rover, objective, mode) in have_image for rover in all_known_rovers)

                if not image_exists:
                    # Cost: take image (1) + nav to image point (1)
                    cost_for_goal += 2

                    # Check if calibration is needed for a suitable camera/rover pair
                    # Find *any* camera `i` on a rover `r` such that `(on_board i r)`, `(supports i m)`, `(equipped_for_imaging r)`.
                    # We need to check if *any* such camera `i` on its rover `r` is calibrated.
                    is_calibrated_for_this_image = False
                    suitable_camera_rover_pairs_exist = False

                    for camera, info in self.camera_info.items():
                        rover = info.get('rover')
                        supported_modes = info.get('modes', set())
                        if rover and mode in supported_modes and self.rover_capabilities.get(rover, {}).get('imaging', False):
                             suitable_camera_rover_pairs_exist = True
                             # Check if this specific camera on this rover is calibrated
                             if (camera, rover) in calibrated_cameras: # Corrected tuple order
                                 is_calibrated_for_this_image = True
                                 break # Found a calibrated suitable camera, no need to check others for calibration

                    # If suitable pairs exist, and none are calibrated, calibration is needed
                    if suitable_camera_rover_pairs_exist and not is_calibrated_for_this_image:
                         # Cost: calibrate (1) + nav to cal point (1)
                         cost_for_goal += 2
                    # Note: If suitable_camera_rover_pairs_exist is False, the goal is impossible.
                    # The heuristic assumes solvable problems, so this case shouldn't occur for goal predicates.

                h += cost_for_goal

        return h
