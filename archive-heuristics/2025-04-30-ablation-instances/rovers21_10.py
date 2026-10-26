from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(at rover1 waypoint2)" -> ["at", "rover1", "waypoint2"]
    if not fact or fact[0] != '(' or fact[-1] != ')':
        # Handle potential malformed facts gracefully, though planner should provide valid ones
        return []
    return fact[1:-1].split()

def match_pattern(fact, pattern_parts):
    """
    Check if a PDDL fact matches a given pattern represented as a list of strings.
    Example: match_pattern("(at rover1 waypoint2)", ["at", "*", "waypoint2"])
    """
    fact_parts = get_parts(fact)
    if len(fact_parts) != len(pattern_parts):
        return False
    return all(fnmatch(f_part, p_part) for f_part, p_part in zip(fact_parts, pattern_parts))

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal
    conditions. It sums up the estimated cost for each unachieved goal fact
    independently, based on the current state of the necessary data/images
    and a simplified estimate of navigation and intermediate steps.

    # Assumptions
    - Each unachieved goal fact (communicated data/image) contributes to the heuristic.
    - The cost for each goal fact is estimated based on whether the required
      sample/image is collected/taken, and whether a camera is calibrated for images.
    - Navigation is estimated with a fixed cost (e.g., 1 action per required move)
      rather than shortest path distance.
    - Resource constraints (like empty stores, specific rover/camera availability)
      are largely ignored, assuming *some* suitable rover/camera exists and can be used.
    - Interactions between goals (e.g., one navigation serving multiple tasks,
      recalibrating camera for multiple images) are simplified or ignored.
    - Assumes soil/rock samples remain available at their initial location until collected.
      If a sample is gone and the goal isn't met, it assumes the sample was collected
      and is held by a rover.
    - Assumes static facts accurately describe the fixed configuration (e.g., which camera is on which rover, which modes are supported).

    # Heuristic Initialization
    - Extract the goal conditions from the task.
    - Pre-process static facts to quickly look up camera capabilities (which camera on which rover supports which modes) and calibration targets. This information is needed to determine if a suitable camera exists and what is required for calibration. Lander location and visible waypoints are also stored, although the simplified navigation cost doesn't strictly require the full graph.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, iterate through each goal fact defined in the task.
    If a goal fact is already present in the current state, its contribution is 0.
    If a goal fact is *not* present in the current state, estimate the remaining
    cost based on the type of goal:

    1.  **For a `(communicated_soil_data ?w)` goal not in state:**
        -   Check if `(have_soil_analysis ?r ?w)` is true for *any* rover `?r` in the state.
        -   If yes: The data is collected. Remaining steps: Navigate to a communication point and communicate. Estimated cost: 1 (navigate) + 1 (communicate) = 2.
        -   If no: The data needs to be collected. This requires navigating to `?w`, sampling, navigating to a communication point, and communicating. Estimated cost: 1 (nav to sample) + 1 (sample) + 1 (nav to comm) + 1 (communicate) = 4. (Assumes sample is at `?w` and a rover is equipped and has an empty store).

    2.  **For a `(communicated_rock_data ?w)` goal not in state:**
        -   Check if `(have_rock_analysis ?r ?w)` is true for *any* rover `?r` in the state.
        -   If yes: Estimated cost: 1 (navigate) + 1 (communicate) = 2.
        -   If no: Estimated cost: 1 (nav to sample) + 1 (sample) + 1 (nav to comm) + 1 (communicate) = 4. (Assumes sample is at `?w` and a rover is equipped and has an empty store).

    3.  **For a `(communicated_image_data ?o ?m)` goal not in state:**
        -   Check if `(have_image ?r ?o ?m)` is true for *any* rover `?r` in the state.
        -   If yes: The image is taken. Remaining steps: Navigate to a communication point and communicate. Estimated cost: 1 (navigate) + 1 (communicate) = 2.
        -   If no: The image needs to be taken. This requires navigating to an image point, taking the image, navigating to a communication point, and communicating. Taking the image requires a calibrated camera.
            -   Check if `(calibrated ?i ?r)` is true for *any* camera `?i` on *any* rover `?r` that supports mode `?m` (using static facts) and is currently calibrated in the state.
            -   If a suitable camera is calibrated: Estimated cost: 1 (nav to image) + 1 (take image) + 1 (nav to comm) + 1 (communicate) = 4.
            -   If no suitable camera is calibrated: Calibration is needed first. This requires navigating to a calibration point and calibrating. Estimated cost: 1 (nav to cal) + 1 (calibrate) + 1 (nav to image) + 1 (take image) + 1 (nav to comm) + 1 (communicate) = 6. (Assumes a suitable camera/rover exists and calibration target is reachable).

    The total heuristic value is the sum of the estimated costs for all unachieved goal facts.
    A goal state is one where all goal facts are in the state, resulting in a heuristic of 0.
    For solvable states, the heuristic value will be finite.
    The heuristic is efficiently computable as it involves iterating through goals and state facts.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static_facts = task.static

        # Pre-process static facts for quick lookups
        # Stores (camera, rover) -> set of modes supported by this camera on this rover
        self._camera_capabilities = {}
        self._camera_cal_target = {} # camera -> target_objective
        self._objective_visible_from = {} # objective -> set of waypoints
        self._lander_location = None
        self._waypoints_visible_from_lander = set() # Waypoints visible from lander

        # Temporary storage to link cameras, rovers, and modes
        camera_on_rover = {} # camera -> rover
        camera_supports_mode = {} # camera -> set of modes

        for fact in self.static_facts:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == 'on_board' and len(parts) == 3:
                camera, rover = parts[1], parts[2]
                camera_on_rover[camera] = rover
            elif parts[0] == 'supports' and len(parts) == 3:
                camera, mode = parts[1], parts[2]
                if camera not in camera_supports_mode:
                    camera_supports_mode[camera] = set()
                camera_supports_mode[camera].add(mode)
            elif parts[0] == 'calibration_target' and len(parts) == 3:
                camera, target = parts[1], parts[2]
                self._camera_cal_target[camera] = target
            elif parts[0] == 'visible_from' and len(parts) == 3:
                objective, waypoint = parts[1], parts[2]
                if objective not in self._objective_visible_from:
                    self._objective_visible_from[objective] = set()
                self._objective_visible_from[objective].add(waypoint)
            elif parts[0] == 'at_lander' and len(parts) == 3:
                 # Assuming only one lander
                 self._lander_location = parts[2]

        # Build _camera_capabilities structure
        for camera, rover in camera_on_rover.items():
             if camera in camera_supports_mode:
                  self._camera_capabilities[(camera, rover)] = camera_supports_mode[camera]
             else:
                  self._camera_capabilities[(camera, rover)] = set() # Camera on board but supports no modes?

        # Find waypoints visible from lander location
        if self._lander_location:
             for fact in self.static_facts:
                 parts = get_parts(fact)
                 if not parts: continue
                 # visible is symmetric, check both directions
                 if parts[0] == 'visible' and len(parts) == 3:
                     w1, w2 = parts[1], parts[2]
                     if w1 == self._lander_location:
                         self._waypoints_visible_from_lander.add(w2)
                     if w2 == self._lander_location:
                         self._waypoints_visible_from_lander.add(w1)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Helper to check if a pattern exists in the state
        def state_has_pattern(state, pattern_parts):
            return any(match_pattern(fact, pattern_parts) for fact in state)

        # Helper to check if any suitable camera is calibrated for an image goal
        def any_suitable_camera_calibrated(state, mode):
            # Iterate through all cameras on board any rover with known capabilities
            for (camera, rover), supported_modes in self._camera_capabilities.items():
                # Check if this camera supports the required mode
                if mode in supported_modes:
                    # Check if this specific camera on this specific rover is calibrated in the current state
                    if f'(calibrated {camera} {rover})' in state:
                        return True
            return False

        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            goal_parts = get_parts(goal)
            if not goal_parts: continue

            predicate = goal_parts[0]

            if predicate == 'communicated_soil_data' and len(goal_parts) == 2:
                waypoint = goal_parts[1]
                # Check if soil analysis is already done by any rover
                if state_has_pattern(state, ['have_soil_analysis', '*', waypoint]):
                    # Data collected, need to navigate to comm point and communicate
                    total_cost += 2 # navigate + communicate
                else:
                    # Need to sample and communicate
                    # Assumes sample is available at waypoint
                    total_cost += 4 # nav to sample + sample + nav to comm + communicate

            elif predicate == 'communicated_rock_data' and len(goal_parts) == 2:
                waypoint = goal_parts[1]
                 # Check if rock analysis is already done by any rover
                if state_has_pattern(state, ['have_rock_analysis', '*', waypoint]):
                    # Data collected, need to navigate to comm point and communicate
                    total_cost += 2 # navigate + communicate
                else:
                    # Need to sample and communicate
                    # Assumes sample is available at waypoint
                    total_cost += 4 # nav to sample + sample + nav to comm + communicate

            elif predicate == 'communicated_image_data' and len(goal_parts) == 3:
                objective, mode = goal_parts[1], goal_parts[2]
                # Check if image is already taken by any rover
                if state_has_pattern(state, ['have_image', '*', objective, mode]):
                    # Image taken, need to navigate to comm point and communicate
                    total_cost += 2 # navigate + communicate
                else:
                    # Need to take image and communicate
                    # Taking image requires calibrated camera
                    if any_suitable_camera_calibrated(state, mode):
                        # Suitable camera is calibrated
                        total_cost += 4 # nav to image + take image + nav to comm + communicate
                    else:
                        # Need to calibrate, take image, and communicate
                        total_cost += 6 # nav to cal + calibrate + nav to image + take image + nav to comm + communicate

        return total_cost
