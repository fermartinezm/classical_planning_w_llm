from fnmatch import fnmatch
# Assuming Heuristic base class is available
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    `fact`: The complete fact as a string.
    `args`: The expected pattern (wildcards `*` allowed).
    Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Using zip means it stops at the shortest list. This works with wildcards like '*'
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to satisfy the goal conditions.
    It counts unsatisfied goals and adds estimated costs for necessary sub-goals
    (sampling, taking images, calibrating), ignoring navigation costs.

    # Assumptions
    - Each required sample (soil or rock) needs a sample action and a communication action.
    - Each required image needs a take_image action and a communication action.
    - Taking an image requires a calibrated camera. If the necessary image hasn't been taken,
      this heuristic checks if *any* suitable camera (on an imaging-equipped rover supporting the mode)
      is calibrated. If none are calibrated, it assumes a calibrate action is needed.
    - It assumes suitable rovers, cameras, and waypoints exist to perform the necessary actions.
    - Navigation costs are ignored.
    - Dropping samples is not explicitly modeled in the cost calculation, assuming stores can be freed up if needed.
    - The existence of soil/rock samples at target waypoints is assumed if the goal requires data from there.
    - The visibility conditions for calibration and imaging are not explicitly checked, only the calibration status.

    # Heuristic Initialization
    - Extracts static information about camera properties (on_board rover, supported modes).
      This information is used to determine which cameras can potentially achieve
      certain image goals and to check calibration status.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize heuristic value `h` to 0.
    2. Identify all goal conditions from `task.goals`.
    3. For each goal condition `g`:
        - If `g` is already present in the current state, it contributes 0 to the heuristic.
        - If `g` is `(communicated_soil_data W)`:
            - Add 1 to `h` for the `communicate_soil_data` action.
            - Check if `(have_soil_analysis R W)` is true for *any* rover R in the current state.
            - If not, add 1 to `h` for the `sample_soil` action.
        - If `g` is `(communicated_rock_data W)`:
            - Add 1 to `h` for the `communicate_rock_data` action.
            - Check if `(have_rock_analysis R W)` is true for *any* rover R in the current state.
            - If not, add 1 to `h` for the `sample_rock` action.
        - If `g` is `(communicated_image_data O M)`:
            - Add 1 to `h` for the `communicate_image_data` action.
            - Check if `(have_image R O M)` is true for *any* rover R in the current state.
            - If not:
                - Add 1 to `h` for the `take_image` action.
                - Check if *any* camera that supports mode M and is on board a rover is currently calibrated in the state.
                - If *no* such camera is calibrated, add 1 to `h` for the `calibrate` action.
    4. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract static information relevant to the heuristic calculation
        self.on_board = {} # camera -> rover
        self.supports = {} # camera -> set of modes

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'on_board':
                self.on_board[parts[1]] = parts[2] # camera -> rover
            elif parts[0] == 'supports':
                camera, mode = parts[1], parts[2]
                if camera not in self.supports:
                    self.supports[camera] = set()
                self.supports[camera].add(mode)

        # We don't need equipped_for_imaging explicitly if we assume
        # any rover with a camera is imaging-equipped.
        # We don't need lander location, visible_from, calibration_target,
        # or visible/can_traverse waypoints for this simple heuristic.


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state
        h = 0

        # Helper to check if soil/rock analysis exists for a waypoint
        def has_analysis(analysis_type, waypoint, current_state):
            predicate = f"have_{analysis_type}_analysis"
            return any(match(fact, predicate, "*", waypoint) for fact in current_state)

        # Helper to check if image exists for an objective/mode
        def has_image(objective, mode, current_state):
             return any(match(fact, "have_image", "*", objective, mode) for fact in current_state)

        # Helper to check if ANY camera supporting the mode and on board a rover is calibrated
        def any_suitable_camera_calibrated(mode, current_state):
            # Find cameras that support the mode
            suitable_cameras = {cam for cam, modes in self.supports.items() if mode in modes}

            for camera in suitable_cameras:
                # Find the rover this camera is on
                rover = self.on_board.get(camera)
                # Check if the camera is on board a rover and is calibrated
                if rover and f"(calibrated {camera} {rover})" in current_state:
                    return True
            return False


        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            predicate = parts[0]

            if predicate == 'communicated_soil_data':
                waypoint = parts[1]
                h += 1 # Cost for communicate action
                if not has_analysis('soil', waypoint, state):
                    h += 1 # Cost for sample action

            elif predicate == 'communicated_rock_data':
                waypoint = parts[1]
                h += 1 # Cost for communicate action
                if not has_analysis('rock', waypoint, state):
                    h += 1 # Cost for sample action

            elif predicate == 'communicated_image_data':
                objective, mode = parts[1], parts[2]
                h += 1 # Cost for communicate action
                if not has_image(objective, mode, state):
                    h += 1 # Cost for take_image action
                    # Check if *any* suitable camera is calibrated.
                    # If NONE are calibrated, we need a calibrate action.
                    if not any_suitable_camera_calibrated(mode, state):
                         h += 1 # Cost for calibrate action

        return h
