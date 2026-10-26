from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if fact has fewer parts than args
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the remaining effort to achieve all communication goals
    by summing up the estimated costs for each unachieved goal. The cost for each
    goal is based on the deepest prerequisite not yet met (e.g., sample collected,
    analysis done, image taken, calibrated) plus an estimated number of moves
    required to reach the necessary locations (sample/image/calibration site,
    lander).

    # Assumptions
    - Each unachieved communication goal contributes independently to the heuristic.
    - The cost of collecting a sample, analyzing data, calibrating a camera,
      taking an image, and communicating data is 1 action each.
    - The cost of moving between any two relevant waypoints (sample/image/calibration
      site, lander location) is a fixed value (estimated as 1 per leg of the journey).
    - The heuristic does not consider resource constraints like rover capabilities
      (beyond checking for existence of required data/calibration), store capacity,
      or pathfinding distance. It assumes a suitable rover exists and can reach
      the required locations in the estimated number of moves.
    - Static facts required for goals (like `at_soil_sample`, `visible_from`,
      `calibration_target`, `equipped_for_...`, `on_board`, `supports`, `at_lander`)
      are assumed to be present if the goal is achievable. The heuristic only
      checks for dynamic facts (`have_...`, `calibrated`, `at`) in the state.

    # Heuristic Initialization
    - Stores the set of goal predicates.
    - Extracts static information about camera capabilities (`supports`),
      calibration targets (`calibration_target`), and camera-rover assignments
      (`on_board`) to efficiently check image goal prerequisites.
    - Extracts the lander's location.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. Pre-process the current state to quickly check for the existence of
       intermediate facts like `have_soil_analysis`, `have_image`, `calibrated`, etc.
       This avoids repeated iteration over the state facts for each goal.
    3. Iterate through each goal predicate in the task's goals.
    4. For each goal predicate, check if it is already true in the current state.
    5. If the goal predicate is NOT true in the current state, calculate its contribution to the heuristic:
       a. Add a base cost of 1 for the final communication action.
       b. Determine the "deepest" unmet prerequisite for this goal type and add costs for the necessary actions and estimated moves:
          - For `(communicated_soil_data W)`:
            - Check if `(have_soil_analysis R W)` exists for any rover R.
            - Check if `(have_soil_sample R W)` exists for any rover R.
            - If no analysis: Add 1 for analyze action.
            - If no sample: Add 1 for collect action.
            - Move cost: 1 (if analysis exists), 2 (if sample exists but no analysis), 3 (if neither sample nor analysis exists).
          - For `(communicated_rock_data W)`:
            - Check if `(have_rock_analysis R W)` exists for any rover R.
            - Check if `(have_rock_sample R W)` exists for any rover R.
            - If no analysis: Add 1 for analyze action.
            - If no sample: Add 1 for collect action.
            - Move cost: 1 (if analysis exists), 2 (if sample exists but no analysis), 3 (if neither sample nor analysis exists).
          - For `(communicated_image_data O M)`:
            - Check if `(have_image R O M)` exists for any rover R.
            - Check if `(calibrated C R)` exists for any camera C and rover R such that C supports M and is a calibration target for O (using pre-calculated static info).
            - If no image: Add 1 for take image action.
            - If not calibrated: Add 1 for calibrate action.
            - Move cost: 1 (if image exists), 2 (if calibrated but no image), 3 (if neither calibrated nor image exists).
       c. Add the calculated cost (actions + moves) to the total heuristic.
    6. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and relevant static facts.
        """
        self.goals = task.goals

        # Extract static information for image goals
        self.camera_supports = {} # {camera: {mode1, mode2, ...}}
        self.calibration_targets = {} # {camera: objective}
        self.camera_on_board = {} # {camera: rover}
        self.lander_location = None # waypoint

        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "supports":
                camera, mode = parts[1], parts[2]
                self.camera_supports.setdefault(camera, set()).add(mode)
            elif parts[0] == "calibration_target":
                camera, objective = parts[1], parts[2]
                self.calibration_targets[camera] = objective
            elif parts[0] == "on_board":
                camera, rover = parts[1], parts[2]
                self.camera_on_board[camera] = rover
            elif parts[0] == "at_lander":
                 # Assuming only one lander and its location is static
                 # The lander object name is parts[1], location is parts[2]
                 self.lander_location = parts[2]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Pre-process state to quickly check for intermediate facts
        state_predicates = {}
        for fact in state:
            parts = get_parts(fact)
            predicate_name = parts[0]
            if predicate_name not in state_predicates:
                state_predicates[predicate_name] = []
            state_predicates[predicate_name].append(parts)

        def check_fact_exists(predicate, *args):
             """Checks if a fact matching the pattern exists in the pre-processed state."""
             if predicate not in state_predicates:
                 return False
             for fact_parts in state_predicates[predicate]:
                 # Check if the number of arguments matches and then use fnmatch
                 if len(fact_parts) == len(args) + 1 and all(fnmatch(part, arg) for part, arg in zip(fact_parts[1:], args)):
                     return True
             return False

        # Check calibration status for image goals considering camera/rover/mode/objective links
        # We need to know if *any* calibrated camera on *any* rover can satisfy the requirement
        # For a goal (communicated_image_data O M), we need *some* rover R with *some* camera C
        # such that on_board(C, R), supports(C, M), calibration_target(C, O), and calibrated(C, R)
        can_satisfy_calibration = {} # {(objective, mode): bool}
        # Iterate through all possible camera/rover pairs based on static facts
        for camera, rover in self.camera_on_board.items():
             # Check if this specific camera on this specific rover is calibrated in the current state
             if check_fact_exists("calibrated", camera, rover):
                 # This camera is calibrated on this rover
                 # Now check which image goals this calibrated pair can contribute to
                 obj = self.calibration_targets.get(camera)
                 if obj: # Check if this camera is a calibration_target for some objective
                     for mode in self.camera_supports.get(camera, set()):
                         # This camera supports this mode and is calibration_target for obj
                         # So, this calibrated camera/rover pair can satisfy the calibration
                         # prerequisite for any image goal involving this obj and mode.
                         can_satisfy_calibration[(obj, mode)] = True


        for goal in self.goals:
            # Check if the goal is already achieved
            if goal in state:
                continue

            parts = get_parts(goal)
            predicate = parts[0]

            if predicate == "communicated_soil_data":
                waypoint = parts[1]
                cost = 1 # communicate

                # Check prerequisites
                analysis_done = check_fact_exists("have_soil_analysis", "*", waypoint)
                sample_collected = check_fact_exists("have_soil_sample", "*", waypoint)

                if not analysis_done:
                    cost += 1 # analyze
                if not sample_collected:
                    cost += 1 # collect

                # Add move cost based on deepest unmet prerequisite
                if analysis_done:
                    cost += 1 # move to lander
                elif sample_collected:
                    cost += 2 # move to analyze location (W), move to lander
                else:
                    cost += 3 # move to collect location (W), move to analyze location (W), move to lander

                total_cost += cost

            elif predicate == "communicated_rock_data":
                waypoint = parts[1]
                cost = 1 # communicate

                # Check prerequisites
                analysis_done = check_fact_exists("have_rock_analysis", "*", waypoint)
                sample_collected = check_fact_exists("have_rock_sample", "*", waypoint)

                if not analysis_done:
                    cost += 1 # analyze
                if not sample_collected:
                    cost += 1 # collect

                # Add move cost based on deepest unmet prerequisite
                if analysis_done:
                    cost += 1 # move to lander
                elif sample_collected:
                    cost += 2 # move to analyze location (W), move to lander
                else:
                    cost += 3 # move to collect location (W), move to analyze location (W), move to lander

                total_cost += cost

            elif predicate == "communicated_image_data":
                objective, mode = parts[1], parts[2]
                cost = 1 # communicate

                # Check prerequisites
                image_taken = check_fact_exists("have_image", "*", objective, mode)
                # Check if *any* suitable camera/rover combination is calibrated
                calibration_done = can_satisfy_calibration.get((objective, mode), False)

                if not image_taken:
                    cost += 1 # take image
                if not calibration_done:
                    cost += 1 # calibrate

                # Add move cost based on deepest unmet prerequisite
                if image_taken:
                    cost += 1 # move to lander
                elif calibration_done:
                    cost += 2 # move to image location, move to lander
                else:
                    cost += 3 # move to calibration location, move to image location, move to lander

                total_cost += cost

            # Note: The rovers domain goals in the examples only include communicated_...
            # If other goal types were possible (e.g., at_rover), they would need
            # to be handled here.

        return total_cost
