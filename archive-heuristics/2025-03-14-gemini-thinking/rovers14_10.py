from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal conditions in the rovers domain.
    It focuses on achieving each goal predicate independently and sums up the estimated costs.
    The heuristic considers the necessary steps for each goal type: communicating data (image, soil, rock).

    # Assumptions:
    - Actions have a uniform cost of 1.
    - The heuristic assumes that for each goal, the required preconditions can be achieved.
    - It does not explicitly plan paths or consider resource contention (e.g., store capacity).
    - It simplifies the problem by assuming direct visibility and traversability when needed, leading to an underestimation in some cases, but maintaining efficiency.

    # Heuristic Initialization
    - Extracts static information from the task, such as visibility relations, traversability, equipment capabilities,
      calibration targets, camera support modes, and lander locations.
    - Stores this information in dictionaries and sets for efficient access during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    For each goal predicate in the goal state:
    1. Check if the goal is already achieved in the current state. If yes, the cost for this goal is 0.
    2. If not achieved, determine the type of goal (communicate_image_data, communicate_soil_data, communicate_rock_data).
    3. Estimate the minimum number of actions required to achieve this goal based on the current state and static information.
       - For communicate_image_data:
         - If image not taken: Estimate cost for taking image (calibrate + take_image).
           - Calibration needed if not already calibrated for the objective and rover.
           - Taking image requires being at a waypoint visible from the objective.
         - Estimate cost for communication (communicate_image_data).
       - For communicate_soil_data:
         - If soil analysis not done: Estimate cost for soil sampling (sample_soil).
           - Sampling requires being at a waypoint with soil sample and having soil analysis equipment and empty store.
         - Estimate cost for communication (communicate_soil_data).
       - For communicate_rock_data:
         - If rock analysis not done: Estimate cost for rock sampling (sample_rock).
           - Sampling requires being at a waypoint with rock sample and having rock analysis equipment and empty store.
         - Estimate cost for communication (communicate_rock_data).
    4. Sum up the estimated costs for all unachieved goal predicates.
    5. Return the total estimated cost.

    This heuristic is admissible if we consider the minimum number of actions for each goal independently and assume optimal conditions (like direct visibility and traversability when needed). However, for efficiency in greedy best-first search, admissibility is not strictly required, and this heuristic aims for a reasonable and fast estimate.
    """

    def __init__(self, task):
        """
        Initialize the rovers heuristic by extracting relevant static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        self.visible = set()
        self.can_traverse = set()
        self.equipped_for_soil_analysis = set()
        self.equipped_for_rock_analysis = set()
        self.equipped_for_imaging = set()
        self.store_of = {}
        self.calibration_target = {}
        self.on_board_camera = {}
        self.supports_mode = {}
        self.visible_from = {}
        self.at_lander = set()

        for fact in static_facts:
            if match(fact, "visible", "*", "*"):
                self.visible.add(tuple(get_parts(fact)[1:]))
            elif match(fact, "can_traverse", "*", "*", "*"):
                self.can_traverse.add(tuple(get_parts(fact)[1:]))
            elif match(fact, "equipped_for_soil_analysis", "*"):
                self.equipped_for_soil_analysis.add(get_parts(fact)[1])
            elif match(fact, "equipped_for_rock_analysis", "*"):
                self.equipped_for_rock_analysis.add(get_parts(fact)[1])
            elif match(fact, "equipped_for_imaging", "*"):
                self.equipped_for_imaging.add(get_parts(fact)[1])
            elif match(fact, "store_of", "*", "*"):
                self.store_of[get_parts(fact)[2]] = get_parts(fact)[1]
            elif match(fact, "calibration_target", "*", "*"):
                camera, objective = get_parts(fact)[1:]
                if camera not in self.calibration_target:
                    self.calibration_target[camera] = set()
                self.calibration_target[camera].add(objective)
            elif match(fact, "on_board", "*", "*"):
                camera, rover = get_parts(fact)[1:]
                if rover not in self.on_board_camera:
                    self.on_board_camera[rover] = set()
                self.on_board_camera[rover].add(camera)
            elif match(fact, "supports", "*", "*"):
                camera, mode = get_parts(fact)[1:]
                if camera not in self.supports_mode:
                    self.supports_mode[camera] = set()
                self.supports_mode[camera].add(mode)
            elif match(fact, "visible_from", "*", "*"):
                objective, waypoint = get_parts(fact)[1:]
                if objective not in self.visible_from:
                    self.visible_from[objective] = set()
                self.visible_from[objective].add(waypoint)
            elif match(fact, "at_lander", "*", "*"):
                self.at_lander.add(tuple(get_parts(fact)[1:]))

    def __call__(self, node):
        """
        Compute the heuristic value for a given state.
        """
        state = node.state
        goal_cost = 0

        for goal in self.goals:
            if goal in state:
                continue

            goal_parts = get_parts(goal)
            goal_predicate = goal_parts[0]

            if goal_predicate == 'communicated_image_data':
                rover_needed = None
                objective, mode = goal_parts[1:3]

                # Check if image is already taken
                image_taken = False
                for fact in state:
                    if match(fact, "have_image", "*", objective, mode):
                        image_taken = True
                        rover_needed = get_parts(fact)[1]
                        break

                if not image_taken:
                    goal_cost += 1 # take_image action
                    calibration_needed = True
                    if rover_needed is None:
                        for r in self.equipped_for_imaging:
                            for cam in self.on_board_camera.get(r, []):
                                if mode in self.supports_mode.get(cam, set()) and objective in self.calibration_target.get(cam, set()):
                                    rover_needed = r
                                    break
                            if rover_needed:
                                break
                    if rover_needed:
                        for fact in state:
                            if match(fact, "calibrated", "*", rover_needed):
                                cam_calibrated = get_parts(fact)[1]
                                if cam_calibrated in self.on_board_camera.get(rover_needed, []) and cam_calibrated in self.supports_mode and mode in self.supports_mode.get(cam_calibrated, set()):
                                    calibration_needed = False
                                    break
                    if calibration_needed and rover_needed:
                        goal_cost += 1 # calibrate action

                goal_cost += 1 # communicate_image_data action

            elif goal_predicate == 'communicated_soil_data':
                waypoint = goal_parts[1]
                soil_sampled = False
                for fact in state:
                    if match(fact, "have_soil_analysis", "*", waypoint):
                        soil_sampled = True
                        break
                if not soil_sampled:
                    goal_cost += 1 # sample_soil action
                goal_cost += 1 # communicate_soil_data action

            elif goal_predicate == 'communicated_rock_data':
                waypoint = goal_parts[1]
                rock_sampled = False
                for fact in state:
                    if match(fact, "have_rock_analysis", "*", waypoint):
                        rock_sampled = True
                        break
                if not rock_sampled:
                    goal_cost += 1 # sample_rock action
                goal_cost += 1 # communicate_rock_data action

        return goal_cost
