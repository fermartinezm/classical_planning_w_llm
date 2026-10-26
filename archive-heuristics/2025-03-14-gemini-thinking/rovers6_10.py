from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import itertools

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
    This heuristic estimates the number of actions required to achieve all goal conditions
    in the rovers domain. It focuses on the necessary actions for sampling, imaging,
    calibration, navigation, and communication, without considering optimal action ordering
    or resource contention. It provides a lower-bound estimate by assuming each goal can be
    achieved independently and ignoring potential action overlaps or optimizations.

    # Assumptions:
    - Each goal predicate (communicated data) is considered independently.
    - Navigation cost is simplified to 1 action per waypoint move when needed.
    - Rovers are always capable of performing actions if preconditions are met (ignoring resource limits beyond predicates).
    - The heuristic is not admissible but aims for efficient computation and informed search guidance.

    # Heuristic Initialization
    - Extracts goal predicates from the task definition.
    - Parses static facts to pre-calculate and store information about:
        - Waypoint visibility (`visible`, `visible_from`).
        - Traversability between waypoints (`can_traverse`).
        - Rover equipment (`equipped_for_soil_analysis`, `equipped_for_rock_analysis`, `equipped_for_imaging`).
        - Camera support for modes (`supports`).
        - Calibration targets (`calibration_target`).
        - Store ownership (`store_of`).
        - Lander location (`at_lander`).

    # Step-By-Step Thinking for Computing Heuristic
    For each goal predicate in the goal state, the heuristic estimates the minimum number of actions required to satisfy it, if it's not already satisfied in the current state. The total heuristic value is the sum of these individual estimates.

    1. **Goal Predicate Check:** Iterate through each goal predicate. If the predicate is already true in the current state, the cost for this goal is 0.

    2. **Estimate Cost for `communicated_soil_data(?waypoint)`:**
       - Check if `(communicated_soil_data ?waypoint)` is already true. If yes, cost is 0.
       - Otherwise, estimate actions:
         - 1 action for `communicate_soil_data`.
         - 1 action for `sample_soil` (if `(have_soil_analysis ?rover ?waypoint)` is not true).
         - 1 action for `navigate` to a waypoint with `(at_soil_sample ?waypoint)` (if rover is not at such a waypoint and `(have_soil_analysis ?rover ?waypoint)` is not true).

    3. **Estimate Cost for `communicated_rock_data(?waypoint)`:**
       - Similar to `communicated_soil_data`, but for rock samples and using `sample_rock` action.

    4. **Estimate Cost for `communicated_image_data(?objective, ?mode)`:**
       - Check if `(communicated_image_data ?objective ?mode)` is already true. If yes, cost is 0.
       - Otherwise, estimate actions:
         - 1 action for `communicate_image_data`.
         - 1 action for `take_image` (if `(have_image ?rover ?objective ?mode)` is not true).
         - 1 action for `calibrate` (if `(calibrated ?camera ?rover)` is not true and `take_image` is needed).
         - 1 action for `navigate` to a waypoint `?waypoint` visible from `?objective` (if rover is not at such a waypoint and `take_image` is needed).

    5. **Summation:** The total heuristic value is the sum of the estimated costs for all unmet goal predicates.

    This heuristic prioritizes achieving individual goals with a minimal action count, making it suitable for greedy best-first search in the rovers domain.
    """

    def __init__(self, task):
        """Initialize the rovers heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.visible_waypoints = set()
        self.visible_from_objectives = {}
        self.can_traverse_pairs = set()
        self.equipped_for_soil = set()
        self.equipped_for_rock = set()
        self.equipped_for_imaging = set()
        self.camera_supports_mode = {}
        self.calibration_targets = {}
        self.store_of_rover = {}
        self.at_lander_location = None

        for fact in static_facts:
            if match(fact, "visible", "*", "*"):
                parts = get_parts(fact)
                self.visible_waypoints.add(tuple(parts[1:]))
            elif match(fact, "visible_from", "*", "*"):
                parts = get_parts(fact)
                objective = parts[1]
                waypoint = parts[2]
                if objective not in self.visible_from_objectives:
                    self.visible_from_objectives[objective] = set()
                self.visible_from_objectives[objective].add(waypoint)
            elif match(fact, "can_traverse", "*", "*", "*"):
                parts = get_parts(fact)
                self.can_traverse_pairs.add(tuple(parts[1:]))
            elif match(fact, "equipped_for_soil_analysis", "*"):
                self.equipped_for_soil.add(get_parts(fact)[1])
            elif match(fact, "equipped_for_rock_analysis", "*"):
                self.equipped_for_rock.add(get_parts(fact)[1])
            elif match(fact, "equipped_for_imaging", "*"):
                self.equipped_for_imaging.add(get_parts(fact)[1])
            elif match(fact, "supports", "*", "*"):
                parts = get_parts(fact)
                camera = parts[1]
                mode = parts[2]
                if camera not in self.camera_supports_mode:
                    self.camera_supports_mode[camera] = set()
                self.camera_supports_mode[camera].add(mode)
            elif match(fact, "calibration_target", "*", "*"):
                parts = get_parts(fact)
                camera = parts[1]
                objective = parts[2]
                self.calibration_targets[camera] = objective
            elif match(fact, "store_of", "*", "*"):
                parts = get_parts(fact)
                store = parts[1]
                rover = parts[2]
                self.store_of_rover[store] = rover
            elif match(fact, "at_lander", "*", "*"):
                self.at_lander_location = get_parts(fact)[2]


    def __call__(self, node):
        """Estimate the number of actions to reach the goal state from the current state."""
        state = node.state
        heuristic_value = 0

        goal_facts = set(self.goals)

        for goal in goal_facts:
            if goal in state:
                continue

            parts = get_parts(goal)
            goal_predicate = parts[0]

            if goal_predicate == 'communicated_soil_data':
                waypoint = parts[1]
                if goal not in state:
                    heuristic_value += 1 # communicate_soil_data action
                    if f'(have_soil_analysis * {waypoint})' not in state:
                        heuristic_value += 1 # sample_soil action (simplified: assuming 1 sample needed)

            elif goal_predicate == 'communicated_rock_data':
                waypoint = parts[1]
                if goal not in state:
                    heuristic_value += 1 # communicate_rock_data action
                    if f'(have_rock_analysis * {waypoint})' not in state:
                        heuristic_value += 1 # sample_rock action (simplified: assuming 1 sample needed)

            elif goal_predicate == 'communicated_image_data':
                objective = parts[1]
                mode = parts[2]
                if goal not in state:
                    heuristic_value += 1 # communicate_image_data action
                    if f'(have_image * {objective} {mode})' not in state:
                        heuristic_value += 1 # take_image action
                        needed_calibration = True
                        for fact in state:
                            if match(fact, 'calibrated', '*', '*'):
                                rover_calibrated = get_parts(fact)[2]
                                camera_calibrated = None
                                for cam, obj in self.calibration_targets.items():
                                    if obj == objective:
                                        camera_calibrated = cam
                                        break
                                if camera_calibrated and rover_calibrated and f'(on_board {camera_calibrated} {rover_calibrated})' in state:
                                    needed_calibration = False
                                    break
                        if needed_calibration:
                            heuristic_value += 1 # calibrate action (simplified: assuming 1 calibration needed if not calibrated)
        return heuristic_value
