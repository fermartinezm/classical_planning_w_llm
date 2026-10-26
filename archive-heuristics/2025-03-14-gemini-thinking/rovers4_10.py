from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_objects_from_fact(fact):
    """
    Extract the objects from a PDDL fact string.

    For example, from '(at rover1 waypoint2)' it extracts ['rover1', 'waypoint2'].
    Ignores the predicate name.
    """
    parts = fact[1:-1].split()
    return parts[1:]  # Return objects, skipping the predicate name


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = fact[1:-1].split()
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal conditions in the rovers domain.
    It focuses on achieving each goal predicate independently and sums up the estimated costs.
    The heuristic considers the actions needed to navigate, sample, calibrate, take images, and communicate data.

    # Assumptions:
    - Each goal predicate is considered independently.
    - The cost of achieving a set of goals is the sum of the costs for each individual goal.
    - Navigation cost is simplified and not based on shortest paths, but rather on reachability.
    - Actions are assumed to have a uniform cost of 1.

    # Heuristic Initialization
    - Pre-processes the static facts to build efficient lookup structures:
        - `visible_waypoints`: A dictionary mapping each waypoint to a set of waypoints visible from it.
        - `can_traverse_rovers`: A dictionary mapping each rover to a set of waypoint pairs it can traverse.
        - `calibration_targets`: A dictionary mapping each camera to its calibration objective.
        - `camera_supports_modes`: A dictionary mapping each camera to the modes it supports.
        - `objective_visible_from`: A dictionary mapping each objective to waypoints visible from it.
        - `lander_location`: The location of the lander.

    # Step-By-Step Thinking for Computing Heuristic
    For each goal predicate in the goal state:
    1. Check if the goal predicate is already satisfied in the current state. If yes, the cost for this goal is 0.
    2. If not satisfied, determine the type of goal predicate:
        - `communicated_soil_data(waypoint)`:
            - Check if `have_soil_analysis(rover, waypoint)` exists. If not, estimate cost to sample soil:
                - 1 (sample_soil action) + navigation cost to a waypoint with soil sample.
            - If `have_soil_analysis` exists, estimate cost to communicate:
                - 1 (communicate_soil_data action) + navigation cost to a waypoint visible from lander.
        - `communicated_rock_data(waypoint)`:
            - Similar to `communicated_soil_data`, but for rock samples and `have_rock_analysis`.
        - `communicated_image_data(objective, mode)`:
            - Check if `have_image(rover, objective, mode)` exists. If not, estimate cost to take image:
                - 1 (take_image action) + 1 (calibrate action, if not calibrated) + navigation cost to a waypoint visible from objective.
            - If `have_image` exists, estimate cost to communicate:
                - 1 (communicate_image_data action) + navigation cost to a waypoint visible from lander.
    3. Sum up the estimated costs for all unsatisfied goal predicates.
    4. The total sum is the heuristic value for the current state.

    Navigation cost is simplified: if a path exists (based on `can_traverse` and `visible` predicates), the cost is considered 1 navigation action.
    If no path is immediately obvious, or for actions like sampling and calibrating, we assume a fixed cost of 1 action plus potential navigation.
    """

    def __init__(self, task):
        """Initialize the rovers domain heuristic by pre-processing static facts and goal information."""
        self.goals = task.goals
        static_facts = task.static

        self.visible_waypoints = {}
        self.can_traverse_rovers = {}
        self.calibration_targets = {}
        self.camera_supports_modes = {}
        self.objective_visible_from = {}
        self.lander_location = None

        for fact in static_facts:
            if match(fact, "visible", "?wp1", "?wp2"):
                wp1, wp2 = get_objects_from_fact(fact)
                self.visible_waypoints.setdefault(wp1, set()).add(wp2)
            elif match(fact, "can_traverse", "?rover", "?wp1", "?wp2"):
                rover, wp1, wp2 = get_objects_from_fact(fact)
                self.can_traverse_rovers.setdefault(rover, set()).add((wp1, wp2))
            elif match(fact, "calibration_target", "?camera", "?objective"):
                camera, objective = get_objects_from_fact(fact)
                self.calibration_targets[camera] = objective
            elif match(fact, "supports", "?camera", "?mode"):
                camera, mode = get_objects_from_fact(fact)
                self.camera_supports_modes.setdefault(camera, set()).add(mode)
            elif match(fact, "visible_from", "?objective", "?waypoint"):
                objective, waypoint = get_objects_from_fact(fact)
                self.objective_visible_from.setdefault(objective, set()).add(waypoint)
            elif match(fact, "at_lander", "*", "?waypoint"):
                self.lander_location = get_objects_from_fact(fact)[1]

    def __call__(self, node):
        """Estimate the number of actions to reach the goal state from the current state."""
        state = node.state
        heuristic_value = 0

        goal_predicates_remaining = []
        for goal in self.goals:
            if goal not in state:
                goal_predicates_remaining.append(goal)

        for goal_predicate in goal_predicates_remaining:
            if match(goal_predicate, "communicated_soil_data", "?waypoint"):
                waypoint = get_objects_from_fact(goal_predicate)[0]
                if not any(match(fact, "have_soil_analysis", "*", waypoint) for fact in state):
                    heuristic_value += 2  # Estimate cost for sample_soil + navigate (simplified)
                heuristic_value += 2  # Estimate cost for communicate_soil_data + navigate (simplified)

            elif match(goal_predicate, "communicated_rock_data", "?waypoint"):
                waypoint = get_objects_from_fact(goal_predicate)[0]
                if not any(match(fact, "have_rock_analysis", "*", waypoint) for fact in state):
                    heuristic_value += 2  # Estimate cost for sample_rock + navigate (simplified)
                heuristic_value += 2  # Estimate cost for communicate_rock_data + navigate (simplified)

            elif match(goal_predicate, "communicated_image_data", "?objective", "?mode"):
                objective, mode = get_objects_from_fact(goal_predicate)
                if not any(match(fact, "have_image", "*", objective, mode) for fact in state):
                    heuristic_value += 3  # Estimate cost for take_image + calibrate + navigate (simplified)
                heuristic_value += 2  # Estimate cost for communicate_image_data + navigate (simplified)

        return heuristic_value
