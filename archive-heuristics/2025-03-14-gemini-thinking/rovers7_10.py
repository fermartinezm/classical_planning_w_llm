from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import itertools

def get_objects_from_fact(fact_str):
    """
    Extract objects from a PDDL fact string.
    For example, from '(at rover1 waypoint1)' it extracts ['rover1', 'waypoint1'].
    Ignores the predicate name.
    """
    parts = fact_str[1:-1].split()
    return parts[1:]

def get_predicate_name(fact_str):
    """
    Extract predicate name from a PDDL fact string.
    For example, from '(at rover1 waypoint1)' it extracts 'at'.
    """
    parts = fact_str[1:-1].split()
    return parts[0]

def match(fact, *args):
    """
    Utility function to check if a PDDL fact matches a given pattern.
    - `fact`: The fact as a string (e.g., "(at ball1 rooma)").
    - `args`: The pattern to match (e.g., "at", "*", "rooma").
    - Returns `True` if the fact matches the pattern, `False` otherwise.
    """
    parts = fact[1:-1].split()  # Remove parentheses and split into individual elements.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal conditions in the rovers domain.
    It considers the necessary steps for each goal, such as navigating to locations,
    sampling soil/rock, calibrating cameras, taking images, and communicating data.
    The heuristic is calculated by summing up the estimated costs for each goal predicate.

    # Assumptions:
    - The heuristic assumes that each goal needs to be achieved independently and sums up the costs.
    - It estimates the cost based on current state and static information like visibility and traversability.
    - It does not consider complex interactions between goals or optimal action ordering.
    - It assumes that for communication, the rover needs to be visible from the lander's location.

    # Heuristic Initialization
    - Pre-processes the static facts to build efficient lookup structures for:
        - `can_traverse`: to check if a rover can move between waypoints.
        - `visible`: to check waypoint visibility.
        - `visible_from`: to check if an objective is visible from a waypoint.
        - `calibration_target`: to identify calibration targets for cameras.
        - `supports`: to check camera mode support.
        - `at_lander`: to find the lander's location.

    # Step-By-Step Thinking for Computing Heuristic
    For each goal in the goal state, the heuristic estimates the minimum actions required as follows:

    1. For `communicated_soil_data(waypoint)` goal:
        - If already achieved, cost is 0.
        - Else, check if `have_soil_analysis(rover, waypoint)` is achieved.
            - If yes, estimate cost to communicate: navigate to a waypoint visible from lander and communicate.
            - If no, estimate cost to sample and communicate:
                - Navigate to the waypoint with soil sample.
                - Sample soil.
                - Navigate to a waypoint visible from lander and communicate.

    2. For `communicated_rock_data(waypoint)` goal:
        - Logic similar to `communicated_soil_data`, but for rock samples and `have_rock_analysis`.

    3. For `communicated_image_data(objective, mode)` goal:
        - If already achieved, cost is 0.
        - Else, check if `have_image(rover, objective, mode)` is achieved.
            - If yes, estimate cost to communicate: navigate to a waypoint visible from lander and communicate.
            - If no, estimate cost to take image and communicate:
                - Navigate to a waypoint visible from the objective.
                - Calibrate camera (if not already calibrated).
                - Take image in the specified mode.
                - Navigate to a waypoint visible from lander and communicate.

    Navigation cost is roughly estimated as 1 action per navigation step.
    Calibration, sampling, taking image, and communication are each estimated as 1 action.
    The heuristic sums up the estimated costs for all unmet goals.
    """

    def __init__(self, task):
        """Initialize the rovers heuristic by pre-processing static facts and goal conditions."""
        self.goals = task.goals
        static_facts = task.static

        self.can_traverse = set()
        self.visible_waypoints = set()
        self.visible_from_objective = {}
        self.calibration_targets = {}
        self.supports_mode = {}
        self.lander_location = None

        for fact in static_facts:
            if match(fact, "can_traverse", "*", "*", "*"):
                self.can_traverse.add(tuple(get_objects_from_fact(fact)))
            elif match(fact, "visible", "*", "*"):
                self.visible_waypoints.add(tuple(get_objects_from_fact(fact)))
            elif match(fact, "visible_from", "*", "*"):
                objective, waypoint = get_objects_from_fact(fact)
                if objective not in self.visible_from_objective:
                    self.visible_from_objective[objective] = set()
                self.visible_from_objective[objective].add(waypoint)
            elif match(fact, "calibration_target", "*", "*"):
                camera, objective = get_objects_from_fact(fact)
                self.calibration_targets[camera] = objective
            elif match(fact, "supports", "*", "*"):
                camera, mode = get_objects_from_fact(fact)
                if camera not in self.supports_mode:
                    self.supports_mode[camera] = set()
                self.supports_mode[camera].add(mode)
            elif match(fact, "at_lander", "*", "*"):
                self.lander_location = get_objects_from_fact(fact)[1] # waypoint where lander is

        self.soil_samples_locations = set()
        self.rock_samples_locations = set()
        for fact in static_facts:
            if match(fact, "at_soil_sample", "*"):
                self.soil_samples_locations.add(get_objects_from_fact(fact)[0])
            elif match(fact, "at_rock_sample", "*"):
                self.rock_samples_locations.add(get_objects_from_fact(fact)[0])


    def __call__(self, node):
        """Estimate the number of actions to reach the goal state from the current node's state."""
        state = node.state
        heuristic_value = 0

        goal_facts = set(self.goals)

        for goal in goal_facts:
            if goal in state:
                continue # Goal already achieved, no cost

            predicate = get_predicate_name(goal)

            if predicate == 'communicated_soil_data':
                waypoint_goal = get_objects_from_fact(goal)[0]
                if match(goal, "communicated_soil_data", "*"):
                    if not match(goal, *get_objects_from_fact(goal)): # Redundant check, but for clarity
                        continue # Goal already achieved

                    if any(match(fact, "have_soil_analysis", "*", waypoint_goal) for fact in state):
                        # Need to communicate soil data
                        heuristic_value += 1 # Estimate 1 action for communicate_soil_data
                    else:
                        # Need to sample and communicate soil data
                        heuristic_value += 2 # Estimate 1 action for sample_soil + 1 for communicate_soil_data

            elif predicate == 'communicated_rock_data':
                waypoint_goal = get_objects_from_fact(goal)[0]
                if match(goal, "communicated_rock_data", "*"):
                    if not match(goal, *get_objects_from_fact(goal)):
                        continue # Goal already achieved

                    if any(match(fact, "have_rock_analysis", "*", waypoint_goal) for fact in state):
                        # Need to communicate rock data
                        heuristic_value += 1 # Estimate 1 action for communicate_rock_data
                    else:
                        # Need to sample and communicate rock data
                        heuristic_value += 2 # Estimate 1 action for sample_rock + 1 for communicate_rock_data

            elif predicate == 'communicated_image_data':
                objective_goal, mode_goal = get_objects_from_fact(goal)
                if match(goal, "communicated_image_data", "*", "*"):
                    if not match(goal, *get_objects_from_fact(goal)):
                        continue # Goal already achieved

                    if any(match(fact, "have_image", "*", objective_goal, mode_goal) for fact in state):
                        # Need to communicate image data
                        heuristic_value += 1 # Estimate 1 action for communicate_image_data
                    else:
                        # Need to take image and communicate image data
                        heuristic_value += 2 # Estimate 1 action for take_image + 1 for communicate_image_data

        return heuristic_value
