from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_objects_from_fact(fact_str):
    """
    Extracts objects from a PDDL fact string.
    For example, from '(at rover1 waypoint1)' it extracts ['rover1', 'waypoint1'].
    Ignores the predicate name.
    """
    parts = fact_str[1:-1].split()
    return parts[1:]

def get_predicate_name(fact_str):
    """
    Extracts the predicate name from a PDDL fact string.
    For example, from '(at rover1 waypoint1)' it extracts 'at'.
    """
    parts = fact_str[1:-1].split()
    return parts[0]

def match_fact(fact, *args):
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
    This heuristic estimates the number of actions required to achieve all goal conditions
    in the rovers domain. It considers the necessary steps for each goal, such as navigating,
    sampling, calibrating, imaging, and communicating data. It prioritizes achieving individual
    goal facts and sums up the estimated costs for each.

    # Assumptions:
    - Each action (navigate, sample, drop, calibrate, take_image, communicate) has a uniform cost of 1.
    - The heuristic focuses on achieving each goal fact independently and sums up the costs.
    - It assumes that for each goal, the rover will need to navigate to a relevant waypoint if not already there.
    - For image goals, it assumes calibration is needed if not already calibrated.
    - For sample goals, it assumes sampling is needed if not already sampled.
    - Communication requires the rover to be in a visible location from the lander.

    # Heuristic Initialization
    - Extracts goal predicates from the task definition.
    - Processes static facts to build data structures for efficient lookup of:
        - `can_traverse` relations for each rover.
        - `calibration_target` for each camera.
        - `supports` modes for each camera.
        - `visible_from` waypoints for each objective.
        - `at_lander` location.
        - `store_of` for each rover.
        - `equipped_for_*` capabilities for each rover.
        - `visible` waypoint connections.

    # Step-By-Step Thinking for Computing Heuristic
    For each goal predicate in the goal state:
    1. Check if the goal predicate is already true in the current state. If yes, the cost for this goal is 0.
    2. If not, determine the type of goal predicate (communicated_soil_data, communicated_rock_data, communicated_image_data).
    3. Estimate the minimum number of actions needed to achieve this goal predicate:
        - For `communicated_soil_data(?w)`:
            - If `(communicated_soil_data ?w)` is not in the state:
                - If `(have_soil_analysis ?r ?w)` is not in the state (for some rover ?r equipped for soil analysis):
                    - Actions: Navigate to ?w (if not at ?w), sample_soil at ?w. Cost: 2 (or 1 if already at ?w).
                - Actions: Navigate to a waypoint visible from lander's location (if not at such a waypoint), communicate_soil_data. Cost: 2 (or 1 if already at visible waypoint).
        - For `communicated_rock_data(?w)`:
            - Similar logic as `communicated_soil_data`, but using `sample_rock` and `have_rock_analysis`.
        - For `communicated_image_data(?o, ?m)`:
            - If `(communicated_image_data ?o ?m)` is not in the state:
                - If `(have_image ?r ?o ?m)` is not in the state (for some rover ?r equipped for imaging):
                    - If `(calibrated ?c ?r)` is not in the state (for a camera ?c on rover ?r and targetting ?o):
                        - Actions: calibrate camera ?c for objective ?o. Cost: 1.
                    - Actions: Navigate to a waypoint visible from ?o (if not at such a waypoint), take_image of ?o in mode ?m. Cost: 2 (or 1 if already at visible waypoint).
                - Actions: Navigate to a waypoint visible from lander's location (if not at such a waypoint), communicate_image_data of ?o in mode ?m. Cost: 2 (or 1 if already at visible waypoint).
    4. Sum up the estimated costs for all unmet goal predicates to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.can_traverse_relations = {} # rover -> {waypoint1: set(waypoint2, ...)}
        self.calibration_targets = {} # camera -> objective
        self.camera_supports_modes = {} # camera -> set(mode, ...)
        self.objective_visible_from = {} # objective -> set(waypoint, ...)
        self.lander_location = None
        self.rover_stores = {} # rover -> store
        self.rover_equipment = {} # rover -> set('soil', 'rock', 'image')
        self.waypoint_visibility = {} # waypoint -> set(waypoint, ...)

        for fact in static_facts:
            if match_fact(fact, "can_traverse", "?r", "?w1", "?w2"):
                rover = get_objects_from_fact(fact)[0]
                waypoint1 = get_objects_from_fact(fact)[1]
                waypoint2 = get_objects_from_fact(fact)[2]
                if rover not in self.can_traverse_relations:
                    self.can_traverse_relations[rover] = {}
                if waypoint1 not in self.can_traverse_relations[rover]:
                    self.can_traverse_relations[rover][waypoint1] = set()
                self.can_traverse_relations[rover][waypoint1].add(waypoint2)
            elif match_fact(fact, "calibration_target", "?c", "?o"):
                camera = get_objects_from_fact(fact)[0]
                objective = get_objects_from_fact(fact)[1]
                self.calibration_targets[camera] = objective
            elif match_fact(fact, "supports", "?c", "?m"):
                camera = get_objects_from_fact(fact)[0]
                mode = get_objects_from_fact(fact)[1]
                if camera not in self.camera_supports_modes:
                    self.camera_supports_modes[camera] = set()
                self.camera_supports_modes[camera].add(mode)
            elif match_fact(fact, "visible_from", "?o", "?w"):
                objective = get_objects_from_fact(fact)[0]
                waypoint = get_objects_from_fact(fact)[1]
                if objective not in self.objective_visible_from:
                    self.objective_visible_from[objective] = set()
                self.objective_visible_from[objective].add(waypoint)
            elif match_fact(fact, "at_lander", "?l", "?w"):
                self.lander_location = get_objects_from_fact(fact)[1]
            elif match_fact(fact, "store_of", "?s", "?r"):
                rover = get_objects_from_fact(fact)[1]
                store = get_objects_from_fact(fact)[0]
                self.rover_stores[rover] = store
            elif match_fact(fact, "equipped_for_soil_analysis", "?r"):
                rover = get_objects_from_fact(fact)[0]
                if rover not in self.rover_equipment:
                    self.rover_equipment[rover] = set()
                self.rover_equipment[rover].add('soil')
            elif match_fact(fact, "equipped_for_rock_analysis", "?r"):
                rover = get_objects_from_fact(fact)[0]
                if rover not in self.rover_equipment:
                    self.rover_equipment[rover] = set()
                self.rover_equipment[rover].add('rock')
            elif match_fact(fact, "equipped_for_imaging", "?r"):
                rover = get_objects_from_fact(fact)[0]
                if rover not in self.rover_equipment:
                    self.rover_equipment[rover] = set()
                self.rover_equipment[rover].add('image')
            elif match_fact(fact, "visible", "?w1", "?w2"):
                waypoint1 = get_objects_from_fact(fact)[0]
                waypoint2 = get_objects_from_fact(fact)[1]
                if waypoint1 not in self.waypoint_visibility:
                    self.waypoint_visibility[waypoint1] = set()
                if waypoint2 not in self.waypoint_visibility:
                    self.waypoint_visibility[waypoint2] = set()
                self.waypoint_visibility[waypoint1].add(waypoint2)
                self.waypoint_visibility[waypoint2].add(waypoint1)


    def __call__(self, node):
        """Estimate the minimum cost to achieve all goal conditions from the current state."""
        state = node.state
        heuristic_value = 0

        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved, no cost

            goal_predicate_name = get_predicate_name(goal)

            if goal_predicate_name == 'communicated_soil_data':
                waypoint_goal = get_objects_from_fact(goal)[0]
                if not match_fact(goal, "communicated_soil_data", "*") in [match_fact(fact, "communicated_soil_data", "*") for fact in state]:
                    rover_capable = None
                    for rover in self.rover_equipment:
                        if 'soil' in self.rover_equipment[rover]:
                            rover_capable = rover
                            break
                    if rover_capable:
                        if not any(match_fact(fact, "have_soil_analysis", rover_capable, waypoint_goal) for fact in state):
                            current_rover_location = None
                            for fact in state:
                                if match_fact(fact, "at", rover_capable, "*"):
                                    current_rover_location = get_objects_from_fact(fact)[1]
                                    break
                            if current_rover_location != waypoint_goal:
                                heuristic_value += 1 # navigate
                            heuristic_value += 1 # sample_soil
                        lander_waypoint = self.lander_location
                        rover_location_for_comm = None
                        for fact in state:
                            if match_fact(fact, "at", rover_capable, "*"):
                                rover_location_for_comm = get_objects_from_fact(fact)[1]
                                break
                        if lander_waypoint and rover_location_for_comm:
                            if lander_waypoint not in self.waypoint_visibility.get(rover_location_for_comm, set()):
                                heuristic_value += 1 # navigate to waypoint visible from lander
                        heuristic_value += 1 # communicate_soil_data

            elif goal_predicate_name == 'communicated_rock_data':
                waypoint_goal = get_objects_from_fact(goal)[0]
                if not match_fact(goal, "communicated_rock_data", "*") in [match_fact(fact, "communicated_rock_data", "*") for fact in state]:
                    rover_capable = None
                    for rover in self.rover_equipment:
                        if 'rock' in self.rover_equipment[rover]:
                            rover_capable = rover
                            break
                    if rover_capable:
                        if not any(match_fact(fact, "have_rock_analysis", rover_capable, waypoint_goal) for fact in state):
                            current_rover_location = None
                            for fact in state:
                                if match_fact(fact, "at", rover_capable, "*"):
                                    current_rover_location = get_objects_from_fact(fact)[1]
                                    break
                            if current_rover_location != waypoint_goal:
                                heuristic_value += 1 # navigate
                            heuristic_value += 1 # sample_rock
                        lander_waypoint = self.lander_location
                        rover_location_for_comm = None
                        for fact in state:
                            if match_fact(fact, "at", rover_capable, "*"):
                                rover_location_for_comm = get_objects_from_fact(fact)[1]
                                break
                        if lander_waypoint and rover_location_for_comm:
                            if lander_waypoint not in self.waypoint_visibility.get(rover_location_for_comm, set()):
                                heuristic_value += 1 # navigate to waypoint visible from lander
                        heuristic_value += 1 # communicate_rock_data

            elif goal_predicate_name == 'communicated_image_data':
                objective_goal = get_objects_from_fact(goal)[0]
                mode_goal = get_objects_from_fact(goal)[1]
                if not match_fact(goal, "communicated_image_data", "*", "*") in [match_fact(fact, "communicated_image_data", "*", "*") for fact in state]:
                    rover_capable = None
                    camera_used = None
                    for rover in self.rover_equipment:
                        if 'image' in self.rover_equipment[rover]:
                            for camera in self.camera_supports_modes:
                                if camera in self.rover_stores and self.rover_stores[camera] == rover and mode_goal in self.camera_supports_modes[camera]: # check if camera is on rover. rover_stores is actually rover_cameras
                                    rover_capable = rover
                                    camera_used = camera
                                    break
                            if rover_capable:
                                break

                    if rover_capable and camera_used:
                        if not any(match_fact(fact, "have_image", rover_capable, objective_goal, mode_goal) for fact in state):
                            if not any(match_fact(fact, "calibrated", camera_used, rover_capable) for fact in state):
                                heuristic_value += 1 # calibrate
                            objective_visible_waypoints = self.objective_visible_from.get(objective_goal, set())
                            current_rover_location = None
                            for fact in state:
                                if match_fact(fact, "at", rover_capable, "*"):
                                    current_rover_location = get_objects_from_fact(fact)[1]
                                    break
                            if current_rover_location not in objective_visible_waypoints:
                                heuristic_value += 1 # navigate to waypoint visible from objective
                            heuristic_value += 1 # take_image
                        lander_waypoint = self.lander_location
                        rover_location_for_comm = None
                        for fact in state:
                            if match_fact(fact, "at", rover_capable, "*"):
                                rover_location_for_comm = get_objects_from_fact(fact)[1]
                                break
                        if lander_waypoint and rover_location_for_comm:
                            if lander_waypoint not in self.waypoint_visibility.get(rover_location_for_comm, set()):
                                heuristic_value += 1 # navigate to waypoint visible from lander
                        heuristic_value += 1 # communicate_image_data

        return heuristic_value
