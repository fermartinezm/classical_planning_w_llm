from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_objects_from_fact(fact):
    """
    Extract the objects from a PDDL fact string.
    For example, from '(at rover1 waypoint2)' it returns ['rover1', 'waypoint2'].
    Ignores the predicate name.
    """
    parts = fact[1:-1].split()
    return parts[1:]


def get_predicate_name(fact):
    """
    Extract the predicate name from a PDDL fact string.
    For example, from '(at rover1 waypoint2)' it returns 'at'.
    """
    parts = fact[1:-1].split()
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
    This heuristic estimates the number of actions required to achieve all goal predicates in the rovers domain.
    It focuses on the necessary steps for each goal type (communicating soil, rock, and image data) and sums up the estimated costs.

    # Assumptions:
    - The heuristic assumes that each goal predicate is independent and calculates the cost for each goal separately.
    - It simplifies navigation cost by assuming a single navigate action is sufficient to reach any visible waypoint.
    - It does not explicitly consider store capacity or dropping actions in detail, assuming a store is always available and large enough when needed.
    - It prioritizes achieving communication goals and estimates the minimum actions needed for each.

    # Heuristic Initialization
    - Extracts static information from the task, such as:
        - `can_traverse` for each rover.
        - `visible` waypoint pairs.
        - `visible_from` objective-waypoint pairs.
        - `calibration_target` for each camera.
        - `supports` modes for each camera.
        - Equipped capabilities for each rover.
        - Lander location.
        - Locations of soil and rock samples.
    - Stores goal predicates for efficient access during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    For each goal predicate in the goal state:
    1. Check if the goal is already achieved in the current state. If yes, the cost for this goal is 0.
    2. If not achieved, determine the type of goal (communicate soil, rock, or image data).
    3. Estimate the minimum number of actions required to achieve this specific goal type:
        - For `communicated_soil_data(waypoint)`:
            - Actions: navigate to soil sample waypoint (if not there), sample soil (if not sampled), navigate to communication waypoint (if not there), communicate soil data. Estimated cost: 4.
        - For `communicated_rock_data(waypoint)`:
            - Actions: navigate to rock sample waypoint (if not there), sample rock (if not sampled), navigate to communication waypoint (if not there), communicate rock data. Estimated cost: 4.
        - For `communicated_image_data(objective, mode)`:
            - Actions: navigate to waypoint visible from objective (if not there), calibrate camera (if not calibrated), take image (if not taken), navigate to communication waypoint (if not there), communicate image data. Estimated cost: 5.
    4. Sum up the estimated costs for all unachieved goal predicates to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.can_traverse_relations = set()
        self.visible_relations = set()
        self.visible_from_relations = set()
        self.calibration_targets = {}
        self.camera_supports = {}
        self.equipped_for_soil_rovers = set()
        self.equipped_for_rock_rovers = set()
        self.equipped_for_imaging_rovers = set()
        self.lander_location = None
        self.soil_sample_locations = set()
        self.rock_sample_locations = set()

        for fact in static_facts:
            if match(fact, 'can_traverse', '*', '*', '*'):
                objs = get_objects_from_fact(fact)
                self.can_traverse_relations.add(tuple(objs))
            elif match(fact, 'visible', '*', '*'):
                objs = get_objects_from_fact(fact)
                self.visible_relations.add(tuple(objs))
            elif match(fact, 'visible_from', '*', '*'):
                objs = get_objects_from_fact(fact)
                self.visible_from_relations.add(tuple(objs))
            elif match(fact, 'calibration_target', '*', '*'):
                camera, objective = get_objects_from_fact(fact)
                self.calibration_targets[camera] = objective
            elif match(fact, 'supports', '*', '*'):
                camera, mode = get_objects_from_fact(fact)
                if camera not in self.camera_supports:
                    self.camera_supports[camera] = set()
                self.camera_supports[camera].add(mode)
            elif match(fact, 'equipped_for_soil_analysis', '*'):
                rover = get_objects_from_fact(fact)[0]
                self.equipped_for_soil_rovers.add(rover)
            elif match(fact, 'equipped_for_rock_analysis', '*'):
                rover = get_objects_from_fact(fact)[0]
                self.equipped_for_rock_rovers.add(rover)
            elif match(fact, 'equipped_for_imaging', '*'):
                rover = get_objects_from_fact(fact)[0]
                self.equipped_for_imaging_rovers.add(rover)
            elif match(fact, 'at_lander', '*', '*'):
                self.lander_location = tuple(get_objects_from_fact(fact)) # (lander, waypoint)
            elif match(fact, 'at_soil_sample', '*'):
                self.soil_sample_locations.add(get_objects_from_fact(fact)[0])
            elif match(fact, 'at_rock_sample', '*'):
                self.rock_sample_locations.add(get_objects_from_fact(fact)[0])


    def __call__(self, node):
        """Estimate the number of actions to reach the goal state from the current state."""
        state = node.state
        goal_cost = 0

        achieved_goals = set()
        for fact in state:
            if fact in self.goals:
                achieved_goals.add(fact)

        unachieved_goals = self.goals - achieved_goals

        for goal_fact in unachieved_goals:
            predicate = get_predicate_name(goal_fact)
            objects = get_objects_from_fact(goal_fact)

            if predicate == 'communicated_soil_data':
                goal_cost += 4 # navigate, sample_soil, navigate, communicate_soil_data
            elif predicate == 'communicated_rock_data':
                goal_cost += 4 # navigate, sample_rock, navigate, communicate_rock_data
            elif predicate == 'communicated_image_data':
                goal_cost += 5 # navigate, calibrate, take_image, navigate, communicate_image_data
        return goal_cost
