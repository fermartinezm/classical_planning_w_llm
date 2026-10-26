from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_objects_from_fact(fact_string):
    """
    Extracts objects from a PDDL fact string.
    For example, from '(at rover1 waypoint2)' it returns ['rover1', 'waypoint2'].
    Ignores the predicate name.
    """
    parts = fact_string[1:-1].split()
    return parts[1:]

def get_predicate_name(fact_string):
    """
    Extracts the predicate name from a PDDL fact string.
    For example, from '(at rover1 waypoint2)' it returns 'at'.
    """
    parts = fact_string[1:-1].split()
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
    It focuses on achieving each goal independently and sums up the estimated costs.
    The heuristic considers the necessary actions for navigation, sampling, calibration, imaging, and communication.

    # Assumptions:
    - The heuristic assumes that each goal predicate needs to be achieved independently.
    - It simplifies the problem by not considering interactions between different goals.
    - It assumes that for each goal, the rover will need to navigate to the required waypoint,
      perform the necessary actions (sample, calibrate, image), and then navigate to a communication point.
    - It does not explicitly consider store capacity constraints or optimize rover movements.

    # Heuristic Initialization
    - Extracts static information from the task, such as:
        - `can_traverse` relations for each rover.
        - Equipped capabilities of each rover (soil, rock, imaging).
        - Stores associated with each rover.
        - Visibility relations between waypoints and from objectives to waypoints.
        - Calibration targets for cameras.
        - Supported modes for cameras.
        - Cameras on board each rover.
        - Lander locations.

    # Step-By-Step Thinking for Computing Heuristic
    For each goal predicate in the goal state:
    1. Check if the goal is already achieved in the current state. If yes, cost is 0.
    2. If not achieved, estimate the cost based on the type of goal:
        - `communicated_soil_data(waypoint)`:
            - Cost for sampling soil (if not already sampled): 1 (sample_soil)
            - Cost for communicating soil data: 1 (communicate_soil_data)
            - Cost for navigation to sample waypoint (simplified to 1 if not at sample waypoint): 1 (navigate - simplified)
            - Cost for navigation to communication waypoint (simplified to 1 if not at communication waypoint): 1 (navigate - simplified)
            Total estimated cost: 3 or 4 (depending on if sampling is needed)
        - `communicated_rock_data(waypoint)`:
            - Cost for sampling rock (if not already sampled): 1 (sample_rock)
            - Cost for communicating rock data: 1 (communicate_rock_data)
            - Cost for navigation to sample waypoint (simplified to 1 if not at sample waypoint): 1 (navigate - simplified)
            - Cost for navigation to communication waypoint (simplified to 1 if not at communication waypoint): 1 (navigate - simplified)
            Total estimated cost: 3 or 4 (depending on if sampling is needed)
        - `communicated_image_data(objective, mode)`:
            - Cost for calibration (if not already calibrated): 1 (calibrate)
            - Cost for taking image: 1 (take_image)
            - Cost for communicating image data: 1 (communicate_image_data)
            - Cost for navigation to imaging waypoint (simplified to 1 if not at imaging waypoint): 1 (navigate - simplified)
            - Cost for navigation to communication waypoint (simplified to 1 if not at communication waypoint): 1 (navigate - simplified)
            Total estimated cost: 4 or 5 (depending on if calibration is needed)

    The heuristic sums up the estimated costs for all unmet goal predicates.
    This is a simplified estimate and does not guarantee optimality or admissibility, but aims for efficiency and informed search guidance.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.can_traverse = {} # {(rover, waypoint1, waypoint2)}
        self.equipped_for_soil_analysis = set() # {rover}
        self.equipped_for_rock_analysis = set() # {rover}
        self.equipped_for_imaging = set() # {rover}
        self.store_of = {} # {store: rover}
        self.visible = set() # {(waypoint1, waypoint2)}
        self.visible_from = {} # {objective: {waypoint}}
        self.calibration_target = {} # {camera: objective}
        self.supports = {} # {camera: {mode}}
        self.on_board = {} # {camera: rover}
        self.at_lander = {} # {lander: waypoint}

        for fact in static_facts:
            predicate = get_predicate_name(fact)
            objects = get_objects_from_fact(fact)
            if predicate == 'can_traverse':
                rover, wp1, wp2 = objects
                if rover not in self.can_traverse:
                    self.can_traverse[rover] = set()
                self.can_traverse[rover].add((wp1, wp2))
            elif predicate == 'equipped_for_soil_analysis':
                self.equipped_for_soil_analysis.add(objects[0])
            elif predicate == 'equipped_for_rock_analysis':
                self.equipped_for_rock_analysis.add(objects[0])
            elif predicate == 'equipped_for_imaging':
                self.equipped_for_imaging.add(objects[0])
            elif predicate == 'store_of':
                self.store_of[objects[0]] = objects[1]
            elif predicate == 'visible':
                self.visible.add((objects[0], objects[1]))
            elif predicate == 'visible_from':
                objective, waypoint = objects
                if objective not in self.visible_from:
                    self.visible_from[objective] = set()
                self.visible_from[objective].add(waypoint)
            elif predicate == 'calibration_target':
                self.calibration_target[objects[0]] = objects[1]
            elif predicate == 'supports':
                camera, mode = objects
                if camera not in self.supports:
                    self.supports[camera] = set()
                self.supports[camera].add(mode)
            elif predicate == 'on_board':
                self.on_board[objects[0]] = objects[1]
            elif predicate == 'at_lander':
                self.at_lander[objects[0]] = objects[1]


    def __call__(self, node):
        """Estimate the number of actions to reach the goal state from the current node's state."""
        state = node.state
        goal_cost = 0

        current_rover_locations = {}
        current_camera_calibrated = {}
        current_have_soil_analysis = set()
        current_have_rock_analysis = set()
        current_have_image = set()
        current_communicated_soil_data = set()
        current_communicated_rock_data = set()
        current_communicated_image_data = set()

        for fact in state:
            predicate = get_predicate_name(fact)
            objects = get_objects_from_fact(fact)
            if predicate == 'at':
                current_rover_locations[objects[0]] = objects[1]
            elif predicate == 'calibrated':
                current_camera_calibrated[objects[0]] = objects[1]
            elif predicate == 'have_soil_analysis':
                current_have_soil_analysis.add(tuple(objects))
            elif predicate == 'have_rock_analysis':
                current_have_rock_analysis.add(tuple(objects))
            elif predicate == 'have_image':
                current_have_image.add(tuple(objects))
            elif predicate == 'communicated_soil_data':
                current_communicated_soil_data.add(objects[0])
            elif predicate == 'communicated_rock_data':
                current_communicated_rock_data.add(objects[0])
            elif predicate == 'communicated_image_data':
                current_communicated_image_data.add(tuple(objects))

        for goal in self.goals:
            if goal in state:
                continue

            predicate = get_predicate_name(goal)
            objects = get_objects_from_fact(goal)

            if predicate == 'communicated_soil_data':
                waypoint = objects[0]
                if waypoint not in current_communicated_soil_data:
                    sample_needed = True
                    for rover in self.equipped_for_soil_analysis:
                        if (rover, waypoint) in current_have_soil_analysis:
                            sample_needed = False
                            break
                    if sample_needed:
                        goal_cost += 1 # sample_soil
                    goal_cost += 1 # communicate_soil_data
                    goal_cost += 1 # navigate to sample waypoint (simplified)
                    goal_cost += 1 # navigate to communication waypoint (simplified)

            elif predicate == 'communicated_rock_data':
                waypoint = objects[0]
                if waypoint not in current_communicated_rock_data:
                    sample_needed = True
                    for rover in self.equipped_for_rock_analysis:
                        if (rover, waypoint) in current_have_rock_analysis:
                            sample_needed = False
                            break
                    if sample_needed:
                        goal_cost += 1 # sample_rock
                    goal_cost += 1 # communicate_rock_data
                    goal_cost += 1 # navigate to sample waypoint (simplified)
                    goal_cost += 1 # navigate to communication waypoint (simplified)

            elif predicate == 'communicated_image_data':
                objective, mode = objects
                if tuple(objects) not in current_communicated_image_data:
                    image_needed = True
                    for rover in self.equipped_for_imaging:
                        if (rover, objective, mode) in current_have_image:
                            image_needed = False
                            break
                    if image_needed:
                        calibration_needed = True
                        for camera, rover in self.on_board.items():
                            if camera in self.calibration_target and self.calibration_target[camera] == objective and camera in self.supports and mode in self.supports[camera]:
                                if camera in current_camera_calibrated and current_camera_calibrated[camera] == rover:
                                    calibration_needed = False
                                    break
                        if calibration_needed:
                            goal_cost += 1 # calibrate
                        goal_cost += 1 # take_image
                    goal_cost += 1 # communicate_image_data
                    goal_cost += 1 # navigate to imaging waypoint (simplified)
                    goal_cost += 1 # navigate to communication waypoint (simplified)

        return goal_cost
