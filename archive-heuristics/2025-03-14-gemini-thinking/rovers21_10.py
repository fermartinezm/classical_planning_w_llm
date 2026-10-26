from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import itertools

def get_objects_from_fact(fact):
    """
    Extract the objects from a PDDL fact string.
    For example, from '(at rover1 waypoint1)' it returns ['rover1', 'waypoint1'].
    """
    fact_content = fact[1:-1].split()
    return fact_content[1:]  # Return objects, skip predicate name

def get_predicate_name(fact):
    """
    Extract the predicate name from a PDDL fact string.
    For example, from '(at rover1 waypoint1)' it returns 'at'.
    """
    fact_content = fact[1:-1].split()
    return fact_content[0]

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
    It considers the necessary steps for each goal, such as navigation, sampling, calibration, imaging, and communication.
    The heuristic is based on a relaxed plan approach, summing up the estimated cost for each goal independently.

    # Assumptions:
    - Achieving each goal predicate is independent of others in terms of action costs.
    - Navigation cost is estimated as 1 if navigation is needed.
    - Sampling, calibration, imaging, and communication actions each have a cost of 1.
    - The heuristic is not guaranteed to be admissible but aims for efficiency and informed search guidance.

    # Heuristic Initialization
    - Extracts goal predicates from the task definition.
    - Pre-processes static facts to build efficient lookup structures for:
        - `can_traverse` routes for each rover.
        - `visible` waypoint pairs.
        - `visible_from` objective-waypoint pairs.
        - `calibration_target` pairs.
        - `supports` camera-mode pairs.
        - `at_soil_sample` waypoints.
        - `at_rock_sample` waypoints.
        - `store_of` rover-store pairs.
        - `on_board` camera-rover pairs.
        - `equipped_for_soil_analysis` rovers.
        - `equipped_for_rock_analysis` rovers.
        - `equipped_for_imaging` rovers.

    # Step-By-Step Thinking for Computing Heuristic
    For each goal predicate in the goal state:
    1. Check if the goal is already achieved in the current state. If yes, the cost for this goal is 0.
    2. If not achieved, estimate the cost based on the goal type:
        - For `communicated_image_data objective mode`:
            a. Cost += 1 (for 'communicate_image_data' action).
            b. Check if 'have_image rover objective mode' is achieved. If not:
               i.  Cost += 1 (for 'take_image' action).
               ii. Check if camera is calibrated for the rover. If not:
                   1. Cost += 1 (for 'calibrate' action).
                   2. Cost += 1 (for navigation to a waypoint visible from calibration target objective, if needed).
               iii. Cost += 1 (for navigation to a waypoint visible from the objective, if needed).
        - For `communicated_soil_data waypoint`:
            a. Cost += 1 (for 'communicate_soil_data' action).
            b. Check if 'have_soil_analysis rover waypoint' is achieved. If not:
                i. Cost += 1 (for 'sample_soil' action).
                ii. Cost += 1 (for navigation to the soil sample waypoint, if needed).
        - For `communicated_rock_data waypoint`:
            a. Cost += 1 (for 'communicate_rock_data' action).
            b. Check if 'have_rock_analysis rover waypoint' is achieved. If not:
                i. Cost += 1 (for 'sample_rock' action).
                ii. Cost += 1 (for navigation to the rock sample waypoint, if needed).

    The total heuristic value is the sum of the estimated costs for each unachieved goal predicate.
    Navigation costs are simplified to 1 if navigation is deemed necessary to reach a required waypoint.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.can_traverse_routes = {}
        self.visible_waypoints = set()
        self.visible_from_objectives = {}
        self.calibration_targets = {}
        self.supports_camera_modes = {}
        self.at_soil_sample_waypoints = set()
        self.at_rock_sample_waypoints = set()
        self.store_of_rovers = {}
        self.on_board_cameras = {}
        self.equipped_for_soil_analysis_rovers = set()
        self.equipped_for_rock_analysis_rovers = set()
        self.equipped_for_imaging_rovers = set()
        self.lander_location = None

        for fact in static_facts:
            if match(fact, "can_traverse", "*", "*", "*"):
                rover, wp1, wp2 = get_objects_from_fact(fact)
                if rover not in self.can_traverse_routes:
                    self.can_traverse_routes[rover] = set()
                self.can_traverse_routes[rover].add(tuple(sorted((wp1, wp2))))
            elif match(fact, "visible", "*", "*"):
                wp1, wp2 = get_objects_from_fact(fact)
                self.visible_waypoints.add(tuple(sorted((wp1, wp2))))
            elif match(fact, "visible_from", "*", "*"):
                objective, waypoint = get_objects_from_fact(fact)
                if objective not in self.visible_from_objectives:
                    self.visible_from_objectives[objective] = set()
                self.visible_from_objectives[objective].add(waypoint)
            elif match(fact, "calibration_target", "*", "*"):
                camera, objective = get_objects_from_fact(fact)
                self.calibration_targets[camera] = objective
            elif match(fact, "supports", "*", "*"):
                camera, mode = get_objects_from_fact(fact)
                if camera not in self.supports_camera_modes:
                    self.supports_camera_modes[camera] = set()
                self.supports_camera_modes[camera].add(mode)
            elif match(fact, "at_soil_sample", "*"):
                waypoint = get_objects_from_fact(fact)[0]
                self.at_soil_sample_waypoints.add(waypoint)
            elif match(fact, "at_rock_sample", "*"):
                waypoint = get_objects_from_fact(fact)[0]
                self.at_rock_sample_waypoints.add(waypoint)
            elif match(fact, "store_of", "*", "*"):
                store, rover = get_objects_from_fact(fact)
                self.store_of_rovers[rover] = store
            elif match(fact, "on_board", "*", "*"):
                camera, rover = get_objects_from_fact(fact)
                if rover not in self.on_board_cameras:
                    self.on_board_cameras[rover] = set()
                self.on_board_cameras[rover].add(camera)
            elif match(fact, "equipped_for_soil_analysis", "*"):
                rover = get_objects_from_fact(fact)[0]
                self.equipped_for_soil_analysis_rovers.add(rover)
            elif match(fact, "equipped_for_rock_analysis", "*"):
                rover = get_objects_from_fact(fact)[0]
                self.equipped_for_rock_analysis_rovers.add(rover)
            elif match(fact, "equipped_for_imaging", "*"):
                rover = get_objects_from_fact(fact)[0]
                self.equipped_for_imaging_rovers.add(rover)
            elif match(fact, "at_lander", "*", "*"):
                _, waypoint = get_objects_from_fact(fact)
                self.lander_location = waypoint


    def __call__(self, node):
        """Estimate the minimum cost to achieve all goal conditions."""
        state = node.state
        heuristic_value = 0

        goal_facts_not_achieved = [goal for goal in self.goals if goal not in state]

        for goal_fact in goal_facts_not_achieved:
            predicate = get_predicate_name(goal_fact)
            objects = get_objects_from_fact(goal_fact)

            if predicate == 'communicated_image_data':
                objective, mode = objects
                rover_needed = None
                camera_needed = None

                # Find a rover capable of imaging and a camera supporting the mode
                for r in self.equipped_for_imaging_rovers:
                    for cam in self.on_board_cameras.get(r, []):
                        if mode in self.supports_camera_modes.get(cam, set()):
                            rover_needed = r
                            camera_needed = cam
                            break
                    if rover_needed:
                        break
                if not rover_needed: # No capable rover found, instance might be unsolvable, but heuristic should still return a value
                    heuristic_value += 100 # High cost if no capable rover
                    continue


                heuristic_value += 1 # communicate_image_data action

                have_image_fact = f'(have_image {rover_needed} {objective} {mode})'
                if have_image_fact not in state:
                    heuristic_value += 1 # take_image action

                    calibrated_fact = f'(calibrated {camera_needed} {rover_needed})'
                    if calibrated_fact not in state:
                        heuristic_value += 1 # calibrate action
                        calibration_objective = self.calibration_targets.get(camera_needed)
                        current_rover_location = None
                        for fact in state:
                            if match(fact, 'at', rover_needed, '*'):
                                current_rover_location = get_objects_from_fact(fact)[1]
                                break
                        visible_calibration_waypoint_needed = False
                        if calibration_objective:
                            visible_calibration_waypoints = self.visible_from_objectives.get(calibration_objective, set())
                            if current_rover_location not in visible_calibration_waypoints:
                                visible_calibration_waypoint_needed = True
                        if visible_calibration_waypoint_needed:
                            heuristic_value += 1 # navigate to waypoint for calibration

                    current_rover_location = None
                    for fact in state:
                        if match(fact, 'at', rover_needed, '*'):
                            current_rover_location = get_objects_from_fact(fact)[1]
                            break

                    visible_objective_waypoints = self.visible_from_objectives.get(objective, set())
                    if current_rover_location not in visible_objective_waypoints:
                        heuristic_value += 1 # navigate to waypoint for imaging

                current_rover_location = None
                for fact in state:
                    if match(fact, 'at', rover_needed, '*'):
                        current_rover_location = get_objects_from_fact(fact)[1]
                        break

                if self.lander_location:
                    visible_lander_waypoint_needed = True
                    if self.lander_location and current_rover_location:
                        if tuple(sorted((current_rover_location, self.lander_location))) in self.visible_waypoints:
                            visible_lander_waypoint_needed = False
                    if visible_lander_waypoint_needed:
                        heuristic_value += 1 # navigate to waypoint for communication


            elif predicate == 'communicated_soil_data':
                waypoint = objects[0]
                rover_needed = None
                for r in self.equipped_for_soil_analysis_rovers:
                    rover_needed = r
                    break
                if not rover_needed:
                    heuristic_value += 100
                    continue

                heuristic_value += 1 # communicate_soil_data action

                have_soil_analysis_fact = f'(have_soil_analysis {rover_needed} {waypoint})'
                if have_soil_analysis_fact not in state:
                    heuristic_value += 1 # sample_soil action

                    current_rover_location = None
                    for fact in state:
                        if match(fact, 'at', rover_needed, '*'):
                            current_rover_location = get_objects_from_fact(fact)[1]
                            break
                    if current_rover_location != waypoint:
                        heuristic_value += 1 # navigate to soil sample waypoint

                current_rover_location = None
                for fact in state:
                    if match(fact, 'at', rover_needed, '*'):
                        current_rover_location = get_objects_from_fact(fact)[1]
                        break
                if self.lander_location:
                    visible_lander_waypoint_needed = True
                    if self.lander_location and current_rover_location:
                        if tuple(sorted((current_rover_location, self.lander_location))) in self.visible_waypoints:
                            visible_lander_waypoint_needed = False
                    if visible_lander_waypoint_needed:
                        heuristic_value += 1 # navigate to waypoint for communication


            elif predicate == 'communicated_rock_data':
                waypoint = objects[0]

                rover_needed = None
                for r in self.equipped_for_rock_analysis_rovers:
                    rover_needed = r
                    break
                if not rover_needed:
                    heuristic_value += 100
                    continue

                heuristic_value += 1 # communicate_rock_data action
                have_rock_analysis_fact = f'(have_rock_analysis {rover_needed} {waypoint})'
                if have_rock_analysis_fact not in state:
                    heuristic_value += 1 # sample_rock action
                    current_rover_location = None
                    for fact in state:
                        if match(fact, 'at', rover_needed, '*'):
                            current_rover_location = get_objects_from_fact(fact)[1]
                            break
                    if current_rover_location != waypoint:
                        heuristic_value += 1 # navigate to rock sample waypoint

                current_rover_location = None
                for fact in state:
                    if match(fact, 'at', rover_needed, '*'):
                        current_rover_location = get_objects_from_fact(fact)[1]
                        break

                if self.lander_location:
                    visible_lander_waypoint_needed = True
                    if self.lander_location and current_rover_location:
                        if tuple(sorted((current_rover_location, self.lander_location))) in self.visible_waypoints:
                            visible_lander_waypoint_needed = False
                    if visible_lander_waypoint_needed:
                        heuristic_value += 1 # navigate to waypoint for communication
        return heuristic_value

