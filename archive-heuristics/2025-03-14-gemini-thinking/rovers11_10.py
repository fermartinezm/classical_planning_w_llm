from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import itertools

def get_objects_from_fact(fact_string):
    """
    Extracts objects from a PDDL fact string.
    For example, from '(at rover1 waypoint1)' it returns ['rover1', 'waypoint1'].
    Ignores the predicate name.
    """
    parts = fact_string[1:-1].split()
    return parts[1:]

def get_predicate_name(fact_string):
    """
    Extracts the predicate name from a PDDL fact string.
    For example, from '(at rover1 waypoint1)' it returns 'at'.
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
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal conditions
    in the Rovers domain. It focuses on the actions needed to collect samples (soil and rock),
    take images, and communicate the collected data back to the lander. The heuristic
    prioritizes achieving each goal condition individually and sums up the estimated costs.

    # Assumptions:
    - Rovers are always capable of performing actions if they have the necessary equipment.
    - Store capacity is not explicitly considered, assuming rovers have enough storage.
    - Navigation cost is estimated as 1 action for each waypoint traversal.
    - Communication can always be done if the rover and lander are at visible waypoints and rover has the data.
    - Calibration is needed before taking each image if the camera is not already calibrated.

    # Heuristic Initialization
    - Pre-processes static facts to build efficient lookup structures for:
        - `can_traverse`: to find reachable waypoints.
        - `visible`: to find visible waypoints from a given waypoint.
        - `visible_from`: to find waypoints visible from an objective.
        - `calibration_target`: to find the target objective for each camera.
        - `supports`: to find supported modes for each camera.
        - `at_lander`: to find the lander's location.

    # Step-By-Step Thinking for Computing Heuristic
    For each goal condition that is not met in the current state:
    1. Identify the goal type: `communicated_soil_data`, `communicated_rock_data`, or `communicated_image_data`.
    2. Estimate the cost to achieve each unmet goal condition:
        - For `communicated_soil_data(waypoint)`:
            - If `communicated_soil_data(waypoint)` is already true, cost is 0.
            - Otherwise, check if `have_soil_analysis(rover, waypoint)` is true.
                - If not, estimate cost to get soil analysis:
                    - Find a rover equipped for soil analysis.
                    - Find a rover with an empty store.
                    - Navigate the rover to the `waypoint` if not already there. (Estimated as 1 navigate action if not at waypoint, otherwise 0).
                    - Perform `sample_soil` action (cost 1).
                - If `have_soil_analysis(rover, waypoint)` is true, estimate cost to communicate:
                    - Find a rover that `have_soil_analysis(rover, waypoint)`.
                    - Navigate the rover to a waypoint visible from the lander's location (estimated as 1 navigate action if not at a visible waypoint, otherwise 0).
                    - Perform `communicate_soil_data` action (cost 1).
        - For `communicated_rock_data(waypoint)`:
            - Similar logic as `communicated_soil_data`, but using `sample_rock`, `have_rock_analysis`, and `communicate_rock_data`.
        - For `communicated_image_data(objective, mode)`:
            - If `communicated_image_data(objective, mode)` is already true, cost is 0.
            - Otherwise, check if `have_image(rover, objective, mode)` is true.
                - If not, estimate cost to get image:
                    - Find a rover equipped for imaging.
                    - Find a rover with a camera that `supports(camera, mode)` and `calibration_target(camera, objective)`.
                    - Check if the camera is `calibrated(camera, rover)`. If not, estimate cost to calibrate:
                        - Navigate the rover to a waypoint `waypoint` from which the `objective` is `visible_from(objective, waypoint)`. (Estimated as 1 navigate action if not at such a waypoint, otherwise 0).
                        - Perform `calibrate` action (cost 1).
                    - Navigate the rover to a waypoint `waypoint` from which the `objective` is `visible_from(objective, waypoint)`. (Estimated as 1 navigate action if not at such a waypoint, otherwise 0).
                    - Perform `take_image` action (cost 1).
                - If `have_image(rover, objective, mode)` is true, estimate cost to communicate:
                    - Find a rover that `have_image(rover, objective, mode)`.
                    - Navigate the rover to a waypoint visible from the lander's location (estimated as 1 navigate action if not at a visible waypoint, otherwise 0).
                    - Perform `communicate_image_data` action (cost 1).
    3. Sum up the estimated costs for all unmet goal conditions.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.can_traverse_map = {} # rover -> { (waypoint1, waypoint2), ... }
        self.visible_map = {} # waypoint -> { visible_waypoint1, ... }
        self.visible_from_map = {} # objective -> { waypoint1, ... }
        self.calibration_target_map = {} # camera -> objective
        self.supports_map = {} # camera -> { mode1, ... }
        self.equipped_for_soil_analysis_rovers = set()
        self.equipped_for_rock_analysis_rovers = set()
        self.equipped_for_imaging_rovers = set()
        self.store_of_map = {} # store -> rover
        self.on_board_cameras = {} # rover -> {camera1, ...}
        self.at_lander_location = None

        for fact in static_facts:
            if match(fact, "can_traverse", "*", "*", "*"):
                rover = get_objects_from_fact(fact)[0]
                wp1 = get_objects_from_fact(fact)[1]
                wp2 = get_objects_from_fact(fact)[2]
                if rover not in self.can_traverse_map:
                    self.can_traverse_map[rover] = set()
                self.can_traverse_map[rover].add((wp1, wp2))
            elif match(fact, "visible", "*", "*"):
                wp1 = get_objects_from_fact(fact)[0]
                wp2 = get_objects_from_fact(fact)[1]
                if wp1 not in self.visible_map:
                    self.visible_map[wp1] = set()
                self.visible_map[wp1].add(wp2)
            elif match(fact, "visible_from", "*", "*"):
                objective = get_objects_from_fact(fact)[0]
                waypoint = get_objects_from_fact(fact)[1]
                if objective not in self.visible_from_map:
                    self.visible_from_map[objective] = set()
                self.visible_from_map[objective].add(waypoint)
            elif match(fact, "calibration_target", "*", "*"):
                camera = get_objects_from_fact(fact)[0]
                objective = get_objects_from_fact(fact)[1]
                self.calibration_target_map[camera] = objective
            elif match(fact, "supports", "*", "*"):
                camera = get_objects_from_fact(fact)[0]
                mode = get_objects_from_fact(fact)[1]
                if camera not in self.supports_map:
                    self.supports_map[camera] = set()
                self.supports_map[camera].add(mode)
            elif match(fact, "equipped_for_soil_analysis", "*"):
                rover = get_objects_from_fact(fact)[0]
                self.equipped_for_soil_analysis_rovers.add(rover)
            elif match(fact, "equipped_for_rock_analysis", "*"):
                rover = get_objects_from_fact(fact)[0]
                self.equipped_for_rock_analysis_rovers.add(rover)
            elif match(fact, "equipped_for_imaging", "*"):
                rover = get_objects_from_fact(fact)[0]
                self.equipped_for_imaging_rovers.add(rover)
            elif match(fact, "store_of", "*", "*"):
                store = get_objects_from_fact(fact)[0]
                rover = get_objects_from_fact(fact)[1]
                self.store_of_map[store] = rover
            elif match(fact, "on_board", "*", "*"):
                camera = get_objects_from_fact(fact)[0]
                rover = get_objects_from_fact(fact)[1]
                if rover not in self.on_board_cameras:
                    self.on_board_cameras[rover] = set()
                self.on_board_cameras[rover].add(camera)
            elif match(fact, "at_lander", "*", "*"):
                self.at_lander_location = get_objects_from_fact(fact)[1]


    def __call__(self, node):
        """Estimate the minimum cost to achieve all goal conditions."""
        state = node.state
        heuristic_value = 0

        goal_conditions_unmet = []
        for goal in self.goals:
            if goal not in state:
                goal_conditions_unmet.append(goal)

        for goal in goal_conditions_unmet:
            predicate = get_predicate_name(goal)
            objects = get_objects_from_fact(goal)

            if predicate == 'communicated_soil_data':
                waypoint_goal = objects[0]
                if goal not in state:
                    rover_with_soil_analysis = None
                    for r in self.equipped_for_soil_analysis_rovers:
                        if f'(have_soil_analysis {r} {waypoint_goal})' in state:
                            rover_with_soil_analysis = r
                            break
                    if not rover_with_soil_analysis:
                        rover_for_sample = next((r for r in self.equipped_for_soil_analysis_rovers if f'(empty {self.store_of_map.get(f"{self.store_of_map.get(r)}store", r)})' in state), None) # Assuming store name is rovername + store
                        if rover_for_sample:
                            rover_location_fact = next((fact for fact in state if match(fact, "at", rover_for_sample, "*")), None)
                            rover_location = get_objects_from_fact(rover_location_fact)[1] if rover_location_fact else None
                            if rover_location != waypoint_goal:
                                heuristic_value += 1 # navigate
                            heuristic_value += 1 # sample_soil
                        else:
                            heuristic_value += 2 # very rough estimate if no rover can sample. In reality might be unsolvable or require store management.

                    rover_for_comm = next((r for r in self.equipped_for_soil_analysis_rovers if f'(have_soil_analysis {r} {waypoint_goal})' in state), None)
                    if rover_for_comm:
                        rover_location_fact = next((fact for fact in state if match(fact, "at", rover_for_comm, "*")), None)
                        rover_location = get_objects_from_fact(rover_location_fact)[1] if rover_location_fact else None
                        visible_from_lander = False
                        if rover_location:
                            if self.at_lander_location and rover_location in self.visible_map and self.at_lander_location in self.visible_map[rover_location]:
                                visible_from_lander = True
                        if not visible_from_lander:
                            heuristic_value += 1 # navigate to communication point
                        heuristic_value += 1 # communicate_soil_data


            elif predicate == 'communicated_rock_data':
                waypoint_goal = objects[0]
                if goal not in state:
                    rover_with_rock_analysis = None
                    for r in self.equipped_for_rock_analysis_rovers:
                        if f'(have_rock_analysis {r} {waypoint_goal})' in state:
                            rover_with_rock_analysis = r
                            break
                    if not rover_with_rock_analysis:
                        rover_for_sample = next((r for r in self.equipped_for_rock_analysis_rovers if f'(empty {self.store_of_map.get(f"{self.store_of_map.get(r)}store", r)})' in state), None) # Assuming store name is rovername + store
                        if rover_for_sample:
                            rover_location_fact = next((fact for fact in state if match(fact, "at", rover_for_sample, "*")), None)
                            rover_location = get_objects_from_fact(rover_location_fact)[1] if rover_location_fact else None
                            if rover_location != waypoint_goal:
                                heuristic_value += 1 # navigate
                            heuristic_value += 1 # sample_rock
                        else:
                            heuristic_value += 2 # very rough estimate if no rover can sample.

                    rover_for_comm = next((r for r in self.equipped_for_rock_analysis_rovers if f'(have_rock_analysis {r} {waypoint_goal})' in state), None)
                    if rover_for_comm:
                        rover_location_fact = next((fact for fact in state if match(fact, "at", rover_for_comm, "*")), None)
                        rover_location = get_objects_from_fact(rover_location_fact)[1] if rover_location_fact else None
                        visible_from_lander = False
                        if rover_location:
                            if self.at_lander_location and rover_location in self.visible_map and self.at_lander_location in self.visible_map[rover_location]:
                                visible_from_lander = True
                        if not visible_from_lander:
                            heuristic_value += 1 # navigate to communication point
                        heuristic_value += 1 # communicate_rock_data


            elif predicate == 'communicated_image_data':
                objective_goal = objects[0]
                mode_goal = objects[1]
                if goal not in state:
                    rover_with_image = None
                    for r in self.equipped_for_imaging_rovers:
                        if f'(have_image {r} {objective_goal} {mode_goal})' in state:
                            rover_with_image = r
                            break
                    if not rover_with_image:
                        rover_for_image = next((r for r in self.equipped_for_imaging_rovers for cam in self.on_board_cameras.get(r, [])
                                                 if objective_goal == self.calibration_target_map.get(cam) and mode_goal in self.supports_map.get(cam, set())), None)

                        if rover_for_image:
                            camera = next(cam for cam in self.on_board_cameras.get(rover_for_image, [])
                                                 if objective_goal == self.calibration_target_map.get(cam) and mode_goal in self.supports_map.get(cam, set()))

                            if f'(calibrated {camera} {rover_for_image})' not in state:
                                calibration_waypoint = next((wp for wp in self.visible_from_map.get(objective_goal, []) if wp), None) # just pick first visible waypoint for calibration
                                rover_location_fact = next((fact for fact in state if match(fact, "at", rover_for_image, "*")), None)
                                rover_location = get_objects_from_fact(rover_location_fact)[1] if rover_location_fact else None
                                if rover_location != calibration_waypoint and calibration_waypoint: # only navigate if there is a calibration waypoint
                                    heuristic_value += 1 # navigate for calibration
                                heuristic_value += 1 # calibrate

                            image_waypoint = next((wp for wp in self.visible_from_map.get(objective_goal, []) if wp), None) # just pick first visible waypoint for imaging
                            rover_location_fact = next((fact for fact in state if match(fact, "at", rover_for_image, "*")), None)
                            rover_location = get_objects_from_fact(rover_location_fact)[1] if rover_location_fact else None
                            if rover_location != image_waypoint and image_waypoint: # only navigate if there is an image waypoint
                                heuristic_value += 1 # navigate for image
                            heuristic_value += 1 # take_image
                        else:
                            heuristic_value += 3 # very rough estimate if no rover can take image.

                    rover_for_comm = next((r for r in self.equipped_for_imaging_rovers if f'(have_image {r} {objective_goal} {mode_goal})' in state), None)
                    if rover_for_comm:
                        rover_location_fact = next((fact for fact in state if match(fact, "at", rover_for_comm, "*")), None)
                        rover_location = get_objects_from_fact(rover_location_fact)[1] if rover_location_fact else None
                        visible_from_lander = False
                        if rover_location:
                            if self.at_lander_location and rover_location in self.visible_map and self.at_lander_location in self.visible_map[rover_location]:
                                visible_from_lander = True
                        if not visible_from_lander:
                            heuristic_value += 1 # navigate to communication point
                        heuristic_value += 1 # communicate_image_data

        return heuristic_value
