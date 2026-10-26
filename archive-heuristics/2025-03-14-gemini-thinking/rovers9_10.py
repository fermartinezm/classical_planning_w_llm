from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_objects_from_fact(fact_string):
    """
    Extracts objects from a PDDL fact string.
    For example, from '(at rover1 waypoint2)' it returns ['rover1', 'waypoint2'].
    """
    fact_content = fact_string[1:-1]  # Remove parentheses
    return fact_content.split()[1:] # Split by space and ignore predicate name

def get_predicate_name(fact_string):
    """
    Extracts predicate name from a PDDL fact string.
    For example, from '(at rover1 waypoint2)' it returns 'at'.
    """
    fact_content = fact_string[1:-1]  # Remove parentheses
    return fact_content.split()[0] # Split by space and take predicate name

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal conditions in the rovers domain.
    It focuses on achieving each goal condition independently and sums up the estimated costs.
    The heuristic considers the necessary actions for sampling, imaging, calibration, navigation, and communication.

    # Assumptions:
    - The heuristic assumes that each goal condition needs to be achieved independently.
    - It simplifies the problem by not explicitly planning for resource contention (e.g., store capacity) or optimal action ordering.
    - It assumes that visibility and traversability are symmetric if stated (though the code checks for stated visibility).
    - It underestimates the cost of navigation by assuming direct paths exist if visibility allows.

    # Heuristic Initialization
    - Extracts goal predicates from the task definition.
    - Preprocesses static facts to build efficient lookup structures for:
        - `visible` waypoints.
        - `can_traverse` paths for each rover.
        - `visible_from` objectives from waypoints.
        - `calibration_target` for each camera.
        - `supports` modes for each camera.
        - `at_lander` location.
        - `equipped_for_...` capabilities of rovers.
        - `store_of` each store and rover.
        - `on_board` cameras for each rover.

    # Step-By-Step Thinking for Computing Heuristic
    For each goal condition, the heuristic estimates the minimum number of actions required to satisfy it:

    1. For each goal condition (communicated_soil_data, communicated_rock_data, communicated_image_data):
        a. Check if the goal is already achieved in the current state. If yes, the cost for this goal is 0.
        b. If not achieved, estimate the cost based on the goal type:

            i. For `communicated_soil_data(waypoint)`:
               - If `communicated_soil_data(waypoint)` is not achieved:
                 - Cost += 1 (communicate_soil_data action)
                 - If `have_soil_analysis(rover, waypoint)` is not achieved:
                   - Cost += 1 (sample_soil action)
                   - If not at the waypoint with soil sample:
                     - Cost += 1 (navigate to waypoint with soil sample) - Assumes shortest path exists if visible

            ii. For `communicated_rock_data(waypoint)`:
                - Similar to `communicated_soil_data`, but for rock samples and `sample_rock` action.

            iii. For `communicated_image_data(objective, mode)`:
                - If `communicated_image_data(objective, mode)` is not achieved:
                  - Cost += 1 (communicate_image_data action)
                  - If `have_image(rover, objective, mode)` is not achieved:
                    - Cost += 1 (take_image action)
                    - If camera is not calibrated for the objective:
                      - Cost += 1 (calibrate action)
                      - If rover is not at a waypoint visible from the objective for calibration:
                        - Cost += 1 (navigate to waypoint visible from objective for calibration) - Assumes shortest path exists if visible
                    - If rover is not at a waypoint visible from the objective for taking image:
                      - Cost += 1 (navigate to waypoint visible from objective for taking image) - Assumes shortest path exists if visible

    2. Sum up the estimated costs for all goal conditions.
    3. Return the total estimated cost.

    This heuristic is admissible if we consider the minimum number of actions for each goal independently and assume optimal navigation.
    However, due to simplifications and greedy nature, it is primarily designed to be efficient and guide the search effectively, not necessarily to be admissible.
    """

    def __init__(self, task):
        """Initialize the rovers heuristic by preprocessing static facts and goals."""
        self.goals = task.goals
        static_facts = task.static

        self.visible_waypoints = {} # visible[waypoint1] = [waypoint2, waypoint3, ...]
        self.can_traverse_paths = {} # can_traverse_paths[rover] = {(waypoint1, waypoint2), ...}
        self.visible_from_objectives = {} # visible_from_objectives[objective] = [waypoint1, waypoint2, ...]
        self.calibration_targets = {} # calibration_targets[camera] = objective
        self.camera_supports_modes = {} # camera_supports_modes[camera] = [mode1, mode2, ...]
        self.lander_location = None
        self.equipped_for_soil_analysis_rovers = set()
        self.equipped_for_rock_analysis_rovers = set()
        self.equipped_for_imaging_rovers = set()
        self.store_of_rover = {} # store_of_rover[store] = rover
        self.on_board_cameras = {} # on_board_cameras[rover] = [camera1, camera2, ...]


        for fact in static_facts:
            predicate = get_predicate_name(fact)
            objects = get_objects_from_fact(fact)
            if predicate == 'visible':
                waypoint1, waypoint2 = objects
                if waypoint1 not in self.visible_waypoints:
                    self.visible_waypoints[waypoint1] = []
                self.visible_waypoints[waypoint1].append(waypoint2)
            elif predicate == 'can_traverse':
                rover, waypoint1, waypoint2 = objects
                if rover not in self.can_traverse_paths:
                    self.can_traverse_paths[rover] = set()
                self.can_traverse_paths[rover].add(tuple(sorted((waypoint1, waypoint2)))) # Treat path as bidirectional
            elif predicate == 'visible_from':
                objective, waypoint = objects
                if objective not in self.visible_from_objectives:
                    self.visible_from_objectives[objective] = []
                self.visible_from_objectives[objective].append(waypoint)
            elif predicate == 'calibration_target':
                camera, objective = objects
                self.calibration_targets[camera] = objective
            elif predicate == 'supports':
                camera, mode = objects
                if camera not in self.camera_supports_modes:
                    self.camera_supports_modes[camera] = []
                self.camera_supports_modes[camera].append(mode)
            elif predicate == 'at_lander':
                lander, waypoint = objects
                self.lander_location = waypoint
            elif predicate == 'equipped_for_soil_analysis':
                rover = objects[0]
                self.equipped_for_soil_analysis_rovers.add(rover)
            elif predicate == 'equipped_for_rock_analysis':
                rover = objects[0]
                self.equipped_for_rock_analysis_rovers.add(rover)
            elif predicate == 'equipped_for_imaging':
                rover = objects[0]
                self.equipped_for_imaging_rovers.add(rover)
            elif predicate == 'store_of':
                store, rover = objects
                self.store_of_rover[store] = rover
            elif predicate == 'on_board':
                camera, rover = objects
                if rover not in self.on_board_cameras:
                    self.on_board_cameras[rover] = []
                self.on_board_cameras[rover].append(camera)


    def __call__(self, node):
        """Compute the heuristic value for a given state."""
        state = node.state
        heuristic_value = 0

        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved, no cost

            predicate = get_predicate_name(goal)
            objects = get_objects_from_fact(goal)

            if predicate == 'communicated_soil_data':
                waypoint = objects[0]
                heuristic_value += 1 # communicate_soil_data action
                achieved_soil_analysis = False
                for fact in state:
                    if get_predicate_name(fact) == 'have_soil_analysis' and get_objects_from_fact(fact)[1] == waypoint:
                        achieved_soil_analysis = True
                        break
                if not achieved_soil_analysis:
                    heuristic_value += 1 # sample_soil action
                    # Navigation cost is simplified to 1 if not at soil sample location.
                    # More accurate navigation cost could be added if needed, e.g., shortest path.
                    at_soil_sample_location = False
                    for fact in state:
                        if get_predicate_name(fact) == 'at_soil_sample' and get_objects_from_fact(fact)[0] == waypoint:
                            at_soil_sample_location = True
                            break
                    if not at_soil_sample_location:
                        heuristic_value += 1 # navigate to waypoint with soil sample (simplified cost)


            elif predicate == 'communicated_rock_data':
                waypoint = objects[0]
                heuristic_value += 1 # communicate_rock_data action
                achieved_rock_analysis = False
                for fact in state:
                    if get_predicate_name(fact) == 'have_rock_analysis' and get_objects_from_fact(fact)[1] == waypoint:
                        achieved_rock_analysis = True
                        break
                if not achieved_rock_analysis:
                    heuristic_value += 1 # sample_rock action
                    at_rock_sample_location = False
                    for fact in state:
                        if get_predicate_name(fact) == 'at_rock_sample' and get_objects_from_fact(fact)[0] == waypoint:
                            at_rock_sample_location = True
                            break
                    if not at_rock_sample_location:
                        heuristic_value += 1 # navigate to waypoint with rock sample (simplified cost)


            elif predicate == 'communicated_image_data':
                objective, mode = objects
                heuristic_value += 1 # communicate_image_data action
                achieved_image = False
                for fact in state:
                    if get_predicate_name(fact) == 'have_image' and get_objects_from_fact(fact)[1] == objective and get_objects_from_fact(fact)[2] == mode:
                        achieved_image = True
                        break
                if not achieved_image:
                    heuristic_value += 1 # take_image action
                    camera_needed = None
                    for cam, target_obj in self.calibration_targets.items():
                        if target_obj == objective and mode in self.camera_supports_modes.get(cam, []):
                            camera_needed = cam
                            break
                    if camera_needed:
                        calibrated = False
                        rover_for_camera = None
                        for rover, cameras in self.on_board_cameras.items():
                            if camera_needed in cameras:
                                rover_for_camera = rover
                                break
                        if rover_for_camera:
                            for fact in state:
                                if get_predicate_name(fact) == 'calibrated' and get_objects_from_fact(fact)[0] == camera_needed and get_objects_from_fact(fact)[1] == rover_for_camera:
                                    calibrated = True
                                    break
                            if not calibrated:
                                heuristic_value += 1 # calibrate action
                                # Navigation for calibration is simplified
                                heuristic_value += 1 # navigate to waypoint for calibration (simplified cost)
                    else:
                        # If no suitable camera is found, this goal might be unachievable, or heuristic is underestimating.
                        pass # In a more robust heuristic, handle cases where goal is impossible or needs more sophisticated estimation.
                    # Navigation for taking image is simplified
                    heuristic_value += 1 # navigate to waypoint for taking image (simplified cost)


        return heuristic_value
