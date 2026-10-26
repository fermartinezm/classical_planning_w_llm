from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
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
    It considers the actions needed for navigation, sampling (soil and rock), calibrating cameras, taking images, and communicating data.

    # Assumptions:
    - Navigation cost between any two visible waypoints is 1.
    - Sampling (soil/rock) cost is 1 if the rover is at the sample location and equipped.
    - Calibration cost is 1 if the rover is at a waypoint visible from the objective and equipped for imaging.
    - Taking an image cost is 1 if the camera is calibrated, on board, supports the mode, and the rover is at a waypoint visible from the objective.
    - Communication cost is 1 if the rover is at a waypoint visible to the lander and has the data.
    - The heuristic is not necessarily admissible but aims to be informative and efficient for greedy best-first search.

    # Heuristic Initialization
    - Extracts static facts from the task to pre-calculate and store information about:
        - Waypoint visibility (`visible`, `visible_from`).
        - Traversability between waypoints (`can_traverse`).
        - Sample locations (`at_soil_sample`, `at_rock_sample`).
        - Rover equipment (`equipped_for_soil_analysis`, `equipped_for_rock_analysis`, `equipped_for_imaging`).
        - Store information (`store_of`).
        - Camera capabilities (`calibration_target`, `on_board`, `supports`).
        - Lander location (`at_lander`).

    # Step-By-Step Thinking for Computing Heuristic
    For each goal condition, the heuristic estimates the minimum number of actions needed to satisfy it, if it's not already satisfied in the current state.
    The total heuristic value is the sum of the estimated costs for all unsatisfied goal conditions.

    1. Initialize the heuristic value to 0.
    2. For each goal condition:
        - Check if the goal condition is already satisfied in the current state. If yes, the cost for this goal is 0.
        - If not, estimate the cost based on the type of goal:
            - `communicated_soil_data ?w`:
                - If `(communicated_soil_data ?w)` is not in the state:
                    - Cost += 1 (communicate_soil_data)
                    - If `(have_soil_analysis ?r ?w)` is not in the state:
                        - Cost += 1 (sample_soil)
                        - If rover is not at waypoint `?w`:
                            - Cost += 1 (navigate to ?w)
                    - If rover is not at a waypoint visible to lander:
                        - Cost += 1 (navigate to waypoint visible to lander)

            - `communicated_rock_data ?w`:
                - Similar to `communicated_soil_data`, but for rock samples and `sample_rock` action.

            - `communicated_image_data ?o ?m`:
                - If `(communicated_image_data ?o ?m)` is not in the state:
                    - Cost += 1 (communicate_image_data)
                    - If `(have_image ?r ?o ?m)` is not in the state:
                        - Cost += 1 (take_image)
                        - If camera is not calibrated for rover:
                            - Cost += 1 (calibrate)
                        - If rover is not at a waypoint visible from objective `?o`:
                            - Cost += 1 (navigate to waypoint visible from objective)
                    - If rover is not at a waypoint visible to lander:
                        - Cost += 1 (navigate to waypoint visible to lander)

    3. Return the total accumulated cost as the heuristic value.
    """

    def __init__(self, task):
        """Initialize the rovers heuristic by pre-calculating static information."""
        self.goals = task.goals
        static_facts = task.static

        self.can_traverse_map = {}
        self.visible_map = {}
        self.visible_from_map = {}
        self.at_soil_sample_waypoints = set()
        self.at_rock_sample_waypoints = set()
        self.equipped_for_soil_analysis_rovers = set()
        self.equipped_for_rock_analysis_rovers = set()
        self.equipped_for_imaging_rovers = set()
        self.store_of_map = {}
        self.calibration_target_map = {}
        self.on_board_map = {}
        self.supports_map = {}
        self.lander_location = None

        for fact in static_facts:
            if match(fact, 'can_traverse', '*', '*', '*'):
                parts = get_parts(fact)
                rover, wp1, wp2 = parts[1], parts[2], parts[3]
                if rover not in self.can_traverse_map:
                    self.can_traverse_map[rover] = set()
                self.can_traverse_map[rover].add((wp1, wp2))
            elif match(fact, 'visible', '*', '*'):
                parts = get_parts(fact)
                wp1, wp2 = parts[1], parts[2]
                if wp1 not in self.visible_map:
                    self.visible_map[wp1] = set()
                self.visible_map[wp1].add(wp2)
            elif match(fact, 'visible_from', '*', '*'):
                parts = get_parts(fact)
                objective, wp = parts[1], parts[2]
                if objective not in self.visible_from_map:
                    self.visible_from_map[objective] = set()
                self.visible_from_map[objective].add(wp)
            elif match(fact, 'at_soil_sample', '*'):
                self.at_soil_sample_waypoints.add(get_parts(fact)[1])
            elif match(fact, 'at_rock_sample', '*'):
                self.at_rock_sample_waypoints.add(get_parts(fact)[1])
            elif match(fact, 'equipped_for_soil_analysis', '*'):
                self.equipped_for_soil_analysis_rovers.add(get_parts(fact)[1])
            elif match(fact, 'equipped_for_rock_analysis', '*'):
                self.equipped_for_rock_analysis_rovers.add(get_parts(fact)[1])
            elif match(fact, 'equipped_for_imaging', '*'):
                self.equipped_for_imaging_rovers.add(get_parts(fact)[1])
            elif match(fact, 'store_of', '*', '*'):
                store, rover = get_parts(fact)[1], get_parts(fact)[2]
                self.store_of_map[store] = rover
            elif match(fact, 'calibration_target', '*', '*'):
                camera, objective = get_parts(fact)[1], get_parts(fact)[2]
                self.calibration_target_map[camera] = objective
            elif match(fact, 'on_board', '*', '*'):
                camera, rover = get_parts(fact)[1], get_parts(fact)[2]
                self.on_board_map[camera] = rover
            elif match(fact, 'supports', '*', '*'):
                camera, mode = get_parts(fact)[1], get_parts(fact)[2]
                if camera not in self.supports_map:
                    self.supports_map[camera] = set()
                self.supports_map[camera].add(mode)
            elif match(fact, 'at_lander', '*', '*'):
                self.lander_location = (get_parts(fact)[1], get_parts(fact)[2]) # (lander, waypoint)


    def __call__(self, node):
        """Calculate the heuristic value for a given state."""
        state = node.state
        heuristic_value = 0

        for goal in self.goals:
            if goal in state:
                continue

            if match(goal, 'communicated_soil_data', '*'):
                waypoint = get_parts(goal)[1]
                heuristic_value += 1 # communicate_soil_data
                rover = None
                for r in self.equipped_for_soil_analysis_rovers:
                    if f'(store_of {r}store {r})' in self.store_of_map and f'(at {r} *)' in state: # Find a rover that can do soil analysis and is in the state.
                        rover_name = r
                        rover_location_fact = next((fact for fact in state if match(fact, 'at', rover_name, '*')), None)
                        if rover_location_fact:
                            rover_location = get_parts(rover_location_fact)[2]
                            rover = rover_name
                            break
                if rover:
                    if f'(have_soil_analysis {rover} {waypoint})' not in state:
                        heuristic_value += 1 # sample_soil
                        if rover_location != waypoint:
                            heuristic_value += 1 # navigate to waypoint

            elif match(goal, 'communicated_rock_data', '*'):
                waypoint = get_parts(goal)[1]
                heuristic_value += 1 # communicate_rock_data
                rover = None
                for r in self.equipped_for_rock_analysis_rovers:
                    if f'(store_of {r}store {r})' in self.store_of_map and f'(at {r} *)' in state: # Find a rover that can do rock analysis and is in the state.
                        rover_name = r
                        rover_location_fact = next((fact for fact in state if match(fact, 'at', rover_name, '*')), None)
                        if rover_location_fact:
                            rover_location = get_parts(rover_location_fact)[2]
                            rover = rover_name
                            break
                if rover:
                    if f'(have_rock_analysis {rover} {waypoint})' not in state:
                        heuristic_value += 1 # sample_rock
                        if rover_location != waypoint:
                            heuristic_value += 1 # navigate to waypoint

            elif match(goal, 'communicated_image_data', '*', '*'):
                objective = get_parts(goal)[1]
                mode = get_parts(goal)[2]
                heuristic_value += 1 # communicate_image_data
                rover = None
                camera = None
                for r in self.equipped_for_imaging_rovers:
                    if f'(on_board * {r})' in self.on_board_map.values() and f'(at {r} *)' in state: # Find a rover with camera on board and is in the state.
                        rover_name = r
                        rover_location_fact = next((fact for fact in state if match(fact, 'at', rover_name, '*')), None)
                        if rover_location_fact:
                            rover_location = get_parts(rover_location_fact)[2]
                            rover = rover_name
                            camera_name = next((cam for cam, rov in self.on_board_map.items() if rov == rover_name), None)
                            camera = camera_name
                            break

                if rover and camera:
                    if f'(have_image {rover} {objective} {mode})' not in state:
                        heuristic_value += 1 # take_image
                        if f'(calibrated {camera} {rover})' not in state:
                            heuristic_value += 1 # calibrate
                        visible_waypoint_found = False
                        if objective in self.visible_from_map:
                            for wp in self.visible_from_map[objective]:
                                if rover_location == wp:
                                    visible_waypoint_found = True
                                    break
                        if not visible_waypoint_found:
                             heuristic_value += 1 # navigate to waypoint visible from objective


        return heuristic_value
