from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming this is the base class

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    uncommunicated goal facts. It sums up the estimated steps for each goal
    independently, including steps for data collection (taking samples or images),
    analysis/processing (analyzing samples, calibrating cameras, taking images),
    and communication, plus a simplified cost for necessary movements.

    # Assumptions
    - The heuristic assumes that if a communication goal is specified for a
      sample at a waypoint or an image of an objective in a mode, the static
      prerequisites for collecting that data exist (e.g., a soil sample exists
      at the waypoint, a rover is equipped for the task, a camera supports the
      mode, an objective is visible from some waypoint, a calibration target
      exists and is visible from some waypoint for imaging tasks).
    - It ignores resource contention (e.g., multiple goals needing the same
      rover, store, or camera simultaneously).
    - Movement cost is simplified to 1 action per required move (e.g., move
      to sample location, move to image location, move to calibration location,
      move to lander for communication). This ignores actual path length.

    # Heuristic Initialization
    The heuristic pre-processes static facts from the task to efficiently
    determine:
    - The lander's location.
    - Which rovers have which capabilities (soil, rock, imaging).
    - Which store belongs to which rover.
    - Information about cameras (which rover they are on, which modes they
      support, and their calibration target).
    - Which objectives are visible from which waypoints.
    - Which waypoints have soil or rock samples.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:
    1. Initialize the total heuristic cost to 0.
    2. Create sets or dictionaries for quick lookup of intermediate facts
       present in the current state (e.g., `have_soil_analysis`, `have_image`,
       `calibrated`).
    3. Iterate through each goal fact specified in the task.
    4. If a goal fact is already present in the current state, it contributes 0
       to the heuristic; continue to the next goal.
    5. If the goal is `(communicated_soil_data ?w)`:
       - Add 1 for the `communicate_data` action.
       - Add 1 for the necessary `move` action to the lander location.
       - If no `(have_soil_analysis ?r ?w)` fact exists in the current state
         for any rover `?r`: Add 1 for the `analyze_soil_sample` action.
       - If no `(have_soil_sample ?s ?w)` fact exists in the current state
         for any store `?s`: Add 1 for the `take_soil_sample` action and
         add 1 for the necessary `move` action to waypoint `?w`.
    6. If the goal is `(communicated_rock_data ?w)`:
       - Add 1 for the `communicate_data` action.
       - Add 1 for the necessary `move` action to the lander location.
       - If no `(have_rock_analysis ?r ?w)` fact exists in the current state
         for any rover `?r`: Add 1 for the `analyze_rock_sample` action.
       - If no `(have_rock_sample ?s ?w)` fact exists in the current state
         for any store `?s`: Add 1 for the `take_rock_sample` action and
         add 1 for the necessary `move` action to waypoint `?w`.
    7. If the goal is `(communicated_image_data ?o ?m)`:
       - Add 1 for the `communicate_data` action.
       - Add 1 for the necessary `move` action to the lander location.
       - If no `(have_image ?r ?o ?m)` fact exists in the current state
         for any rover `?r`: Add 1 for the `take_image` action and
         add 1 for the necessary `move` action to a waypoint visible from `?o`.
       - Check if *any* camera `?c` on *any* rover `?r` that supports mode `?m`
         (using pre-processed static info) is currently calibrated
         (`(calibrated ?c ?r)` in state). If no such calibrated camera is found:
         Add 1 for the `calibrate_camera` action and add 1 for the necessary
         `move` action to a waypoint visible from the calibration target of
         a suitable camera.
    8. The total heuristic value is the sum of costs calculated for all
       unachieved goals.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        self.lander_location = None
        self.rover_capabilities = {} # {rover: {soil: bool, rock: bool, imaging: bool}}
        self.rover_store = {} # {rover: store}
        self.camera_info = {} # {camera: {rover: rover, modes: set, cal_target: objective}}
        self.objective_visibility = {} # {objective: set of waypoints}
        self.soil_sample_locations = set()
        self.rock_sample_locations = set()

        # Collect all rovers mentioned in store_of first to initialize capabilities dict
        all_rovers = {get_parts(fact)[2] for fact in static_facts if match(fact, "store_of", "*", "*")}
        for rover in all_rovers:
            self.rover_capabilities[rover] = {'soil': False, 'rock': False, 'imaging': False}

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]

            if predicate == "at_lander" and len(parts) == 3:
                self.lander_location = parts[2]
            elif predicate == "equipped_for_soil_analysis" and len(parts) == 2:
                rover = parts[1]
                if rover in self.rover_capabilities:
                     self.rover_capabilities[rover]['soil'] = True
            elif predicate == "equipped_for_rock_analysis" and len(parts) == 2:
                rover = parts[1]
                if rover in self.rover_capabilities:
                     self.rover_capabilities[rover]['rock'] = True
            elif predicate == "equipped_for_imaging" and len(parts) == 2:
                rover = parts[1]
                if rover in self.rover_capabilities:
                     self.rover_capabilities[rover]['imaging'] = True
            elif predicate == "store_of" and len(parts) == 3:
                store, rover = parts[1], parts[2]
                self.rover_store[rover] = store
            elif predicate == "supports" and len(parts) == 3:
                camera, mode = parts[1], parts[2]
                if camera not in self.camera_info:
                    self.camera_info[camera] = {'rover': None, 'modes': set(), 'cal_target': None}
                self.camera_info[camera]['modes'].add(mode)
            elif predicate == "on_board" and len(parts) == 3:
                camera, rover = parts[1], parts[2]
                if camera not in self.camera_info:
                    self.camera_info[camera] = {'rover': None, 'modes': set(), 'cal_target': None}
                self.camera_info[camera]['rover'] = rover
            elif predicate == "calibration_target" and len(parts) == 3:
                camera, objective = parts[1], parts[2]
                if camera not in self.camera_info:
                    self.camera_info[camera] = {'rover': None, 'modes': set(), 'cal_target': None}
                self.camera_info[camera]['cal_target'] = objective
            elif predicate == "visible_from" and len(parts) == 3:
                objective, waypoint = parts[1], parts[2]
                if objective not in self.objective_visibility:
                    self.objective_visibility[objective] = set()
                self.objective_visibility[objective].add(waypoint)
            elif predicate == "at_soil_sample" and len(parts) == 2:
                self.soil_sample_locations.add(parts[1])
            elif predicate == "at_rock_sample" and len(parts) == 2:
                self.rock_sample_locations.add(parts[1])

        # Ensure rover is set for cameras if on_board fact appeared later
        for fact in static_facts:
             if match(fact, "on_board", "?c", "?r"):
                 camera, rover = get_parts(fact)[1], get_parts(fact)[2]
                 if camera in self.camera_info:
                     self.camera_info[camera]['rover'] = rover
                 else:
                      # This case implies supports/cal_target were not seen for this camera,
                      # but on_board was. Initialize camera info.
                      self.camera_info[camera] = {'rover': rover, 'modes': set(), 'cal_target': None}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Pre-process state facts for faster lookups
        have_soil_analysis = {get_parts(fact)[2] for fact in state if match(fact, "have_soil_analysis", "*", "*")}
        have_rock_analysis = {get_parts(fact)[2] for fact in state if match(fact, "have_rock_analysis", "*", "*")}
        have_soil_sample = {get_parts(fact)[2] for fact in state if match(fact, "have_soil_sample", "*", "*")}
        have_rock_sample = {get_parts(fact)[2] for fact in state if match(fact, "have_rock_sample", "*", "*")}
        have_image = {(get_parts(fact)[2], get_parts(fact)[3]) for fact in state if match(fact, "have_image", "*", "*", "*")} # Store as (objective, mode)
        calibrated_cameras = {(get_parts(fact)[1], get_parts(fact)[2]) for fact in state if match(fact, "calibrated", "*", "*")} # Store as (camera, rover)
        # Rover locations are not strictly needed for this simplified movement heuristic

        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts: continue # Skip malformed goals

            predicate = parts[0]

            if predicate == "communicated_soil_data" and len(parts) == 2:
                waypoint = parts[1]
                cost_g = 0
                cost_g += 1 # communicate
                cost_g += 1 # move to lander

                if waypoint not in have_soil_analysis:
                    cost_g += 1 # analyze
                    if waypoint not in have_soil_sample:
                        cost_g += 1 # take_sample
                        cost_g += 1 # move to waypoint

                total_cost += cost_g

            elif predicate == "communicated_rock_data" and len(parts) == 2:
                waypoint = parts[1]
                cost_g = 0
                cost_g += 1 # communicate
                cost_g += 1 # move to lander

                if waypoint not in have_rock_analysis:
                    cost_g += 1 # analyze
                    if waypoint not in have_rock_sample:
                        cost_g += 1 # take_sample
                        cost_g += 1 # move to waypoint

                total_cost += cost_g

            elif predicate == "communicated_image_data" and len(parts) == 3:
                objective = parts[1]
                mode = parts[2]
                cost_g = 0
                cost_g += 1 # communicate
                cost_g += 1 # move to lander

                if (objective, mode) not in have_image:
                    cost_g += 1 # take_image
                    cost_g += 1 # move to image waypoint (waypoint visible from objective)

                    # Check if *any* suitable camera is calibrated
                    suitable_calibrated_found = False
                    for camera, info in self.camera_info.items():
                        if mode in info['modes'] and info['rover'] is not None: # Camera supports mode and is on a rover
                            rover = info['rover']
                            if (camera, rover) in calibrated_cameras:
                                suitable_calibrated_found = True
                                break # Found one

                    if not suitable_calibrated_found:
                        cost_g += 1 # calibrate
                        cost_g += 1 # move to calibration waypoint (waypoint visible from cal target)

                total_cost += cost_g

            # Add other goal types if they exist in the domain but not examples
            # elif predicate == "communicated_data" and len(parts) == 2: # Example of a generic data communication goal
            #     data_item = parts[1]
            #     # Logic would depend on what 'data_item' represents and its prerequisites
            #     pass # Placeholder


        return total_cost
