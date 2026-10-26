from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Used for infinity

def get_parts(fact):
    """Helper function to parse a PDDL fact string into its components."""
    # Remove parentheses and split by spaces
    return fact[1:-1].split()

def match(fact, *args):
    """Helper function to check if a fact matches a pattern using fnmatch."""
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments
    if len(parts) != len(args):
        return False
    # Use fnmatch for pattern matching on each part
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Estimates the number of actions required to reach a goal state.
    Designed for greedy best-first search, so it is not admissible.
    Focuses on the main steps required for each goal (acquire data, communicate data)
    with simplified navigation costs.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static facts from the task.

        Heuristic Initialization:
        Static facts are parsed to build data structures representing:
        - The lander's location.
        - Which rovers have which equipment (soil, rock, imaging).
        - Which store belongs to which rover.
        - Information about cameras (which rover they are on, which modes they support,
          and their calibration target objective).
        - Which waypoints are visible from which objectives (used for imaging and calibration).
        - Which waypoints are visible from the lander's location (for communication).
        """
        self.goals = task.goals
        static_facts = task.static

        # Store static information
        self.lander_location = {}
        self.rover_equipment = {} # {rover: {equipment_type}}
        self.store_owners = {} # {store: rover}
        self.camera_info = {} # {camera: {'rover': r, 'modes': {m}, 'calibration_target': t}}
        self.objective_visibility = {} # {objective: {waypoint}}
        self.lander_visible_waypoints = set()

        # Parse static facts
        lander_waypoint = None
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'at_lander':
                lander, waypoint = parts[1], parts[2]
                self.lander_location[lander] = waypoint
                lander_waypoint = waypoint
            elif parts[0] == 'equipped_for_soil_analysis':
                rover = parts[1]
                self.rover_equipment.setdefault(rover, set()).add('soil')
            elif parts[0] == 'equipped_for_rock_analysis':
                rover = parts[1]
                self.rover_equipment.setdefault(rover, set()).add('rock')
            elif parts[0] == 'equipped_for_imaging':
                rover = parts[1]
                self.rover_equipment.setdefault(rover, set()).add('imaging')
            elif parts[0] == 'store_of':
                store, rover = parts[1], parts[2]
                self.store_owners[store] = rover
            elif parts[0] == 'on_board':
                camera, rover = parts[1], parts[2]
                self.camera_info.setdefault(camera, {}).update({'rover': rover})
            elif parts[0] == 'supports':
                camera, mode = parts[1], parts[2]
                self.camera_info.setdefault(camera, {}).setdefault('modes', set()).add(mode)
            elif parts[0] == 'calibration_target':
                camera, objective = parts[1], parts[2]
                self.camera_info.setdefault(camera, {}).update({'calibration_target': objective})
            elif parts[0] == 'visible_from':
                objective, waypoint = parts[1], parts[2]
                self.objective_visibility.setdefault(objective, set()).add(waypoint)
            # Note: 'visible' and 'can_traverse' are used by the navigate action
            # but for this simplified heuristic, we only need 'visible' relative to the lander
            # and assume 1-step navigation cost if not at the target waypoint.

        # Determine waypoints visible from the lander
        if lander_waypoint:
             for fact in static_facts:
                 # Check visible from lander to another waypoint
                 if match(fact, 'visible', lander_waypoint, '*'):
                      self.lander_visible_waypoints.add(get_parts(fact)[2])
                 # Check visible from another waypoint to lander
                 if match(fact, 'visible', '*', lander_waypoint):
                     self.lander_visible_waypoints.add(get_parts(fact)[1])


    def __call__(self, node):
        """
        Computes the heuristic value for the given state.

        Step-By-Step Thinking for Computing Heuristic:
        1. Initialize the total heuristic value to 0.
        2. Parse the current state to extract dynamic information:
           - Current location of each rover.
           - Status of each store (empty or full).
           - Data already collected (soil, rock, image).
           - Calibrated status of cameras.
        3. Iterate through each goal fact defined in the task.
        4. For each goal fact:
           - If the goal fact is already present in the current state, it is achieved; continue to the next goal.
           - If the goal fact is not achieved, calculate the estimated minimum cost to achieve it from the current state. This calculation is specific to the type of goal (soil, rock, or image data communication).
           - For soil/rock data communication goal `(communicated_soil_data/rock_data ?w)`:
             - Find all rovers equipped for the required analysis type.
             - For each equipped rover, calculate the cost to achieve this specific goal:
               - Cost to acquire data:
                 - If the rover does not already have the data `(have_soil/rock_analysis ?r ?w)`:
                   - Add 1 for navigation to waypoint `?w` (if not already there).
                   - Check the rover's store status. If full, add 1 for the `drop` action.
                   - Add 1 for the `sample_soil/rock` action.
               - Cost to communicate data:
                 - Add 1 for navigation to a waypoint visible from the lander (if not already at such a waypoint).
                 - Add 1 for the `communicate_soil/rock_data` action.
               - The total cost for this rover is the sum of acquisition and communication costs.
             - Find the minimum total cost among all equipped rovers. Add this minimum cost to the total heuristic value.
           - For image data communication goal `(communicated_image_data ?o ?m)`:
             - Find all rover/camera pairs where the rover is equipped for imaging, has the camera on board, and the camera supports mode `?m`.
             - For each suitable pair (`?r`, `?i`):
               - Cost to acquire image:
                 - If the rover does not already have the image `(have_image ?r ?o ?m)`:
                   - Add 1 for navigation to a waypoint visible from objective `?o` (if not already at such a waypoint).
                   - Check camera calibration. If camera `?i` is not calibrated for rover `?r`:
                     - Add 1 for navigation to a waypoint visible from the camera's calibration target (if not already at such a waypoint).
                     - Add 1 for the `calibrate` action.
                   - Add 1 for the `take_image` action.
               - Cost to communicate data:
                 - Add 1 for navigation to a waypoint visible from the lander (if not already at such a waypoint).
                 - Add 1 for the `communicate_image_data` action.
               - The total cost for this pair is the sum of acquisition and communication costs.
             - Find the minimum total cost among all suitable rover/camera pairs. Add this minimum cost to the total heuristic value.
        5. Return the total heuristic value.

        Assumptions:
        - Navigation between any two visible waypoints by a capable rover costs a simplified 1 action, regardless of actual graph distance or `can_traverse` predicates (beyond initial equipment check).
        - Resource constraints (like store capacity beyond full/empty, multiple rovers needing the same resource) are ignored.
        - The existence of required samples, calibration targets, and visible waypoints for imaging/calibration is assumed for goals that require them (as is typical in solvable instances).
        - The heuristic sums costs for each unachieved goal independently, ignoring potential synergies or conflicts between goals.
        - A large number (infinity) is used if a goal is determined to be impossible based on static capabilities (e.g., no rover equipped for soil analysis if a soil goal exists, or no visible waypoint for imaging/calibration). In solvable instances, this should not occur.
        """
        state = node.state

        # Parse dynamic state facts
        rover_locations = {} # {rover: waypoint}
        store_status = {} # {store: 'empty' or 'full'}
        collected_data = set() # {(predicate, arg1, ...)} - stored as original fact strings
        calibrated_cameras = set() # {(camera, rover)}

        for fact_str in state:
            parts = get_parts(fact_str)
            predicate = parts[0]
            if predicate == 'at':
                rover, waypoint = parts[1], parts[2]
                rover_locations[rover] = waypoint
            elif predicate == 'empty':
                store = parts[1]
                store_status[store] = 'empty'
            elif predicate == 'full':
                store = parts[1]
                store_status[store] = 'full'
            elif predicate in ['have_soil_analysis', 'have_rock_analysis', 'have_image']:
                 collected_data.add(fact_str)
            elif predicate == 'calibrated':
                camera, rover = parts[1], parts[2]
                calibrated_cameras.add((camera, rover))

        h = 0

        # Iterate through goals and calculate cost for unachieved ones
        for goal_fact_str in self.goals:
            if goal_fact_str in state:
                continue # Goal already achieved

            parts = get_parts(goal_fact_str)
            goal_predicate = parts[0]

            if goal_predicate == 'communicated_soil_data':
                waypoint_w = parts[1]
                min_cost_goal = math.inf

                # Find rovers equipped for soil analysis
                soil_rovers = {r for r, equip in self.rover_equipment.items() if 'soil' in equip}

                for rover in soil_rovers:
                    cost_acquire = 0
                    # Check if rover already has the soil data for this waypoint
                    have_data = f'(have_soil_analysis {rover} {waypoint_w})' in collected_data

                    if not have_data:
                        # Need to acquire data: navigate, maybe drop, sample
                        current_w = rover_locations.get(rover)
                        # Cost to navigate to waypoint_w
                        if current_w != waypoint_w:
                             cost_acquire += 1 # navigate

                        # Check store status for this rover
                        rover_store = None
                        for s, r_owner in self.store_owners.items():
                            if r_owner == rover:
                                rover_store = s
                                break

                        if rover_store and store_status.get(rover_store) == 'full':
                            cost_acquire += 1 # drop

                        cost_acquire += 1 # sample_soil

                    # Need to communicate data: navigate, communicate
                    cost_communicate = 0
                    current_w = rover_locations.get(rover)
                    # Cost to navigate to a communication point (waypoint visible from lander)
                    if current_w is None or current_w not in self.lander_visible_waypoints:
                         cost_communicate += 1 # navigate

                    cost_communicate += 1 # communicate_soil_data

                    min_cost_goal = min(min_cost_goal, cost_acquire + cost_communicate)

                if math.isinf(min_cost_goal):
                     # Should not happen in solvable instances, but handle defensively
                     h += 1000 # Large penalty
                else:
                     h += min_cost_goal

            elif goal_predicate == 'communicated_rock_data':
                waypoint_w = parts[1]
                min_cost_goal = math.inf

                # Find rovers equipped for rock analysis
                rock_rovers = {r for r, equip in self.rover_equipment.items() if 'rock' in equip}

                for rover in rock_rovers:
                    cost_acquire = 0
                    # Check if rover already has the rock data for this waypoint
                    have_data = f'(have_rock_analysis {rover} {waypoint_w})' in collected_data

                    if not have_data:
                        # Need to acquire data: navigate, maybe drop, sample
                        current_w = rover_locations.get(rover)
                        # Cost to navigate to waypoint_w
                        if current_w != waypoint_w:
                             cost_acquire += 1 # navigate

                        # Check store status for this rover
                        rover_store = None
                        for s, r_owner in self.store_owners.items():
                            if r_owner == rover:
                                rover_store = s
                                break

                        if rover_store and store_status.get(rover_store) == 'full':
                            cost_acquire += 1 # drop

                        cost_acquire += 1 # sample_rock

                    # Need to communicate data: navigate, communicate
                    cost_communicate = 0
                    current_w = rover_locations.get(rover)
                    # Cost to navigate to a communication point (waypoint visible from lander)
                    if current_w is None or current_w not in self.lander_visible_waypoints:
                         cost_communicate += 1 # navigate

                    cost_communicate += 1 # communicate_rock_data

                    min_cost_goal = min(min_cost_goal, cost_acquire + cost_communicate)

                if math.isinf(min_cost_goal):
                     h += 1000 # Large penalty
                else:
                     h += min_cost_goal

            elif goal_predicate == 'communicated_image_data':
                objective_o = parts[1]
                mode_m = parts[2]
                min_cost_goal = math.inf

                # Find suitable rover/camera pairs
                suitable_pairs = [] # [(rover, camera)]
                for camera, info in self.camera_info.items():
                    rover = info.get('rover')
                    supported_modes = info.get('modes', set())
                    if rover and 'imaging' in self.rover_equipment.get(rover, set()) and mode_m in supported_modes:
                        suitable_pairs.append((rover, camera))

                for rover, camera in suitable_pairs:
                    cost_acquire = 0
                    # Check if rover already has the image data for this objective and mode
                    have_data = f'(have_image {rover} {objective_o} {mode_m})' in collected_data

                    if not have_data:
                        # Need to acquire image: navigate to image waypoint, maybe calibrate, take image
                        current_w = rover_locations.get(rover)

                        # Cost to navigate to an image waypoint visible from objective_o
                        image_ws = self.objective_visibility.get(objective_o, set())
                        if not image_ws:
                            cost_acquire = math.inf # Cannot image this objective
                        elif current_w is None or current_w not in image_ws:
                             cost_acquire += 1 # navigate to image waypoint

                        if not math.isinf(cost_acquire): # Only proceed if navigation is possible
                            # Check camera calibration
                            calibrated = (camera, rover) in calibrated_cameras

                            if not calibrated:
                                # Need to calibrate: navigate to calibration waypoint, calibrate
                                cal_target = self.camera_info.get(camera, {}).get('calibration_target')
                                if not cal_target:
                                    cost_acquire = math.inf # Cannot calibrate this camera
                                else:
                                    cal_ws = self.objective_visibility.get(cal_target, set())
                                    if not cal_ws:
                                        cost_acquire = math.inf # Cannot calibrate from any visible waypoint
                                    elif current_w is None or current_w not in cal_ws:
                                         cost_acquire += 1 # navigate to calibration waypoint

                                if not math.isinf(cost_acquire): # Only proceed if calibration navigation is possible
                                    cost_acquire += 1 # calibrate

                            if not math.isinf(cost_acquire): # Only proceed if calibration was possible
                                cost_acquire += 1 # take_image

                    if not math.isinf(cost_acquire): # Only proceed if acquisition was possible
                        # Need to communicate data: navigate, communicate
                        cost_communicate = 0
                        current_w = rover_locations.get(rover)
                        # Cost to navigate to a communication point (waypoint visible from lander)
                        if current_w is None or current_w not in self.lander_visible_waypoints:
                             cost_communicate += 1 # navigate

                        cost_communicate += 1 # communicate_image_data

                        min_cost_goal = min(min_cost_goal, cost_acquire + cost_communicate)
                    # else: cost_acquire was infinity, this pair cannot achieve the goal

                if math.isinf(min_cost_goal):
                     h += 1000 # Large penalty
                else:
                     h += min_cost_goal

        return h
