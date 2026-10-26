from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict

# Helper functions
def parse_fact(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    """
    parts = parse_fact(fact)
    # Use zip, which stops at the shortest sequence
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to satisfy all goal
    conditions. It counts the final communication/drop action for each
    unmet goal, plus the necessary data/image acquisition actions (sample,
    calibrate, take_image) if the required data/image is not yet available.
    It also adds a fixed cost (1) for each distinct navigation leg required
    to reach sampling, imaging, calibration, or communication locations for
    the unmet goals.

    # Assumptions
    - Each action (sample, drop, calibrate, take_image, communicate, navigate)
      is assumed to have a cost of 1 for heuristic calculation purposes.
    - The heuristic does not model complex interactions between goals or
      rover assignments, potentially overcounting costs, but aims to guide
      towards states where more goal prerequisites are met.
    - Assumes solvable instances (e.g., equipped rovers exist for required tasks).
    - Navigation cost is a simplified fixed cost per required location visit type.

    # Heuristic Initialization
    The constructor extracts static information from the task:
    - Lander location.
    - Rover capabilities (soil, rock, imaging).
    - Rover store assignments.
    - Camera information (on which rover, supported modes, calibration target).
    - Waypoint visibility for communication points (visible from lander).
    - Waypoint visibility for objectives and calibration targets.
    - Initial locations of soil and rock samples (though these change, their initial presence is static).
    - Set of all rovers.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. Identify the lander's location and determine the set of waypoints from
       which the lander is visible (communication points).
    3. Iterate through each goal condition specified in the task.
    4. For each goal condition that is *not* currently true in the state:
       a. Add 1 to the cost for the final communication action required for this goal.
       b. Determine the type of data/image required (soil, rock, or image) and its parameters (waypoint, objective, mode).
       c. Check if the required data/image is already available on *any* rover
          (e.g., `(have_soil_analysis ?r w)`, `(have_image ?r o m)`).
       d. If the required data/image is *not* available:
          i. Add 1 to the cost for the acquisition action (sample_soil, sample_rock, or take_image).
          ii. Add 1 to the cost for navigating to the required acquisition location
              (waypoint for sample, waypoint visible from objective for image).
          iii. If it's a sample goal (soil or rock):
              - Check if any rover equipped for this task has a full store. If yes, add 1 for a drop action.
          iv. If it's an image goal:
              - Check if a suitable camera (on an equipped rover, supporting the mode) is calibrated. If not:
                - Add 1 for the calibrate action.
                - Add 1 for navigating to a waypoint visible from the camera's calibration target (using *any* suitable camera).
       e. Add 1 to the cost for navigating to a communication point.
    5. Return the total accumulated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.lander_location = None
        self.rover_capabilities = defaultdict(set) # rover -> {soil, rock, imaging}
        self.rover_stores = {} # rover -> store
        self.camera_info = {} # camera -> {'rover': rover, 'modes': set(), 'calibration_target': objective}
        self.visible_from_map = defaultdict(set) # objective_or_target -> {waypoint}
        self.initial_soil_samples = set() # {waypoint}
        self.initial_rock_samples = set() # {waypoint}
        self.communication_points = set() # {waypoint}
        self.all_rovers = set() # {rover}

        # Extract static facts
        # We only need to iterate through static_facts once
        for fact in static_facts:
            parts = parse_fact(fact)
            predicate = parts[0]

            if predicate == 'at_lander':
                lander, wp = parts[1], parts[2]
                self.lander_location = wp
            elif predicate == 'rover':
                 self.all_rovers.add(parts[1])
            elif predicate.startswith('equipped_for_'):
                capability = predicate.split('_')[-2] # soil, rock, imaging
                rover = parts[1]
                self.rover_capabilities[rover].add(capability)
            elif predicate == 'store_of':
                store, rover = parts[1], parts[2]
                self.rover_stores[rover] = store
            elif predicate == 'supports':
                camera, mode = parts[1], parts[2]
                if camera not in self.camera_info:
                    self.camera_info[camera] = {'rover': None, 'modes': set(), 'calibration_target': None}
                self.camera_info[camera]['modes'].add(mode)
            elif predicate == 'visible': # Waypoint visibility for communication points
                wp1, wp2 = parts[1], parts[2]
                # Check if wp2 is the lander location
                # Note: visible ?w ?p means ?w is visible from ?p.
                # Communication needs rover at ?x, lander at ?y, (visible ?x ?y).
                # So, lander location ?y must be visible from rover location ?x.
                # Communication points are waypoints ?x such that (visible ?x lander_wp) is true.
                if self.lander_location and wp2 == self.lander_location:
                     self.communication_points.add(wp1)
            elif predicate == 'at_soil_sample':
                wp = parts[1]
                self.initial_soil_samples.add(wp)
            elif predicate == 'at_rock_sample':
                wp = parts[1]
                self.initial_rock_samples.add(wp)
            elif predicate == 'visible_from':
                obj_or_target, wp = parts[1], parts[2]
                self.visible_from_map[obj_or_target].add(wp)
            elif predicate == 'calibration_target':
                camera, target = parts[1], parts[2]
                if camera not in self.camera_info:
                     self.camera_info[camera] = {'rover': None, 'modes': set(), 'calibration_target': None}
                self.camera_info[camera]['calibration_target'] = target
            elif predicate == 'on_board':
                camera, rover = parts[1], parts[2]
                if camera not in self.camera_info:
                     self.camera_info[camera] = {'rover': None, 'modes': set(), 'calibration_target': None}
                self.camera_info[camera]['rover'] = rover

        # Store goal types for easier lookup
        self.goal_types = defaultdict(list) # 'soil' -> [wp1, wp2], 'rock' -> [wp3], 'image' -> [(o1, m1), (o2, m2)]
        for goal in self.goals:
            parts = parse_fact(goal)
            predicate = parts[0]
            if predicate == 'communicated_soil_data':
                self.goal_types['soil'].append(parts[1]) # waypoint
            elif predicate == 'communicated_rock_data':
                self.goal_types['rock'].append(parts[1]) # waypoint
            elif predicate == 'communicated_image_data':
                self.goal_types['image'].append((parts[1], parts[2])) # (objective, mode)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Helper to find rovers with a specific capability
        def get_equipped_rovers(capability):
            return [r for r, caps in self.rover_capabilities.items() if capability in caps]

        # Helper to find rovers with a camera supporting a specific mode
        def get_rovers_with_camera_for_mode(mode):
            suitable_rovers = []
            for camera, info in self.camera_info.items():
                if mode in info['modes'] and info['rover']:
                    suitable_rovers.append((info['rover'], camera))
            return suitable_rovers

        # Helper to find the store for a rover
        def get_rover_store(rover):
            return self.rover_stores.get(rover)

        # Helper to find the calibration target for a camera
        def get_camera_calibration_target(camera):
             return self.camera_info.get(camera, {}).get('calibration_target')

        # Helper to find waypoints visible from an objective or target
        def get_visible_from_waypoints(obj_or_target):
             return self.visible_from_map.get(obj_or_target, set())


        # --- Heuristic Calculation ---

        # 1. Soil Data Goals
        for w in self.goal_types.get('soil', []):
            goal_fact = f'(communicated_soil_data {w})'
            if goal_fact not in state:
                total_cost += 1 # Cost for communicate_soil_data

                # Check if sample is already acquired by any rover
                has_sample = any(f'(have_soil_analysis {r} {w})' in state for r in self.all_rovers)

                if not has_sample:
                    total_cost += 1 # Cost for sample_soil
                    total_cost += 1 # Navigation to waypoint w

                    # Check if any equipped rover needs to drop before sampling
                    equipped_soil_rovers = get_equipped_rovers('soil')
                    if equipped_soil_rovers:
                        # If *any* equipped soil rover has a full store, we might need a drop
                        if any(f'(full {get_rover_store(r)})' in state for r in equipped_soil_rovers if get_rover_store(r)):
                            total_cost += 1 # Cost for drop

                # Need to navigate to a communication point
                if self.communication_points:
                    total_cost += 1 # Navigation to a communication point
                # else: problem might be unsolvable if no comm points, but heuristic assumes solvable

        # 2. Rock Data Goals
        for w in self.goal_types.get('rock', []):
            goal_fact = f'(communicated_rock_data {w})'
            if goal_fact not in state:
                total_cost += 1 # Cost for communicate_rock_data

                # Check if sample is already acquired by any rover
                has_sample = any(f'(have_rock_analysis {r} {w})' in state for r in self.all_rovers)

                if not has_sample:
                    total_cost += 1 # Cost for sample_rock
                    total_cost += 1 # Navigation to waypoint w

                    # Check if any equipped rover needs to drop before sampling
                    equipped_rock_rovers = get_equipped_rovers('rock')
                    if equipped_rock_rovers:
                        # If *any* equipped rock rover has a full store, we might need a drop
                        if any(f'(full {get_rover_store(r)})' in state for r in equipped_rock_rovers if get_rover_store(r)):
                            total_cost += 1 # Cost for drop

                # Need to navigate to a communication point
                if self.communication_points:
                    total_cost += 1 # Navigation to a communication point

        # 3. Image Data Goals
        for o, m in self.goal_types.get('image', []):
            goal_fact = f'(communicated_image_data {o} {m})'
            if goal_fact not in state:
                total_cost += 1 # Cost for communicate_image_data

                # Find rovers equipped for imaging with camera supporting mode m
                suitable_rover_camera_pairs = get_rovers_with_camera_for_mode(m)

                # Check if image is already acquired by any rover
                has_image = any(f'(have_image {r} {o} {m})' in state for r in self.all_rovers)

                if not has_image:
                    total_cost += 1 # Cost for take_image
                    # Need to navigate to an image point for objective o
                    image_wps = get_visible_from_waypoints(o)
                    if image_wps:
                         total_cost += 1 # Navigation to an image point

                    # Check if a suitable camera is calibrated
                    calibrated = False
                    if suitable_rover_camera_pairs:
                        # If *any* suitable camera is calibrated on its rover
                        calibrated = any(f'(calibrated {i} {r})' in state for r, i in suitable_rover_camera_pairs)

                    if not calibrated:
                        total_cost += 1 # Cost for calibrate
                        # Need to navigate to a calibration point for a suitable camera
                        # Pick the first suitable camera found for cost estimation
                        if suitable_rover_camera_pairs:
                            # Find a camera from the suitable pairs that has a calibration target defined
                            camera_for_cal = None
                            for _, cam in suitable_rover_camera_pairs:
                                if get_camera_calibration_target(cam):
                                    camera_for_cal = cam
                                    break

                            if camera_for_cal:
                                cal_target = get_camera_calibration_target(camera_for_cal)
                                cal_wps = get_visible_from_waypoints(cal_target)
                                if cal_wps:
                                    total_cost += 1 # Navigation to a calibration point


                # Need to navigate to a communication point
                if self.communication_points:
                    total_cost += 1 # Navigation to a communication point


        return total_cost
