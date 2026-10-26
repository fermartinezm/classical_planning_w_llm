from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve all communication goals
    (soil data, rock data, and image data) by considering:
    - The distance rovers need to travel to sample soil/rock or take images
    - Whether cameras need calibration
    - The need to return to a lander for communication
    - The current state of stores (empty/full)

    # Assumptions:
    - Each rover can carry only one sample at a time (due to single store)
    - Communication requires being at a waypoint visible to the lander
    - Soil/rock samples can only be collected if present at waypoint
    - Images require proper camera calibration and visibility conditions

    # Heuristic Initialization
    - Extract goal conditions (what needs to be communicated)
    - Extract static information about:
      - Rover capabilities (equipment)
      - Waypoint connectivity (can_traverse)
      - Camera support and calibration targets
      - Sample locations (static)
      - Objective visibility

    # Step-By-Step Thinking for Computing Heuristic
    1. For each communication goal (soil, rock, image):
       a. If already communicated, skip (0 cost)
       b. Otherwise:
          i. For soil/rock:
             - Find closest rover with appropriate equipment
             - Estimate travel to sample location
             - Add sample and communication actions
          ii. For images:
              - Find rover with appropriate camera
              - Estimate calibration if needed
              - Estimate travel to imaging location
              - Add imaging and communication actions
    2. For each rover:
       a. If carrying a sample (full store), add cost to communicate it
       b. If calibrated camera, add cost to take image before losing calibration
    3. Sum all estimated actions
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static
        
        # Extract static information
        self.rover_capabilities = {}
        self.waypoint_connections = {}
        self.camera_info = {}
        self.sample_locations = {'soil': set(), 'rock': set()}
        self.objective_visibility = {}
        self.lander_location = None
        
        for fact in self.static:
            parts = get_parts(fact)
            
            # Rover capabilities
            if match(fact, "equipped_for_*", "*"):
                capability = parts[0].split('_')[-2]
                rover = parts[1]
                if rover not in self.rover_capabilities:
                    self.rover_capabilities[rover] = set()
                self.rover_capabilities[rover].add(capability)
            
            # Waypoint connections
            elif match(fact, "can_traverse", "*", "*", "*"):
                rover, wp1, wp2 = parts[1], parts[2], parts[3]
                if rover not in self.waypoint_connections:
                    self.waypoint_connections[rover] = set()
                self.waypoint_connections[rover].add((wp1, wp2))
            
            # Camera information
            elif match(fact, "supports", "*", "*"):
                camera, mode = parts[1], parts[2]
                if camera not in self.camera_info:
                    self.camera_info[camera] = {'supports': set(), 'target': None, 'on_board': None}
                self.camera_info[camera]['supports'].add(mode)
            elif match(fact, "calibration_target", "*", "*"):
                camera, objective = parts[1], parts[2]
                if camera not in self.camera_info:
                    self.camera_info[camera] = {'supports': set(), 'target': None, 'on_board': None}
                self.camera_info[camera]['target'] = objective
            elif match(fact, "on_board", "*", "*"):
                camera, rover = parts[1], parts[2]
                if camera not in self.camera_info:
                    self.camera_info[camera] = {'supports': set(), 'target': None, 'on_board': None}
                self.camera_info[camera]['on_board'] = rover
            
            # Sample locations (initial state)
            elif match(fact, "at_*_sample", "*"):
                sample_type = parts[0].split('_')[1]
                waypoint = parts[1]
                self.sample_locations[sample_type].add(waypoint)
            
            # Objective visibility
            elif match(fact, "visible_from", "*", "*"):
                objective, waypoint = parts[1], parts[2]
                if objective not in self.objective_visibility:
                    self.objective_visibility[objective] = set()
                self.objective_visibility[objective].add(waypoint)
            
            # Lander location
            elif match(fact, "at_lander", "*", "*"):
                self.lander_location = parts[2]

    def __call__(self, node):
        """Compute heuristic estimate for given state."""
        state = node.state
        total_cost = 0
        
        # Check which goals are already satisfied
        unsatisfied_goals = {
            'soil': set(),
            'rock': set(),
            'image': set()
        }
        
        for goal in self.goals:
            parts = get_parts(goal)
            if match(goal, "communicated_soil_data", "*"):
                wp = parts[1]
                if goal not in state:
                    unsatisfied_goals['soil'].add(wp)
            elif match(goal, "communicated_rock_data", "*"):
                wp = parts[1]
                if goal not in state:
                    unsatisfied_goals['rock'].add(wp)
            elif match(goal, "communicated_image_data", "*", "*"):
                obj, mode = parts[1], parts[2]
                if goal not in state:
                    unsatisfied_goals['image'].add((obj, mode))
        
        # Check rover states
        rover_states = {}
        for rover in self.rover_capabilities:
            rover_states[rover] = {
                'location': None,
                'store_empty': True,
                'has_soil': set(),
                'has_rock': set(),
                'has_images': set(),
                'calibrated_cameras': set()
            }
        
        for fact in state:
            parts = get_parts(fact)
            
            # Rover location
            if match(fact, "at", "*", "*"):
                rover, wp = parts[1], parts[2]
                if rover in rover_states:
                    rover_states[rover]['location'] = wp
            
            # Store status
            elif match(fact, "empty", "*"):
                store = parts[1]
                for rover in rover_states:
                    if match(fact, "empty", f"{rover}store"):
                        rover_states[rover]['store_empty'] = True
            elif match(fact, "full", "*"):
                store = parts[1]
                for rover in rover_states:
                    if match(fact, "full", f"{rover}store"):
                        rover_states[rover]['store_empty'] = False
            
            # Samples carried
            elif match(fact, "have_soil_analysis", "*", "*"):
                rover, wp = parts[1], parts[2]
                rover_states[rover]['has_soil'].add(wp)
            elif match(fact, "have_rock_analysis", "*", "*"):
                rover, wp = parts[1], parts[2]
                rover_states[rover]['has_rock'].add(wp)
            
            # Images taken
            elif match(fact, "have_image", "*", "*", "*"):
                rover, obj, mode = parts[1], parts[2], parts[3]
                rover_states[rover]['has_images'].add((obj, mode))
            
            # Calibrated cameras
            elif match(fact, "calibrated", "*", "*"):
                camera, rover = parts[1], parts[2]
                rover_states[rover]['calibrated_cameras'].add(camera)
        
        # Estimate cost for unsatisfied soil goals
        for wp in unsatisfied_goals['soil']:
            # Check if any rover already has this sample
            found = False
            for rover, state in rover_states.items():
                if wp in state['has_soil']:
                    # Just need to communicate it
                    total_cost += 2  # navigate to lander-visible wp + communicate
                    found = True
                    break
            
            if not found:
                # Need to find a rover to collect it
                min_cost = float('inf')
                for rover, state in rover_states.items():
                    if 'soil_analysis' in self.rover_capabilities[rover]:
                        # Estimate distance to sample location
                        dist = 1  # optimistic estimate
                        # Sample collection (2 actions: sample + drop if store was full)
                        cost = dist + 2
                        # Communication (navigate + communicate)
                        cost += 2
                        min_cost = min(min_cost, cost)
                
                if min_cost != float('inf'):
                    total_cost += min_cost
        
        # Estimate cost for unsatisfied rock goals (similar to soil)
        for wp in unsatisfied_goals['rock']:
            found = False
            for rover, state in rover_states.items():
                if wp in state['has_rock']:
                    total_cost += 2
                    found = True
                    break
            
            if not found:
                min_cost = float('inf')
                for rover, state in rover_states.items():
                    if 'rock_analysis' in self.rover_capabilities[rover]:
                        dist = 1
                        cost = dist + 2 + 2
                        min_cost = min(min_cost, cost)
                
                if min_cost != float('inf'):
                    total_cost += min_cost
        
        # Estimate cost for unsatisfied image goals
        for (obj, mode) in unsatisfied_goals['image']:
            # Check if any rover already has this image
            found = False
            for rover, state in rover_states.items():
                if (obj, mode) in state['has_images']:
                    total_cost += 2  # navigate + communicate
                    found = True
                    break
            
            if not found:
                min_cost = float('inf')
                for rover, state in rover_states.items():
                    if 'imaging' in self.rover_capabilities[rover]:
                        # Find a camera on this rover that supports the mode
                        for camera, info in self.camera_info.items():
                            if info['on_board'] == rover and mode in info['supports']:
                                # Check if camera is calibrated
                                if camera in state['calibrated_cameras']:
                                    cal_cost = 0
                                else:
                                    cal_cost = 2  # navigate to calibration wp + calibrate
                                
                                # Find visible waypoint for this objective
                                if obj in self.objective_visibility:
                                    dist = 1  # optimistic distance
                                    # take image (1 action) + communicate (2 actions)
                                    cost = cal_cost + dist + 1 + 2
                                    min_cost = min(min_cost, cost)
                
                if min_cost != float('inf'):
                    total_cost += min_cost
        
        # Add cost for rovers with full stores that haven't communicated yet
        for rover, state in rover_states.items():
            if not state['store_empty']:
                # Need to communicate any samples
                if state['has_soil'] or state['has_rock']:
                    total_cost += 2  # navigate + communicate
        
        # Add cost for calibrated cameras that haven't taken images yet
        for rover, state in rover_states.items():
            if 'imaging' in self.rover_capabilities[rover]:
                for camera in state['calibrated_cameras']:
                    # Check if there are unfulfilled image goals this camera could take
                    for (obj, mode) in unsatisfied_goals['image']:
                        cam_info = self.camera_info.get(camera, {})
                        if (mode in cam_info.get('supports', set()) and 
                            obj == cam_info.get('target')):
                            total_cost += 1  # take image action
        
        return total_cost
