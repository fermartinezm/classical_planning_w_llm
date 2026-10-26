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
    This heuristic estimates the number of actions needed to achieve all goals by:
    1) Collecting required samples (soil/rock)
    2) Taking required images
    3) Communicating all data to the lander
    It considers the current state of rovers (position, equipped instruments, stored samples)
    and the remaining tasks needed to achieve the goals.

    # Assumptions:
    - Each rover can only carry one sample at a time (due to single store)
    - Communication requires being at a waypoint visible to the lander
    - Calibration is needed before taking images
    - Samples can only be collected if the rover is at the sample location with proper equipment

    # Heuristic Initialization
    - Extract goal conditions (communicated data)
    - Build mapping of waypoints to samples (rock/soil)
    - Build mapping of objectives to their visible waypoints
    - Identify lander location
    - Identify rover capabilities (equipment)

    # Step-By-Step Thinking for Computing Heuristic
    1) For each uncommunicated soil/rock data:
       - If not collected yet: 
           * Move to sample location (if needed)
           * Sample (if possible)
       - Move to waypoint visible to lander
       * Communicate data
    2) For each uncommunicated image data:
       - If image not taken yet:
           * Calibrate camera (if needed)
           * Move to waypoint where objective is visible
           * Take image
       - Move to waypoint visible to lander
       * Communicate image
    3) Sum all required actions
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static
        
        # Extract static information
        self.lander_location = None
        self.sample_locations = {'soil': set(), 'rock': set()}
        self.objective_visibility = {}
        self.rover_capabilities = {}
        self.camera_info = {}
        
        for fact in self.static:
            if match(fact, "at_lander", "*", "*"):
                _, lander, loc = get_parts(fact)
                self.lander_location = loc
            elif match(fact, "at_soil_sample", "*"):
                self.sample_locations['soil'].add(get_parts(fact)[1])
            elif match(fact, "at_rock_sample", "*"):
                self.sample_locations['rock'].add(get_parts(fact)[1])
            elif match(fact, "visible_from", "*", "*"):
                _, obj, wp = get_parts(fact)
                if obj not in self.objective_visibility:
                    self.objective_visibility[obj] = set()
                self.objective_visibility[obj].add(wp)
            elif match(fact, "equipped_for_*", "*"):
                capability = get_parts(fact)[0]
                rover = get_parts(fact)[1]
                if rover not in self.rover_capabilities:
                    self.rover_capabilities[rover] = set()
                self.rover_capabilities[rover].add(capability)
            elif match(fact, "calibration_target", "*", "*"):
                cam, obj = get_parts(fact)
                if cam not in self.camera_info:
                    self.camera_info[cam] = {}
                self.camera_info[cam]['target'] = obj
            elif match(fact, "on_board", "*", "*"):
                cam, rover = get_parts(fact)
                if cam not in self.camera_info:
                    self.camera_info[cam] = {}
                self.camera_info[cam]['rover'] = rover
            elif match(fact, "supports", "*", "*"):
                cam, mode = get_parts(fact)
                if cam not in self.camera_info:
                    self.camera_info[cam] = {}
                if 'modes' not in self.camera_info[cam]:
                    self.camera_info[cam]['modes'] = set()
                self.camera_info[cam]['modes'].add(mode)
        
        # Extract goal information
        self.required_soil_data = set()
        self.required_rock_data = set()
        self.required_image_data = set()
        
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "communicated_soil_data":
                self.required_soil_data.add(parts[1])
            elif parts[0] == "communicated_rock_data":
                self.required_rock_data.add(parts[1])
            elif parts[0] == "communicated_image_data":
                self.required_image_data.add((parts[1], parts[2]))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0
        
        # Track current rover positions and states
        rover_positions = {}
        stored_samples = {'soil': {}, 'rock': {}}  # {rover: waypoint}
        calibrated_cameras = set()
        taken_images = set()
        
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at", "*", "*"):
                rover, wp = parts[1], parts[2]
                rover_positions[rover] = wp
            elif match(fact, "have_soil_analysis", "*", "*"):
                rover, wp = parts[1], parts[2]
                stored_samples['soil'][rover] = wp
            elif match(fact, "have_rock_analysis", "*", "*"):
                rover, wp = parts[1], parts[2]
                stored_samples['rock'][rover] = wp
            elif match(fact, "calibrated", "*", "*"):
                cam, rover = parts[1], parts[2]
                calibrated_cameras.add((cam, rover))
            elif match(fact, "have_image", "*", "*", "*"):
                rover, obj, mode = parts[1], parts[2], parts[3]
                taken_images.add((rover, obj, mode))
            elif match(fact, "communicated_soil_data", "*"):
                wp = parts[1]
                if wp in self.required_soil_data:
                    self.required_soil_data.remove(wp)
            elif match(fact, "communicated_rock_data", "*"):
                wp = parts[1]
                if wp in self.required_rock_data:
                    self.required_rock_data.remove(wp)
            elif match(fact, "communicated_image_data", "*", "*"):
                obj, mode = parts[1], parts[2]
                if (obj, mode) in self.required_image_data:
                    self.required_image_data.remove((obj, mode))
        
        # If all goals are satisfied, return 0
        if not (self.required_soil_data or self.required_rock_data or self.required_image_data):
            return 0
        
        # Estimate cost for remaining soil samples
        for wp in self.required_soil_data:
            # Find a rover that can collect soil samples
            suitable_rover = None
            for rover, capabilities in self.rover_capabilities.items():
                if "equipped_for_soil_analysis" in capabilities:
                    suitable_rover = rover
                    break
            
            if not suitable_rover:
                continue  # No rover can collect this sample
            
            # Check if sample is already collected by this rover
            if suitable_rover in stored_samples['soil'] and stored_samples['soil'][suitable_rover] == wp:
                # Just need to communicate
                total_cost += 2  # move to visible waypoint + communicate
            else:
                # Need to collect and communicate
                total_cost += 4  # move to sample + sample + move to visible + communicate
        
        # Estimate cost for remaining rock samples
        for wp in self.required_rock_data:
            # Find a rover that can collect rock samples
            suitable_rover = None
            for rover, capabilities in self.rover_capabilities.items():
                if "equipped_for_rock_analysis" in capabilities:
                    suitable_rover = rover
                    break
            
            if not suitable_rover:
                continue  # No rover can collect this sample
            
            # Check if sample is already collected by this rover
            if suitable_rover in stored_samples['rock'] and stored_samples['rock'][suitable_rover] == wp:
                # Just need to communicate
                total_cost += 2  # move to visible waypoint + communicate
            else:
                # Need to collect and communicate
                total_cost += 4  # move to sample + sample + move to visible + communicate
        
        # Estimate cost for remaining images
        for obj, mode in self.required_image_data:
            # Find a camera that supports this mode and can see the objective
            suitable_camera = None
            suitable_rover = None
            visible_waypoints = self.objective_visibility.get(obj, set())
            
            for cam, info in self.camera_info.items():
                if mode in info.get('modes', set()):
                    suitable_camera = cam
                    suitable_rover = info.get('rover')
                    break
            
            if not suitable_camera or not suitable_rover:
                continue  # No suitable camera
            
            # Check if image is already taken
            image_taken = any((r, o, m) in taken_images 
                          for r in rover_positions.keys() 
                          for o, m in [(obj, mode)])
            
            if image_taken:
                # Just need to communicate
                total_cost += 2  # move to visible waypoint + communicate
            else:
                # Need to calibrate, take image, and communicate
                # Check if camera is already calibrated
                is_calibrated = (suitable_camera, suitable_rover) in calibrated_cameras
                
                if not is_calibrated:
                    total_cost += 1  # calibrate
                
                total_cost += 4  # move to calibration target + calibrate (if needed) + move to visible + take image + move to lander visible + communicate
        
        return total_cost
