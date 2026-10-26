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
    - The need to calibrate cameras before taking images
    - The need to communicate data back to the lander

    # Assumptions:
    - Each rover can only carry one sample at a time (due to single store)
    - Communication requires being at a waypoint visible to the lander
    - Soil/rock samples must be collected before communication
    - Images must be taken before communication

    # Heuristic Initialization
    - Extract goal conditions (what needs to be communicated)
    - Extract static information about:
        - Rover capabilities (equipped_for_*)
        - Camera configurations
        - Waypoint visibility and traversal
        - Sample locations
        - Objective visibility

    # Step-By-Step Thinking for Computing Heuristic
    1. For each communication goal (soil, rock, image):
        a. If already communicated, skip
        b. Otherwise:
            i. For soil/rock:
                - Find closest rover with appropriate equipment
                - Add distance to sample location
                - Add distance to lander-visible waypoint
                - Add actions for sampling and communicating
            ii. For images:
                - Find rover with appropriate camera and equipment
                - Add distance to calibration location (if needed)
                - Add distance to imaging location
                - Add distance to lander-visible waypoint
                - Add actions for calibration, imaging, and communicating
    2. Sum all required actions across all goals
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract static information into more accessible data structures
        self.rover_capabilities = {}
        self.camera_info = {}
        self.waypoint_visibility = set()
        self.can_traverse = set()
        self.sample_locations = {'soil': set(), 'rock': set()}
        self.objective_visibility = {}

        for fact in self.static:
            parts = get_parts(fact)
            if not parts:
                continue

            if match(fact, "equipped_for_*", "*"):
                capability = parts[0].split('_')[-1]  # soil_analysis, rock_analysis, imaging
                rover = parts[1]
                if rover not in self.rover_capabilities:
                    self.rover_capabilities[rover] = set()
                self.rover_capabilities[rover].add(capability)
            
            elif match(fact, "on_board", "*", "*"):
                camera, rover = parts[1], parts[2]
                if rover not in self.camera_info:
                    self.camera_info[rover] = []
                self.camera_info[rover].append(camera)
            
            elif match(fact, "supports", "*", "*"):
                camera, mode = parts[1], parts[2]
                if camera not in self.camera_info:
                    self.camera_info[camera] = {'modes': set()}
                self.camera_info[camera]['modes'].add(mode)
            
            elif match(fact, "visible", "*", "*"):
                self.waypoint_visibility.add((parts[1], parts[2]))
            
            elif match(fact, "can_traverse", "*", "*", "*"):
                self.can_traverse.add((parts[1], parts[2], parts[3]))
            
            elif match(fact, "at_soil_sample", "*"):
                self.sample_locations['soil'].add(parts[1])
            
            elif match(fact, "at_rock_sample", "*"):
                self.sample_locations['rock'].add(parts[1])
            
            elif match(fact, "visible_from", "*", "*"):
                objective, waypoint = parts[1], parts[2]
                if objective not in self.objective_visibility:
                    self.objective_visibility[objective] = set()
                self.objective_visibility[objective].add(waypoint)
            
            elif match(fact, "calibration_target", "*", "*"):
                camera, objective = parts[1], parts[2]
                if camera not in self.camera_info:
                    self.camera_info[camera] = {}
                self.camera_info[camera]['target'] = objective

        # Extract lander location
        self.lander_location = None
        for fact in self.static:
            if match(fact, "at_lander", "*", "*"):
                parts = get_parts(fact)
                self.lander_location = parts[2]
                break

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Check which goals are already satisfied
        unsatisfied_soil = set()
        unsatisfied_rock = set()
        unsatisfied_images = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts:
                continue

            if match(goal, "communicated_soil_data", "*"):
                wp = parts[1]
                if goal not in state:
                    unsatisfied_soil.add(wp)
            
            elif match(goal, "communicated_rock_data", "*"):
                wp = parts[1]
                if goal not in state:
                    unsatisfied_rock.add(wp)
            
            elif match(goal, "communicated_image_data", "*", "*"):
                obj, mode = parts[1], parts[2]
                if goal not in state:
                    unsatisfied_images.add((obj, mode))

        # Helper function to estimate distance between waypoints
        def estimate_distance(wp1, wp2):
            # Simple estimation: 1 action per waypoint hop
            # In a real implementation, we might use a more sophisticated distance measure
            return 1

        # Process soil data goals
        for wp in unsatisfied_soil:
            # Find if any rover already has this sample
            rover_has_sample = False
            for fact in state:
                if match(fact, "have_soil_analysis", "*", wp):
                    rover_has_sample = True
                    rover = get_parts(fact)[1]
                    break
            
            if rover_has_sample:
                # Rover needs to communicate the data
                # Find rover's current location
                rover_loc = None
                for fact in state:
                    if match(fact, "at", rover, "*"):
                        rover_loc = get_parts(fact)[2]
                        break
                
                if rover_loc:
                    # Find closest waypoint visible to lander
                    comm_wp = self.lander_location  # Simplification
                    if (rover_loc, comm_wp) in self.waypoint_visibility:
                        total_cost += 1  # communicate action
                    else:
                        # Need to navigate to visible waypoint
                        total_cost += estimate_distance(rover_loc, comm_wp) + 1
            else:
                # Need to sample and communicate
                # Find closest rover with soil analysis capability
                closest_rover = None
                min_dist = float('inf')
                
                for rover, capabilities in self.rover_capabilities.items():
                    if 'soil' not in capabilities:
                        continue
                    
                    # Find rover location
                    rover_loc = None
                    for fact in state:
                        if match(fact, "at", rover, "*"):
                            rover_loc = get_parts(fact)[2]
                            break
                    
                    if rover_loc:
                        dist = estimate_distance(rover_loc, wp)
                        if dist < min_dist:
                            min_dist = dist
                            closest_rover = rover
                
                if closest_rover:
                    # Cost to navigate to sample location
                    total_cost += min_dist
                    # Cost to sample
                    total_cost += 1
                    # Cost to navigate to communication point
                    comm_wp = self.lander_location  # Simplification
                    total_cost += estimate_distance(wp, comm_wp)
                    # Cost to communicate
                    total_cost += 1

        # Process rock data goals (similar to soil)
        for wp in unsatisfied_rock:
            rover_has_sample = False
            for fact in state:
                if match(fact, "have_rock_analysis", "*", wp):
                    rover_has_sample = True
                    rover = get_parts(fact)[1]
                    break
            
            if rover_has_sample:
                rover_loc = None
                for fact in state:
                    if match(fact, "at", rover, "*"):
                        rover_loc = get_parts(fact)[2]
                        break
                
                if rover_loc:
                    comm_wp = self.lander_location
                    if (rover_loc, comm_wp) in self.waypoint_visibility:
                        total_cost += 1
                    else:
                        total_cost += estimate_distance(rover_loc, comm_wp) + 1
            else:
                closest_rover = None
                min_dist = float('inf')
                
                for rover, capabilities in self.rover_capabilities.items():
                    if 'rock' not in capabilities:
                        continue
                    
                    rover_loc = None
                    for fact in state:
                        if match(fact, "at", rover, "*"):
                            rover_loc = get_parts(fact)[2]
                            break
                    
                    if rover_loc:
                        dist = estimate_distance(rover_loc, wp)
                        if dist < min_dist:
                            min_dist = dist
                            closest_rover = rover
                
                if closest_rover:
                    total_cost += min_dist
                    total_cost += 1
                    comm_wp = self.lander_location
                    total_cost += estimate_distance(wp, comm_wp)
                    total_cost += 1

        # Process image data goals
        for obj, mode in unsatisfied_images:
            # Check if any rover already has this image
            rover_has_image = False
            for fact in state:
                if match(fact, "have_image", "*", obj, mode):
                    rover_has_image = True
                    rover = get_parts(fact)[1]
                    break
            
            if rover_has_image:
                # Rover needs to communicate the image
                rover_loc = None
                for fact in state:
                    if match(fact, "at", rover, "*"):
                        rover_loc = get_parts(fact)[2]
                        break
                
                if rover_loc:
                    comm_wp = self.lander_location
                    if (rover_loc, comm_wp) in self.waypoint_visibility:
                        total_cost += 1
                    else:
                        total_cost += estimate_distance(rover_loc, comm_wp) + 1
            else:
                # Need to take and communicate image
                # Find rover with appropriate camera and mode
                suitable_rover = None
                suitable_camera = None
                
                for rover, cameras in self.camera_info.items():
                    if not isinstance(cameras, list):
                        continue  # Skip if not a list of cameras
                    
                    for camera in cameras:
                        cam_info = self.camera_info.get(camera, {})
                        if (cam_info.get('modes', set()) and 
                            mode in cam_info['modes'] and 
                            cam_info.get('target') == obj):
                            suitable_rover = rover
                            suitable_camera = camera
                            break
                    if suitable_rover:
                        break
                
                if suitable_rover:
                    # Check if camera is calibrated
                    calibrated = False
                    for fact in state:
                        if match(fact, "calibrated", suitable_camera, suitable_rover):
                            calibrated = True
                            break
                    
                    # Find rover location
                    rover_loc = None
                    for fact in state:
                        if match(fact, "at", suitable_rover, "*"):
                            rover_loc = get_parts(fact)[2]
                            break
                    
                    if rover_loc:
                        # Find imaging waypoint
                        imaging_wp = next(iter(self.objective_visibility.get(obj, set())), None)
                        
                        if imaging_wp:
                            # Cost to calibrate if needed
                            if not calibrated:
                                # Need to go to calibration target waypoint
                                calib_wp = next(iter(
                                    self.objective_visibility.get(
                                        self.camera_info[suitable_camera]['target'], set())), None)
                                if calib_wp:
                                    total_cost += estimate_distance(rover_loc, calib_wp)
                                    total_cost += 1  # calibrate action
                                    rover_loc = calib_wp  # Update rover location
                            
                            # Cost to navigate to imaging waypoint
                            total_cost += estimate_distance(rover_loc, imaging_wp)
                            # Cost to take image
                            total_cost += 1
                            # Cost to navigate to communication point
                            comm_wp = self.lander_location
                            total_cost += estimate_distance(imaging_wp, comm_wp)
                            # Cost to communicate
                            total_cost += 1

        return total_cost
