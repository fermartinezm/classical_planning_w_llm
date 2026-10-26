from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """Check if a PDDL fact matches a given pattern with wildcards."""
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve all goals by:
    1. Counting unsatisfied communication goals (soil, rock, image data)
    2. Estimating the steps required to:
       - Move to sample locations
       - Take samples (soil/rock)
       - Calibrate cameras and take images
       - Communicate data to the lander

    # Assumptions:
    - Each unsatisfied goal requires at least one action (communication)
    - Sampling requires moving to the location and performing the sample action
    - Imaging requires calibration, moving to a visible location, and taking the image
    - Communication requires moving to a location visible to the lander
    - The heuristic is not admissible (may overestimate)

    # Heuristic Initialization
    - Extract goal conditions (what needs to be communicated)
    - Extract static information about:
      - Rover capabilities (equipment)
      - Waypoint connectivity (can_traverse)
      - Camera capabilities and calibration targets
      - Sample locations (static rock/soil samples)
      - Visibility relationships

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify unsatisfied communication goals (soil, rock, image data)
    2. For each unsatisfied goal:
       a. If it's soil/rock data:
          - Check if rover has the analysis (then just need to communicate)
          - Otherwise, check if sample is still available (need to sample first)
          - If sample not available, another rover must have taken it (problematic)
       b. If it's image data:
          - Check if rover has the image (then just need to communicate)
          - Otherwise, need to calibrate camera and take image
    3. Estimate movement costs between current locations and target locations
    4. Sum all required actions (move, sample, calibrate, image, communicate)
    """

    def __init__(self, task):
        """Initialize by extracting goal conditions and static information."""
        self.goals = task.goals
        self.static = task.static

        # Extract static information into useful data structures
        self.lander_location = None
        self.rover_capabilities = {}
        self.camera_info = {}
        self.sample_locations = {'rock': set(), 'soil': set()}
        self.visible_from = {}
        self.waypoint_connections = {}

        for fact in self.static:
            parts = get_parts(fact)
            
            if match(fact, "at_lander", "*", "*"):
                self.lander_location = parts[2]
            
            elif match(fact, "equipped_for_*", "*"):
                capability = parts[0].split('_')[-2]  # soil_analysis -> soil
                rover = parts[1]
                if rover not in self.rover_capabilities:
                    self.rover_capabilities[rover] = set()
                self.rover_capabilities[rover].add(capability)
            
            elif match(fact, "on_board", "*", "*"):
                camera, rover = parts[1], parts[2]
                if rover not in self.camera_info:
                    self.camera_info[rover] = []
                self.camera_info[rover].append(camera)
            
            elif match(fact, "calibration_target", "*", "*"):
                camera, objective = parts[1], parts[2]
                if camera not in self.camera_info:
                    self.camera_info[camera] = {}
                self.camera_info[camera]['target'] = objective
            
            elif match(fact, "supports", "*", "*"):
                camera, mode = parts[1], parts[2]
                if camera not in self.camera_info:
                    self.camera_info[camera] = {}
                if 'modes' not in self.camera_info[camera]:
                    self.camera_info[camera]['modes'] = set()
                self.camera_info[camera]['modes'].add(mode)
            
            elif match(fact, "at_rock_sample", "*"):
                self.sample_locations['rock'].add(parts[1])
            
            elif match(fact, "at_soil_sample", "*"):
                self.sample_locations['soil'].add(parts[1])
            
            elif match(fact, "visible_from", "*", "*"):
                objective, waypoint = parts[1], parts[2]
                if objective not in self.visible_from:
                    self.visible_from[objective] = set()
                self.visible_from[objective].add(waypoint)
            
            elif match(fact, "can_traverse", "*", "*", "*"):
                rover, wp1, wp2 = parts[1], parts[2], parts[3]
                if rover not in self.waypoint_connections:
                    self.waypoint_connections[rover] = set()
                self.waypoint_connections[rover].add((wp1, wp2))

    def __call__(self, node):
        """Compute heuristic estimate for the given state."""
        state = node.state
        total_cost = 0

        # Check if we've already reached the goal
        if self.goals <= state:
            return 0

        # Extract current rover positions and what they're carrying
        rover_positions = {}
        rover_analyses = {'soil': {}, 'rock': {}}
        rover_images = {}
        calibrated_cameras = set()
        stored_samples = set()

        for fact in state:
            parts = get_parts(fact)
            
            if match(fact, "at", "*", "*"):
                rover, waypoint = parts[1], parts[2]
                rover_positions[rover] = waypoint
            
            elif match(fact, "have_soil_analysis", "*", "*"):
                rover, waypoint = parts[1], parts[2]
                rover_analyses['soil'][waypoint] = rover
            
            elif match(fact, "have_rock_analysis", "*", "*"):
                rover, waypoint = parts[1], parts[2]
                rover_analyses['rock'][waypoint] = rover
            
            elif match(fact, "have_image", "*", "*", "*"):
                rover, objective, mode = parts[1], parts[2], parts[3]
                if rover not in rover_images:
                    rover_images[rover] = set()
                rover_images[rover].add((objective, mode))
            
            elif match(fact, "calibrated", "*", "*"):
                camera, rover = parts[1], parts[2]
                calibrated_cameras.add((camera, rover))
            
            elif match(fact, "full", "*"):
                store = parts[1]
                stored_samples.add(store)

        # Process each goal
        for goal in self.goals:
            parts = get_parts(goal)
            
            # Soil data communication goal
            if match(goal, "communicated_soil_data", "*"):
                waypoint = parts[1]
                
                # Check if already communicated
                if goal in state:
                    continue
                
                # Check if any rover has the analysis
                if waypoint in rover_analyses['soil']:
                    rover = rover_analyses['soil'][waypoint]
                    # Need to communicate (1 action) plus possibly move to visible location
                    total_cost += 1
                    # Estimate movement cost (at least 1 move action)
                    total_cost += 1
                else:
                    # Need to sample first (if sample still available)
                    if waypoint in self.sample_locations['soil']:
                        # Find a rover with soil analysis capability
                        for rover, capabilities in self.rover_capabilities.items():
                            if 'soil' in capabilities:
                                # Sample action (1) plus move to location (at least 1)
                                total_cost += 2
                                # Then communicate (1) plus possible move (1)
                                total_cost += 2
                                break
                    else:
                        # Sample already taken by another rover - problematic case
                        total_cost += 4  # conservative estimate
            
            # Rock data communication goal
            elif match(goal, "communicated_rock_data", "*"):
                waypoint = parts[1]
                
                if goal in state:
                    continue
                
                if waypoint in rover_analyses['rock']:
                    rover = rover_analyses['rock'][waypoint]
                    total_cost += 1  # communicate
                    total_cost += 1  # move
                else:
                    if waypoint in self.sample_locations['rock']:
                        for rover, capabilities in self.rover_capabilities.items():
                            if 'rock' in capabilities:
                                total_cost += 2  # sample + move
                                total_cost += 2  # communicate + move
                                break
                    else:
                        total_cost += 4
            
            # Image data communication goal
            elif match(goal, "communicated_image_data", "*", "*"):
                objective, mode = parts[1], parts[2]
                
                if goal in state:
                    continue
                
                # Check if any rover has the image
                image_found = False
                for rover in rover_images:
                    if (objective, mode) in rover_images[rover]:
                        image_found = True
                        total_cost += 1  # communicate
                        total_cost += 1  # move
                        break
                
                if not image_found:
                    # Need to take image first
                    # Find a rover with imaging capability and suitable camera
                    for rover, capabilities in self.rover_capabilities.items():
                        if 'imaging' in capabilities and rover in self.camera_info:
                            for camera in self.camera_info[rover]:
                                if (isinstance(camera, str) and mode in self.camera_info[camera].get('modes', set()):
                                    # Calibrate (1), take image (1), communicate (1)
                                    total_cost += 3
                                    # Movement to calibration and imaging locations (estimate 2)
                                    total_cost += 2
                                    image_found = True
                                    break
                            if image_found:
                                break
                    
                    if not image_found:
                        total_cost += 5  # conservative estimate

        return total_cost
