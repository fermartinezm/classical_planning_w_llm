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
    - Each rover can carry only one sample at a time (due to single store)
    - Communication requires being at a waypoint visible to the lander
    - Soil/rock samples must be collected before communication
    - Images require proper camera calibration before capture

    # Heuristic Initialization
    - Extract goal conditions (what needs to be communicated)
    - Build mapping of waypoints to their samples (soil/rock)
    - Build mapping of objectives to their visible waypoints
    - Identify lander location
    - Identify rover capabilities and initial positions

    # Step-By-Step Thinking for Computing Heuristic
    1. For each communication goal (soil, rock, image):
        a. If already communicated, skip
        b. Otherwise:
            i. For soil/rock:
                - Find closest rover with soil/rock capability
                - Estimate distance to sample waypoint
                - Add actions for sampling and communicating
            ii. For images:
                - Find rover with appropriate camera
                - Estimate distance to calibration waypoint
                - Estimate distance to imaging waypoint
                - Add actions for calibration, imaging, and communicating
    2. Sum all estimated actions
    3. Add penalty for rovers needing to drop samples before new collection
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract static information
        self.lander_location = None
        self.soil_samples = set()
        self.rock_samples = set()
        self.objective_waypoints = {}
        self.rover_capabilities = {}
        self.rover_positions = {}
        self.camera_info = {}
        self.visible_map = {}

        for fact in self.static:
            if match(fact, "at_lander", "*", "*"):
                self.lander_location = get_parts(fact)[2]
            elif match(fact, "at_soil_sample", "*"):
                self.soil_samples.add(get_parts(fact)[1])
            elif match(fact, "at_rock_sample", "*"):
                self.rock_samples.add(get_parts(fact)[1])
            elif match(fact, "visible_from", "*", "*"):
                obj, wp = get_parts(fact)[1], get_parts(fact)[2]
                if obj not in self.objective_waypoints:
                    self.objective_waypoints[obj] = set()
                self.objective_waypoints[obj].add(wp)
            elif match(fact, "visible", "*", "*"):
                wp1, wp2 = get_parts(fact)[1], get_parts(fact)[2]
                if wp1 not in self.visible_map:
                    self.visible_map[wp1] = set()
                self.visible_map[wp1].add(wp2)
            elif match(fact, "equipped_for_*", "*"):
                capability, rover = get_parts(fact)[0].split('_')[2], get_parts(fact)[1]
                if rover not in self.rover_capabilities:
                    self.rover_capabilities[rover] = set()
                self.rover_capabilities[rover].add(capability)
            elif match(fact, "on_board", "*", "*"):
                camera, rover = get_parts(fact)[1], get_parts(fact)[2]
                if rover not in self.camera_info:
                    self.camera_info[rover] = set()
                self.camera_info[rover].add(camera)
            elif match(fact, "supports", "*", "*"):
                camera, mode = get_parts(fact)[1], get_parts(fact)[2]
                # Store camera modes if needed

    def __call__(self, node):
        """Estimate the number of actions needed to achieve all communication goals."""
        state = node.state
        total_cost = 0

        # Check which goals are already satisfied
        unsatisfied_soil = set()
        unsatisfied_rock = set()
        unsatisfied_images = set()

        for goal in self.goals:
            if goal not in state:
                parts = get_parts(goal)
                if parts[0] == "communicated_soil_data":
                    unsatisfied_soil.add(parts[1])
                elif parts[0] == "communicated_rock_data":
                    unsatisfied_rock.add(parts[1])
                elif parts[0] == "communicated_image_data":
                    unsatisfied_images.add((parts[1], parts[2]))

        # Update rover positions from current state
        for fact in state:
            if match(fact, "at", "*", "*"):
                rover, wp = get_parts(fact)[1], get_parts(fact)[2]
                self.rover_positions[rover] = wp

        # Estimate cost for soil data communication
        for wp in unsatisfied_soil:
            # Find closest rover with soil capability
            min_cost = float('inf')
            for rover, capabilities in self.rover_capabilities.items():
                if 'soil' in capabilities and rover in self.rover_positions:
                    current_pos = self.rover_positions[rover]
                    # Simple estimate: 1 action per waypoint distance
                    distance = 1  # Minimum cost even if at same waypoint
                    
                    # Check if rover already has the sample
                    has_sample = False
                    for fact in state:
                        if match(fact, "have_soil_analysis", rover, wp):
                            has_sample = True
                            break
                    
                    if has_sample:
                        # Just need to communicate
                        comm_cost = 1 if self._can_communicate(rover, current_pos) else 2
                        cost = comm_cost
                    else:
                        # Need to navigate, sample, and communicate
                        nav_cost = distance
                        sample_cost = 1
                        comm_cost = 1 if self._can_communicate(rover, wp) else 2
                        cost = nav_cost + sample_cost + comm_cost
                    
                    if cost < min_cost:
                        min_cost = cost
            
            if min_cost != float('inf'):
                total_cost += min_cost

        # Estimate cost for rock data communication (similar to soil)
        for wp in unsatisfied_rock:
            min_cost = float('inf')
            for rover, capabilities in self.rover_capabilities.items():
                if 'rock' in capabilities and rover in self.rover_positions:
                    current_pos = self.rover_positions[rover]
                    distance = 1
                    
                    has_sample = False
                    for fact in state:
                        if match(fact, "have_rock_analysis", rover, wp):
                            has_sample = True
                            break
                    
                    if has_sample:
                        comm_cost = 1 if self._can_communicate(rover, current_pos) else 2
                        cost = comm_cost
                    else:
                        nav_cost = distance
                        sample_cost = 1
                        comm_cost = 1 if self._can_communicate(rover, wp) else 2
                        cost = nav_cost + sample_cost + comm_cost
                    
                    if cost < min_cost:
                        min_cost = cost
            
            if min_cost != float('inf'):
                total_cost += min_cost

        # Estimate cost for image data communication
        for (obj, mode) in unsatisfied_images:
            min_cost = float('inf')
            for rover, cameras in self.camera_info.items():
                if rover not in self.rover_positions:
                    continue
                    
                current_pos = self.rover_positions[rover]
                
                # Check if rover already has the image
                has_image = False
                for fact in state:
                    if match(fact, "have_image", rover, obj, mode):
                        has_image = True
                        break
                
                if has_image:
                    comm_cost = 1 if self._can_communicate(rover, current_pos) else 2
                    cost = comm_cost
                else:
                    # Need to calibrate and take image
                    # Find a waypoint where the objective is visible
                    if obj not in self.objective_waypoints:
                        continue
                    
                    # Find closest waypoint where objective is visible
                    target_wp = next(iter(self.objective_waypoints[obj]))  # Just pick first for estimation
                    
                    # Check if camera is calibrated
                    calibrated = False
                    for camera in cameras:
                        for fact in state:
                            if match(fact, "calibrated", camera, rover):
                                calibrated = True
                                break
                        if calibrated:
                            break
                    
                    nav_to_calib = 1  # Estimate distance to calibration waypoint
                    calib_cost = 1 if not calibrated else 0
                    nav_to_image = 1  # Estimate distance to imaging waypoint
                    image_cost = 1
                    comm_cost = 1 if self._can_communicate(rover, target_wp) else 2
                    
                    cost = nav_to_calib + calib_cost + nav_to_image + image_cost + comm_cost
                
                if cost < min_cost:
                    min_cost = cost
            
            if min_cost != float('inf'):
                total_cost += min_cost

        return total_cost

    def _can_communicate(self, rover, rover_pos):
        """Check if rover can communicate from its current position."""
        return (rover_pos in self.visible_map and 
                self.lander_location in self.visible_map[rover_pos])
