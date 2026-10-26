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
    - Soil/rock samples can only be collected if present at waypoint
    - Images require proper camera calibration and visibility conditions

    # Heuristic Initialization
    - Extract goal conditions (what needs to be communicated)
    - Extract static information about:
        - Rover capabilities (equipment)
        - Waypoint connectivity (can_traverse)
        - Camera configurations
        - Sample locations
        - Objective visibility

    # Step-By-Step Thinking for Computing Heuristic
    1. For each communication goal (soil, rock, image):
        a. If already communicated, skip
        b. Otherwise determine the steps needed:
            - For soil/rock:
                * Find closest rover with proper equipment
                * Calculate distance to sample waypoint
                * Add sample and return-to-lander actions
            - For images:
                * Find rover with proper camera
                * Calculate calibration and image capture steps
                * Add return-to-lander actions
    2. Sum all required actions across all goals
    3. Add penalty for rovers that need to drop samples before new collection
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract static information into more accessible data structures
        self.rover_capabilities = {}
        self.waypoint_connections = {}
        self.camera_info = {}
        self.sample_locations = {'soil': set(), 'rock': set()}
        self.objective_visibility = {}

        for fact in self.static:
            parts = get_parts(fact)
            if not parts:
                continue

            # Rover capabilities
            if match(fact, "equipped_for_*", "*"):
                capability = parts[0].split('_')[-2]  # soil/rock/imaging
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
            elif match(fact, "on_board", "*", "*"):
                camera, rover = parts[1], parts[2]
                if rover not in self.camera_info:
                    self.camera_info[rover] = []
                self.camera_info[rover].append(camera)
            
            elif match(fact, "supports", "*", "*"):
                camera, mode = parts[1], parts[2]
                # Store camera modes
                pass
            
            elif match(fact, "calibration_target", "*", "*"):
                camera, objective = parts[1], parts[2]
                # Store calibration targets
                pass
            
            # Sample locations
            elif match(fact, "at_soil_sample", "*"):
                self.sample_locations['soil'].add(parts[1])
            
            elif match(fact, "at_rock_sample", "*"):
                self.sample_locations['rock'].add(parts[1])
            
            # Objective visibility
            elif match(fact, "visible_from", "*", "*"):
                objective, waypoint = parts[1], parts[2]
                if objective not in self.objective_visibility:
                    self.objective_visibility[objective] = set()
                self.objective_visibility[objective].add(waypoint)

        # Find lander location
        self.lander_location = None
        for fact in self.static:
            if match(fact, "at_lander", "*", "*"):
                self.lander_location = get_parts(fact)[2]
                break

    def __call__(self, node):
        """Estimate the number of actions needed to achieve all communication goals."""
        state = node.state
        total_cost = 0

        # Check which goals are already satisfied
        unsatisfied_goals = set()
        for goal in self.goals:
            if goal not in state:
                unsatisfied_goals.add(goal)

        if not unsatisfied_goals:
            return 0  # All goals satisfied

        # Process each unsatisfied goal
        for goal in unsatisfied_goals:
            parts = get_parts(goal)
            goal_type = parts[0].split('_')[1]  # soil/rock/image

            if goal_type == 'soil':
                waypoint = parts[2]
                total_cost += self._estimate_soil_cost(state, waypoint)
            elif goal_type == 'rock':
                waypoint = parts[2]
                total_cost += self._estimate_rock_cost(state, waypoint)
            elif goal_type == 'image':
                objective, mode = parts[2], parts[3]
                total_cost += self._estimate_image_cost(state, objective, mode)

        return total_cost

    def _estimate_soil_cost(self, state, waypoint):
        """Estimate actions needed to communicate soil data from given waypoint."""
        cost = 0
        
        # Check if already have analysis
        for fact in state:
            if match(fact, "have_soil_analysis", "*", waypoint):
                rover = get_parts(fact)[1]
                # Add communication cost
                return 1 + self._estimate_communication_cost(state, rover)
        
        # Need to collect sample
        cost += 3  # navigate to waypoint, sample, navigate to lander
        
        # Find closest rover with soil capability
        min_distance = float('inf')
        for rover in self.rover_capabilities:
            if 'soil' in self.rover_capabilities[rover]:
                distance = self._estimate_distance(state, rover, waypoint)
                if distance < min_distance:
                    min_distance = distance
        
        if min_distance != float('inf'):
            cost += min_distance
        
        # Add communication cost (1 action)
        cost += 1
        
        return cost

    def _estimate_rock_cost(self, state, waypoint):
        """Estimate actions needed to communicate rock data from given waypoint."""
        cost = 0
        
        # Check if already have analysis
        for fact in state:
            if match(fact, "have_rock_analysis", "*", waypoint):
                rover = get_parts(fact)[1]
                # Add communication cost
                return 1 + self._estimate_communication_cost(state, rover)
        
        # Need to collect sample
        cost += 3  # navigate to waypoint, sample, navigate to lander
        
        # Find closest rover with rock capability
        min_distance = float('inf')
        for rover in self.rover_capabilities:
            if 'rock' in self.rover_capabilities[rover]:
                distance = self._estimate_distance(state, rover, waypoint)
                if distance < min_distance:
                    min_distance = distance
        
        if min_distance != float('inf'):
            cost += min_distance
        
        # Add communication cost (1 action)
        cost += 1
        
        return cost

    def _estimate_image_cost(self, state, objective, mode):
        """Estimate actions needed to communicate image data for given objective and mode."""
        cost = 0
        
        # Check if already have image
        for fact in state:
            if match(fact, "have_image", "*", objective, mode):
                rover = get_parts(fact)[1]
                # Add communication cost
                return 1 + self._estimate_communication_cost(state, rover)
        
        # Need to take image
        cost += 4  # calibrate, take_image, navigate to lander, communicate
        
        # Find rover with appropriate camera
        for rover in self.camera_info:
            if 'imaging' in self.rover_capabilities.get(rover, set()):
                for camera in self.camera_info[rover]:
                    # Check if camera supports required mode
                    for fact in self.static:
                        if match(fact, "supports", camera, mode):
                            # Add distance to calibration waypoint
                            calibration_wp = self._find_calibration_waypoint(objective, rover)
                            if calibration_wp:
                                distance = self._estimate_distance(state, rover, calibration_wp)
                                cost += distance
                                break
        
        return cost

    def _estimate_distance(self, state, rover, target_waypoint):
        """Estimate navigation distance between rover's current position and target waypoint."""
        # Find current position
        current_pos = None
        for fact in state:
            if match(fact, "at", rover, "*"):
                current_pos = get_parts(fact)[2]
                break
        
        if not current_pos or current_pos == target_waypoint:
            return 0
        
        # Simple estimation: 1 action per waypoint (could be improved with pathfinding)
        return 1

    def _estimate_communication_cost(self, state, rover):
        """Estimate cost to communicate data from rover's current position."""
        # Check if already at visible waypoint to lander
        rover_pos = None
        for fact in state:
            if match(fact, "at", rover, "*"):
                rover_pos = get_parts(fact)[2]
                break
        
        if not rover_pos:
            return 1  # Need to navigate
        
        # Check visibility to lander
        for fact in self.static:
            if match(fact, "visible", rover_pos, self.lander_location):
                return 1  # Can communicate directly
        
        return 2  # Need to navigate + communicate

    def _find_calibration_waypoint(self, objective, rover):
        """Find a waypoint where the rover can calibrate for given objective."""
        # Find calibration target for rover's cameras
        target = None
        for fact in self.static:
            if match(fact, "calibration_target", "*", objective):
                camera = get_parts(fact)[1]
                # Check if rover has this camera
                if camera in self.camera_info.get(rover, []):
                    target = objective
                    break
        
        if not target:
            return None
        
        # Find visible waypoint for calibration
        for fact in self.static:
            if match(fact, "visible_from", objective, "*"):
                waypoint = get_parts(fact)[2]
                return waypoint
        
        return None
