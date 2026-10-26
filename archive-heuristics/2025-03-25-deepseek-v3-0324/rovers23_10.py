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
    - The distance rovers need to travel to collect samples or take images
    - The need to calibrate cameras before taking images
    - The need to communicate data back to the lander

    # Assumptions:
    - Each rover can only carry one sample at a time (due to single store)
    - Communication requires being at a waypoint visible to the lander
    - Soil/rock samples can only be collected once
    - Images can be taken multiple times if needed (but require recalibration)

    # Heuristic Initialization
    - Extract goal conditions (what needs to be communicated)
    - Extract static information about:
        - Rover capabilities (equipment)
        - Waypoint connectivity (can_traverse)
        - Camera support and calibration targets
        - Sample locations
        - Objective visibility

    # Step-By-Step Thinking for Computing Heuristic
    1. For each communication goal (soil, rock, image):
        a. If already communicated, skip
        b. Otherwise:
            i. For soil/rock:
                - Find closest rover with appropriate equipment
                - Estimate distance to sample waypoint
                - Add actions for: navigate, sample, navigate to lander-visible waypoint, communicate
            ii. For images:
                - Find rover with appropriate camera
                - Estimate distance to calibration waypoint (if needed)
                - Estimate distance to imaging waypoint
                - Add actions for: calibrate, navigate, take_image, navigate to lander-visible waypoint, communicate
    2. Sum all estimated actions
    3. Add penalty for rovers needing to drop samples before new collection
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract rover capabilities
        self.equipped_for = {
            'soil': set(),
            'rock': set(),
            'imaging': set()
        }
        
        # Extract waypoint connectivity
        self.can_traverse = {}
        
        # Extract camera information
        self.camera_info = {}
        self.calibration_targets = {}
        
        # Extract sample locations
        self.initial_soil_samples = set()
        self.initial_rock_samples = set()
        
        # Extract objective visibility
        self.objective_visibility = {}
        
        # Extract lander position
        self.lander_position = None
        
        for fact in self.static:
            if match(fact, "equipped_for_soil_analysis", "*"):
                rover = get_parts(fact)[1]
                self.equipped_for['soil'].add(rover)
            elif match(fact, "equipped_for_rock_analysis", "*"):
                rover = get_parts(fact)[1]
                self.equipped_for['rock'].add(rover)
            elif match(fact, "equipped_for_imaging", "*"):
                rover = get_parts(fact)[1]
                self.equipped_for['imaging'].add(rover)
            elif match(fact, "can_traverse", "*", "*", "*"):
                rover, wp1, wp2 = get_parts(fact)[1:4]
                if rover not in self.can_traverse:
                    self.can_traverse[rover] = set()
                self.can_traverse[rover].add((wp1, wp2))
            elif match(fact, "calibration_target", "*", "*"):
                camera, objective = get_parts(fact)[1:3]
                self.calibration_targets[camera] = objective
            elif match(fact, "on_board", "*", "*"):
                camera, rover = get_parts(fact)[1:3]
                if rover not in self.camera_info:
                    self.camera_info[rover] = []
                self.camera_info[rover].append(camera)
            elif match(fact, "supports", "*", "*"):
                camera, mode = get_parts(fact)[1:3]
                # Store camera modes if needed
            elif match(fact, "at_soil_sample", "*"):
                wp = get_parts(fact)[1]
                self.initial_soil_samples.add(wp)
            elif match(fact, "at_rock_sample", "*"):
                wp = get_parts(fact)[1]
                self.initial_rock_samples.add(wp)
            elif match(fact, "visible_from", "*", "*"):
                objective, wp = get_parts(fact)[1:3]
                if objective not in self.objective_visibility:
                    self.objective_visibility[objective] = set()
                self.objective_visibility[objective].add(wp)
            elif match(fact, "at_lander", "*", "*"):
                lander, wp = get_parts(fact)[1:3]
                self.lander_position = wp

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal from the current state."""
        state = node.state
        
        # Check if goal is already reached
        if all(goal in state for goal in self.goals):
            return 0
            
        total_cost = 0
        
        # Process communication goals
        for goal in self.goals:
            if goal in state:
                continue  # Already communicated
                
            parts = get_parts(goal)
            if parts[0] == "communicated_soil_data":
                wp = parts[1]
                total_cost += self._estimate_soil_communication_cost(state, wp)
            elif parts[0] == "communicated_rock_data":
                wp = parts[1]
                total_cost += self._estimate_rock_communication_cost(state, wp)
            elif parts[0] == "communicated_image_data":
                objective, mode = parts[1:3]
                total_cost += self._estimate_image_communication_cost(state, objective, mode)
        
        return total_cost

    def _estimate_soil_communication_cost(self, state, waypoint):
        """Estimate cost to communicate soil data from given waypoint."""
        cost = 0
        
        # Check if we already have the analysis
        for fact in state:
            if match(fact, "have_soil_analysis", "*", waypoint):
                return self._estimate_communication_cost(state, waypoint)
        
        # Need to collect sample first
        cost += 3  # navigate to waypoint, sample, navigate to lander-visible waypoint
        cost += self._estimate_communication_cost(state, waypoint)
        return cost

    def _estimate_rock_communication_cost(self, state, waypoint):
        """Estimate cost to communicate rock data from given waypoint."""
        cost = 0
        
        # Check if we already have the analysis
        for fact in state:
            if match(fact, "have_rock_analysis", "*", waypoint):
                return self._estimate_communication_cost(state, waypoint)
        
        # Need to collect sample first
        cost += 3  # navigate to waypoint, sample, navigate to lander-visible waypoint
        cost += self._estimate_communication_cost(state, waypoint)
        return cost

    def _estimate_image_communication_cost(self, state, objective, mode):
        """Estimate cost to communicate image data for given objective and mode."""
        cost = 0
        
        # Check if we already have the image
        for fact in state:
            if match(fact, "have_image", "*", objective, mode):
                return self._estimate_communication_cost(state, None, objective)
        
        # Need to take image first
        cost += 4  # calibrate (if needed), navigate, take_image, navigate to lander-visible waypoint
        cost += self._estimate_communication_cost(state, None, objective)
        return cost

    def _estimate_communication_cost(self, state, waypoint=None, objective=None):
        """Estimate cost to communicate data (either from waypoint or for objective)."""
        # Assume we're already at a waypoint visible to lander
        # If not, add 1 for navigation
        return 1  # communicate action
