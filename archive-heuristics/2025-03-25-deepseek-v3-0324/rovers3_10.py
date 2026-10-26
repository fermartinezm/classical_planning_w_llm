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
    - Calculating the cost for each uncommunicated data type (soil, rock, image)
    - Considering the rover's current state (location, equipped capabilities, stored samples)
    - Accounting for necessary calibration and navigation actions

    # Assumptions:
    - Each rover can only carry one sample at a time (due to single store)
    - Calibration is required before each imaging action
    - Communication requires being at a waypoint visible from the lander
    - Soil/rock samples can only be collected if the rover is at the waypoint with the sample

    # Heuristic Initialization
    - Extract goal conditions (what needs to be communicated)
    - Extract static information about:
        - Rover capabilities (equipped_for_* predicates)
        - Waypoint visibility and traversal
        - Camera support and calibration targets
        - Sample locations (static for initial state)

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all uncommunicated goals (soil, rock, image data)
    2. For each uncommunicated goal type:
        a. If it's soil/rock data:
            - Check if rover has the analysis (if yes, just need to communicate)
            - If not, check if sample exists and rover can collect it
            - Add costs for navigation, sampling, and communication
        b. If it's image data:
            - Check if rover has the image (if yes, just need to communicate)
            - If not, check if camera is calibrated and can take the image
            - Add costs for calibration, navigation, imaging, and communication
    3. Consider rover assignments to minimize total cost (assign tasks to best rover)
    4. Sum all required actions for the most efficient plan
    """

    def __init__(self, task):
        """Initialize by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static
        
        # Extract rover capabilities
        self.rover_capabilities = {}
        for fact in self.static:
            if match(fact, "equipped_for_*", "*"):
                parts = get_parts(fact)
                capability = parts[0]
                rover = parts[1]
                if rover not in self.rover_capabilities:
                    self.rover_capabilities[rover] = set()
                self.rover_capabilities[rover].add(capability)
        
        # Extract waypoint visibility
        self.visible = set()
        for fact in self.static:
            if match(fact, "visible", "*", "*"):
                parts = get_parts(fact)
                self.visible.add((parts[1], parts[2]))
        
        # Extract can_traverse relationships
        self.can_traverse = set()
        for fact in self.static:
            if match(fact, "can_traverse", "*", "*", "*"):
                parts = get_parts(fact)
                self.can_traverse.add((parts[1], parts[2], parts[3]))
        
        # Extract camera information
        self.camera_info = {}
        for fact in self.static:
            if match(fact, "supports", "*", "*"):
                parts = get_parts(fact)
                camera, mode = parts[1], parts[2]
                if camera not in self.camera_info:
                    self.camera_info[camera] = {'supports': set(), 'on_board': None, 'target': None}
                self.camera_info[camera]['supports'].add(mode)
            elif match(fact, "on_board", "*", "*"):
                parts = get_parts(fact)
                camera, rover = parts[1], parts[2]
                if camera not in self.camera_info:
                    self.camera_info[camera] = {'supports': set(), 'on_board': None, 'target': None}
                self.camera_info[camera]['on_board'] = rover
            elif match(fact, "calibration_target", "*", "*"):
                parts = get_parts(fact)
                camera, objective = parts[1], parts[2]
                if camera not in self.camera_info:
                    self.camera_info[camera] = {'supports': set(), 'on_board': None, 'target': None}
                self.camera_info[camera]['target'] = objective
        
        # Extract visible_from relationships
        self.visible_from = {}
        for fact in self.static:
            if match(fact, "visible_from", "*", "*"):
                parts = get_parts(fact)
                objective, waypoint = parts[1], parts[2]
                if objective not in self.visible_from:
                    self.visible_from[objective] = set()
                self.visible_from[objective].add(waypoint)
        
        # Extract lander location
        self.lander_location = None
        for fact in self.static:
            if match(fact, "at_lander", "*", "*"):
                parts = get_parts(fact)
                self.lander_location = parts[2]
                break

    def __call__(self, node):
        """Compute heuristic estimate for given state."""
        state = node.state
        
        # Check if goal is already satisfied
        if self.goals <= state:
            return 0
        
        # Initialize cost
        total_cost = 0
        
        # Track which goals are already satisfied
        satisfied_goals = set()
        for goal in self.goals:
            if goal in state:
                satisfied_goals.add(goal)
        
        # Process uncommunicated soil data goals
        soil_goals = set()
        for goal in self.goals:
            if match(goal, "communicated_soil_data", "*") and goal not in satisfied_goals:
                waypoint = get_parts(goal)[1]
                soil_goals.add(waypoint)
        
        # Process uncommunicated rock data goals
        rock_goals = set()
        for goal in self.goals:
            if match(goal, "communicated_rock_data", "*") and goal not in satisfied_goals:
                waypoint = get_parts(goal)[1]
                rock_goals.add(waypoint)
        
        # Process uncommunicated image data goals
        image_goals = set()
        for goal in self.goals:
            if match(goal, "communicated_image_data", "*", "*") and goal not in satisfied_goals:
                objective, mode = get_parts(goal)[1], get_parts(goal)[2]
                image_goals.add((objective, mode))
        
        # Helper function to estimate navigation cost between two waypoints
        def estimate_nav_cost(from_wp, to_wp):
            # Simple estimate: 1 action per hop (assuming direct path exists)
            # In a more sophisticated version, we could implement pathfinding
            if from_wp == to_wp:
                return 0
            return 1  # Optimistic estimate
        
        # Process soil goals
        for waypoint in soil_goals:
            # Check if any rover already has the analysis
            rover_has_analysis = False
            for fact in state:
                if match(fact, "have_soil_analysis", "*", waypoint):
                    rover_has_analysis = True
                    rover = get_parts(fact)[1]
                    break
            
            if rover_has_analysis:
                # Just need to communicate (1 action)
                total_cost += 1
            else:
                # Need to sample and communicate (at least 3 actions: navigate, sample, communicate)
                total_cost += 3
        
        # Process rock goals
        for waypoint in rock_goals:
            # Check if any rover already has the analysis
            rover_has_analysis = False
            for fact in state:
                if match(fact, "have_rock_analysis", "*", waypoint):
                    rover_has_analysis = True
                    rover = get_parts(fact)[1]
                    break
            
            if rover_has_analysis:
                # Just need to communicate (1 action)
                total_cost += 1
            else:
                # Need to sample and communicate (at least 3 actions: navigate, sample, communicate)
                total_cost += 3
        
        # Process image goals
        for objective, mode in image_goals:
            # Check if any rover already has the image
            rover_has_image = False
            for fact in state:
                if match(fact, "have_image", "*", objective, mode):
                    rover_has_image = True
                    rover = get_parts(fact)[1]
                    break
            
            if rover_has_image:
                # Just need to communicate (1 action)
                total_cost += 1
            else:
                # Need to calibrate, take image, and communicate (at least 3 actions)
                total_cost += 3
        
        return total_cost
