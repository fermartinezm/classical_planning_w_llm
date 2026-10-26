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
       - Perform sampling (soil/rock)
       - Move to imaging locations
       - Calibrate cameras
       - Take images
       - Move to lander locations
       - Communicate data

    # Assumptions:
    - Each unsatisfied goal requires at least one action to communicate it
    - Sampling requires at least one move and one sample action
    - Imaging requires calibration, moving, and taking the image
    - Communication requires moving to a visible lander location
    - The heuristic is not admissible (may overestimate)

    # Heuristic Initialization
    - Extract goal conditions (what needs to be communicated)
    - Extract static information about:
      - Landers and their locations
      - Rover capabilities
      - Waypoint connectivity
      - Camera capabilities
      - Sample locations
      - Objective visibility

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unsatisfied communication goal:
       a) If it's soil data:
          - Check if rover has the analysis (count as 0 if yes)
          - Else estimate moves to sample location + sample action
       b) If it's rock data:
          - Similar to soil data
       c) If it's image data:
          - Check if rover has the image (count as 0 if yes)
          - Else estimate calibration + moves + imaging actions
    2. For each needed communication:
       - Estimate moves to reach lander from current position
       - Add communication action
    3. Sum all estimated actions
    """

    def __init__(self, task):
        """Initialize by extracting goal conditions and static information."""
        self.goals = task.goals
        self.static = task.static
        
        # Extract lander locations
        self.lander_locations = {
            get_parts(fact)[1]: get_parts(fact)[2]  # lander -> waypoint
            for fact in self.static 
            if match(fact, "at_lander", "*", "*")
        }
        
        # Extract rover capabilities
        self.rover_capabilities = {
            "soil": set(),
            "rock": set(),
            "imaging": set()
        }
        for fact in self.static:
            if match(fact, "equipped_for_soil_analysis", "*"):
                self.rover_capabilities["soil"].add(get_parts(fact)[1])
            elif match(fact, "equipped_for_rock_analysis", "*"):
                self.rover_capabilities["rock"].add(get_parts(fact)[1])
            elif match(fact, "equipped_for_imaging", "*"):
                self.rover_capabilities["imaging"].add(get_parts(fact)[1])
        
        # Extract waypoint connectivity
        self.visible_waypoints = set()
        self.can_traverse = set()
        for fact in self.static:
            if match(fact, "visible", "*", "*"):
                wp1, wp2 = get_parts(fact)[1], get_parts(fact)[2]
                self.visible_waypoints.add((wp1, wp2))
            elif match(fact, "can_traverse", "*", "*", "*"):
                rover, wp1, wp2 = get_parts(fact)[1], get_parts(fact)[2], get_parts(fact)[3]
                self.can_traverse.add((rover, wp1, wp2))
        
        # Extract camera information
        self.camera_info = {}
        for fact in self.static:
            if match(fact, "on_board", "*", "*"):
                cam, rover = get_parts(fact)[1], get_parts(fact)[2]
                if cam not in self.camera_info:
                    self.camera_info[cam] = {}
                self.camera_info[cam]["rover"] = rover
            elif match(fact, "supports", "*", "*"):
                cam, mode = get_parts(fact)[1], get_parts(fact)[2]
                if cam not in self.camera_info:
                    self.camera_info[cam] = {}
                if "modes" not in self.camera_info[cam]:
                    self.camera_info[cam]["modes"] = set()
                self.camera_info[cam]["modes"].add(mode)
            elif match(fact, "calibration_target", "*", "*"):
                cam, obj = get_parts(fact)[1], get_parts(fact)[2]
                if cam not in self.camera_info:
                    self.camera_info[cam] = {}
                self.camera_info[cam]["target"] = obj
        
        # Extract sample locations (initial state)
        self.initial_soil_samples = set()
        self.initial_rock_samples = set()
        for fact in task.initial_state:
            if match(fact, "at_soil_sample", "*"):
                self.initial_soil_samples.add(get_parts(fact)[1])
            elif match(fact, "at_rock_sample", "*"):
                self.initial_rock_samples.add(get_parts(fact)[1])
        
        # Extract objective visibility
        self.objective_visibility = {}
        for fact in self.static:
            if match(fact, "visible_from", "*", "*"):
                obj, wp = get_parts(fact)[1], get_parts(fact)[2]
                if obj not in self.objective_visibility:
                    self.objective_visibility[obj] = set()
                self.objective_visibility[obj].add(wp)

    def __call__(self, node):
        """Compute heuristic estimate for the given state."""
        state = node.state
        total_cost = 0
        
        # Check which goals are already satisfied
        unsatisfied_goals = self.goals - state
        
        # If all goals are satisfied, return 0
        if not unsatisfied_goals:
            return 0
        
        # Process each unsatisfied goal
        for goal in unsatisfied_goals:
            parts = get_parts(goal)
            
            # Soil data communication goal
            if match(goal, "communicated_soil_data", "*"):
                wp = parts[1]
                
                # Check if any rover already has this analysis
                has_analysis = any(
                    match(fact, "have_soil_analysis", "*", wp) 
                    for fact in state
                )
                
                if not has_analysis:
                    # Estimate cost to get the sample
                    # 1 move to sample location + 1 sample action
                    total_cost += 2
                    
            # Rock data communication goal
            elif match(goal, "communicated_rock_data", "*"):
                wp = parts[1]
                
                # Check if any rover already has this analysis
                has_analysis = any(
                    match(fact, "have_rock_analysis", "*", wp) 
                    for fact in state
                )
                
                if not has_analysis:
                    # Estimate cost to get the sample
                    # 1 move to sample location + 1 sample action
                    total_cost += 2
                    
            # Image data communication goal
            elif match(goal, "communicated_image_data", "*", "*"):
                obj, mode = parts[1], parts[2]
                
                # Check if any rover already has this image
                has_image = any(
                    match(fact, "have_image", "*", obj, mode) 
                    for fact in state
                )
                
                if not has_image:
                    # Estimate cost to get the image:
                    # 1 calibrate + 1 move to imaging location + 1 take_image
                    total_cost += 3
        
        # Estimate communication costs
        # For each unsatisfied goal, we need to communicate it
        # 1 move to lander location + 1 communicate action per goal
        # We'll assume we can communicate multiple goals from one location
        # So we'll add 1 move (to reach lander) + 1 per communication
        if unsatisfied_goals:
            total_cost += 1 + len(unsatisfied_goals)
        
        return total_cost
