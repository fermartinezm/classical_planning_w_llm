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
    This heuristic estimates the number of actions required to achieve all goals by:
    1) Counting unsatisfied communication goals (soil, rock, image data)
    2) Estimating the cost to:
       - Move rovers to sample locations
       - Perform sampling/imaging
       - Move to lander locations
       - Communicate data

    # Assumptions:
    - Each rover can handle only one task at a time (sample or image)
    - Communication requires being at a waypoint visible to the lander
    - Soil/rock samples can only be collected once per waypoint
    - Images can be taken multiple times (if recalibrated)

    # Heuristic Initialization
    - Extract goal conditions (communicated data)
    - Extract static information about:
      - Rover capabilities
      - Waypoint visibility
      - Sample locations
      - Camera capabilities
      - Lander positions

    # Step-By-Step Thinking for Computing Heuristic
    1) For each unsatisfied communication goal:
       a) If it's soil/rock data:
          - Find closest rover with soil/rock analysis capability
          - Estimate moves to sample location + moves to lander
          - Add sampling and communication actions
       b) If it's image data:
          - Find rover with appropriate camera
          - Estimate moves to calibration location + imaging location
          - Add calibration, imaging, and communication actions
    2) Sum all estimated actions
    3) Add penalty for rovers needing to drop samples before new collection
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract static information
        self.lander_positions = {}
        self.rover_capabilities = {}
        self.waypoint_connections = set()
        self.sample_locations = {'soil': set(), 'rock': set()}
        self.camera_info = {}
        self.objective_locations = {}

        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "at_lander":
                self.lander_positions[parts[1]] = parts[2]
            elif parts[0] == "equipped_for_soil_analysis":
                self.rover_capabilities.setdefault(parts[1], set()).add('soil')
            elif parts[0] == "equipped_for_rock_analysis":
                self.rover_capabilities.setdefault(parts[1], set()).add('rock')
            elif parts[0] == "equipped_for_imaging":
                self.rover_capabilities.setdefault(parts[1], set()).add('imaging')
            elif parts[0] == "visible":
                self.waypoint_connections.add((parts[1], parts[2]))
            elif parts[0] == "at_soil_sample":
                self.sample_locations['soil'].add(parts[1])
            elif parts[0] == "at_rock_sample":
                self.sample_locations['rock'].add(parts[1])
            elif parts[0] == "calibration_target":
                self.camera_info.setdefault(parts[1], {})['target'] = parts[2]
            elif parts[0] == "on_board":
                self.camera_info.setdefault(parts[1], {})['rover'] = parts[2]
            elif parts[0] == "supports":
                self.camera_info.setdefault(parts[1], {}).setdefault('modes', set()).add(parts[2])
            elif parts[0] == "visible_from":
                self.objective_locations.setdefault(parts[1], set()).add(parts[2])

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal from the current state."""
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
            
            if parts[0] == "communicated_soil_data":
                waypoint = parts[1]
                # Find a rover that can collect soil samples
                for rover, capabilities in self.rover_capabilities.items():
                    if 'soil' in capabilities:
                        # Estimate cost to collect and communicate
                        total_cost += self.estimate_soil_sample_cost(rover, waypoint, state)
                        break
            
            elif parts[0] == "communicated_rock_data":
                waypoint = parts[1]
                # Find a rover that can collect rock samples
                for rover, capabilities in self.rover_capabilities.items():
                    if 'rock' in capabilities:
                        # Estimate cost to collect and communicate
                        total_cost += self.estimate_rock_sample_cost(rover, waypoint, state)
                        break
            
            elif parts[0] == "communicated_image_data":
                objective = parts[1]
                mode = parts[2]
                # Find a rover with a camera that supports this mode
                for camera, info in self.camera_info.items():
                    if mode in info.get('modes', set()):
                        rover = info['rover']
                        # Estimate cost to take and communicate image
                        total_cost += self.estimate_image_cost(rover, camera, objective, mode, state)
                        break

        return total_cost

    def estimate_soil_sample_cost(self, rover, waypoint, state):
        """Estimate actions needed to collect and communicate soil sample."""
        cost = 0
        
        # Check if rover already has this sample
        has_sample = any(
            match(fact, "have_soil_analysis", rover, waypoint)
            for fact in state
        )
        
        if not has_sample:
            # Need to collect sample first
            cost += 1  # sample_soil action
            
            # Check if rover needs to move to sample location
            if not any(match(fact, "at", rover, waypoint) for fact in state):
                cost += 1  # navigate action (optimistic estimate)
        
        # Check if rover needs to move to communicate
        lander_pos = next(iter(self.lander_positions.values()))  # Assuming single lander
        if not any(match(fact, "at", rover, "*") and 
                  any(match(fact2, "visible", get_parts(fact)[2], lander_pos) 
                      for fact2 in self.static)
                  for fact in state):
            cost += 1  # navigate action to visible waypoint
        
        cost += 1  # communicate_soil_data action
        return cost

    def estimate_rock_sample_cost(self, rover, waypoint, state):
        """Estimate actions needed to collect and communicate rock sample."""
        cost = 0
        
        # Check if rover already has this sample
        has_sample = any(
            match(fact, "have_rock_analysis", rover, waypoint)
            for fact in state
        )
        
        if not has_sample:
            # Need to collect sample first
            cost += 1  # sample_rock action
            
            # Check if rover needs to move to sample location
            if not any(match(fact, "at", rover, waypoint) for fact in state):
                cost += 1  # navigate action (optimistic estimate)
        
        # Check if rover needs to move to communicate
        lander_pos = next(iter(self.lander_positions.values()))  # Assuming single lander
        if not any(match(fact, "at", rover, "*") and 
                  any(match(fact2, "visible", get_parts(fact)[2], lander_pos) 
                  for fact2 in self.static)
                  for fact in state):
            cost += 1  # navigate action to visible waypoint
        
        cost += 1  # communicate_rock_data action
        return cost

    def estimate_image_cost(self, rover, camera, objective, mode, state):
        """Estimate actions needed to take and communicate image."""
        cost = 0
        
        # Check if rover already has this image
        has_image = any(
            match(fact, "have_image", rover, objective, mode)
            for fact in state
        )
        
        if not has_image:
            # Need to take image first
            cost += 1  # take_image action
            
            # Check if camera needs calibration
            if not any(match(fact, "calibrated", camera, rover) for fact in state):
                cost += 1  # calibrate action
                
                # Need to move to calibration target location
                target = self.camera_info[camera]['target']
                visible_from = self.objective_locations.get(target, set())
                if not any(match(fact, "at", rover, wp) for wp in visible_from for fact in state):
                    cost += 1  # navigate action
            
            # Need to move to imaging location
            visible_from = self.objective_locations.get(objective, set())
            if not any(match(fact, "at", rover, wp) for wp in visible_from for fact in state):
                cost += 1  # navigate action
        
        # Check if rover needs to move to communicate
        lander_pos = next(iter(self.lander_positions.values()))  # Assuming single lander
        if not any(match(fact, "at", rover, "*") and 
                  any(match(fact2, "visible", get_parts(fact)[2], lander_pos) 
                      for fact2 in self.static)
                  for fact in state):
            cost += 1  # navigate action to visible waypoint
        
        cost += 1  # communicate_image_data action
        return cost
