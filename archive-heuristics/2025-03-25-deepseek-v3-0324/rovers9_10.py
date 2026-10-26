from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """Check if a PDDL fact matches a given pattern."""
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve all goals by:
    1. Counting remaining uncommunicated data (soil, rock, images)
    2. Estimating the steps required to collect each piece of data
    3. Estimating the steps required to communicate each piece of data
    4. Considering rover equipment and current state

    # Assumptions:
    - Each rover can only carry one sample at a time (due to single store)
    - Soil/rock samples must be collected before communication
    - Images must be taken before communication
    - Calibration is required before taking images
    - Rovers may need to navigate between waypoints

    # Heuristic Initialization
    - Extract goal conditions (what needs to be communicated)
    - Extract static information about:
      - Rover capabilities (equipment)
      - Waypoint connectivity (can_traverse)
      - Sample locations (static soil/rock samples)
      - Camera capabilities and targets
      - Lander locations

    # Step-By-Step Thinking for Computing Heuristic
    1. For each uncommunicated soil/rock data:
       - If not collected yet: 
         * Find closest rover with appropriate equipment
         * Estimate navigation steps to sample location
         * Add sample action
       - If collected but not communicated:
         * Estimate navigation steps to lander
         * Add communication action
    2. For each uncommunicated image:
       - If not taken yet:
         * Find rover with appropriate camera
         * Estimate calibration steps if needed
         * Estimate navigation to visible waypoint
         * Add take_image action
       - If taken but not communicated:
         * Estimate navigation to lander
         * Add communication action
    3. Sum all estimated actions
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static
        
        # Extract static information
        self.rover_equipment = {
            'soil': set(),
            'rock': set(),
            'imaging': set()
        }
        self.rovers = set()
        self.waypoints = set()
        self.landers = {}
        self.sample_locations = {
            'soil': set(),
            'rock': set()
        }
        self.camera_info = {}
        self.visible_from = {}
        self.can_traverse = {}
        
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'equipped_for_soil_analysis':
                self.rover_equipment['soil'].add(parts[1])
                self.rovers.add(parts[1])
            elif parts[0] == 'equipped_for_rock_analysis':
                self.rover_equipment['rock'].add(parts[1])
                self.rovers.add(parts[1])
            elif parts[0] == 'equipped_for_imaging':
                self.rover_equipment['imaging'].add(parts[1])
                self.rovers.add(parts[1])
            elif parts[0] == 'at_lander':
                self.landers[parts[1]] = parts[2]
            elif parts[0] == 'at_soil_sample':
                self.sample_locations['soil'].add(parts[1])
            elif parts[0] == 'at_rock_sample':
                self.sample_locations['rock'].add(parts[1])
            elif parts[0] == 'can_traverse':
                rover, wp1, wp2 = parts[1], parts[2], parts[3]
                if rover not in self.can_traverse:
                    self.can_traverse[rover] = set()
                self.can_traverse[rover].add((wp1, wp2))
                self.waypoints.update({wp1, wp2})
            elif parts[0] == 'calibration_target':
                camera, objective = parts[1], parts[2]
                if camera not in self.camera_info:
                    self.camera_info[camera] = {'target': None, 'supports': set()}
                self.camera_info[camera]['target'] = objective
            elif parts[0] == 'supports':
                camera, mode = parts[1], parts[2]
                if camera not in self.camera_info:
                    self.camera_info[camera] = {'target': None, 'supports': set()}
                self.camera_info[camera]['supports'].add(mode)
            elif parts[0] == 'on_board':
                camera, rover = parts[1], parts[2]
                if camera not in self.camera_info:
                    self.camera_info[camera] = {'target': None, 'supports': set()}
                self.camera_info[camera]['rover'] = rover
            elif parts[0] == 'visible_from':
                objective, waypoint = parts[1], parts[2]
                if objective not in self.visible_from:
                    self.visible_from[objective] = set()
                self.visible_from[objective].add(waypoint)
        
        # Extract goal information
        self.goal_soil = set()
        self.goal_rock = set()
        self.goal_images = set()
        
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'communicated_soil_data':
                self.goal_soil.add(parts[1])
            elif parts[0] == 'communicated_rock_data':
                self.goal_rock.add(parts[1])
            elif parts[0] == 'communicated_image_data':
                self.goal_images.add((parts[1], parts[2]))

    def __call__(self, node):
        """Compute heuristic estimate for given state."""
        state = node.state
        total_cost = 0
        
        # Check which goals are already satisfied
        satisfied_soil = set()
        satisfied_rock = set()
        satisfied_images = set()
        
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'communicated_soil_data' and parts[1] in self.goal_soil:
                satisfied_soil.add(parts[1])
            elif parts[0] == 'communicated_rock_data' and parts[1] in self.goal_rock:
                satisfied_rock.add(parts[1])
            elif parts[0] == 'communicated_image_data' and (parts[1], parts[2]) in self.goal_images:
                satisfied_images.add((parts[1], parts[2]))
        
        # Soil data collection and communication
        for wp in self.goal_soil - satisfied_soil:
            # Check if sample is already collected
            collected = any(
                match(fact, 'have_soil_analysis', '*', wp) for fact in state
            )
            
            if collected:
                # Need to communicate - estimate 2 actions (navigate to lander and communicate)
                total_cost += 2
            else:
                # Need to collect and communicate - estimate 4 actions:
                # navigate to sample, sample, navigate to lander, communicate
                total_cost += 4
        
        # Rock data collection and communication
        for wp in self.goal_rock - satisfied_rock:
            collected = any(
                match(fact, 'have_rock_analysis', '*', wp) for fact in state
            )
            
            if collected:
                total_cost += 2
            else:
                total_cost += 4
        
        # Image data collection and communication
        for (obj, mode) in self.goal_images - satisfied_images:
            # Check if image is already taken
            taken = any(
                match(fact, 'have_image', '*', obj, mode) for fact in state
            )
            
            if taken:
                # Just need to communicate - 2 actions
                total_cost += 2
            else:
                # Need to take image and communicate - estimate more steps
                # Find a camera that supports this mode and can see the objective
                camera_found = False
                for camera, info in self.camera_info.items():
                    if mode in info['supports'] and info['target'] == obj:
                        camera_found = True
                        break
                
                if camera_found:
                    # Estimate steps: calibrate (1), navigate (1), take image (1), navigate to lander (1), communicate (1)
                    total_cost += 5
                else:
                    # If no suitable camera, this goal is impossible - return large number
                    return float('inf')
        
        return total_cost
