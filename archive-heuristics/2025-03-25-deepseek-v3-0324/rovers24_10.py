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
    1) Counting unsatisfied communication goals (soil, rock, image data)
    2) Estimating the cost to:
       - Move rovers to sample locations
       - Take samples/images
       - Calibrate cameras when needed
       - Communicate data to the lander

    # Assumptions:
    - Each rover can handle only one task at a time (simplified estimation)
    - Navigation paths between waypoints are direct (no pathfinding complexity)
    - Soil/rock samples and images can be collected independently
    - Communication requires being at a waypoint visible to the lander

    # Heuristic Initialization
    - Extract goal conditions (what needs to be communicated)
    - Build maps of:
      - Waypoint connectivity (can_traverse)
      - Sample locations (soil/rock)
      - Objective visibility
      - Camera capabilities
      - Lander location

    # Step-By-Step Thinking for Computing Heuristic
    1) For each unsatisfied communication goal:
       a) If it's soil data:
          - Find closest rover with soil analysis capability
          - Estimate moves to sample location + moves to communication point
          - Add actions for sampling and communicating
       b) If it's rock data:
          - Similar to soil but with rock analysis capability
       c) If it's image data:
          - Find rover with appropriate camera
          - Estimate moves to calibration point if needed
          - Estimate moves to imaging location
          - Estimate moves to communication point
          - Add actions for calibration, imaging, communicating
    2) Sum all estimated actions
    3) Add penalty if multiple tasks assigned to same rover (simplified conflict)
    """

    def __init__(self, task):
        """Initialize by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract static information into useful data structures
        self.lander_location = None
        self.waypoint_connections = {}  # {rover: {from_wp: [to_wps]}}
        self.soil_locations = set()
        self.rock_locations = set()
        self.objective_visibility = {}  # {objective: [visible_waypoints]}
        self.camera_info = {}  # {camera: {'on': rover, 'supports': [modes], 'target': objective}}
        self.rover_capabilities = {}  # {rover: {'soil': bool, 'rock': bool, 'imaging': bool}}

        for fact in self.static:
            parts = get_parts(fact)
            
            if match(fact, "at_lander", "*", "*"):
                self.lander_location = parts[2]
                
            elif match(fact, "can_traverse", "*", "*", "*"):
                rover, wp1, wp2 = parts[1], parts[2], parts[3]
                if rover not in self.waypoint_connections:
                    self.waypoint_connections[rover] = {}
                if wp1 not in self.waypoint_connections[rover]:
                    self.waypoint_connections[rover][wp1] = []
                self.waypoint_connections[rover][wp1].append(wp2)
                
            elif match(fact, "at_soil_sample", "*"):
                self.soil_locations.add(parts[1])
                
            elif match(fact, "at_rock_sample", "*"):
                self.rock_locations.add(parts[1])
                
            elif match(fact, "visible_from", "*", "*"):
                obj, wp = parts[1], parts[2]
                if obj not in self.objective_visibility:
                    self.objective_visibility[obj] = []
                self.objective_visibility[obj].append(wp)
                
            elif match(fact, "on_board", "*", "*"):
                cam, rover = parts[1], parts[2]
                if cam not in self.camera_info:
                    self.camera_info[cam] = {}
                self.camera_info[cam]['on'] = rover
                
            elif match(fact, "supports", "*", "*"):
                cam, mode = parts[1], parts[2]
                if cam not in self.camera_info:
                    self.camera_info[cam] = {}
                if 'supports' not in self.camera_info[cam]:
                    self.camera_info[cam]['supports'] = []
                self.camera_info[cam]['supports'].append(mode)
                
            elif match(fact, "calibration_target", "*", "*"):
                cam, obj = parts[1], parts[2]
                if cam not in self.camera_info:
                    self.camera_info[cam] = {}
                self.camera_info[cam]['target'] = obj
                
            elif match(fact, "equipped_for_soil_analysis", "*"):
                rover = parts[1]
                if rover not in self.rover_capabilities:
                    self.rover_capabilities[rover] = {}
                self.rover_capabilities[rover]['soil'] = True
                
            elif match(fact, "equipped_for_rock_analysis", "*"):
                rover = parts[1]
                if rover not in self.rover_capabilities:
                    self.rover_capabilities[rover] = {}
                self.rover_capabilities[rover]['rock'] = True
                
            elif match(fact, "equipped_for_imaging", "*"):
                rover = parts[1]
                if rover not in self.rover_capabilities:
                    self.rover_capabilities[rover] = {}
                self.rover_capabilities[rover]['imaging'] = True

    def __call__(self, node):
        """Compute heuristic estimate for given state."""
        state = node.state
        
        # If all goals are satisfied, return 0
        if self.goals <= state:
            return 0
            
        total_cost = 0
        rover_positions = {}  # {rover: waypoint}
        calibrated_cameras = set()  # {(camera, rover)}
        stored_samples = {}  # {rover: {'soil': [waypoints], 'rock': [waypoints]}}
        captured_images = set()  # {(rover, objective, mode)}
        
        # Extract current state information
        for fact in state:
            parts = get_parts(fact)
            
            if match(fact, "at", "*", "*"):
                rover, wp = parts[1], parts[2]
                rover_positions[rover] = wp
                
            elif match(fact, "calibrated", "*", "*"):
                cam, rover = parts[1], parts[2]
                calibrated_cameras.add((cam, rover))
                
            elif match(fact, "have_soil_analysis", "*", "*"):
                rover, wp = parts[1], parts[2]
                if rover not in stored_samples:
                    stored_samples[rover] = {'soil': [], 'rock': []}
                stored_samples[rover]['soil'].append(wp)
                
            elif match(fact, "have_rock_analysis", "*", "*"):
                rover, wp = parts[1], parts[2]
                if rover not in stored_samples:
                    stored_samples[rover] = {'soil': [], 'rock': []}
                stored_samples[rover]['rock'].append(wp)
                
            elif match(fact, "have_image", "*", "*", "*"):
                rover, obj, mode = parts[1], parts[2], parts[3]
                captured_images.add((rover, obj, mode))
                
        # Process each unsatisfied goal
        for goal in self.goals:
            if goal in state:
                continue
                
            parts = get_parts(goal)
            
            # Soil data communication goal
            if match(goal, "communicated_soil_data", "*"):
                wp = parts[1]
                cost = self._estimate_soil_communication_cost(
                    wp, rover_positions, stored_samples, self.lander_location)
                total_cost += cost
                
            # Rock data communication goal
            elif match(goal, "communicated_rock_data", "*"):
                wp = parts[1]
                cost = self._estimate_rock_communication_cost(
                    wp, rover_positions, stored_samples, self.lander_location)
                total_cost += cost
                
            # Image data communication goal
            elif match(goal, "communicated_image_data", "*", "*"):
                obj, mode = parts[1], parts[2]
                cost = self._estimate_image_communication_cost(
                    obj, mode, rover_positions, captured_images, 
                    calibrated_cameras, self.lander_location)
                total_cost += cost
                
        return total_cost

    def _estimate_soil_communication_cost(self, target_wp, rover_positions, 
                                        stored_samples, lander_wp):
        """
        Estimate cost to communicate soil data from target_wp.
        Returns minimal cost across all capable rovers.
        """
        min_cost = float('inf')
        
        for rover, capabilities in self.rover_capabilities.items():
            if not capabilities.get('soil', False):
                continue
                
            # Case 1: Rover already has the sample
            if rover in stored_samples and target_wp in stored_samples[rover]['soil']:
                # Just need to communicate (move to visible wp + communicate)
                current_pos = rover_positions.get(rover, None)
                if current_pos:
                    move_cost = self._estimate_move_cost(rover, current_pos, lander_wp)
                    return move_cost + 1  # 1 for communicate action
                continue
                
            # Case 2: Need to sample and then communicate
            current_pos = rover_positions.get(rover, None)
            if not current_pos:
                continue
                
            # Move to sample location
            move1_cost = self._estimate_move_cost(rover, current_pos, target_wp)
            if move1_cost == float('inf'):
                continue
                
            # Sample action
            sample_cost = 1
            
            # Move to lander-visible waypoint
            move2_cost = self._estimate_move_cost(rover, target_wp, lander_wp)
            if move2_cost == float('inf'):
                continue
                
            # Communicate action
            communicate_cost = 1
            
            total_cost = move1_cost + sample_cost + move2_cost + communicate_cost
            if total_cost < min_cost:
                min_cost = total_cost
                
        return min_cost if min_cost != float('inf') else 1000  # Large penalty if impossible

    def _estimate_rock_communication_cost(self, target_wp, rover_positions, 
                                         stored_samples, lander_wp):
        """
        Estimate cost to communicate rock data from target_wp.
        Similar to soil but with rock capabilities.
        """
        min_cost = float('inf')
        
        for rover, capabilities in self.rover_capabilities.items():
            if not capabilities.get('rock', False):
                continue
                
            # Case 1: Rover already has the sample
            if rover in stored_samples and target_wp in stored_samples[rover]['rock']:
                current_pos = rover_positions.get(rover, None)
                if current_pos:
                    move_cost = self._estimate_move_cost(rover, current_pos, lander_wp)
                    return move_cost + 1
                continue
                
            # Case 2: Need to sample and then communicate
            current_pos = rover_positions.get(rover, None)
            if not current_pos:
                continue
                
            move1_cost = self._estimate_move_cost(rover, current_pos, target_wp)
            if move1_cost == float('inf'):
                continue
                
            sample_cost = 1
            move2_cost = self._estimate_move_cost(rover, target_wp, lander_wp)
            if move2_cost == float('inf'):
                continue
                
            communicate_cost = 1
            total_cost = move1_cost + sample_cost + move2_cost + communicate_cost
            if total_cost < min_cost:
                min_cost = total_cost
                
        return min_cost if min_cost != float('inf') else 1000

    def _estimate_image_communication_cost(self, target_obj, target_mode, 
                                          rover_positions, captured_images, 
                                          calibrated_cameras, lander_wp):
        """
        Estimate cost to communicate image data for target_obj in target_mode.
        """
        min_cost = float('inf')
        
        # Find cameras that support the required mode
        suitable_cameras = []
        for cam, info in self.camera_info.items():
            if target_mode in info.get('supports', []):
                suitable_cameras.append((cam, info['on'], info.get('target', None)))
        
        for cam, rover, calibration_obj in suitable_cameras:
            if not self.rover_capabilities.get(rover, {}).get('imaging', False):
                continue
                
            # Case 1: Image already captured
            if (rover, target_obj, target_mode) in captured_images:
                current_pos = rover_positions.get(rover, None)
                if current_pos:
                    move_cost = self._estimate_move_cost(rover, current_pos, lander_wp)
                    return move_cost + 1
                continue
                
            # Case 2: Need to capture image and then communicate
            current_pos = rover_positions.get(rover, None)
            if not current_pos:
                continue
                
            # Find visible waypoint for the objective
            visible_wps = self.objective_visibility.get(target_obj, [])
            if not visible_wps:
                continue
                
            # Pick first visible waypoint (simplified)
            image_wp = visible_wps[0]
            
            # Check if camera needs calibration
            calibration_needed = (cam, rover) not in calibrated_cameras
            calibration_cost = 0
            move_calibration_cost = 0
            
            if calibration_needed:
                if calibration_obj != target_obj:
                    # Need to calibrate on different objective
                    calib_wps = self.objective_visibility.get(calibration_obj, [])
                    if not calib_wps:
                        continue
                    calib_wp = calib_wps[0]
                    
                    # Move to calibration waypoint
                    move_calibration_cost = self._estimate_move_cost(rover, current_pos, calib_wp)
                    if move_calibration_cost == float('inf'):
                        continue
                        
                    # Calibrate action
                    calibration_cost = 1
                    
                    # Move from calibration to imaging waypoint
                    move1_cost = self._estimate_move_cost(rover, calib_wp, image_wp)
                else:
                    # Can calibrate at imaging waypoint
                    move1_cost = self._estimate_move_cost(rover, current_pos, image_wp)
                    calibration_cost = 1
            else:
                # No calibration needed
                move1_cost = self._estimate_move_cost(rover, current_pos, image_wp)
            
            if move1_cost == float('inf'):
                continue
                
            # Take image action
            image_cost = 1
            
            # Move to lander-visible waypoint
            move2_cost = self._estimate_move_cost(rover, image_wp, lander_wp)
            if move2_cost == float('inf'):
                continue
                
            # Communicate action
            communicate_cost = 1
            
            total_cost = (move_calibration_cost + calibration_cost + 
                          move1_cost + image_cost + move2_cost + communicate_cost)
            if total_cost < min_cost:
                min_cost = total_cost
                
        return min_cost if min_cost != float('inf') else 1000

    def _estimate_move_cost(self, rover, from_wp, to_wp):
        """
        Estimate number of navigate actions needed to move between waypoints.
        Uses simplified assumption of 1 action per hop (no pathfinding).
        """
        if from_wp == to_wp:
            return 0
            
        if rover not in self.waypoint_connections:
            return float('inf')
            
        # Check if direct connection exists
        if from_wp in self.waypoint_connections[rover]:
            if to_wp in self.waypoint_connections[rover][from_wp]:
                return 1
                
        # Simplified: assume at most 2 hops (real implementation could use BFS)
        # This is a placeholder - a real implementation would do proper pathfinding
        for intermediate in self.waypoint_connections[rover].get(from_wp, []):
            if to_wp in self.waypoint_connections[rover].get(intermediate, []):
                return 2
                
        return float('inf')  # No path found
