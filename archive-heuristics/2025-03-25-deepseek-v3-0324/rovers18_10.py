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
    This heuristic estimates the number of actions needed to achieve all goals by:
    1) Counting unsatisfied communication goals (soil, rock, image data)
    2) Estimating the steps needed to collect samples/take images
    3) Estimating the movement steps between waypoints
    4) Considering equipment requirements and calibration needs

    # Assumptions:
    - Each rover can handle only one task at a time (sample, image, communicate)
    - The most efficient path is used for movement estimates
    - Calibration is done once per camera when needed
    - Stores are emptied after communication (simplifying assumption)

    # Heuristic Initialization
    - Extract goal conditions (what needs to be communicated)
    - Extract static information about:
        - Rover capabilities (equipment)
        - Waypoint connectivity (can_traverse)
        - Camera capabilities and calibration targets
        - Sample locations and objective visibility

    # Step-By-Step Thinking for Computing Heuristic
    1) For each unsatisfied communication goal:
        a) If it's soil data:
            - Find closest rover with soil analysis capability
            - Estimate steps to reach sample waypoint
            - Add steps for sampling and communicating
        b) If it's rock data:
            - Find closest rover with rock analysis capability
            - Estimate steps to reach sample waypoint
            - Add steps for sampling and communicating
        c) If it's image data:
            - Find rover with appropriate camera
            - Estimate steps to calibrate (if needed)
            - Estimate steps to reach visible waypoint
            - Add steps for taking image and communicating
    2) Sum all estimated actions
    3) Add penalty for rovers that need to switch tasks (e.g., from sampling to imaging)
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static
        
        # Extract rover capabilities
        self.rover_capabilities = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "equipped_for_soil_analysis":
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, set()).add("soil")
            elif parts[0] == "equipped_for_rock_analysis":
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, set()).add("rock")
            elif parts[0] == "equipped_for_imaging":
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, set()).add("imaging")
        
        # Extract store information
        self.rover_stores = {}
        for fact in self.static:
            if match(fact, "store_of", "*", "*"):
                store, rover = get_parts(fact)[1:]
                self.rover_stores[rover] = store
        
        # Extract waypoint connectivity
        self.traversal_graph = {}
        for fact in self.static:
            if match(fact, "can_traverse", "*", "*", "*"):
                rover, wp1, wp2 = get_parts(fact)[1:]
                self.traversal_graph.setdefault(rover, {}).setdefault(wp1, set()).add(wp2)
                self.traversal_graph.setdefault(rover, {}).setdefault(wp2, set()).add(wp1)
        
        # Extract camera information
        self.camera_info = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "on_board":
                camera, rover = parts[1:]
                self.camera_info.setdefault(camera, {})["rover"] = rover
            elif parts[0] == "supports":
                camera, mode = parts[1:]
                self.camera_info.setdefault(camera, {}).setdefault("modes", set()).add(mode)
            elif parts[0] == "calibration_target":
                camera, objective = parts[1:]
                self.camera_info.setdefault(camera, {})["target"] = objective
        
        # Extract sample locations (initial state)
        self.initial_soil_samples = set()
        self.initial_rock_samples = set()
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts[0] == "at_soil_sample":
                self.initial_soil_samples.add(parts[1])
            elif parts[0] == "at_rock_sample":
                self.initial_rock_samples.add(parts[1])
        
        # Extract objective visibility
        self.objective_visibility = {}
        for fact in self.static:
            if match(fact, "visible_from", "*", "*"):
                obj, wp = get_parts(fact)[1:]
                self.objective_visibility.setdefault(obj, set()).add(wp)
        
        # Extract lander location
        self.lander_location = None
        for fact in self.static:
            if match(fact, "at_lander", "*", "*"):
                self.lander_location = get_parts(fact)[2]

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
            if parts[0] == "communicated_soil_data":
                wp = parts[1]
                cost = self._estimate_soil_communication_cost(state, wp)
                total_cost += cost
            
            # Rock data communication goal
            elif parts[0] == "communicated_rock_data":
                wp = parts[1]
                cost = self._estimate_rock_communication_cost(state, wp)
                total_cost += cost
            
            # Image data communication goal
            elif parts[0] == "communicated_image_data":
                obj, mode = parts[1:]
                cost = self._estimate_image_communication_cost(state, obj, mode)
                total_cost += cost
        
        return total_cost

    def _estimate_soil_communication_cost(self, state, waypoint):
        """
        Estimate cost to communicate soil data from given waypoint.
        Steps:
        1) Find rover with soil capability and empty store
        2) Estimate movement to sample waypoint
        3) Sample soil (1 action)
        4) Move to lander-visible waypoint (if not already there)
        5) Communicate (1 action)
        """
        min_cost = float('inf')
        
        # Find suitable rovers
        for rover, capabilities in self.rover_capabilities.items():
            if "soil" not in capabilities:
                continue
                
            # Check if rover has empty store
            store = self.rover_stores.get(rover)
            if not store or f"(empty {store})" not in state:
                continue
                
            # Find rover's current location
            rover_loc = None
            for fact in state:
                if match(fact, "at", rover, "*"):
                    rover_loc = get_parts(fact)[2]
                    break
            if not rover_loc:
                continue
                
            # Estimate movement cost to sample waypoint
            movement_cost = self._estimate_movement_cost(rover, rover_loc, waypoint)
            if movement_cost is None:
                continue
                
            # Estimate movement cost from sample waypoint to lander-visible waypoint
            comm_cost = 1  # communicate action
            if waypoint != rover_loc:  # need to move back if not already at lander-visible waypoint
                lander_visible_cost = self._estimate_movement_cost(rover, waypoint, self.lander_location)
                if lander_visible_cost is None:
                    continue
                movement_cost += lander_visible_cost
                
            total_cost = movement_cost + 1 + comm_cost  # move + sample + communicate
            if total_cost < min_cost:
                min_cost = total_cost
                
        return min_cost if min_cost != float('inf') else 1000  # large penalty if no path found

    def _estimate_rock_communication_cost(self, state, waypoint):
        """
        Estimate cost to communicate rock data from given waypoint.
        Similar to soil communication but for rock samples.
        """
        min_cost = float('inf')
        
        for rover, capabilities in self.rover_capabilities.items():
            if "rock" not in capabilities:
                continue
                
            store = self.rover_stores.get(rover)
            if not store or f"(empty {store})" not in state:
                continue
                
            rover_loc = None
            for fact in state:
                if match(fact, "at", rover, "*"):
                    rover_loc = get_parts(fact)[2]
                    break
            if not rover_loc:
                continue
                
            movement_cost = self._estimate_movement_cost(rover, rover_loc, waypoint)
            if movement_cost is None:
                continue
                
            comm_cost = 1
            if waypoint != rover_loc:
                lander_visible_cost = self._estimate_movement_cost(rover, waypoint, self.lander_location)
                if lander_visible_cost is None:
                    continue
                movement_cost += lander_visible_cost
                
            total_cost = movement_cost + 1 + comm_cost
            if total_cost < min_cost:
                min_cost = total_cost
                
        return min_cost if min_cost != float('inf') else 1000

    def _estimate_image_communication_cost(self, state, objective, mode):
        """
        Estimate cost to communicate image data for given objective and mode.
        Steps:
        1) Find rover with camera supporting the mode
        2) Estimate calibration cost if needed
        3) Estimate movement to visible waypoint
        4) Take image (1 action)
        5) Move to lander-visible waypoint
        6) Communicate (1 action)
        """
        min_cost = float('inf')
        
        # Find suitable cameras
        for camera, info in self.camera_info.items():
            if "modes" not in info or mode not in info["modes"]:
                continue
                
            rover = info.get("rover")
            if not rover:
                continue
                
            # Check if rover is equipped for imaging
            if "imaging" not in self.rover_capabilities.get(rover, set()):
                continue
                
            # Find rover's current location
            rover_loc = None
            for fact in state:
                if match(fact, "at", rover, "*"):
                    rover_loc = get_parts(fact)[2]
                    break
            if not rover_loc:
                continue
                
            # Check calibration status
            calibration_cost = 0
            if f"(calibrated {camera} {rover})" not in state:
                # Need to calibrate - find calibration target waypoint
                target_obj = info.get("target")
                if not target_obj:
                    continue
                    
                visible_wps = self.objective_visibility.get(target_obj, set())
                if not visible_wps:
                    continue
                    
                # Find closest waypoint where we can calibrate
                min_calib_movement = float('inf')
                for wp in visible_wps:
                    cost = self._estimate_movement_cost(rover, rover_loc, wp)
                    if cost is not None and cost < min_calib_movement:
                        min_calib_movement = cost
                        
                if min_calib_movement == float('inf'):
                    continue
                    
                calibration_cost = min_calib_movement + 1  # move + calibrate
                rover_loc = list(visible_wps)[0]  # update assumed location after calibration
                
            # Find visible waypoint for the target objective
            target_wps = self.objective_visibility.get(objective, set())
            if not target_wps:
                continue
                
            # Find closest waypoint where we can take the image
            min_image_movement = float('inf')
            best_wp = None
            for wp in target_wps:
                cost = self._estimate_movement_cost(rover, rover_loc, wp)
                if cost is not None and cost < min_image_movement:
                    min_image_movement = cost
                    best_wp = wp
                    
            if min_image_movement == float('inf'):
                continue
                
            # Estimate movement to lander-visible waypoint
            comm_cost = 1
            if best_wp != rover_loc:
                lander_visible_cost = self._estimate_movement_cost(rover, best_wp, self.lander_location)
                if lander_visible_cost is None:
                    continue
                min_image_movement += lander_visible_cost
                
            total_cost = calibration_cost + min_image_movement + 1 + comm_cost  # calib + move + image + communicate
            if total_cost < min_cost:
                min_cost = total_cost
                
        return min_cost if min_cost != float('inf') else 1000

    def _estimate_movement_cost(self, rover, start, end):
        """
        Estimate movement cost between two waypoints using BFS.
        Returns number of steps or None if no path exists.
        """
        if start == end:
            return 0
            
        if rover not in self.traversal_graph:
            return None
            
        visited = set()
        queue = [(start, 0)]
        
        while queue:
            current, steps = queue.pop(0)
            if current == end:
                return steps
                
            if current in visited:
                continue
            visited.add(current)
            
            for neighbor in self.traversal_graph[rover].get(current, set()):
                queue.append((neighbor, steps + 1))
                
        return None
