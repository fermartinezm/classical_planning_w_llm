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
    This heuristic estimates the number of actions required to achieve all communication goals
    (soil data, rock data, and image data) by considering:
    - The distance rovers need to travel to sample soil/rock or take images
    - The need to calibrate cameras before taking images
    - The need to communicate data back to the lander

    # Assumptions:
    - Each rover can carry only one sample at a time (due to single store)
    - Communication requires being at a waypoint visible to the lander
    - Soil/rock samples must be collected before communication
    - Images require calibration before capture

    # Heuristic Initialization
    - Extract goal conditions (what needs to be communicated)
    - Extract static information about:
        - Rover capabilities (equipped_for_*)
        - Waypoint visibility and traversal
        - Camera support and calibration targets
        - Objective visibility

    # Step-By-Step Thinking for Computing Heuristic
    1. For each communication goal (soil, rock, image):
        a. If already communicated, skip (0 cost)
        b. Otherwise:
            i. Find which rover can achieve this goal
            ii. Estimate actions needed:
                - For soil/rock: navigate to sample location + sample + navigate to communication point + communicate
                - For images: navigate to calibration point + calibrate + navigate to imaging point + take image + navigate to communication point + communicate
    2. Consider parallel actions by different rovers
    3. Sum the most critical path (longest sequence of dependent actions)
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract static information
        self.rover_capabilities = {}
        self.waypoint_visibility = set()
        self.can_traverse = set()
        self.camera_info = {}
        self.objective_visibility = {}
        self.lander_location = None

        for fact in self.static:
            parts = get_parts(fact)
            if match(fact, "equipped_for_*", "*"):
                capability, rover = parts
                if rover not in self.rover_capabilities:
                    self.rover_capabilities[rover] = set()
                self.rover_capabilities[rover].add(capability)
            elif match(fact, "visible", "*", "*"):
                self.waypoint_visibility.add((parts[1], parts[2]))
            elif match(fact, "can_traverse", "*", "*", "*"):
                self.can_traverse.add((parts[1], parts[2], parts[3]))
            elif match(fact, "supports", "*", "*"):
                camera, mode = parts[1], parts[2]
                if camera not in self.camera_info:
                    self.camera_info[camera] = {'supports': set(), 'on_board': None, 'target': None}
                self.camera_info[camera]['supports'].add(mode)
            elif match(fact, "on_board", "*", "*"):
                camera, rover = parts[1], parts[2]
                if camera not in self.camera_info:
                    self.camera_info[camera] = {'supports': set(), 'on_board': None, 'target': None}
                self.camera_info[camera]['on_board'] = rover
            elif match(fact, "calibration_target", "*", "*"):
                camera, objective = parts[1], parts[2]
                if camera not in self.camera_info:
                    self.camera_info[camera] = {'supports': set(), 'on_board': None, 'target': None}
                self.camera_info[camera]['target'] = objective
            elif match(fact, "visible_from", "*", "*"):
                objective, waypoint = parts[1], parts[2]
                if objective not in self.objective_visibility:
                    self.objective_visibility[objective] = set()
                self.objective_visibility[objective].add(waypoint)
            elif match(fact, "at_lander", "*", "*"):
                self.lander_location = parts[2]

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal from the current state."""
        state = node.state
        total_cost = 0

        # Check which goals are already satisfied
        unsatisfied_goals = self.goals - state

        # If all goals are satisfied, return 0
        if not unsatisfied_goals:
            return 0

        # Process each type of goal
        for goal in unsatisfied_goals:
            parts = get_parts(goal)
            if match(goal, "communicated_soil_data", "*"):
                waypoint = parts[1]
                cost = self._estimate_soil_communication_cost(state, waypoint)
            elif match(goal, "communicated_rock_data", "*"):
                waypoint = parts[1]
                cost = self._estimate_rock_communication_cost(state, waypoint)
            elif match(goal, "communicated_image_data", "*", "*"):
                objective, mode = parts[1], parts[2]
                cost = self._estimate_image_communication_cost(state, objective, mode)
            else:
                continue  # Unknown goal type
            
            total_cost += cost

        return total_cost

    def _estimate_soil_communication_cost(self, state, waypoint):
        """Estimate actions needed to communicate soil data from given waypoint."""
        # Check if already have soil analysis
        for fact in state:
            if match(fact, "have_soil_analysis", "*", waypoint):
                rover = get_parts(fact)[1]
                return self._estimate_communication_cost(state, rover, waypoint)
        
        # Need to collect sample first
        min_cost = float('inf')
        for rover in self.rover_capabilities:
            if "equipped_for_soil_analysis" not in self.rover_capabilities[rover]:
                continue
            
            # Find rover's current location
            rover_loc = None
            for fact in state:
                if match(fact, "at", rover, "*"):
                    rover_loc = get_parts(fact)[2]
                    break
            
            if not rover_loc:
                continue
            
            # Estimate navigation to sample location + sample + navigate to communication point + communicate
            nav1_cost = 1  # Simplified: assume 1 action to reach sample
            sample_cost = 1
            nav2_cost = 1  # Simplified: assume 1 action to reach communication point
            comm_cost = 1
            
            total = nav1_cost + sample_cost + nav2_cost + comm_cost
            if total < min_cost:
                min_cost = total
        
        return min_cost if min_cost != float('inf') else 10  # Fallback cost

    def _estimate_rock_communication_cost(self, state, waypoint):
        """Estimate actions needed to communicate rock data from given waypoint."""
        # Check if already have rock analysis
        for fact in state:
            if match(fact, "have_rock_analysis", "*", waypoint):
                rover = get_parts(fact)[1]
                return self._estimate_communication_cost(state, rover, waypoint)
        
        # Need to collect sample first
        min_cost = float('inf')
        for rover in self.rover_capabilities:
            if "equipped_for_rock_analysis" not in self.rover_capabilities[rover]:
                continue
            
            # Find rover's current location
            rover_loc = None
            for fact in state:
                if match(fact, "at", rover, "*"):
                    rover_loc = get_parts(fact)[2]
                    break
            
            if not rover_loc:
                continue
            
            # Estimate navigation to sample location + sample + navigate to communication point + communicate
            nav1_cost = 1  # Simplified: assume 1 action to reach sample
            sample_cost = 1
            nav2_cost = 1  # Simplified: assume 1 action to reach communication point
            comm_cost = 1
            
            total = nav1_cost + sample_cost + nav2_cost + comm_cost
            if total < min_cost:
                min_cost = total
        
        return min_cost if min_cost != float('inf') else 10  # Fallback cost

    def _estimate_image_communication_cost(self, state, objective, mode):
        """Estimate actions needed to communicate image data for given objective and mode."""
        # Check if already have image
        for fact in state:
            if match(fact, "have_image", "*", objective, mode):
                rover = get_parts(fact)[1]
                return self._estimate_communication_cost(state, rover, objective)
        
        # Need to take image first
        min_cost = float('inf')
        for camera in self.camera_info:
            if mode not in self.camera_info[camera]['supports']:
                continue
            
            rover = self.camera_info[camera]['on_board']
            if not rover:
                continue
            
            # Find rover's current location
            rover_loc = None
            for fact in state:
                if match(fact, "at", rover, "*"):
                    rover_loc = get_parts(fact)[2]
                    break
            
            if not rover_loc:
                continue
            
            # Check if camera is calibrated
            calibrated = False
            for fact in state:
                if match(fact, "calibrated", camera, rover):
                    calibrated = True
                    break
            
            # Estimate actions needed:
            # 1. Navigate to calibration point (if not calibrated)
            # 2. Calibrate (if not calibrated)
            # 3. Navigate to imaging point
            # 4. Take image
            # 5. Navigate to communication point
            # 6. Communicate
            
            calib_cost = 0
            if not calibrated:
                calib_cost = 2  # 1 navigate + 1 calibrate
            
            nav1_cost = 1  # Simplified: assume 1 action to reach imaging point
            image_cost = 1
            nav2_cost = 1  # Simplified: assume 1 action to reach communication point
            comm_cost = 1
            
            total = calib_cost + nav1_cost + image_cost + nav2_cost + comm_cost
            if total < min_cost:
                min_cost = total
        
        return min_cost if min_cost != float('inf') else 10  # Fallback cost

    def _estimate_communication_cost(self, state, rover, target):
        """Estimate actions needed to communicate data from rover to lander."""
        # Check if rover is already at a point visible to lander
        rover_loc = None
        for fact in state:
            if match(fact, "at", rover, "*"):
                rover_loc = get_parts(fact)[2]
                break
        
        if not rover_loc:
            return 2  # Fallback: navigate + communicate
        
        # Check if current location is visible to lander
        if (rover_loc, self.lander_location) in self.waypoint_visibility:
            return 1  # Just communicate
        
        # Need to navigate to visible point
        return 2  # 1 navigate + 1 communicate
