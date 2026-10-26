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
    - The need to communicate data to the lander
    - The current state of rovers (position, equipped capabilities, stored samples)

    # Assumptions:
    - Each rover can only carry one sample at a time (due to single store)
    - Communication requires being at a waypoint visible to the lander
    - Soil/rock samples can only be collected once
    - Images can be retaken if needed (but calibration is required each time)

    # Heuristic Initialization
    - Extract goal conditions (what needs to be communicated)
    - Extract static information about:
        - Rover capabilities
        - Waypoint visibility
        - Camera support and calibration targets
        - Sample locations
        - Lander position

    # Step-By-Step Thinking for Computing Heuristic
    1. For each communication goal (soil, rock, image):
        a. If already communicated, skip (0 cost)
        b. Otherwise:
            i. For soil/rock:
                - Find closest rover with appropriate equipment
                - Calculate distance to sample location
                - Add cost for sampling and communicating
            ii. For images:
                - Find rover with appropriate camera
                - Calculate distance to calibration waypoint
                - Calculate distance to imaging waypoint
                - Add cost for calibration, imaging, and communicating
    2. For each rover:
        - If carrying a sample, add cost to communicate it
        - If calibrated camera, add cost to use it before losing calibration
    3. Sum all estimated action costs
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract static information
        self.lander_positions = {}
        self.rover_capabilities = {}
        self.waypoint_visibility = set()
        self.can_traverse = set()
        self.camera_info = {}
        self.sample_locations = {'soil': set(), 'rock': set()}
        self.objective_visibility = {}

        for fact in self.static:
            if match(fact, "at_lander", "*", "*"):
                parts = get_parts(fact)
                self.lander_positions[parts[1]] = parts[2]
            elif match(fact, "equipped_for_*", "*"):
                parts = get_parts(fact)
                capability = parts[0].split('_')[-1]
                rover = parts[1]
                if rover not in self.rover_capabilities:
                    self.rover_capabilities[rover] = set()
                self.rover_capabilities[rover].add(capability)
            elif match(fact, "visible", "*", "*"):
                self.waypoint_visibility.add((get_parts(fact)[1], get_parts(fact)[2]))
            elif match(fact, "can_traverse", "*", "*", "*"):
                self.can_traverse.add((get_parts(fact)[1], get_parts(fact)[2], get_parts(fact)[3]))
            elif match(fact, "on_board", "*", "*"):
                parts = get_parts(fact)
                camera, rover = parts[1], parts[2]
                if rover not in self.camera_info:
                    self.camera_info[rover] = []
                self.camera_info[rover].append(camera)
            elif match(fact, "supports", "*", "*"):
                parts = get_parts(fact)
                camera, mode = parts[1], parts[2]
                for rover in self.camera_info:
                    if camera in self.camera_info[rover]:
                        if 'modes' not in self.camera_info[rover]:
                            self.camera_info[rover]['modes'] = set()
                        self.camera_info[rover]['modes'].add((camera, mode))
            elif match(fact, "calibration_target", "*", "*"):
                parts = get_parts(fact)
                camera, objective = parts[1], parts[2]
                for rover in self.camera_info:
                    if camera in self.camera_info[rover]:
                        self.camera_info[rover]['target'] = (camera, objective)
            elif match(fact, "visible_from", "*", "*"):
                parts = get_parts(fact)
                objective, waypoint = parts[1], parts[2]
                if objective not in self.objective_visibility:
                    self.objective_visibility[objective] = set()
                self.objective_visibility[objective].add(waypoint)
            elif match(fact, "at_soil_sample", "*"):
                self.sample_locations['soil'].add(get_parts(fact)[1])
            elif match(fact, "at_rock_sample", "*"):
                self.sample_locations['rock'].add(get_parts(fact)[1])

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Check which goals are already satisfied
        unsatisfied_soil = set()
        unsatisfied_rock = set()
        unsatisfied_images = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if match(goal, "communicated_soil_data", "*"):
                if goal not in state:
                    unsatisfied_soil.add(parts[1])
            elif match(goal, "communicated_rock_data", "*"):
                if goal not in state:
                    unsatisfied_rock.add(parts[1])
            elif match(goal, "communicated_image_data", "*", "*"):
                if goal not in state:
                    unsatisfied_images.add((parts[1], parts[2]))

        # Extract current rover states
        rover_positions = {}
        rover_samples = {'soil': {}, 'rock': {}}
        rover_images = {}
        calibrated_cameras = set()
        full_stores = set()

        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                if parts[1].startswith('rover'):
                    rover_positions[parts[1]] = parts[2]
            elif match(fact, "have_soil_analysis", "*", "*"):
                parts = get_parts(fact)
                rover_samples['soil'][parts[2]] = parts[1]
            elif match(fact, "have_rock_analysis", "*", "*"):
                parts = get_parts(fact)
                rover_samples['rock'][parts[2]] = parts[1]
            elif match(fact, "have_image", "*", "*", "*"):
                parts = get_parts(fact)
                rover_images[(parts[2], parts[3])] = parts[1]
            elif match(fact, "calibrated", "*", "*"):
                parts = get_parts(fact)
                calibrated_cameras.add((parts[1], parts[2]))
            elif match(fact, "full", "*"):
                full_stores.add(get_parts(fact)[1])

        # Estimate cost for unsatisfied soil samples
        for waypoint in unsatisfied_soil:
            # Check if a rover already has this sample
            if waypoint in rover_samples['soil']:
                rover = rover_samples['soil'][waypoint]
                # Need to communicate it
                total_cost += 1  # communicate action
            else:
                # Find closest rover with soil analysis capability
                min_cost = float('inf')
                for rover, pos in rover_positions.items():
                    if 'soil' in self.rover_capabilities.get(rover, set()):
                        # Estimate distance (number of navigate actions)
                        distance = self.estimate_distance(pos, waypoint)
                        cost = distance + 1  # sample action
                        # Check if rover has empty store
                        if any(match(f, "store_of", "*", rover) and f"(empty {get_parts(f)[1]})" in state for f in self.static):
                            cost += 1  # communicate action
                        else:
                            cost += 2  # drop and communicate actions
                        if cost < min_cost:
                            min_cost = cost
                if min_cost != float('inf'):
                    total_cost += min_cost

        # Estimate cost for unsatisfied rock samples (similar to soil)
        for waypoint in unsatisfied_rock:
            if waypoint in rover_samples['rock']:
                rover = rover_samples['rock'][waypoint]
                total_cost += 1  # communicate action
            else:
                min_cost = float('inf')
                for rover, pos in rover_positions.items():
                    if 'rock' in self.rover_capabilities.get(rover, set()):
                        distance = self.estimate_distance(pos, waypoint)
                        cost = distance + 1  # sample action
                        if any(match(f, "store_of", "*", rover) and f"(empty {get_parts(f)[1]})" in state for f in self.static):
                            cost += 1  # communicate action
                        else:
                            cost += 2  # drop and communicate actions
                        if cost < min_cost:
                            min_cost = cost
                if min_cost != float('inf'):
                    total_cost += min_cost

        # Estimate cost for unsatisfied images
        for (objective, mode) in unsatisfied_images:
            # Check if a rover already has this image
            if (objective, mode) in rover_images:
                rover = rover_images[(objective, mode)]
                # Need to communicate it
                total_cost += 1  # communicate action
            else:
                # Find rover with appropriate camera
                min_cost = float('inf')
                for rover, pos in rover_positions.items():
                    if 'imaging' in self.rover_capabilities.get(rover, set()):
                        if rover in self.camera_info and 'modes' in self.camera_info[rover]:
                            # Check if rover has camera supporting this mode
                            for (camera, supported_mode) in self.camera_info[rover]['modes']:
                                if supported_mode == mode:
                                    # Find calibration target
                                    if 'target' in self.camera_info[rover]:
                                        target_camera, target_obj = self.camera_info[rover]['target']
                                        if target_obj == objective:
                                            # Find visible waypoint for calibration
                                            cal_waypoint = None
                                            for wp in self.objective_visibility.get(objective, set()):
                                                if (pos, wp) in self.waypoint_visibility:
                                                    cal_waypoint = wp
                                                    break
                                            if cal_waypoint:
                                                # Find visible waypoint for imaging
                                                img_waypoint = None
                                                for wp in self.objective_visibility.get(objective, set()):
                                                    if (pos, wp) in self.waypoint_visibility:
                                                        img_waypoint = wp
                                                        break
                                                if img_waypoint:
                                                    # Calculate costs
                                                    cal_distance = self.estimate_distance(pos, cal_waypoint)
                                                    img_distance = self.estimate_distance(cal_waypoint, img_waypoint)
                                                    cost = cal_distance + 1  # calibrate
                                                    cost += img_distance + 1  # take_image
                                                    cost += 1  # communicate
                                                    if cost < min_cost:
                                                        min_cost = cost
                if min_cost != float('inf'):
                    total_cost += min_cost

        # Add cost for rovers carrying samples that need to be communicated
        for waypoint, rover in rover_samples['soil'].items():
            if f"(communicated_soil_data {waypoint})" not in self.goals or f"(communicated_soil_data {waypoint})" not in state:
                total_cost += 1  # communicate action

        for waypoint, rover in rover_samples['rock'].items():
            if f"(communicated_rock_data {waypoint})" not in self.goals or f"(communicated_rock_data {waypoint})" not in state:
                total_cost += 1  # communicate action

        # Add cost for calibrated cameras that should be used
        for (camera, rover) in calibrated_cameras:
            if rover in rover_positions:
                # Check if there are any image goals left that this camera could help with
                for (objective, mode) in unsatisfied_images:
                    if rover in self.camera_info and 'modes' in self.camera_info[rover]:
                        if (camera, mode) in self.camera_info[rover]['modes']:
                            if 'target' in self.camera_info[rover] and self.camera_info[rover]['target'][1] == objective:
                                total_cost += 1  # take_image action
                                break

        return total_cost

    def estimate_distance(self, from_waypoint, to_waypoint):
        """Estimate the number of navigate actions needed between two waypoints."""
        # Simple BFS to find shortest path
        if from_waypoint == to_waypoint:
            return 0
            
        visited = set()
        queue = [(from_waypoint, 0)]
        
        while queue:
            current, dist = queue.pop(0)
            if current == to_waypoint:
                return dist
            if current in visited:
                continue
            visited.add(current)
            
            # Find all directly reachable waypoints
            for (rover, wp1, wp2) in self.can_traverse:
                if wp1 == current and (wp1, wp2) in self.waypoint_visibility:
                    queue.append((wp2, dist + 1))
                elif wp2 == current and (wp2, wp1) in self.waypoint_visibility:
                    queue.append((wp1, dist + 1))
        
        # If no path found, return a large number (but finite)
        return 10
