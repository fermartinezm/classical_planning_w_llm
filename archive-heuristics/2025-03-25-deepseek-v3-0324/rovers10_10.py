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
    This heuristic estimates the number of actions required to achieve all goal conditions
    by considering the following tasks:
    - Collecting soil/rock samples
    - Taking images of objectives
    - Communicating data to the lander
    - Navigating between waypoints

    # Assumptions:
    - Each rover can carry only one sample at a time (due to single store).
    - Communication requires the rover to be at a waypoint visible from the lander.
    - Calibration is required before taking images.
    - The heuristic assumes optimal path planning between waypoints (shortest path).

    # Heuristic Initialization
    - Extract goal conditions (communicated data).
    - Extract static information about:
        - Waypoint connectivity (can_traverse, visible)
        - Rover capabilities (equipped_for_*)
        - Camera and objective relationships (calibration_target, supports)
        - Lander location (at_lander)

    # Step-By-Step Thinking for Computing Heuristic
    1. For each uncommunicated soil/rock data:
        - Find the rover that can collect it (equipped_for_*).
        - Estimate the cost to navigate to the sample location, collect it, and communicate it.
    2. For each uncommunicated image data:
        - Find the rover with a suitable camera (supports mode).
        - Estimate the cost to calibrate, navigate to a visible waypoint, take the image, and communicate it.
    3. Sum all estimated action costs:
        - Navigation: 1 per waypoint in the shortest path.
        - Sample collection: 1 (sample_soil/sample_rock).
        - Communication: 1 (communicate_*).
        - Calibration: 1 (calibrate).
        - Imaging: 1 (take_image).
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract lander location
        self.lander_location = None
        for fact in self.static:
            if match(fact, "at_lander", "*", "*"):
                _, lander, location = get_parts(fact)
                self.lander_location = location
                break

        # Build waypoint graph (adjacency list)
        self.waypoint_graph = {}
        for fact in self.static:
            if match(fact, "can_traverse", "*", "*", "*"):
                _, rover, wp1, wp2 = get_parts(fact)
                if wp1 not in self.waypoint_graph:
                    self.waypoint_graph[wp1] = set()
                if wp2 not in self.waypoint_graph:
                    self.waypoint_graph[wp2] = set()
                self.waypoint_graph[wp1].add(wp2)
                self.waypoint_graph[wp2].add(wp1)

        # Extract rover capabilities
        self.rover_capabilities = {}
        for fact in self.static:
            if match(fact, "equipped_for_*", "*"):
                predicate, rover = get_parts(fact)
                capability = predicate.split("_")[2]  # soil/rock/imaging
                if rover not in self.rover_capabilities:
                    self.rover_capabilities[rover] = set()
                self.rover_capabilities[rover].add(capability)

        # Extract camera information
        self.camera_info = {}
        for fact in self.static:
            if match(fact, "on_board", "*", "*"):
                _, camera, rover = get_parts(fact)
                if rover not in self.camera_info:
                    self.camera_info[rover] = []
                self.camera_info[rover].append(camera)

        # Extract calibration targets
        self.calibration_targets = {}
        for fact in self.static:
            if match(fact, "calibration_target", "*", "*"):
                _, camera, objective = get_parts(fact)
                self.calibration_targets[camera] = objective

        # Extract supported modes
        self.supported_modes = {}
        for fact in self.static:
            if match(fact, "supports", "*", "*"):
                _, camera, mode = get_parts(fact)
                if camera not in self.supported_modes:
                    self.supported_modes[camera] = set()
                self.supported_modes[camera].add(mode)

        # Extract visible_from relationships
        self.visible_from = {}
        for fact in self.static:
            if match(fact, "visible_from", "*", "*"):
                _, objective, waypoint = get_parts(fact)
                if objective not in self.visible_from:
                    self.visible_from[objective] = set()
                self.visible_from[objective].add(waypoint)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Check which goals are already satisfied
        unsatisfied_goals = self.goals - state

        # If all goals are satisfied, return 0
        if not unsatisfied_goals:
            return 0

        # Process soil data goals
        for goal in unsatisfied_goals:
            if match(goal, "communicated_soil_data", "*"):
                _, waypoint = get_parts(goal)
                # Find a rover that can collect soil samples
                for rover in self.rover_capabilities:
                    if "soil" in self.rover_capabilities[rover]:
                        # Estimate cost to collect and communicate
                        total_cost += self._estimate_soil_cost(rover, waypoint, state)
                        break

        # Process rock data goals
        for goal in unsatisfied_goals:
            if match(goal, "communicated_rock_data", "*"):
                _, waypoint = get_parts(goal)
                # Find a rover that can collect rock samples
                for rover in self.rover_capabilities:
                    if "rock" in self.rover_capabilities[rover]:
                        # Estimate cost to collect and communicate
                        total_cost += self._estimate_rock_cost(rover, waypoint, state)
                        break

        # Process image data goals
        for goal in unsatisfied_goals:
            if match(goal, "communicated_image_data", "*", "*"):
                _, objective, mode = get_parts(goal)
                # Find a rover with a suitable camera
                for rover in self.camera_info:
                    for camera in self.camera_info[rover]:
                        if mode in self.supported_modes.get(camera, set()):
                            # Estimate cost to take and communicate image
                            total_cost += self._estimate_image_cost(
                                rover, camera, objective, mode, state
                            )
                            break

        return total_cost

    def _estimate_soil_cost(self, rover, waypoint, state):
        """Estimate cost to collect and communicate soil sample."""
        cost = 0

        # Check if already have the sample
        have_sample = False
        for fact in state:
            if match(fact, "have_soil_analysis", rover, waypoint):
                have_sample = True
                break

        if not have_sample:
            # Need to navigate to waypoint and collect sample
            cost += self._estimate_navigation_cost(rover, waypoint, state)
            cost += 1  # sample_soil action

        # Communicate the data
        cost += self._estimate_communication_cost(rover, waypoint, state)

        return cost

    def _estimate_rock_cost(self, rover, waypoint, state):
        """Estimate cost to collect and communicate rock sample."""
        cost = 0

        # Check if already have the sample
        have_sample = False
        for fact in state:
            if match(fact, "have_rock_analysis", rover, waypoint):
                have_sample = True
                break

        if not have_sample:
            # Need to navigate to waypoint and collect sample
            cost += self._estimate_navigation_cost(rover, waypoint, state)
            cost += 1  # sample_rock action

        # Communicate the data
        cost += self._estimate_communication_cost(rover, waypoint, state)

        return cost

    def _estimate_image_cost(self, rover, camera, objective, mode, state):
        """Estimate cost to take and communicate image."""
        cost = 0

        # Check if already have the image
        have_image = False
        for fact in state:
            if match(fact, "have_image", rover, objective, mode):
                have_image = True
                break

        if not have_image:
            # Need to calibrate if not already done
            calibrated = False
            for fact in state:
                if match(fact, "calibrated", camera, rover):
                    calibrated = True
                    break

            if not calibrated:
                cost += 1  # calibrate action
                # Need to navigate to calibration waypoint
                target = self.calibration_targets.get(camera, None)
                if target:
                    visible_waypoints = self.visible_from.get(target, set())
                    if visible_waypoints:
                        # Find the closest visible waypoint
                        min_cost = float("inf")
                        for wp in visible_waypoints:
                            nav_cost = self._estimate_navigation_cost(rover, wp, state)
                            if nav_cost < min_cost:
                                min_cost = nav_cost
                        cost += min_cost

            # Need to navigate to waypoint visible from objective
            visible_waypoints = self.visible_from.get(objective, set())
            if visible_waypoints:
                # Find the closest visible waypoint
                min_cost = float("inf")
                for wp in visible_waypoints:
                    nav_cost = self._estimate_navigation_cost(rover, wp, state)
                    if nav_cost < min_cost:
                        min_cost = nav_cost
                cost += min_cost

            cost += 1  # take_image action

        # Communicate the image
        cost += self._estimate_communication_cost(rover, None, state)

        return cost

    def _estimate_communication_cost(self, rover, waypoint, state):
        """Estimate cost to communicate data to lander."""
        cost = 0

        # Find rover's current location
        rover_location = None
        for fact in state:
            if match(fact, "at", rover, "*"):
                _, _, location = get_parts(fact)
                rover_location = location
                break

        if rover_location and self.lander_location:
            # Check if already at a waypoint visible from lander
            visible = False
            for fact in self.static:
                if match(fact, "visible", rover_location, self.lander_location):
                    visible = True
                    break

            if not visible:
                # Need to navigate to a waypoint visible from lander
                # Find all waypoints visible from lander
                visible_waypoints = set()
                for fact in self.static:
                    if match(fact, "visible", "*", self.lander_location):
                        _, wp, _ = get_parts(fact)
                        visible_waypoints.add(wp)

                # Find the closest visible waypoint
                min_cost = float("inf")
                for wp in visible_waypoints:
                    nav_cost = self._estimate_navigation_cost(rover, wp, state)
                    if nav_cost < min_cost:
                        min_cost = nav_cost
                cost += min_cost

        cost += 1  # communicate action
        return cost

    def _estimate_navigation_cost(self, rover, target_waypoint, state):
        """Estimate navigation cost between current location and target waypoint."""
        # Find rover's current location
        current_location = None
        for fact in state:
            if match(fact, "at", rover, "*"):
                _, _, location = get_parts(fact)
                current_location = location
                break

        if not current_location or current_location == target_waypoint:
            return 0

        # Simple heuristic: assume direct path exists (minimum 1 action)
        # In a more sophisticated version, we could implement BFS here
        return 1
