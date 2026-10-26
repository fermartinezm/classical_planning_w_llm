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
    - The need to communicate data back to the lander

    # Assumptions:
    - Each rover can carry only one sample at a time (due to single store)
    - Communication requires being at a waypoint visible to the lander
    - Soil/rock samples must be collected before communication
    - Images require camera calibration before capture

    # Heuristic Initialization
    - Extract goal conditions (what needs to be communicated)
    - Extract static information about:
      - Rover capabilities (equipment)
      - Waypoint connectivity (can_traverse)
      - Camera-objective relationships
      - Lander position

    # Step-By-Step Thinking for Computing Heuristic
    1. For each communication goal (soil, rock, image):
       a. If already communicated, skip
       b. Otherwise:
          i. For soil/rock:
             - Find closest rover with appropriate equipment
             - Estimate distance to sample location
             - Add actions for sampling and communicating
          ii. For images:
             - Find rover with appropriate camera
             - Estimate distance to calibration waypoint
             - Estimate distance to imaging waypoint
             - Add actions for calibration, imaging, and communicating
    2. Sum all estimated actions across all goals
    3. Add any necessary drop actions if rovers need to free their stores
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract lander position
        self.lander_pos = None
        for fact in self.static:
            if match(fact, "at_lander", "*", "*"):
                _, lander, pos = get_parts(fact)
                self.lander_pos = pos
                break

        # Build waypoint graph for pathfinding
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
                capability, rover = get_parts(fact)
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

        # Extract camera modes
        self.camera_modes = {}
        for fact in self.static:
            if match(fact, "supports", "*", "*"):
                _, camera, mode = get_parts(fact)
                if camera not in self.camera_modes:
                    self.camera_modes[camera] = set()
                self.camera_modes[camera].add(mode)

        # Extract visible_from relationships
        self.visible_from = {}
        for fact in self.static:
            if match(fact, "visible_from", "*", "*"):
                _, objective, waypoint = get_parts(fact)
                if objective not in self.visible_from:
                    self.visible_from[objective] = set()
                self.visible_from[objective].add(waypoint)

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
                total_cost += self._estimate_soil_communication_cost(state, waypoint)
            elif parts[0] == "communicated_rock_data":
                waypoint = parts[1]
                total_cost += self._estimate_rock_communication_cost(state, waypoint)
            elif parts[0] == "communicated_image_data":
                objective, mode = parts[1], parts[2]
                total_cost += self._estimate_image_communication_cost(state, objective, mode)

        return total_cost

    def _estimate_soil_communication_cost(self, state, waypoint):
        """Estimate cost to communicate soil data from given waypoint."""
        cost = 0

        # Check if we already have the analysis
        has_analysis = any(
            match(fact, "have_soil_analysis", "*", waypoint) for fact in state
        )

        if has_analysis:
            # Just need to communicate
            cost += self._estimate_communication_cost(state)
        else:
            # Need to sample and communicate
            cost += 2  # sample_soil + communicate_soil_data
            cost += self._estimate_navigation_cost(state, waypoint)

        return cost

    def _estimate_rock_communication_cost(self, state, waypoint):
        """Estimate cost to communicate rock data from given waypoint."""
        cost = 0

        # Check if we already have the analysis
        has_analysis = any(
            match(fact, "have_rock_analysis", "*", waypoint) for fact in state
        )

        if has_analysis:
            # Just need to communicate
            cost += self._estimate_communication_cost(state)
        else:
            # Need to sample and communicate
            cost += 2  # sample_rock + communicate_rock_data
            cost += self._estimate_navigation_cost(state, waypoint)

        return cost

    def _estimate_image_communication_cost(self, state, objective, mode):
        """Estimate cost to communicate image data for given objective and mode."""
        cost = 0

        # Check if we already have the image
        has_image = any(
            match(fact, "have_image", "*", objective, mode) for fact in state
        )

        if has_image:
            # Just need to communicate
            cost += self._estimate_communication_cost(state)
        else:
            # Need to calibrate, take image, and communicate
            cost += 3  # calibrate + take_image + communicate_image_data
            
            # Find a waypoint where we can take the image
            imaging_waypoints = self.visible_from.get(objective, set())
            if imaging_waypoints:
                cost += self._estimate_navigation_cost(state, next(iter(imaging_waypoints)))

        return cost

    def _estimate_communication_cost(self, state):
        """Estimate cost to communicate data back to lander."""
        # Simple estimate: 1 action if we're at a visible waypoint, else 2 (move + communicate)
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, rover, waypoint = get_parts(fact)
                # Check if lander is visible from this waypoint
                if any(match(f, "visible", waypoint, self.lander_pos) for f in self.static):
                    return 1
        return 2

    def _estimate_navigation_cost(self, state, target_waypoint):
        """Estimate navigation cost to reach target waypoint."""
        # Simple estimate: 1 if we're at adjacent waypoint, else 2
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, rover, current_waypoint = get_parts(fact)
                if current_waypoint == target_waypoint:
                    return 0
                if target_waypoint in self.waypoint_graph.get(current_waypoint, set()):
                    return 1
        return 2
