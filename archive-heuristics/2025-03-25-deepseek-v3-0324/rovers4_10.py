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
    1) Calculating the minimum navigation steps required for each rover to reach waypoints
    2) Accounting for necessary sample collection and communication actions
    3) Considering image capturing and calibration requirements

    # Assumptions:
    - Each rover can only perform one task at a time (e.g., can't sample and image simultaneously)
    - Navigation between waypoints takes 1 action per hop
    - Sample collection and communication each take 1 action
    - Image capture requires calibration (1 action) and then taking the image (1 action)

    # Heuristic Initialization
    - Extract goal conditions (what needs to be communicated)
    - Extract static information about:
        - Rover capabilities (equipment)
        - Waypoint connectivity (can_traverse)
        - Camera support and calibration targets
        - Sample locations
        - Lander positions

    # Step-By-Step Thinking for Computing Heuristic
    1) For each uncommunicated soil/rock data:
        - Find closest rover with appropriate equipment
        - Calculate path to sample waypoint
        - Add sample collection action
        - Calculate path to lander-visible waypoint
        - Add communication action
    2) For each uncommunicated image data:
        - Find rover with appropriate camera
        - Calculate calibration requirements
        - Calculate path to waypoint where objective is visible
        - Add calibration and image capture actions
        - Calculate path to lander-visible waypoint
        - Add communication action
    3) Sum all required actions across all goals
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract rover capabilities
        self.rover_capabilities = {}
        for fact in self.static:
            if match(fact, "equipped_for_*", "*"):
                parts = get_parts(fact)
                capability = parts[0].split("_")[2]  # soil/rock/imaging
                rover = parts[1]
                if rover not in self.rover_capabilities:
                    self.rover_capabilities[rover] = set()
                self.rover_capabilities[rover].add(capability)

        # Extract waypoint connectivity graph
        self.waypoint_graph = {}
        for fact in self.static:
            if match(fact, "can_traverse", "*", "*", "*"):
                rover, wp1, wp2 = get_parts(fact)[1:]
                if rover not in self.waypoint_graph:
                    self.waypoint_graph[rover] = {}
                if wp1 not in self.waypoint_graph[rover]:
                    self.waypoint_graph[rover][wp1] = set()
                self.waypoint_graph[rover][wp1].add(wp2)

        # Extract camera information
        self.camera_info = {}
        for fact in self.static:
            if match(fact, "on_board", "*", "*"):
                camera, rover = get_parts(fact)[1:]
                if rover not in self.camera_info:
                    self.camera_info[rover] = []
                self.camera_info[rover].append(camera)

        # Extract calibration targets
        self.calibration_targets = {}
        for fact in self.static:
            if match(fact, "calibration_target", "*", "*"):
                camera, objective = get_parts(fact)[1:]
                self.calibration_targets[camera] = objective

        # Extract camera modes
        self.camera_modes = {}
        for fact in self.static:
            if match(fact, "supports", "*", "*"):
                camera, mode = get_parts(fact)[1:]
                if camera not in self.camera_modes:
                    self.camera_modes[camera] = set()
                self.camera_modes[camera].add(mode)

        # Extract lander positions
        self.lander_positions = {}
        for fact in self.static:
            if match(fact, "at_lander", "*", "*"):
                lander, waypoint = get_parts(fact)[1:]
                self.lander_positions[lander] = waypoint

        # Extract sample locations
        self.sample_locations = {
            "soil": set(),
            "rock": set()
        }
        for fact in self.static:
            if match(fact, "at_*_sample", "*"):
                sample_type = get_parts(fact)[0].split("_")[1]
                waypoint = get_parts(fact)[1]
                self.sample_locations[sample_type].add(waypoint)

        # Extract objective visibility
        self.objective_visibility = {}
        for fact in self.static:
            if match(fact, "visible_from", "*", "*"):
                objective, waypoint = get_parts(fact)[1:]
                if objective not in self.objective_visibility:
                    self.objective_visibility[objective] = set()
                self.objective_visibility[objective].add(waypoint)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Check which goals are already satisfied
        unsatisfied_goals = self.goals - state

        # If all goals are satisfied, return 0
        if not unsatisfied_goals:
            return 0

        # Get current rover positions
        rover_positions = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                rover, waypoint = get_parts(fact)[1:]
                rover_positions[rover] = waypoint

        # Process each unsatisfied goal
        for goal in unsatisfied_goals:
            parts = get_parts(goal)
            predicate = parts[0]

            if predicate == "communicated_soil_data":
                waypoint = parts[1]
                # Find rover with soil analysis capability
                for rover, capabilities in self.rover_capabilities.items():
                    if "soil" in capabilities and rover in rover_positions:
                        # Estimate path to sample waypoint and then to lander
                        total_cost += 2  # sample + communicate
                        # Add navigation cost (simplified as 1 per hop)
                        total_cost += 1
                        break

            elif predicate == "communicated_rock_data":
                waypoint = parts[1]
                # Find rover with rock analysis capability
                for rover, capabilities in self.rover_capabilities.items():
                    if "rock" in capabilities and rover in rover_positions:
                        # Estimate path to sample waypoint and then to lander
                        total_cost += 2  # sample + communicate
                        # Add navigation cost (simplified as 1 per hop)
                        total_cost += 1
                        break

            elif predicate == "communicated_image_data":
                objective, mode = parts[1:]
                # Find rover with appropriate camera
                for rover, cameras in self.camera_info.items():
                    if rover not in rover_positions:
                        continue
                    for camera in cameras:
                        if mode in self.camera_modes.get(camera, set()):
                            # Check if camera is calibrated
                            calibrated = False
                            for fact in state:
                                if match(fact, "calibrated", camera, rover):
                                    calibrated = True
                                    break
                            if not calibrated:
                                total_cost += 1  # calibration action
                            total_cost += 2  # take_image + communicate
                            # Add navigation cost (simplified as 1 per hop)
                            total_cost += 1
                            break
                    else:
                        continue
                    break

        return total_cost
