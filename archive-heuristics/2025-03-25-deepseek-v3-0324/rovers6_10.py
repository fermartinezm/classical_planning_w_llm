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
    1) Collecting required samples (soil/rock)
    2) Taking required images
    3) Communicating all data to the lander
    It considers the current state of rovers (location, equipped instruments, stored samples)
    and the remaining tasks needed to achieve the goals.

    # Assumptions:
    - Each rover can perform only one task at a time (sample, image, communicate).
    - The most efficient path is used for estimation (may not be admissible).
    - Communication requires being at a waypoint visible to the lander.
    - Imaging requires calibration and being at a waypoint visible to the objective.

    # Heuristic Initialization
    - Extract goal conditions (communicated data requirements).
    - Extract static information about rover capabilities, waypoint visibility, etc.
    - Build data structures to efficiently access this information.

    # Step-By-Step Thinking for Computing Heuristic
    1) For each uncommunicated soil/rock data:
       - Find if any rover already has the sample (have_soil/rock_analysis)
       - If not, find if the sample is still available at a waypoint
       - Estimate actions needed to collect and communicate it
    2) For each uncommunicated image data:
       - Find if any rover already has the image (have_image)
       - If not, estimate actions needed to calibrate, take image, and communicate it
    3) For each required action, estimate navigation steps between waypoints
    4) Sum all estimated actions to get the total heuristic value
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract all waypoints, rovers, objectives from static facts
        self.waypoints = set()
        self.rovers = set()
        self.objectives = set()
        self.landers = set()
        
        # Mapping from rover to its store
        self.rover_stores = {}
        
        # Rover capabilities
        self.equipped_for_soil = set()
        self.equipped_for_rock = set()
        self.equipped_for_imaging = set()
        
        # Waypoint visibility graph
        self.visible = set()
        
        # Rover traversal capabilities
        self.can_traverse = set()
        
        # Camera information
        self.cameras = set()
        self.on_board = {}
        self.supports = {}
        self.calibration_targets = {}
        
        # Objective visibility
        self.visible_from = set()
        
        # Landers positions
        self.lander_positions = {}

        for fact in self.static:
            parts = get_parts(fact)
            if match(fact, "waypoint", "*"):
                self.waypoints.add(parts[1])
            elif match(fact, "rover", "*"):
                self.rovers.add(parts[1])
            elif match(fact, "objective", "*"):
                self.objectives.add(parts[1])
            elif match(fact, "lander", "*"):
                self.landers.add(parts[1])
            elif match(fact, "store_of", "*", "*"):
                self.rover_stores[parts[2]] = parts[1]
            elif match(fact, "equipped_for_soil_analysis", "*"):
                self.equipped_for_soil.add(parts[1])
            elif match(fact, "equipped_for_rock_analysis", "*"):
                self.equipped_for_rock.add(parts[1])
            elif match(fact, "equipped_for_imaging", "*"):
                self.equipped_for_imaging.add(parts[1])
            elif match(fact, "visible", "*", "*"):
                self.visible.add((parts[1], parts[2]))
            elif match(fact, "can_traverse", "*", "*", "*"):
                self.can_traverse.add((parts[1], parts[2], parts[3]))
            elif match(fact, "camera", "*"):
                self.cameras.add(parts[1])
            elif match(fact, "on_board", "*", "*"):
                self.on_board[(parts[1], parts[2])] = True
            elif match(fact, "supports", "*", "*"):
                if parts[1] not in self.supports:
                    self.supports[parts[1]] = set()
                self.supports[parts[1]].add(parts[2])
            elif match(fact, "calibration_target", "*", "*"):
                self.calibration_targets[parts[1]] = parts[2]
            elif match(fact, "visible_from", "*", "*"):
                self.visible_from.add((parts[1], parts[2]))
            elif match(fact, "at_lander", "*", "*"):
                self.lander_positions[parts[1]] = parts[2]

        # Extract goal requirements
        self.required_soil_data = set()
        self.required_rock_data = set()
        self.required_image_data = set()
        
        for goal in self.goals:
            parts = get_parts(goal)
            if match(goal, "communicated_soil_data", "*"):
                self.required_soil_data.add(parts[1])
            elif match(goal, "communicated_rock_data", "*"):
                self.required_rock_data.add(parts[1])
            elif match(goal, "communicated_image_data", "*", "*"):
                self.required_image_data.add((parts[1], parts[2]))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0
        
        # Check which goals are already satisfied
        satisfied_soil = set()
        satisfied_rock = set()
        satisfied_images = set()
        
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "communicated_soil_data", "*"):
                satisfied_soil.add(parts[1])
            elif match(fact, "communicated_rock_data", "*"):
                satisfied_rock.add(parts[1])
            elif match(fact, "communicated_image_data", "*", "*"):
                satisfied_images.add((parts[1], parts[2]))
        
        # Soil data collection and communication
        for wp in self.required_soil_data - satisfied_soil:
            # Check if any rover already has this sample
            has_sample = False
            for rover in self.rovers:
                if f"(have_soil_analysis {rover} {wp})" in state:
                    has_sample = True
                    # Estimate communication cost
                    total_cost += self._estimate_communication_cost(rover, wp, state)
                    break
            
            if not has_sample:
                # Check if sample is still available
                if f"(at_soil_sample {wp})" in state:
                    # Find best rover to collect it
                    best_cost = float('inf')
                    for rover in self.rovers:
                        if rover in self.equipped_for_soil:
                            # Estimate collection + communication cost
                            cost = self._estimate_sample_collection_cost(rover, wp, "soil", state)
                            cost += self._estimate_communication_cost(rover, wp, state)
                            if cost < best_cost:
                                best_cost = cost
                    if best_cost != float('inf'):
                        total_cost += best_cost
        
        # Rock data collection and communication
        for wp in self.required_rock_data - satisfied_rock:
            # Check if any rover already has this sample
            has_sample = False
            for rover in self.rovers:
                if f"(have_rock_analysis {rover} {wp})" in state:
                    has_sample = True
                    # Estimate communication cost
                    total_cost += self._estimate_communication_cost(rover, wp, state)
                    break
            
            if not has_sample:
                # Check if sample is still available
                if f"(at_rock_sample {wp})" in state:
                    # Find best rover to collect it
                    best_cost = float('inf')
                    for rover in self.rovers:
                        if rover in self.equipped_for_rock:
                            # Estimate collection + communication cost
                            cost = self._estimate_sample_collection_cost(rover, wp, "rock", state)
                            cost += self._estimate_communication_cost(rover, wp, state)
                            if cost < best_cost:
                                best_cost = cost
                    if best_cost != float('inf'):
                        total_cost += best_cost
        
        # Image data collection and communication
        for obj, mode in self.required_image_data - satisfied_images:
            # Check if any rover already has this image
            has_image = False
            for rover in self.rovers:
                if f"(have_image {rover} {obj} {mode})" in state:
                    has_image = True
                    # Estimate communication cost
                    total_cost += self._estimate_image_communication_cost(rover, obj, mode, state)
                    break
            
            if not has_image:
                # Find best rover to take the image
                best_cost = float('inf')
                for rover in self.rovers:
                    if rover in self.equipped_for_imaging:
                        # Find suitable camera
                        for camera in self.cameras:
                            if (camera, rover) in self.on_board and mode in self.supports.get(camera, set()):
                                # Estimate imaging + communication cost
                                cost = self._estimate_imaging_cost(rover, camera, obj, mode, state)
                                cost += self._estimate_image_communication_cost(rover, obj, mode, state)
                                if cost < best_cost:
                                    best_cost = cost
                if best_cost != float('inf'):
                    total_cost += best_cost
        
        return total_cost

    def _estimate_navigation_cost(self, rover, from_wp, to_wp):
        """Estimate the number of navigate actions needed to move between waypoints."""
        # Simple estimation: 1 action per navigation (ignoring path planning)
        return 1 if from_wp != to_wp else 0

    def _estimate_sample_collection_cost(self, rover, wp, sample_type, state):
        """Estimate actions needed to collect a sample (soil or rock)."""
        cost = 0
        
        # Find rover's current position
        rover_pos = None
        for fact in state:
            if match(fact, "at", rover, "*"):
                rover_pos = get_parts(fact)[2]
                break
        
        if rover_pos:
            # Navigate to sample waypoint
            cost += self._estimate_navigation_cost(rover, rover_pos, wp)
            # Sample action
            cost += 1
            # Check if store is empty
            store = self.rover_stores.get(rover)
            if store and f"(empty {store})" not in state:
                # Need to drop sample first
                cost += 1
        
        return cost

    def _estimate_communication_cost(self, rover, wp, state):
        """Estimate actions needed to communicate soil/rock data."""
        cost = 0
        
        # Find rover's current position (after sample collection)
        rover_pos = None
        for fact in state:
            if match(fact, "at", rover, "*"):
                rover_pos = get_parts(fact)[2]
                break
        
        if rover_pos:
            # Find lander position
            lander_pos = next(iter(self.lander_positions.values()), None)
            if lander_pos:
                # Check if we can communicate from current position
                if (rover_pos, lander_pos) not in self.visible:
                    # Need to navigate to a waypoint visible to lander
                    cost += 1  # Simplified estimation
                # Communication action
                cost += 1
        
        return cost

    def _estimate_imaging_cost(self, rover, camera, obj, mode, state):
        """Estimate actions needed to take an image."""
        cost = 0
        
        # Find rover's current position
        rover_pos = None
        for fact in state:
            if match(fact, "at", rover, "*"):
                rover_pos = get_parts(fact)[2]
                break
        
        if rover_pos:
            # Check if camera is calibrated
            if f"(calibrated {camera} {rover})" not in state:
                # Need to calibrate first
                target = self.calibration_targets.get(camera)
                if target:
                    # Find waypoint visible to target
                    visible_wp = None
                    for (o, w) in self.visible_from:
                        if o == target:
                            visible_wp = w
                            break
                    if visible_wp:
                        # Navigate to calibration waypoint
                        cost += self._estimate_navigation_cost(rover, rover_pos, visible_wp)
                        # Calibrate action
                        cost += 1
            
            # Find waypoint visible to objective
            imaging_wp = None
            for (o, w) in self.visible_from:
                if o == obj:
                    imaging_wp = w
                    break
            
            if imaging_wp:
                # Navigate to imaging waypoint (if not already there)
                cost += self._estimate_navigation_cost(rover, rover_pos, imaging_wp)
                # Take image action
                cost += 1
        
        return cost

    def _estimate_image_communication_cost(self, rover, obj, mode, state):
        """Estimate actions needed to communicate image data."""
        cost = 0
        
        # Find rover's current position (after imaging)
        rover_pos = None
        for fact in state:
            if match(fact, "at", rover, "*"):
                rover_pos = get_parts(fact)[2]
                break
        
        if rover_pos:
            # Find lander position
            lander_pos = next(iter(self.lander_positions.values()), None)
            if lander_pos:
                # Check if we can communicate from current position
                if (rover_pos, lander_pos) not in self.visible:
                    # Need to navigate to a waypoint visible to lander
                    cost += 1  # Simplified estimation
                # Communication action
                cost += 1
        
        return cost
