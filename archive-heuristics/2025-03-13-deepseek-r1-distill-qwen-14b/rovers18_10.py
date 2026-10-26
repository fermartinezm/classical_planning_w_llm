from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions needed to communicate all required data points.

    # Assumptions:
    - Each rover can handle multiple tasks but may need to switch between them.
    - Navigation between visible waypoints takes one action.
    - Calibration of a camera for an objective is a one-time requirement.
    - Communicating data takes one action per data point.

    # Heuristic Initialization
    - Extracts goal conditions and static facts from the task.
    - Maps waypoints with samples and objectives' visibility.

    # Step-by-Step Thinking for Computing Heuristic
    1. For each soil and rock sample:
       - If not communicated, add actions to collect and communicate.
    2. For each image:
       - If not communicated, add actions to calibrate, take, and communicate.
    3. Sum all required actions across all data points.
    """

    def __init__(self, task):
        """Initialize the heuristic with task information."""
        self.goals = task.goals
        static_facts = task.static

        # Extract static information into useful data structures
        self.waypoint_samples = {}
        self.visible_from = {}
        self.calibration_targets = {}
        self.camera_equipment = {}
        self.rover_equipment = {}

        # Populate waypoint samples
        for fact in static_facts:
            if match(fact, "at_soil_sample", "*"):
                w = get_parts(fact)[1]
                self.waypoint_samples[w] = "soil"
            elif match(fact, "at_rock_sample", "*"):
                w = get_parts(fact)[1]
                self.waypoint_samples[w] = "rock"

        # Populate visible_from information
        for fact in static_facts:
            if match(fact, "visible_from", "*", "*"):
                obj, w = get_parts(fact)[1], get_parts(fact)[2]
                if w not in self.visible_from:
                    self.visible_from[w] = []
                self.visible_from[w].append(obj)

        # Populate calibration targets
        for fact in static_facts:
            if match(fact, "calibration_target", "*", "*"):
                cam, obj = get_parts(fact)[1], get_parts(fact)[2]
                self.calibration_targets[cam] = obj

        # Populate camera equipment on rovers
        for fact in static_facts:
            if match(fact, "equipped_for_imaging", "*"):
                rover = get_parts(fact)[1]
                if rover not in self.camera_equipment:
                    self.camera_equipment[rover] = []
            if match(fact, "on_board", "*", "*"):
                cam, rover = get_parts(fact)[1], get_parts(fact)[2]
                if rover not in self.camera_equipment:
                    self.camera_equipment[rover] = []
                self.camera_equipment[rover].append(cam)

    def __call__(self, node):
        """Compute the heuristic value for the current state."""
        state = node.state
        static = node.static

        def get_parts(fact):
            """Extract components of a PDDL fact."""
            return fact[1:-1].split()

        def match(fact, *args):
            """Check if a fact matches a given pattern."""
            parts = get_parts(fact)
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        # Count required actions
        total_actions = 0

        # Check for soil and rock communication
        for fact in self.goals:
            if match(fact, "communicated_soil_data", "*"):
                w = get_parts(fact)[1]
                if f"(at_soil_sample {w})" in state and f"(have_soil_analysis * {w})" not in state:
                    total_actions += 2  # Sample and communicate
            elif match(fact, "communicated_rock_data", "*"):
                w = get_parts(fact)[1]
                if f"(at_rock_sample {w})" in state and f"(have_rock_analysis * {w})" not in state:
                    total_actions += 2  # Sample and communicate

        # Check for image communication
        for fact in self.goals:
            if match(fact, "communicated_image_data", "*", "*"):
                obj, m = get_parts(fact)[1], get_parts(fact)[2]
                # Check if image is already taken
                image_taken = any(f.match(f"(have_image * {obj} {m})") for f in state)
                if not image_taken:
                    # Find a rover equipped for imaging and camera
                    rover = None
                    for r in self.camera_equipment:
                        if any(cam for cam in self.camera_equipment[r] if self.calibration_targets[cam] == obj):
                            rover = r
                            break
                    if rover:
                        # Navigation to waypoint visible from objective
                        visible_ways = [w for w in self.visible_from.get(obj, []) if f"(at {rover} {w})" in state]
                        if not visible_ways:
                            total_actions += 2  # Move to visible waypoint and calibrate
                        # Calibration and image taking
                        total_actions += 3  # Calibrate, take image, communicate
                    else:
                        total_actions += 5  # Assume worst case if no rover is found

        return total_actions
