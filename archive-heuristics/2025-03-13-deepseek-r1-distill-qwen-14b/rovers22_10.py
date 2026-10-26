from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions needed to:
    1. Collect and communicate soil and rock samples.
    2. Calibrate cameras and take images of objectives in different modes.
    3. Communicate the collected data back to the lander.

    # Assumptions:
    - Each rover can carry one sample at a time.
    - Communication actions require visibility between the rover and the lander.
    - Calibration is required before taking images.
    - Multiple rovers can work in parallel.

    # Heuristic Initialization
    - Extracts goal conditions to identify which data needs to be communicated.
    - Parses static facts to identify available samples, objectives, and calibration targets.

    # Step-by-Step Thinking for Computing Heuristic
    1. Identify the number of soil, rock, and image data points that need communication.
    2. For each soil/rock sample:
       - If not already communicated, estimate 3 actions: navigate to waypoint, sample, and communicate.
    3. For each image:
       - If not already taken, estimate 4 actions: calibrate, take image, and communicate.
       - If calibration is already done, estimate 3 actions: take image, communicate.
    4. Sum the actions for all required data points.
    5. If multiple rovers are available, reduce the total actions by the number of rovers (as they can work in parallel).
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract goal locations for each waypoint and objective
        self.goal_waypoints = set()
        self.goal_objectives = set()
        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == "communicated_soil_data":
                self.goal_waypoints.add(parts[1])
            elif parts[0] == "communicated_rock_data":
                self.goal_waypoints.add(parts[1])
            elif parts[0] == "communicated_image_data":
                self.goal_objectives.add(parts[1])

        # Parse static facts
        self.waypoints_with_soil = set()
        self.waypoints_with_rock = set()
        self.waypoints_with_samples = set()
        self.objectives = set()
        self.calibration_targets = set()
        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] == "at_soil_sample":
                self.waypoints_with_soil.add(parts[1])
            elif parts[0] == "at_rock_sample":
                self.waypoints_with_rock.add(parts[1])
            elif parts[0] == "at_soil_sample":
                self.waypoints_with_samples.add(parts[1])
            elif parts[0] == "at_rock_sample":
                self.waypoints_with_samples.add(parts[1])
            elif parts[0] == "calibration_target":
                self.calibration_targets.add(parts[2])
            elif parts[0] == "objective":
                self.objectives.add(parts[1])

        # Count available rovers
        self.num_rovers = sum(1 for fact in task.objects if fact == " rover")

    def __call__(self, node):
        """Estimate the minimum number of actions to reach the goal state."""
        state = node.state

        def get_parts(fact):
            """Extract components of a PDDL fact."""
            return fact[1:-1].split()

        def match(fact, *args):
            """Check if a fact matches a given pattern."""
            parts = get_parts(fact)
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        # Initialize action counters
        soil_needed = 0
        rock_needed = 0
        image_needed = 0

        # Track communicated data
        communicated_soil = set()
        communicated_rock = set()
        communicated_images = set()

        # Track taken images
        taken_images = set()

        # Track calibration status
        calibrated_cameras = set()

        # Track current samples
        current_samples = set()

        # Process state
        for fact in state:
            if match(fact, "communicated_soil_data", "*"):
                communicated_soil.add(fact.split()[1])
            elif match(fact, "communicated_rock_data", "*"):
                communicated_rock.add(fact.split()[1])
            elif match(fact, "communicated_image_data", "*", "*", "*"):
                obj = fact.split()[1]
                mode = fact.split()[2]
                communicated_images.add((obj, mode))
            elif match(fact, "have_image", "*", "*", "*"):
                obj = fact.split()[1]
                mode = fact.split()[2]
                taken_images.add((obj, mode))
            elif match(fact, "calibrated", "*", "*"):
                calibrated_cameras.add(fact.split()[1])
            elif match(fact, "have_soil_analysis", "*", "*"):
                current_samples.add((fact.split()[1], "soil", fact.split()[2]))
            elif match(fact, "have_rock_analysis", "*", "*"):
                current_samples.add((fact.split()[1], "rock", fact.split()[2]))

        # Calculate required actions for soil and rock
        for wp in self.goal_waypoints:
            if wp in self.waypoints_with_soil and wp not in communicated_soil:
                soil_needed += 1
            if wp in self.waypoints_with_rock and wp not in communicated_rock:
                rock_needed += 1

        # Calculate required actions for images
        for obj in self.goal_objectives:
            for mode in ["high_res", "low_res", "colour"]:
                if (obj, mode) not in communicated_images:
                    image_needed += 1

        # Estimate actions for soil and rock communication
        total_actions = 0
        total_actions += soil_needed * 3
        total_actions += rock_needed * 3

        # Estimate actions for images
        for img in range(image_needed):
            # Calibration needed if not already done
            if f"calibrated camera{img % 2 + 1}" not in calibrated_cameras:
                total_actions += 1
            # Taking the image
            total_actions += 1
            # Communication
            total_actions += 1

        # Adjust for parallel actions if multiple rovers are available
        if self.num_rovers > 1:
            total_actions = (total_actions // self.num_rovers) + (total_actions % self.num_rovers)

        return total_actions
