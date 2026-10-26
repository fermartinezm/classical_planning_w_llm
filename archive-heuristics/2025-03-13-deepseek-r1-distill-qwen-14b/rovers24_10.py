from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goal by:
    1. Communicating soil and rock analysis data from waypoints.
    2. Capturing and communicating image data from objectives.
    3. Calculating the minimal steps required for each type of data.

    # Assumptions:
    - Each rover can carry one sample at a time.
    - Communication requires visibility between the rover's location and the lander.
    - Image capture requires calibration and specific modes.

    # Heuristic Initialization
    - Extract static facts including navigation capabilities, calibration targets, and visibility.
    - Track required data points from the goal.

    # Step-by-Step Thinking for Computing Heuristic
    1. **Extract Goal Requirements**: Identify which data points (soil, rock, image) need to be communicated.
    2. **Evaluate Current State**: Determine the status of each data point and the rover's positions.
    3. **Calculate Required Actions**:
       - For soil and rock: Navigation to sample, sampling, and communication.
       - For images: Calibration, image capture, and communication.
    4. **Sum Costs**: Aggregate the costs for all required actions to estimate the total steps needed.
    """

    def __init__(self, task):
        """Initialize the heuristic with static facts and goal conditions."""
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        # Extract can_traverse information for navigation
        self.can_traverse = {}
        for fact in static_facts:
            if fnmatch(fact, '(can_traverse * * *)'):
                rover, from_w, to_w = fact[1:-1].split()
                if rover not in self.can_traverse:
                    self.can_traverse[rover] = set()
                self.can_traverse[rover].add((from_w, to_w))

        # Extract calibration targets
        self.calibration_targets = {}
        for fact in static_facts:
            if fnmatch(fact, '(calibration_target * *)'):
                camera, obj = fact[1:-1].split()
                self.calibration_targets[camera] = obj

        # Extract visible_from information
        self.visible_from = {}
        for fact in static_facts:
            if fnmatch(fact, '(visible_from * *)'):
                obj, w = fact[1:-1].split()
                self.visible_from[obj] = w

    def __call__(self, node):
        """Compute the heuristic value for the current state."""
        state = node.state
        goal_met = self.task.goal_reached(state)
        if goal_met:
            return 0

        def get_parts(fact):
            """Extract components of a PDDL fact."""
            return fact[1:-1].split()

        def match(fact, pattern):
            """Check if a fact matches a given pattern."""
            parts = get_parts(fact)
            return all(fnmatch(part, p) for part, p in zip(parts, pattern))

        soil_required = set()
        rock_required = set()
        image_required = set()

        # Identify which data needs to be communicated
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'communicated_soil_data':
                soil_required.add(parts[1])
            elif parts[0] == 'communicated_rock_data':
                rock_required.add(parts[1])
            elif parts[0] == 'communicated_image_data':
                image_required.add((parts[1], parts[2], parts[3]))

        total_cost = 0

        # Track rover positions and sample status
        rover_pos = {}
        for fact in state:
            if match(fact, '(at * *)'):
                rover, w = get_parts(fact)
                rover_pos[rover] = w
            elif match(fact, '(have_soil_analysis * *)'):
                rover, w = get_parts(fact)
                if w in soil_required:
                    soil_required.remove(w)
            elif match(fact, '(have_rock_analysis * *)'):
                rover, w = get_parts(fact)
                if w in rock_required:
                    rock_required.remove(w)

        # Track communicated data
        communicated_soil = set()
        communicated_rock = set()
        communicated_image = set()
        for fact in state:
            if match(fact, '(communicated_soil_data *)'):
                w = get_parts(fact)[1]
                communicated_soil.add(w)
            elif match(fact, '(communicated_rock_data *)'):
                w = get_parts(fact)[1]
                communicated_rock.add(w)
            elif match(fact, '(communicated_image_data * * *)'):
                obj, mode, w = get_parts(fact)
                communicated_image.add((obj, mode))

        # Calculate remaining soil and rock communications
        for w in soil_required:
            total_cost += 2  # Sample and communicate
        for w in rock_required:
            total_cost += 2  # Sample and communicate

        # Calculate image communications
        for (obj, mode) in image_required:
            if (obj, mode) in communicated_image:
                continue
            # Find if calibration is done
            calibrated = False
            for fact in state:
                if match(fact, '(calibrated * *)'):
                    camera, rover = get_parts(fact)
                    if camera in self.calibration_targets:
                        target = self.calibration_targets[camera]
                        if (obj == target) and (rover in rover_pos):
                            calibrated = True
                            break
            if not calibrated:
                total_cost += 3  # Calibrate, take image, communicate
            else:
                total_cost += 2  # Take image, communicate

        return total_cost
