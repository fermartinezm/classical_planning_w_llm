from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve all goal conditions, including:
    - Communicating soil and rock analysis data from waypoints.
    - Communicating image data from objectives in specific modes.
    - Moving rovers between waypoints, collecting samples, calibrating cameras, and taking images.

    # Assumptions:
    - Each rover can perform any task as long as it has the necessary equipment.
    - Moving between adjacent waypoints takes one action.
    - Collecting a sample and communicating it takes two actions.
    - Taking an image requires calibration and communication, taking three actions.
    - Multiple rovers can work in parallel to achieve the goals.

    # Heuristic Initialization
    - Extract goal conditions for each type of data (soil, rock, image).
    - Extract static facts about can_traverse, visible waypoints, and calibration targets.

    # Step-by-Step Thinking for Computing Heuristic
    1. Identify all missing communication goals.
    2. For each missing soil or rock data:
       a. Find a rover equipped for the task.
       b. Calculate the moves needed to reach the waypoint.
       c. Add actions for collecting the sample and communicating it.
    3. For each missing image data:
       a. Assign a rover with the necessary camera equipment.
       b. Ensure the camera is calibrated (if not, add calibration step).
       c. Calculate moves to reach the waypoint.
       d. Add actions for taking the image and communicating it.
    4. Sum all required actions to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract static information into useful data structures
        self.can_traverse = {}
        for fact in self.static:
            if match(fact, "can_traverse", "*", "*", "*"):
                rover, from_w, to_w = get_parts(fact)[1:]
                if rover not in self.can_traverse:
                    self.can_traverse[rover] = {}
                self.can_traverse[rover][(from_w, to_w)] = True

        self.visible = {}
        for fact in self.static:
            if match(fact, "visible", "*", "*"):
                from_w, to_w = get_parts(fact)[1:]
                if from_w not in self.visible:
                    self.visible[from_w] = []
                self.visible[from_w].append(to_w)

        self.calibration_targets = {}
        for fact in self.static:
            if match(fact, "calibration_target", "*", "*"):
                camera, obj = get_parts(fact)[1:]
                self.calibration_targets[(camera, obj)] = True

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        def get_parts(fact):
            """Extract components of a PDDL fact."""
            return fact[1:-1].split()

        def match(fact, *args):
            """Check if a fact matches a given pattern."""
            parts = get_parts(fact)
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        # Extract current state information
        current_at = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                rover, waypoint = get_parts(fact)[1:]
                current_at[rover] = waypoint

        current_samples = {}
        for fact in state:
            if match(fact, "have_soil_analysis", "*", "*"):
                rover, waypoint = get_parts(fact)[1:]
                if rover not in current_samples:
                    current_samples[rover] = []
                current_samples[rover].append(('soil', waypoint))
            if match(fact, "have_rock_analysis", "*", "*"):
                rover, waypoint = get_parts(fact)[1:]
                if rover not in current_samples:
                    current_samples[rover] = []
                current_samples[rover].append(('rock', waypoint))

        current_images = {}
        for fact in state:
            if match(fact, "have_image", "*", "*", "*"):
                rover, obj, mode = get_parts(fact)[1:]
                key = (rover, obj, mode)
                current_images[key] = True

        current_communicated_soil = set()
        current_communicated_rock = set()
        current_communicated_image = set()
        for fact in state:
            if match(fact, "communicated_soil_data", "*"):
                waypoint = get_parts(fact)[1]
                current_communicated_soil.add(waypoint)
            if match(fact, "communicated_rock_data", "*"):
                waypoint = get_parts(fact)[1]
                current_communicated_rock.add(waypoint)
            if match(fact, "communicated_image_data", "*", "*", "*"):
                obj, mode = get_parts(fact)[1], get_parts(fact)[2]
                current_communicated_image.add((obj, mode))

        # Count missing goals
        total_actions = 0

        # Count missing soil and rock communications
        for fact in self.goals:
            if match(fact, "communicated_soil_data", "*"):
                waypoint = get_parts(fact)[1]
                if waypoint not in current_communicated_soil:
                    # Find a rover that can collect soil
                    for rover in current_at:
                        if (rover, 'soil') in current_samples.get(rover, []):
                            break
                    # Assume at least one rover can do it
                    total_actions += 2  # Collect and communicate
            elif match(fact, "communicated_rock_data", "*"):
                waypoint = get_parts(fact)[1]
                if waypoint not in current_communicated_rock:
                    # Find a rover that can collect rock
                    for rover in current_at:
                        if (rover, 'rock') in current_samples.get(rover, []):
                            break
                    # Assume at least one rover can do it
                    total_actions += 2  # Collect and communicate

        # Count missing image communications
        for fact in self.goals:
            if match(fact, "communicated_image_data", "*", "*", "*"):
                obj, mode = get_parts(fact)[1], get_parts(fact)[2]
                if (obj, mode) not in current_communicated_image:
                    # Find a rover with the necessary camera
                    for rover in current_at:
                        for camera in [c for c in state if match(c, "on_board", "*", rover)]:
                            if (camera, obj) in self.calibration_targets:
                                # Check if camera is calibrated
                                calibrated = any(match(f, "calibrated", camera, rover) for f in state)
                                if not calibrated:
                                    total_actions += 1  # Calibrate
                                # Move to waypoint (assuming visible)
                                total_actions += 1  # Move
                                total_actions += 2  # Take image and communicate
                                break
                        else:
                            continue
                        break

        return total_actions
