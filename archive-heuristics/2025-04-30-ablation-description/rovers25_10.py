from fnmatch import fnmatch
# Assuming Heuristic base class is available in a 'heuristics' module
# from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't try to match more args than parts
    if len(args) > len(parts):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Define the heuristic class
# Inherit from Heuristic if available in the environment
# class roversHeuristic(Heuristic):
class roversHeuristic:
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the cost to satisfy the goal conditions by summing
    up the estimated costs for each unsatisfied goal predicate. The cost for
    each goal depends on the stage of completion (e.g., data collected, image taken,
    camera calibrated) and the type of data (soil, rock, image).

    # Assumptions
    - The cost of movement between waypoints is abstracted away or assumed to be constant (e.g., 1).
    - The cost of actions like sampling, calibrating, taking images, dropping, and communicating is assumed to be 1.
    - The heuristic does not consider resource constraints like store capacity beyond checking if sampling is needed.
    - The heuristic does not consider specific rover locations for movement costs, only whether the data is collected or image is taken/calibrated.
    - The heuristic assumes that if a goal is specified, there exists a configuration of rovers/cameras/waypoints that can achieve it (reachability is not checked).

    # Heuristic Initialization
    - Stores the task goals.
    - Extracts static information relevant to image goals: identifies which (rover, camera) pairs can achieve which (objective, mode) image goals based on static facts like `on_board`, `supports`, and `equipped_for_imaging`.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is the sum of costs for each goal predicate in the task's goal set that is not currently true in the state.

    For each unsatisfied goal `g`:
    1.  **If `g` is `(communicated_soil_data ?w)`:**
        - Check if `(have_soil_analysis ?r ?w)` is true for *any* rover `?r` in the current state.
        - If yes: The soil sample is collected. The remaining cost is just communication. Add 1 to the total cost.
        - If no: The soil sample needs to be collected. This requires sampling and then communicating. Add 2 to the total cost (1 for sampling, 1 for communicating).
    2.  **If `g` is `(communicated_rock_data ?w)`:**
        - Check if `(have_rock_analysis ?r ?w)` is true for *any* rover `?r` in the current state.
        - If yes: The rock sample is collected. The remaining cost is just communication. Add 1 to the total cost.
        - If no: The rock sample needs to be collected. This requires sampling and then communicating. Add 2 to the total cost (1 for sampling, 1 for communicating).
    3.  **If `g` is `(communicated_image_data ?o ?m)`:**
        - Check if `(have_image ?r ?o ?m)` is true for *any* rover `?r` in the current state.
        - If yes: The image is taken. The remaining cost is just communication. Add 1 to the total cost.
        - If no: The image needs to be taken and then communicated.
            - To take the image, a suitable camera must be calibrated. A camera `?i` on rover `?r` is suitable if `(equipped_for_imaging ?r)`, `(on_board ?i ?r)`, and `(supports ?i ?m)` are static facts.
            - Check if `(calibrated ?i ?r)` is true in the current state for *any* suitable `(rover, camera)` pair `(?r, ?i)`.
            - If a suitable camera is calibrated: The remaining cost is taking the image and communicating. Add 2 to the total cost (1 for taking image, 1 for communicating).
            - If no suitable camera is calibrated: The remaining cost is calibrating, taking the image, and communicating. Add 3 to the total cost (1 for calibrating, 1 for taking image, 1 for communicating).

    The total heuristic value is the sum of these costs for all unsatisfied goals.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        Pre-calculates which (rover, camera) pairs can achieve which (objective, mode)
        image goals.
        """
        # super().__init__(task) # Uncomment if inheriting from Heuristic
        self.goals = task.goals
        self.static_facts = task.static

        # Pre-calculate capability for image goals: (objective, mode) -> set of (rover, camera)
        self.image_capabilities = {}
        # Extract static info needed for capabilities
        equipped_imaging = {get_parts(f)[1] for f in self.static_facts if match(f, "equipped_for_imaging", "*")}
        on_board = {get_parts(f)[1]: get_parts(f)[2] for f in self.static_facts if match(f, "on_board", "*", "*")} # camera -> rover
        supports = {} # camera -> set of modes
        for f in self.static_facts:
            if match(f, "supports", "*", "*"):
                camera, mode = get_parts(f)[1:]
                supports.setdefault(camera, set()).add(mode)

        # Iterate through image goals to find capabilities
        for goal in self.goals:
            if match(goal, "communicated_image_data", "*", "*"):
                _, objective, mode = get_parts(goal)
                obj_mode_key = (objective, mode)
                self.image_capabilities[obj_mode_key] = set()

                # Find all (rover, camera) pairs that can achieve this (objective, mode)
                for camera, rover in on_board.items():
                    if rover in equipped_imaging and mode in supports.get(camera, set()):
                         self.image_capabilities[obj_mode_key].add((rover, camera))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Helper sets for quick lookup in the state
        have_soil_waypoints = {get_parts(f)[2] for f in state if match(f, "have_soil_analysis", "*", "*")}
        have_rock_waypoints = {get_parts(f)[2] for f in state if match(f, "have_rock_analysis", "*", "*")}
        have_image_obj_modes = {(get_parts(f)[2], get_parts(f)[3]) for f in state if match(f, "have_image", "*", "*", "*")}
        calibrated_cameras_rovers = {tuple(get_parts(f)[1:]) for f in state if match(f, "calibrated", "*", "*")} # (camera, rover)

        for goal in self.goals:
            # If the goal is already satisfied, it contributes 0 to the heuristic
            if goal in state:
                continue

            predicate, *args = get_parts(goal)

            if predicate == "communicated_soil_data":
                waypoint = args[0]
                if waypoint in have_soil_waypoints:
                    # Data collected, need to communicate
                    total_cost += 1
                else:
                    # Need to sample and communicate
                    total_cost += 2

            elif predicate == "communicated_rock_data":
                waypoint = args[0]
                if waypoint in have_rock_waypoints:
                    # Data collected, need to communicate
                    total_cost += 1
                else:
                    # Need to sample and communicate
                    total_cost += 2

            elif predicate == "communicated_image_data":
                objective, mode = args
                obj_mode_key = (objective, mode)

                if obj_mode_key in have_image_obj_modes:
                    # Image taken, need to communicate
                    total_cost += 1
                else:
                    # Need to take image and communicate
                    # Check if ANY suitable camera is calibrated
                    suitable_rovers_cameras = self.image_capabilities.get(obj_mode_key, set())
                    # Check if any (camera, rover) in calibrated_cameras_rovers matches a (rover, camera) in suitable_rovers_cameras
                    is_calibrated = any((camera, rover) in calibrated_cameras_rovers for (rover, camera) in suitable_rovers_cameras)

                    if is_calibrated:
                        # Camera calibrated, need to take image and communicate
                        total_cost += 2
                    else:
                        # Need to calibrate, take image, and communicate
                        total_cost += 3

        return total_cost
