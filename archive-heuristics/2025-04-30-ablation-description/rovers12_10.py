from fnmatch import fnmatch
# Assume the Heuristic base class is provided elsewhere or use the dummy one below.

# Dummy Heuristic base class (remove or comment out if the real one is available)
# class Heuristic:
#     def __init__(self, task):
#         self.task = task
#         self.process_static_facts(task.static)
#
#     def process_static_facts(self, static_facts):
#         pass # To be implemented by subclass
#
#     def __call__(self, node):
#         raise NotImplementedError

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by spaces
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing up the
    "stages" remaining for each unachieved goal condition. It provides a
    simple estimate based on how far along the process (sampling/imaging,
    having the data, communicating the data) each goal is.

    # Assumptions
    - Goals are always of the form (communicated_soil_data ?w),
      (communicated_rock_data ?w), or (communicated_image_data ?o ?m).
    - Each stage (sampling/imaging, having data, communicating) requires
      at least one action. Movement costs are implicitly included within
      these stages or ignored for simplicity in this relaxed heuristic.
    - The heuristic is not admissible but aims to guide a greedy search
      towards states where goal conditions are closer to being met.

    # Heuristic Initialization
    - Extracts static information from the task, such as which rovers are
      equipped for which tasks, which cameras are on which rovers, and which
      modes cameras support. This information is needed to determine if a
      rover/camera combination is relevant for achieving an image goal stage.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic is calculated by iterating through each goal condition
    and adding a cost based on its current state:

    1.  Initialize total heuristic cost to 0.
    2.  For each goal condition in the task's goals:
        a.  Parse the goal fact to identify its type (soil, rock, or image)
            and parameters (waypoint or objective/mode).
        b.  Check if the goal condition is already true in the current state.
            If yes, the cost for this goal is 0; continue to the next goal.
        c.  If the goal is (communicated_soil_data ?w) and not achieved:
            - Check if (have_soil_analysis ?r ?w) is true for *any* rover ?r.
            - If yes, add 1 to the total cost (need to communicate).
            - If no, add 2 to the total cost (need to sample and communicate).
        d.  If the goal is (communicated_rock_data ?w) and not achieved:
            - Check if (have_rock_analysis ?r ?w) is true for *any* rover ?r.
            - If yes, add 1 to the total cost (need to communicate).
            - If no, add 2 to the total cost (need to sample and communicate).
        e.  If the goal is (communicated_image_data ?o ?m) and not achieved:
            - Check if (have_image ?r ?o ?m) is true for *any* rover ?r.
            - If yes, add 1 to the total cost (need to communicate).
            - If no:
                - Check if (calibrated ?i ?r) is true for *any* camera ?i
                  on *any* rover ?r such that:
                    - ?r is equipped for imaging ((equipped_for_imaging ?r) is static).
                    - ?i is on board ?r ((on_board ?i ?r) is static).
                    - ?i supports mode ?m ((supports ?i ?m) is static).
                - If such a calibrated camera/rover exists, add 2 to the total
                  cost (need to take image and communicate).
                - If no such calibrated camera/rover exists, add 3 to the total
                  cost (need to calibrate, take image, and communicate).
    3.  Return the total calculated cost.
    """

    def process_static_facts(self, static_facts):
        """
        Extracts static information relevant to the heuristic.
        """
        self.equipped_for_imaging = set()
        self.on_board = {} # camera -> rover
        self.supports = {} # camera -> set of modes
        # calibration_target is not strictly needed for this heuristic logic,
        # as we only check for calibrated cameras that *can* support the goal mode.
        # self.calibration_target = {} # camera -> objective

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "equipped_for_imaging":
                self.equipped_for_imaging.add(parts[1])
            elif predicate == "on_board":
                camera, rover = parts[1], parts[2]
                self.on_board[camera] = rover
            elif predicate == "supports":
                camera, mode = parts[1], parts[2]
                if camera not in self.supports:
                    self.supports[camera] = set()
                self.supports[camera].add(mode)
            # elif predicate == "calibration_target":
            #     camera, objective = parts[1], parts[2]
            #     self.calibration_target[camera] = objective

    def __call__(self, node):
        """
        Computes the heuristic value for the given state.
        """
        state = node.state
        total_cost = 0

        # Helper to check if a pattern exists in the state
        def state_has(pattern_parts):
             return any(match(fact, *pattern_parts) for fact in state)

        for goal in self.task.goals:
            goal_parts = get_parts(goal)
            if not goal_parts: continue # Skip malformed goals

            goal_predicate = goal_parts[0]

            # If the goal is already achieved, cost is 0 for this goal
            if goal in state:
                continue

            # Goal: (communicated_soil_data ?w)
            if goal_predicate == "communicated_soil_data":
                waypoint = goal_parts[1]
                # Check if sample is collected
                if state_has(["have_soil_analysis", "*", waypoint]):
                    total_cost += 1 # Need to communicate
                else:
                    total_cost += 2 # Need to sample and communicate

            # Goal: (communicated_rock_data ?w)
            elif goal_predicate == "communicated_rock_data":
                waypoint = goal_parts[1]
                # Check if sample is collected
                if state_has(["have_rock_analysis", "*", waypoint]):
                    total_cost += 1 # Need to communicate
                else:
                    total_cost += 2 # Need to sample and communicate

            # Goal: (communicated_image_data ?o ?m)
            elif goal_predicate == "communicated_image_data":
                objective, mode = goal_parts[1], goal_parts[2]
                # Check if image is taken
                if state_has(["have_image", "*", objective, mode]):
                    total_cost += 1 # Need to communicate
                else:
                    # Check if a relevant camera is calibrated
                    # A relevant camera is one on an imaging-equipped rover
                    # that supports the required mode.
                    relevant_camera_calibrated = False
                    # Iterate through cameras and their rovers from static facts
                    for camera, rover in self.on_board.items():
                        # Check if the rover is equipped for imaging
                        if rover in self.equipped_for_imaging:
                            # Check if the camera supports the required mode
                            if camera in self.supports and mode in self.supports[camera]:
                                # Check if this specific camera on this specific rover is calibrated in the current state
                                if state_has(["calibrated", camera, rover]):
                                    relevant_camera_calibrated = True
                                    break # Found one calibrated relevant camera

                    if relevant_camera_calibrated:
                        total_cost += 2 # Need to take image and communicate
                    else:
                        total_cost += 3 # Need to calibrate, take image, and communicate

        return total_cost
