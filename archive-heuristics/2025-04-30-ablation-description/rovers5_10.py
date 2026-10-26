from fnmatch import fnmatch
# Assuming Heuristic base class is available in this path
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or invalid format gracefully, though PDDL facts are structured.
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing up costs
    for each unachieved goal condition. The cost for an unachieved goal
    includes the final communication step and, if necessary, the preceding
    sampling/imaging steps and potentially a calibration step for images.
    It ignores navigation costs and resource constraints (like store capacity).

    # Assumptions
    - The heuristic counts the number of unachieved goal predicates.
    - For soil/rock goals, it adds a cost for sampling if no sample data
      for that waypoint exists in the state for any equipped rover.
    - For image goals, it adds a cost for taking the image if no image data
      for that objective and mode exists in the state for any rover.
    - For image goals where the image data is missing, it adds a cost for
      calibration if *no* camera on *any* imaging-equipped rover that
      supports the required mode is currently calibrated. This is a
      simplification that doesn't track specific camera/rover calibration
      needs but checks for the general availability of a calibrated camera
      for the mode.
    - Navigation costs are ignored.
    - Store capacity constraints for samples are ignored.
    - The existence of suitable rovers/equipment is assumed based on static facts.

    # Heuristic Initialization
    The constructor extracts relevant static information from the task:
    - Which rovers are equipped for soil, rock, and imaging analysis.
    - Which cameras are on which rovers.
    - Which modes each camera supports.
    - The set of goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize the total heuristic cost to 0.
    2. Iterate through each goal literal specified in the task.
    3. For each goal literal:
       - If the goal literal is already present in the current state, it contributes 0 to the heuristic.
       - If the goal literal is NOT present in the current state:
         - Add 1 to the total cost (representing the final communication action).
         - Identify the type of goal:
           - If it's `(communicated_soil_data ?w)`:
             - Check if `(have_soil_analysis ?r ?w)` exists in the state for *any* rover `?r` equipped for soil analysis.
             - If no such fact exists, add 1 to the total cost (representing the sample_soil action).
           - If it's `(communicated_rock_data ?w)`:
             - Check if `(have_rock_analysis ?r ?w)` exists in the state for *any* rover `?r` equipped for rock analysis.
             - If no such fact exists, add 1 to the total cost (representing the sample_rock action).
           - If it's `(communicated_image_data ?o ?m)`:
             - Check if `(have_image ?r ?o ?m)` exists in the state for *any* rover `?r`.
             - If no such fact exists, add 1 to the total cost (representing the take_image action).
             - Additionally, if the image data is missing, check if *any* camera `?i` on *any* imaging-equipped rover `?r` that supports mode `?m` is currently calibrated (`(calibrated ?i ?r)` is in the state).
             - If *no* such calibrated camera/rover combination exists for the required mode, add 1 to the total cost (representing the calibrate action).
    4. The total cost accumulated is the heuristic value for the state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals  # Goal conditions (frozenset of strings).
        self.static = task.static # Static facts (frozenset of strings).

        # Pre-process static facts for quick lookups
        self.equipped_soil = {get_parts(fact)[1] for fact in self.static if match(fact, "equipped_for_soil_analysis", "*")}
        self.equipped_rock = {get_parts(fact)[1] for fact in self.static if match(fact, "equipped_for_rock_analysis", "*")}
        self.equipped_imaging = {get_parts(fact)[1] for fact in self.static if match(fact, "equipped_for_imaging", "*")}

        self.camera_on_rover = {get_parts(fact)[1]: get_parts(fact)[2] for fact in self.static if match(fact, "on_board", "*", "*")} # camera -> rover
        self.camera_supports_mode = {} # camera -> set of modes
        for fact in self.static:
            if match(fact, "supports", "*", "*"):
                cam, mode = get_parts(fact)[1:3]
                self.camera_supports_mode.setdefault(cam, set()).add(mode)

        # Lander location (assuming one lander) - not strictly needed for this heuristic but good practice
        # self.lander_location = next(get_parts(fact)[2] for fact in self.static if match(fact, "at_lander", "*", "*"))

        # Visible waypoints (graph structure) - not strictly needed for this heuristic
        # self.visible_waypoints = {(get_parts(fact)[1], get_parts(fact)[2]) for fact in self.static if match(fact, "visible", "*", "*")}

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state from the current state.
        """
        state = node.state  # Current world state (frozenset of strings).
        total_cost = 0  # Initialize action cost counter.

        for goal in self.goals:
            # If the goal is already achieved, it contributes 0 to the heuristic.
            if goal in state:
                continue

            # Goal is not achieved, add cost based on the type of goal.
            parts = get_parts(goal)
            if not parts: # Should not happen with valid PDDL goals
                continue

            predicate = parts[0]

            if predicate == 'communicated_soil_data':
                waypoint = parts[1]
                total_cost += 1  # Cost for the 'communicate_soil_data' action

                # Check if the soil sample has been collected for this waypoint by any equipped rover.
                sample_collected = any(match(fact, "have_soil_analysis", r, waypoint)
                                       for fact in state for r in self.equipped_soil)

                if not sample_collected:
                    total_cost += 1  # Cost for the 'sample_soil' action

            elif predicate == 'communicated_rock_data':
                waypoint = parts[1]
                total_cost += 1  # Cost for the 'communicate_rock_data' action

                # Check if the rock sample has been collected for this waypoint by any equipped rover.
                sample_collected = any(match(fact, "have_rock_analysis", r, waypoint)
                                       for fact in state for r in self.equipped_rock)

                if not sample_collected:
                    total_cost += 1  # Cost for the 'sample_rock' action

            elif predicate == 'communicated_image_data':
                objective, mode = parts[1:3]
                total_cost += 1  # Cost for the 'communicate_image_data' action

                # Check if the image has been taken for this objective and mode by any rover.
                image_taken = any(match(fact, "have_image", r, objective, mode)
                                  for fact in state) # Check any rover

                if not image_taken:
                    total_cost += 1  # Cost for the 'take_image' action

                    # If the image is not taken, check if calibration is needed.
                    # Calibration is needed if *no* suitable camera on *any* imaging rover
                    # that supports the mode is currently calibrated.
                    calibration_exists_for_mode = False
                    for fact in state:
                        if match(fact, "calibrated", "*", "*"):
                            camera, rover = get_parts(fact)[1:3]
                            # Check if this calibrated camera/rover combination is suitable for this mode
                            if (rover in self.equipped_imaging and
                                camera in self.camera_on_rover and
                                self.camera_on_rover[camera] == rover and
                                mode in self.camera_supports_mode.get(camera, set())):
                                calibration_exists_for_mode = True
                                break # Found a suitable calibrated camera, no need to check others

                    if not calibration_exists_for_mode:
                        total_cost += 1  # Cost for the 'calibrate' action

        return total_cost
