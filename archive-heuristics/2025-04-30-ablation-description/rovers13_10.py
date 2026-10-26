from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Dummy Heuristic class for local testing if needed
# class Heuristic:
#     def __init__(self, task):
#         self.task = task
#         self.goals = task.goals
#         self.static = task.static
#     def __call__(self, node):
#         raise NotImplementedError

from fnmatch import fnmatch

# Helper functions (copied from examples)
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the estimated
    number of actions required for each unsatisfied goal predicate. It considers
    the dependencies between actions (e.g., sampling before communicating,
    calibrating before imaging) but ignores navigation costs and resource
    constraints (like multiple rovers needing the same resource or location).

    # Assumptions
    - The heuristic assumes that necessary equipment (equipped rovers, cameras, stores)
      exists in the problem instance to achieve the goals.
    - Navigation costs are ignored.
    - Resource conflicts (e.g., multiple rovers needing the same store or camera)
      are ignored.
    - The presence of samples (`at_soil_sample`, `at_rock_sample`) and objective
      visibility (`visible_from`) required for goals is assumed if the goal is
      part of the problem definition.

    # Heuristic Initialization
    - Stores the set of goal predicates.
    - Extracts static facts about rover capabilities (`equipped_for_imaging`),
      camera properties (`on_board`, `supports`), and calibration targets
      (`calibration_target`) to quickly check prerequisites during heuristic
      calculation.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is calculated as the sum of estimated costs for each
    individual goal predicate that is not yet satisfied in the current state.
    The cost for an unsatisfied goal is determined by the sequence of actions
    required to achieve it and its prerequisites, counting 1 for each major
    action type in the chain, ignoring navigation.

    For each goal `g` in the task's goals:
    1. If `g` is already true in the current state, its contribution to the
       heuristic is 0.
    2. If `g` is `(communicated_soil_data ?w)` and is not in the state:
       - Add 1 to the cost (for the `communicate_soil_data` action).
       - Check if `(have_soil_analysis ?r ?w)` is true for *any* rover `r`
         in the current state.
       - If not, add 1 to the cost (for the `sample_soil` action).
    3. If `g` is `(communicated_rock_data ?w)` and is not in the state:
       - Add 1 to the cost (for the `communicate_rock_data` action).
       - Check if `(have_rock_analysis ?r ?w)` is true for *any* rover `r`
         in the current state.
       - If not, add 1 to the cost (for the `sample_rock` action).
    4. If `g` is `(communicated_image_data ?o ?m)` and is not in the state:
       - Add 1 to the cost (for the `communicate_image_data` action).
       - Check if `(have_image ?r ?o ?m)` is true for *any* rover `r`
         in the current state.
       - If not:
         - Add 1 to the cost (for the `take_image` action).
         - Check if `(calibrated ?i ?r)` is true for *any* camera `i` on *any*
           rover `r` such that rover `r` is equipped for imaging, camera `i`
           is on board `r`, and camera `i` supports mode `m`.
         - If no such calibrated camera/rover pair exists in the current state,
           add 1 to the cost (for the `calibrate` action).

    The total heuristic value is the sum of these costs for all unsatisfied goals.
    A goal state will have a heuristic value of 0.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static_facts = task.static

        # Extract relevant static facts for quick lookup
        self.imaging_rovers = set()
        self.on_board_cameras = set() # Store as (camera, rover) tuples
        self.camera_modes_tuples = set() # Store as (camera, mode) tuples

        for fact in self.static_facts:
            if match(fact, "equipped_for_imaging", "?r"):
                self.imaging_rovers.add(get_parts(fact)[1])
            elif match(fact, "on_board", "?i", "?r"):
                self.on_board_cameras.add((get_parts(fact)[1], get_parts(fact)[2]))
            elif match(fact, "supports", "?i", "?m"):
                self.camera_modes_tuples.add((get_parts(fact)[1], get_parts(fact)[2]))


    def __call__(self, node):
        """Estimate the minimum cost to satisfy all goal conditions."""
        state = node.state
        total_cost = 0

        # Parse state facts efficiently into a dictionary for quick lookup
        state_facts_dict = {}
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate not in state_facts_dict:
                state_facts_dict[predicate] = []
            state_facts_dict[predicate].append(parts)

        # Helper to check for facts in state_facts_dict using pattern matching
        def check_state_fact(predicate_name, *args):
            if predicate_name not in state_facts_dict:
                return False
            # Check if any fact with the predicate matches the arguments pattern
            for parts in state_facts_dict[predicate_name]:
                 # parts[0] is the predicate name, args match parts[1:]
                 if len(parts) - 1 != len(args):
                     continue # Mismatch in number of arguments
                 if all(fnmatch(part, arg) for part, arg in zip(parts[1:], args)):
                     return True # Found a matching fact
            return False # No matching fact found

        # Iterate through each goal predicate
        for goal in self.goals:
            # If the goal is already satisfied, contribute 0 to the heuristic
            if goal in state:
                continue

            # Goal is not satisfied, calculate its contribution
            goal_parts = get_parts(goal)
            predicate = goal_parts[0]

            if predicate == "communicated_soil_data":
                waypoint = goal_parts[1]
                total_cost += 1 # Cost for the communicate_soil_data action

                # Check if the required soil sample is held by any rover
                has_sample = check_state_fact("have_soil_analysis", "*", waypoint)

                if not has_sample:
                    total_cost += 1 # Cost for the sample_soil action

            elif predicate == "communicated_rock_data":
                waypoint = goal_parts[1]
                total_cost += 1 # Cost for the communicate_rock_data action

                # Check if the required rock sample is held by any rover
                has_sample = check_state_fact("have_rock_analysis", "*", waypoint)

                if not has_sample:
                    total_cost += 1 # Cost for the sample_rock action

            elif predicate == "communicated_image_data":
                objective = goal_parts[1]
                mode = goal_parts[2]
                total_cost += 1 # Cost for the communicate_image_data action

                # Check if the required image is held by any rover
                has_image = check_state_fact("have_image", "*", objective, mode)

                if not has_image:
                    total_cost += 1 # Cost for the take_image action

                    # Check if calibration is needed for *any* suitable camera/rover
                    # A suitable camera/rover is one that is imaging-equipped,
                    # has the camera on board, and the camera supports the mode.
                    # Calibration is needed if *none* of the suitable camera/rover
                    # pairs are currently calibrated.
                    any_suitable_calibrated_camera_exists = False

                    # Iterate through all cameras on board rovers (from static facts)
                    for camera, rover in self.on_board_cameras:
                         # Check if the rover is equipped for imaging (from static facts)
                         if rover in self.imaging_rovers:
                             # Check if the camera supports the required mode (from static facts)
                             if (camera, mode) in self.camera_modes_tuples:
                                 # This is a suitable camera/rover pair. Check if calibrated in state.
                                 if check_state_fact("calibrated", camera, rover):
                                     any_suitable_calibrated_camera_exists = True
                                     break # Found one suitable and calibrated camera/rover

                    if not any_suitable_calibrated_camera_exists:
                         total_cost += 1 # Cost for the calibrate action

        return total_cost
