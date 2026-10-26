from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Return the original string split by space if it doesn't look like a fact
        return fact.split()
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing up the number
    of unsatisfied goal conditions and adding costs for necessary intermediate
    steps (sampling, imaging, calibrating) that haven't been completed yet.
    It provides a relaxed estimate by ignoring navigation costs and resource
    constraints (like store capacity or specific rover capabilities beyond
    what's implied by the need for samples/images).

    # Assumptions
    - Each unsatisfied goal requires at least one action (communication).
    - If a sample (soil/rock) required by a goal hasn't been collected,
      it adds the cost of the sampling action.
    - If an image required by a goal hasn't been taken, it adds the cost
      of the take_image action and the preceding calibrate action.
    - Navigation costs are ignored.
    - Resource constraints (like needing an empty store for sampling, or
      a specific rover being equipped) are largely ignored, assuming that
      if a sample/image is needed, it *can* eventually be obtained by *some*
      suitable rover.

    # Heuristic Initialization
    - Stores the set of goal conditions from the task.
    - Static facts are not explicitly processed in the constructor for this
      simple version, as the heuristic only checks for the presence of
      intermediate facts and goal facts in the current state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value `h_value` to 0.
    2. Iterate through each goal condition specified in the task (`self.goals`).
    3. For each goal condition `g`:
       - Check if `g` is present in the current state (`node.state`).
       - If `g` is *not* in the current state:
         - Increment `h_value` by 1. This accounts for the final action
           (e.g., `communicate_soil_data`, `communicate_rock_data`,
           `communicate_image_data`) needed to satisfy this goal.
         - Parse the goal predicate and arguments using `get_parts`.
         - Based on the predicate:
           - If `g` is `(communicated_soil_data ?w)`:
             - Check if `(have_soil_analysis ?r ?w)` exists for *any* rover `r`
               in the current state using `match`.
             - If no such `have_soil_analysis` fact exists, it means the soil
               sample from waypoint `w` has not been collected yet. Add 1 to
               `h_value` for the `sample_soil` action.
           - If `g` is `(communicated_rock_data ?w)`:
             - Check if `(have_rock_analysis ?r ?w)` exists for *any* rover `r`
               in the current state using `match`.
             - If no such `have_rock_analysis` fact exists, it means the rock
               sample from waypoint `w` has not been collected yet. Add 1 to
               `h_value` for the `sample_rock` action.
           - If `g` is `(communicated_image_data ?o ?m)`:
             - Check if `(have_image ?r ?o ?m)` exists for *any* rover `r`
               in the current state using `match`.
             - If no such `have_image` fact exists, it means the image of
               objective `o` in mode `m` has not been taken yet. Add 1 to
               `h_value` for the `take_image` action. Since `take_image`
               requires a calibrated camera, and calibration is a separate
               step that is consumed by `take_image`, also add 1 to `h_value`
               for the `calibrate` action.
               Total added cost for an un-taken image goal: 1 (communicate) +
               1 (take_image) + 1 (calibrate) = 3.
               If the image *has* been taken but not communicated, only 1
               (communicate) is added (as per the initial `h_value += 1` for
               the unsatisfied goal).
    4. Return the final `h_value`.
    """

    def __init__(self, task):
        """Initialize the heuristic by storing goal conditions."""
        self.goals = task.goals
        # Static facts are not strictly needed for this simple heuristic,
        # but we could store them if a more complex version required them.
        # self.static_facts = task.static

    def __call__(self, node):
        """Estimate the cost to reach the goal state."""
        state = node.state
        h_value = 0

        for goal in self.goals:
            # Check if the goal is already satisfied in the current state
            if goal not in state:
                # Goal is not satisfied, add cost

                # Add cost for the final communication/achievement action
                h_value += 1

                # Parse the goal predicate and arguments
                parts = get_parts(goal)
                # Ensure parts list is not empty before accessing index 0
                if not parts:
                    # This shouldn't happen with valid PDDL facts, but for robustness
                    continue # Skip this goal if parsing failed

                predicate = parts[0]

                if predicate == "communicated_soil_data":
                    # Ensure there's a waypoint argument
                    if len(parts) > 1:
                        waypoint = parts[1]
                        # Check if the intermediate step (having the sample) is done
                        soil_sample_obtained = any(match(fact, "have_soil_analysis", "*", waypoint) for fact in state)
                        if not soil_sample_obtained:
                            # Sample not obtained, add cost for sampling action
                            h_value += 1
                    # else: Invalid soil goal format, ignore or handle error

                elif predicate == "communicated_rock_data":
                     # Ensure there's a waypoint argument
                    if len(parts) > 1:
                        waypoint = parts[1]
                        # Check if the intermediate step (having the sample) is done
                        rock_sample_obtained = any(match(fact, "have_rock_analysis", "*", waypoint) for fact in state)
                        if not rock_sample_obtained:
                            # Sample not obtained, add cost for sampling action
                            h_value += 1
                    # else: Invalid rock goal format, ignore or handle error

                elif predicate == "communicated_image_data":
                    # Ensure there are objective and mode arguments
                    if len(parts) > 2:
                        objective = parts[1]
                        mode = parts[2]
                        # Check if the intermediate step (having the image) is done
                        image_obtained = any(match(fact, "have_image", "*", objective, mode) for fact in state)
                        if not image_obtained:
                            # Image not obtained, add cost for take_image action
                            h_value += 1
                            # Taking an image requires calibration, add cost for calibrate action
                            h_value += 1
                    # else: Invalid image goal format, ignore or handle error

        return h_value
