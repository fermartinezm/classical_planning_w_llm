from fnmatch import fnmatch
# Assuming Heuristic base class is available in this path
# from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle potential non-string inputs or malformed facts gracefully
        # For this domain, we expect valid fact strings.
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
    # The fact must have the same number of components as the pattern arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic: # Inherit from Heuristic if available in the environment
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing up costs for each uncommunicated goal fact. The cost for a goal fact depends on whether the required data (soil analysis, rock analysis, or image) has already been acquired by any rover. It provides a simple estimate of the remaining work based on achieving intermediate 'have_...' states and then communicating.

    # Assumptions
    - The heuristic uses a simplified cost model based on stages of achieving a goal:
        - Communicating data that has already been acquired costs 1 unit.
        - Acquiring (sampling) and then communicating soil/rock data costs 2 units (1 for sampling, 1 for communication setup).
        - Acquiring (calibrating and taking image) and then communicating image data costs 3 units (1 for calibration, 1 for taking image, 1 for communication setup).
    - This heuristic is non-admissible as it does not consider movement costs, resource constraints (like empty stores or camera calibration state), or specific rover capabilities beyond checking for the existence of 'have_...' facts. It is designed to guide a greedy best-first search.

    # Heuristic Initialization
    - The heuristic stores the set of goal facts from the task. Static facts are not explicitly used in this simple version of the heuristic, as the heuristic only checks for the presence of dynamic 'have_...' facts relative to the goals.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Initialize the total heuristic cost `h` to 0.
    2. Iterate through each goal fact specified in the task's goals.
    3. For each goal fact:
       - Check if the goal fact is already present in the current state. If it is, this goal is satisfied, contributes 0 to the heuristic, and we move to the next goal.
       - If the goal fact is not in the current state, it needs to be achieved. Determine the type of goal:
         - If the goal is `(communicated_soil_data ?w)`:
           - Check if any rover currently possesses the soil analysis for waypoint `?w`. This is done by searching the current state for any fact matching the pattern `(have_soil_analysis * ?w)`.
           - If such a fact exists (meaning the soil sample has been taken), the remaining task is primarily communication. Add 1 to the total cost `h`.
           - If no such fact exists, the soil sample still needs to be taken *and* then communicated. Add 2 to the total cost `h`.
         - If the goal is `(communicated_rock_data ?w)`:
           - Check if any rover currently possesses the rock analysis for waypoint `?w` by searching for facts matching `(have_rock_analysis * ?w)`.
           - If such a fact exists, add 1 to `h` (communication needed).
           - If no such fact exists, add 2 to `h` (sampling and communication needed).
         - If the goal is `(communicated_image_data ?o ?m)`:
           - Check if any rover currently possesses the image for objective `?o` in mode `?m` by searching for facts matching `(have_image * ?o ?m)`.
           - If such a fact exists, add 1 to `h` (communication needed).
           - If no such fact exists, the image still needs to be acquired (calibration + taking image) *and* then communicated. Add 3 to the total cost `h`.
    4. After iterating through all goal facts, the accumulated value in `h` is the heuristic estimate for the current state. Return `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by storing the goal facts.

        Args:
            task: The planning task object containing initial state, goals, etc.
        """
        self.goals = task.goals
        # Static facts are available in task.static but are not used in this
        # simple heuristic version.

    def __call__(self, node):
        """
        Compute the heuristic estimate for the given state.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer representing the estimated cost to reach the goal.
        """
        state = node.state

        total_cost = 0

        # Iterate through each goal fact defined in the problem
        for goal in self.goals:
            # If the goal fact is already true in the current state, it's satisfied
            if goal in state:
                continue

            # If the goal is not satisfied, determine the cost based on its type
            parts = get_parts(goal)
            if not parts: # Skip malformed goal facts if any
                continue

            predicate = parts[0]

            if predicate == "communicated_soil_data":
                # Goal: (communicated_soil_data ?w)
                waypoint = parts[1]
                # Check if the intermediate goal (have_soil_analysis ?r ?w) is met for any rover
                # We iterate through the state to find any matching 'have_soil_analysis' fact
                found_analysis = False
                for fact in state:
                    if match(fact, "have_soil_analysis", "*", waypoint):
                        found_analysis = True
                        break # Found one, no need to check further

                if found_analysis:
                    # Soil analysis is done, just need to communicate
                    total_cost += 1
                else:
                    # Need to sample soil AND communicate
                    total_cost += 2

            elif predicate == "communicated_rock_data":
                # Goal: (communicated_rock_data ?w)
                waypoint = parts[1]
                # Check if the intermediate goal (have_rock_analysis ?r ?w) is met for any rover
                found_analysis = False
                for fact in state:
                    if match(fact, "have_rock_analysis", "*", waypoint):
                        found_analysis = True
                        break # Found one, no need to check further

                if found_analysis:
                    # Rock analysis is done, just need to communicate
                    total_cost += 1
                else:
                    # Need to sample rock AND communicate
                    total_cost += 2

            elif predicate == "communicated_image_data":
                # Goal: (communicated_image_data ?o ?m)
                objective = parts[1]
                mode = parts[2]
                # Check if the intermediate goal (have_image ?r ?o ?m) is met for any rover
                found_image = False
                for fact in state:
                    if match(fact, "have_image", "*", objective, mode):
                        found_image = True
                        break # Found one, no need to check further

                if found_image:
                    # Image is taken, just need to communicate
                    total_cost += 1
                else:
                    # Need to calibrate, take image AND communicate
                    total_cost += 3

            # Add checks for other potential goal types if the domain were extended
            # else:
            #     # Unknown goal type, maybe add a default cost or ignore?
            #     # For this domain, we only have these three types.
            #     pass

        return total_cost

