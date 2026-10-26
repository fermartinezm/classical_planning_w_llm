from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or malformed facts defensively
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
         # Or raise an error, depending on expected input robustness
         return []
    return fact[1:-1].split()

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    uncommunicated goals (soil data, rock data, image data). It sums the
    estimated cost for each individual uncommunicated goal fact.

    # Assumptions
    - The heuristic assumes that for any uncommunicated goal, there exists
      a rover with the necessary equipment (soil analysis, rock analysis,
      or imaging) and that required samples, objectives, calibration targets,
      and lander-visible waypoints are reachable.
    - It ignores resource constraints like store capacity beyond a simple
      check for 'have_..._analysis' or 'have_image'.
    - It ignores negative effects like camera becoming uncalibrated after
      taking an image, or a store becoming full after sampling.
    - It assumes a fixed cost for sequences of actions (e.g., navigate + sample + navigate + communicate).

    # Heuristic Initialization
    - The heuristic stores the set of goal facts from the task. Static facts
      are not explicitly used in this simplified calculation but are available
      via `task.static` if needed for a more complex heuristic.

    # Step-By-Step Thinking for Computing Heuristic
    For each goal fact that is not yet true in the current state:

    1.  **Identify the type of goal:** Is it `communicated_soil_data`,
        `communicated_rock_data`, or `communicated_image_data`?

    2.  **Estimate cost based on intermediate state:**
        *   **For `communicated_soil_data(?w)`:**
            *   If the state contains `have_soil_analysis(?r, ?w)` for any rover `?r`:
                Estimated cost = 2 actions (Navigate to lander-visible waypoint + Communicate).
            *   Otherwise (need to sample):
                Estimated cost = 4 actions (Navigate to sample waypoint + Sample + Navigate to lander-visible waypoint + Communicate).
                (This simplifies the sequence: move to sample, sample, move to lander, communicate).
        *   **For `communicated_rock_data(?w)`:**
            *   If the state contains `have_rock_analysis(?r, ?w)` for any rover `?r`:
                Estimated cost = 2 actions (Navigate to lander-visible waypoint + Communicate).
            *   Otherwise (need to sample):
                Estimated cost = 4 actions (Navigate to sample waypoint + Sample + Navigate to lander-visible waypoint + Communicate).
        *   **For `communicated_image_data(?o, ?m)`:**
            *   If the state contains `have_image(?r, ?o, ?m)` for any rover `?r`:
                Estimated cost = 2 actions (Navigate to lander-visible waypoint + Communicate).
            *   Otherwise (need to take image):
                Estimated cost = 5 actions (Navigate to calibration waypoint + Calibrate + Navigate to image waypoint + Take Image + Navigate to lander-visible waypoint + Communicate).
                (This simplifies the sequence: move to cal, calibrate, move to img, take image, move to lander, communicate).

    3.  **Sum costs:** The total heuristic value is the sum of the estimated costs for all uncommunicated goal facts.

    This heuristic is non-admissible but aims to guide the search towards states where intermediate steps (like having samples or images) are achieved, and then towards communication.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by storing the goal conditions.
        """
        self.goals = task.goals

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state
        total_cost = 0

        # Pre-parse state facts for quicker lookup/matching
        # Convert frozenset to list for easier iteration and access
        parsed_state = [get_parts(fact) for fact in state]

        for goal in self.goals:
            # If the goal is already achieved, it contributes 0 to the heuristic.
            if goal in state:
                continue

            # Parse the goal fact
            goal_parts = get_parts(goal)
            if not goal_parts: # Skip malformed goals
                continue

            predicate = goal_parts[0]

            if predicate == "communicated_soil_data":
                # Goal: (communicated_soil_data ?w)
                if len(goal_parts) != 2: continue # Malformed goal

                waypoint = goal_parts[1]
                # Check if (have_soil_analysis ?r waypoint) exists for any rover ?r
                have_sample = any(
                    parts[0] == "have_soil_analysis" and len(parts) == 3 and parts[2] == waypoint
                    for parts in parsed_state
                )

                if have_sample:
                    # Need to navigate to lander + communicate
                    total_cost += 2
                else:
                    # Need to navigate to sample + sample + navigate to lander + communicate
                    total_cost += 4

            elif predicate == "communicated_rock_data":
                # Goal: (communicated_rock_data ?w)
                if len(goal_parts) != 2: continue # Malformed goal

                waypoint = goal_parts[1]
                # Check if (have_rock_analysis ?r waypoint) exists for any rover ?r
                have_sample = any(
                    parts[0] == "have_rock_analysis" and len(parts) == 3 and parts[2] == waypoint
                    for parts in parsed_state
                )

                if have_sample:
                    # Need to navigate to lander + communicate
                    total_cost += 2
                else:
                    # Need to navigate to sample + sample + navigate to lander + communicate
                    total_cost += 4

            elif predicate == "communicated_image_data":
                # Goal: (communicated_image_data ?o ?m)
                if len(goal_parts) != 3: continue # Malformed goal

                objective = goal_parts[1]
                mode = goal_parts[2]
                # Check if (have_image ?r objective mode) exists for any rover ?r
                have_image = any(
                    parts[0] == "have_image" and len(parts) == 4 and parts[2] == objective and parts[3] == mode
                    for parts in parsed_state
                )

                if have_image:
                    # Need to navigate to lander + communicate
                    total_cost += 2
                else:
                    # Need to navigate to cal + calibrate + navigate to img + take image + navigate to lander + communicate
                    total_cost += 5 # Simplified cost estimate

            # Ignore other goal types if any exist (based on domain, only these three)

        return total_cost
