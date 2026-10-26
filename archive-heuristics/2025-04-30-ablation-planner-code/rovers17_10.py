from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential whitespace issues and empty facts robustly
    fact = fact.strip()
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
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
    This heuristic estimates the number of actions required to achieve each uncommunicated goal fact (soil data, rock data, image data). It assigns a fixed cost based on whether the required data (sample or image) has already been collected/taken.

    # Assumptions
    - The heuristic assumes that for any uncommunicated goal, the necessary resources (equipped rover, camera, calibration target, sample location, lander visibility) exist and are reachable, as long as the problem is solvable.
    - It assigns fixed costs for different stages of achieving a goal (collecting data vs. communicating data), ignoring specific locations, distances, or resource availability details (like store fullness or camera calibration state beyond the 'have_image' fact).
    - Costs are estimated as:
        - Communicate data (if data is already collected): 2 actions (move to lander visibility + communicate).
        - Collect and Communicate soil/rock data (if sample exists but not collected): 5 actions (move to sample + sample + move to lander visibility + communicate, potentially including a drop).
        - Collect and Communicate image data (if image not taken): 6 actions (move to calibrate + calibrate + move to image location + take image + move to lander visibility + communicate).

    # Heuristic Initialization
    - The heuristic stores the set of goal facts.
    - It extracts and stores static information that might be useful, although the current simple version primarily relies on goal facts and dynamic 'have' facts. Static facts like initial sample locations could be used to verify if a sample goal is valid, but for simplicity, we assume valid goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. Iterate through each goal fact specified in the problem.
    3. For each goal fact:
        a. Check if the goal fact is already present in the current state. If yes, this goal is achieved, add 0 to the total cost, and proceed to the next goal.
        b. If the goal fact is not in the current state, determine its type (communicated_soil_data, communicated_rock_data, or communicated_image_data).
        c. If the goal is `(communicated_soil_data ?w)`:
            - Check if any rover currently has the soil analysis for that waypoint (`(have_soil_analysis ?r ?w)` is in the state for any rover `?r`).
            - If yes, the data is collected but not communicated. Estimate the remaining cost as 2 (move to lander visibility + communicate). Add 2 to the total cost.
            - If no, the data has not been collected. Estimate the cost to collect and communicate as 5 (move to sample + sample + move to lander visibility + communicate, including potential drop). Add 5 to the total cost.
        d. If the goal is `(communicated_rock_data ?w)`:
            - Check if any rover currently has the rock analysis for that waypoint (`(have_rock_analysis ?r ?w)` is in the state for any rover `?r`).
            - If yes, the data is collected but not communicated. Estimate the remaining cost as 2 (move to lander visibility + communicate). Add 2 to the total cost.
            - If no, the data has not been collected. Estimate the cost to collect and communicate as 5 (move to sample + sample + move to lander visibility + communicate, including potential drop). Add 5 to the total cost.
        e. If the goal is `(communicated_image_data ?o ?m)`:
            - Check if any rover currently has the required image (`(have_image ?r ?o ?m)` is in the state for any rover `?r`).
            - If yes, the image is taken but not communicated. Estimate the remaining cost as 2 (move to lander visibility + communicate). Add 2 to the total cost.
            - If no, the image has not been taken. Estimate the cost to take and communicate as 6 (move to calibrate + calibrate + move to image location + take image + move to lander visibility + communicate). Add 6 to the total cost.
    4. Return the total calculated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by storing goal conditions."""
        self.goals = task.goals
        # Although this simple heuristic doesn't strictly require pre-processing
        # static facts beyond the goals, the constructor is the place to do it
        # for more complex heuristics. Example:
        # self.lander_location = None
        # for fact in task.static:
        #     if match(fact, "at_lander", "*", "*"):
        #         self.lander_location = get_parts(fact)[2]
        #         break
        pass # No complex static fact processing needed for this version

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Convert state to a set for efficient lookup
        state_set = set(state)

        for goal in self.goals:
            # If the goal is already achieved, it costs 0.
            if goal in state_set:
                continue

            # Parse the goal fact
            parts = get_parts(goal)
            if not parts: # Should not happen with valid goals
                continue

            predicate = parts[0]

            if predicate == "communicated_soil_data":
                waypoint = parts[1]
                # Check if the soil analysis is already available
                have_soil = False
                # Iterate through state facts to find (have_soil_analysis ?r waypoint)
                for fact in state_set:
                    if match(fact, "have_soil_analysis", "*", waypoint):
                        have_soil = True
                        break

                if have_soil:
                    # Data collected, need to communicate
                    total_cost += 2 # Move to lander visibility + communicate
                else:
                    # Data not collected, need to sample and communicate
                    total_cost += 5 # Move to sample + sample + move to lander visibility + communicate (+ potential drop)

            elif predicate == "communicated_rock_data":
                waypoint = parts[1]
                # Check if the rock analysis is already available
                have_rock = False
                # Iterate through state facts to find (have_rock_analysis ?r waypoint)
                for fact in state_set:
                    if match(fact, "have_rock_analysis", "*", waypoint):
                        have_rock = True
                        break

                if have_rock:
                    # Data collected, need to communicate
                    total_cost += 2 # Move to lander visibility + communicate
                else:
                    # Data not collected, need to sample and communicate
                    total_cost += 5 # Move to sample + sample + move to lander visibility + communicate (+ potential drop)

            elif predicate == "communicated_image_data":
                objective = parts[1]
                mode = parts[2]
                # Check if the image is already available
                have_image = False
                # Iterate through state facts to find (have_image ?r objective mode)
                for fact in state_set:
                    if match(fact, "have_image", "*", objective, mode):
                        have_image = True
                        break

                if have_image:
                    # Image taken, need to communicate
                    total_cost += 2 # Move to lander visibility + communicate
                else:
                    # Image not taken, need to take image and communicate
                    total_cost += 6 # Move to calibrate + calibrate + move to image + take image + move to lander visibility + communicate

            # Add other goal types if any, though the domain only lists these communication goals.

        return total_cost
