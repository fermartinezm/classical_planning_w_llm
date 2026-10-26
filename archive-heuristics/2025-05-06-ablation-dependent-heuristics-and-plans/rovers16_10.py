from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # This should not happen with standard PDDL fact strings, but handle defensively.
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint2)".
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
    This heuristic estimates the number of actions required to achieve all
    uncommunicated goals (soil data, rock data, image data). It sums up
    estimated costs for each individual unachieved goal, simplifying complex
    interactions and resource constraints (like shared stores or cameras)
    for computational efficiency. It is designed for greedy best-first search
    and is not admissible.

    # Assumptions
    - The heuristic is non-admissible.
    - Navigation between any two required waypoints (sample location, image location,
      calibration location, communication location) is estimated as a fixed cost of 1 action.
      This ignores actual pathfinding distance but keeps computation fast.
    - Taking a sample (soil or rock) costs 1 action.
    - Dropping a sample costs 1 action, added only if all relevant equipped rovers' stores are full.
    - Calibrating a camera costs 1 action.
    - Taking an image costs 1 action.
    - Communicating data (soil, rock, or image) costs 1 action.
    - Each unachieved goal is treated somewhat independently, summing up minimum
      required steps to complete that specific goal from scratch if necessary.
    - For sampling goals, a drop action is assumed necessary only if *all* rovers
      equipped for that sample type currently have full stores.

    # Heuristic Initialization
    - Stores the set of goal conditions from the task.
    - Extracts relevant static facts from the task for quick lookups:
        - Rovers equipped for soil analysis.
        - Rovers equipped for rock analysis.
        - Mapping from rover to its store.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is the sum of estimated costs for each goal fact
    that is not yet true in the current state.

    For each unachieved goal `g`:

    1.  **If `g` is `(communicated_soil_data ?w)`:**
        -   Check if `(have_soil_analysis ?r ?w)` is true for any rover `r` in the current state.
        -   If yes (sample is already collected):
            -   Estimated cost = 1 (move to communication point) + 1 (communicate) = 2 actions.
        -   If no (sample needs to be collected):
            -   Estimated base cost = 1 (move to waypoint `w`) + 1 (sample soil) + 1 (move to communication point) + 1 (communicate) = 4 actions.
            -   Check if a `drop` action is likely needed: Find all rovers equipped for soil analysis. Find their corresponding stores using static facts. If *all* such stores are currently `full` in the state, add 1 action (for dropping a sample to free up a store).
            -   Total estimated cost = 4 or 5 actions.

    2.  **If `g` is `(communicated_rock_data ?w)`:**
        -   Check if `(have_rock_analysis ?r ?w)` is true for any rover `r` in the current state.
        -   If yes (sample is already collected):
            -   Estimated cost = 1 (move to communication point) + 1 (communicate) = 2 actions.
        -   If no (sample needs to be collected):
            -   Estimated base cost = 1 (move to waypoint `w`) + 1 (sample rock) + 1 (move to communication point) + 1 (communicate) = 4 actions.
            -   Check if a `drop` action is likely needed: Find all rovers equipped for rock analysis. Find their corresponding stores using static facts. If *all* such stores are currently `full` in the state, add 1 action (for dropping a sample).
            -   Total estimated cost = 4 or 5 actions.

    3.  **If `g` is `(communicated_image_data ?o ?m)`:**
        -   Check if `(have_image ?r ?o ?m)` is true for any rover `r` in the current state.
        -   If yes (image is already taken):
            -   Estimated cost = 1 (move to communication point) + 1 (communicate) = 2 actions.
        -   If no (image needs to be taken):
            -   Estimated cost = 1 (move to calibration waypoint) + 1 (calibrate camera) + 1 (move to image waypoint) + 1 (take image) + 1 (move to communication point) + 1 (communicate) = 6 actions.
            -   This assumes a calibration cycle is needed before taking the image, which is true because taking an image uncalibrates the camera. This cost is added regardless of the camera's current calibration status, simplifying the estimate.

    The total heuristic value is the sum of these estimated costs for all unachieved goals.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and relevant static facts.
        """
        self.goals = task.goals
        self.static = task.static

        # Pre-process static facts for quick lookups
        self.equipped_soil_rovers = {get_parts(fact)[1] for fact in self.static if match(fact, "equipped_for_soil_analysis", "*")}
        self.equipped_rock_rovers = {get_parts(fact)[1] for fact in self.static if match(fact, "equipped_for_rock_analysis", "*")}
        # Map rover name to its store name
        self.rover_stores = {get_parts(fact)[2]: get_parts(fact)[1] for fact in self.static if match(fact, "store_of", "*", "*")}


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach a goal state.
        """
        state = node.state
        h = 0

        # Iterate through all goal conditions
        for goal in self.goals:
            # If the goal is already achieved in the current state, it costs 0
            if goal in state:
                continue

            # Parse the goal fact
            parts = get_parts(goal)
            if not parts: # Handle potential parsing error
                 continue

            predicate = parts[0]

            if predicate == "communicated_soil_data":
                # Goal: (communicated_soil_data ?w)
                waypoint = parts[1]
                # Check if the soil sample from this waypoint is already collected by any rover
                sample_held = any(match(fact, "have_soil_analysis", "*", waypoint) for fact in state)

                if sample_held:
                    # Sample is held, just need to move to comm point and communicate
                    h += 2 # 1 (move_to_comm) + 1 (communicate)
                else:
                    # Need to sample, move to comm point, and communicate
                    h += 4 # 1 (move_to_w) + 1 (sample) + 1 (move_to_comm) + 1 (communicate)

                    # Check if a drop action is needed before sampling
                    # A drop is needed if all equipped soil rovers have full stores
                    all_equipped_stores_full = True
                    if self.equipped_soil_rovers: # Check if there are any equipped rovers
                        for rover in self.equipped_soil_rovers:
                            store = self.rover_stores.get(rover)
                            # Check if the store exists and is NOT empty in the current state
                            if store and f"(empty {store})" in state:
                                # Found at least one equipped rover with an empty store
                                all_equipped_stores_full = False
                                break
                        if all_equipped_stores_full:
                            h += 1 # Add cost for drop action

            elif predicate == "communicated_rock_data":
                # Goal: (communicated_rock_data ?w)
                waypoint = parts[1]
                # Check if the rock sample from this waypoint is already collected by any rover
                sample_held = any(match(fact, "have_rock_analysis", "*", waypoint) for fact in state)

                if sample_held:
                    # Sample is held, just need to move to comm point and communicate
                    h += 2 # 1 (move_to_comm) + 1 (communicate)
                else:
                    # Need to sample, move to comm point, and communicate
                    h += 4 # 1 (move_to_w) + 1 (sample) + 1 (move_to_comm) + 1 (communicate)

                    # Check if a drop action is needed before sampling
                    # A drop is needed if all equipped rock rovers have full stores
                    all_equipped_stores_full = True
                    if self.equipped_rock_rovers: # Check if there are any equipped rovers
                         for rover in self.equipped_rock_rovers:
                            store = self.rover_stores.get(rover)
                            # Check if the store exists and is NOT empty in the current state
                            if store and f"(empty {store})" in state:
                                # Found at least one equipped rover with an empty store
                                all_equipped_stores_full = False
                                break
                         if all_equipped_stores_full:
                            h += 1 # Add cost for drop action

            elif predicate == "communicated_image_data":
                # Goal: (communicated_image_data ?o ?m)
                objective, mode = parts[1], parts[2]
                # Check if the image is already taken by any rover
                image_held = any(match(fact, "have_image", "*", objective, mode) for fact in state)

                if image_held:
                    # Image is held, just need to move to comm point and communicate
                    h += 2 # 1 (move_to_comm) + 1 (communicate)
                else:
                    # Need to take image and communicate
                    # This requires calibration, moving to image location, taking image,
                    # moving to comm location, and communicating.
                    h += 6 # 1 (move_to_calib) + 1 (calibrate) + 1 (move_to_image) + 1 (take_image) + 1 (move_to_comm) + 1 (communicate)

            # Add checks for other potential goal predicates if the domain were extended
            # else:
            #     # Unknown goal type, maybe add a large penalty or ignore
            #     pass

        return h
