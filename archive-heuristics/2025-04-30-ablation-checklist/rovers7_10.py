from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by whitespace
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding pattern argument
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to satisfy the goal
    conditions. It counts the necessary data collection (sampling or imaging),
    calibration (for images), and communication actions for each unmet goal.
    It also adds a simplified cost for necessary travel types (to sample sites,
    image sites, calibration sites, and communication sites), without
    considering specific rover locations or path distances.

    # Assumptions
    - The heuristic assumes that a suitable rover (equipped for the task)
      exists for every required action.
    - It ignores resource constraints like store capacity beyond the initial
      sampling action requirement.
    - It ignores specific rover assignments and capabilities beyond equipment
      type (soil, rock, imaging).
    - Camera calibration is assumed to be needed for each image task unless
      the specific image data is already held.
    - Travel cost is estimated by adding a fixed cost (1) for each *type*
      of location that needs to be visited across all unmet goals (sample,
      image, calibrate, communicate), if that type of task is required.

    # Heuristic Initialization
    - Stores the goal conditions.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. Initialize heuristic value `h = 0`.
    2. Identify which goal conditions are not met in the current state.
    3. Determine which data (soil samples, rock samples, images) are currently
       held by any rover.
    4. Initialize boolean flags to track if travel to specific types of locations
       is needed: `needs_soil_sample_travel`, `needs_rock_sample_travel`,
       `needs_image_take_travel`, `needs_calibration_travel`, `needs_comm_travel`.
    5. Iterate through each unmet goal condition:
       - If the goal is `(communicated_soil_data ?w)`:
         - Add 1 to `h` for the `communicate_soil_data` action.
         - Check if `(have_soil_analysis ?r ?w)` is true for any rover `?r`.
         - If not, add 1 to `h` for the `sample_soil` action, and set
           `needs_soil_sample_travel = True`.
         - Set `needs_comm_travel = True`.
       - If the goal is `(communicated_rock_data ?w)`:
         - Add 1 to `h` for the `communicate_rock_data` action.
         - Check if `(have_rock_analysis ?r ?w)` is true for any rover `?r`.
         - If not, add 1 to `h` for the `sample_rock` action, and set
           `needs_rock_sample_travel = True`.
         - Set `needs_comm_travel = True`.
       - If the goal is `(communicated_image_data ?o ?m)`:
         - Add 1 to `h` for the `communicate_image_data` action.
         - Check if `(have_image ?r ?o ?m)` is true for any rover `?r`.
         - If not, add 1 to `h` for the `take_image` action, add 1 to `h` for
           the `calibrate` action, and set `needs_image_take_travel = True`
           and `needs_calibration_travel = True`.
         - Set `needs_comm_travel = True`.
    6. Add travel costs based on the flags: Add 1 to `h` for each flag that is True.
    7. Return the total `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by storing goal conditions."""
        self.goals = task.goals

    def __call__(self, node):
        """Compute the heuristic estimate for the given state."""
        state = node.state
        h = 0

        # --- Step 3: Determine currently held data ---
        # Use sets for efficient lookup
        soil_held = set()
        rock_held = set()
        images_held = set() # Stores tuples (objective, mode)

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "have_soil_analysis" and len(parts) == 3:
                # Fact is (have_soil_analysis ?r ?w)
                waypoint = parts[2]
                soil_held.add(waypoint)
            elif predicate == "have_rock_analysis" and len(parts) == 3:
                # Fact is (have_rock_analysis ?r ?w)
                waypoint = parts[2]
                rock_held.add(waypoint)
            elif predicate == "have_image" and len(parts) == 4:
                # Fact is (have_image ?r ?o ?m)
                objective = parts[2]
                mode = parts[3]
                images_held.add((objective, mode))

        # --- Step 4: Initialize travel flags ---
        needs_soil_sample_travel = False
        needs_rock_sample_travel = False
        needs_image_take_travel = False
        needs_calibration_travel = False
        needs_comm_travel = False

        # --- Step 5: Iterate through unmet goal conditions ---
        for goal_fact in self.goals:
            if goal_fact in state:
                continue # Goal already met

            parts = get_parts(goal_fact)
            if not parts: continue # Skip malformed goals

            predicate = parts[0]

            if predicate == "communicated_soil_data" and len(parts) == 2:
                # Goal is (communicated_soil_data ?w)
                waypoint = parts[1]
                h += 1 # Cost for communicate action

                # Check if soil data from this waypoint is held by any rover
                if waypoint not in soil_held:
                    h += 1 # Cost for sample action
                    needs_soil_sample_travel = True # Need to travel to sample site

                needs_comm_travel = True # Need to travel to communication site

            elif predicate == "communicated_rock_data" and len(parts) == 2:
                # Goal is (communicated_rock_data ?w)
                waypoint = parts[1]
                h += 1 # Cost for communicate action

                # Check if rock data from this waypoint is held by any rover
                if waypoint not in rock_held:
                    h += 1 # Cost for sample action
                    needs_rock_sample_travel = True # Need to travel to sample site

                needs_comm_travel = True # Need to travel to communication site

            elif predicate == "communicated_image_data" and len(parts) == 3:
                # Goal is (communicated_image_data ?o ?m)
                objective = parts[1]
                mode = parts[2]
                h += 1 # Cost for communicate action

                # Check if this specific image is held by any rover
                if (objective, mode) not in images_held:
                    h += 1 # Cost for take_image action
                    h += 1 # Cost for calibrate action (calibration is consumed by take_image)
                    needs_image_take_travel = True # Need to travel to image site
                    needs_calibration_travel = True # Need to travel to calibration site

                needs_comm_travel = True # Need to travel to communication site

            # Add checks for other potential goal predicates if the domain changes,
            # though the provided domain only has these three communicated_data goals.

        # --- Step 6: Add travel costs based on flags ---
        if needs_soil_sample_travel:
            h += 1
        if needs_rock_sample_travel:
            h += 1
        if needs_image_take_travel:
            h += 1
        if needs_calibration_travel:
            h += 1
        if needs_comm_travel:
            h += 1

        # --- Step 7: Return the total heuristic value ---
        return h

