from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal conditions.
    It counts the number of unachieved communication goals (soil, rock, image) and adds
    an estimated cost for each, based on whether the prerequisite data/image has already
    been collected/taken. It simplifies navigation costs and ignores resource conflicts.

    # Assumptions
    - Each unachieved communication goal requires a fixed number of actions, regardless
      of the specific rover, location, or other state details (like store capacity or
      exact navigation path).
    - Navigation between any two relevant points (sample location, objective visibility,
      calibration target visibility, lander visibility) is estimated as a single action.
    - Calibration is assumed to be needed for an image goal if the image hasn't been taken yet.
    - Store management (dropping samples) is largely ignored in the cost estimation.
    - Any equipped/capable rover can perform the necessary tasks for any goal.
    - The costs are estimated based on a simplified sequence of actions:
        - Soil/Rock (not collected): Nav to sample + Sample + Nav to lander + Communicate = 4 actions.
        - Soil/Rock (collected): Nav to lander + Communicate = 2 actions.
        - Image (not taken): Nav to obj + Take Image + Nav to cal target + Calibrate + Nav to lander + Communicate = 6 actions.
        - Image (taken): Nav to lander + Communicate = 2 actions.

    # Heuristic Initialization
    - Extracts the set of all rovers from static facts (assuming rovers relevant to goals
      will appear in static facts like equipped_for_* or on_board).
    - Stores the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is the sum of estimated costs for each unachieved goal fact.

    For each goal fact `g` present in `self.goals` but not in the current `state`:
    1.  Parse the goal fact `g` to identify its predicate and parameters.
    2.  If the predicate is `communicated_soil_data` with parameter `?w`:
        -   Check if the fact `(have_soil_analysis ?r ?w)` exists in the current state for *any* rover `?r` in the domain.
        -   If it exists (sample collected), add 2 to the heuristic (estimated cost for navigation to lander and communication).
        -   If it does not exist (sample not collected), add 4 to the heuristic (estimated cost for navigation to sample, sampling, navigation to lander, and communication).
    3.  If the predicate is `communicated_rock_data` with parameter `?w`:
        -   Follow the same logic as `communicated_soil_data`, checking for `(have_rock_analysis ?r ?w)`. Add 2 if collected, 4 if not.
    4.  If the predicate is `communicated_image_data` with parameters `?o` and `?m`:
        -   Check if the fact `(have_image ?r ?o ?m)` exists in the current state for *any* rover `?r` in the domain.
        -   If it exists (image taken), add 2 to the heuristic (estimated cost for navigation to lander and communication).
        -   If it does not exist (image not taken), add 6 to the heuristic (estimated cost for navigation to objective, taking image, navigation to calibration target, calibrating, navigation to lander, and communication).
    5.  Sum the costs added for all unachieved goal facts. This sum is the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract the set of all rovers from static facts.
        # This assumes any rover relevant to the problem appears in at least one
        # static fact like equipped_for_*, on_board, or store_of.
        self.rovers = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate in ['equipped_for_soil_analysis', 'equipped_for_rock_analysis', 'equipped_for_imaging']:
                if len(parts) > 1: self.rovers.add(parts[1])
            elif predicate == 'on_board':
                 if len(parts) > 2: self.rovers.add(parts[2]) # parts[1] is camera, parts[2] is rover
            elif predicate == 'store_of':
                 if len(parts) > 2: self.rovers.add(parts[2]) # parts[1] is store, parts[2] is rover

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0

        for goal in self.goals:
            if goal not in state:
                parts = get_parts(goal)
                if not parts: continue

                predicate = parts[0]

                if predicate == 'communicated_soil_data':
                    if len(parts) > 1:
                        waypoint_w = parts[1]
                        # Check if sample is collected by *any* rover
                        sample_collected = any(f'(have_soil_analysis {r} {waypoint_w})' in state for r in self.rovers)
                        if not sample_collected:
                            # Need to sample and communicate
                            # Estimated cost: Nav to sample (1) + Sample (1) + Nav to lander (1) + Communicate (1) = 4
                            h += 4
                        else:
                            # Sample collected, just need to communicate
                            # Estimated cost: Nav to lander (1) + Communicate (1) = 2
                            h += 2

                elif predicate == 'communicated_rock_data':
                    if len(parts) > 1:
                        waypoint_w = parts[1]
                        # Check if sample is collected by *any* rover
                        sample_collected = any(f'(have_rock_analysis {r} {waypoint_w})' in state for r in self.rovers)
                        if not sample_collected:
                            # Need to sample and communicate
                            # Estimated cost: Nav to sample (1) + Sample (1) + Nav to lander (1) + Communicate (1) = 4
                            h += 4
                        else:
                            # Sample collected, just need to communicate
                            # Estimated cost: Nav to lander (1) + Communicate (1) = 2
                            h += 2

                elif predicate == 'communicated_image_data':
                    if len(parts) > 2:
                        objective_o = parts[1]
                        mode_m = parts[2]
                        # Check if image is taken by *any* rover
                        image_taken = any(f'(have_image {r} {objective_o} {mode_m})' in state for r in self.rovers)
                        if not image_taken:
                            # Need to take image and communicate
                            # Estimated cost: Nav to obj (1) + Take Image (1) + Nav to cal target (1) + Calibrate (1) + Nav to lander (1) + Communicate (1) = 6
                            h += 6
                        else:
                            # Image taken, just need to communicate
                            # Estimated cost: Nav to lander (1) + Communicate (1) = 2
                            h += 2

        return h
