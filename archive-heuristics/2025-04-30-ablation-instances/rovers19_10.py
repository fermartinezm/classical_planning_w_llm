# Import necessary modules
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential malformed strings gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
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

# Define the domain-dependent heuristic class
class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goal
    conditions, focusing on the core tasks of sampling, imaging, and communication,
    and adding a fixed cost estimate for necessary navigation steps and resource management (dropping samples).

    # Assumptions
    - The heuristic counts the cost for each uncommunicated goal independently.
    - Navigation cost is estimated by adding a fixed value (2 for soil/rock, 3 for image)
      if the required data (sample/image) is not yet acquired, representing moves
      to the acquisition location(s) and then to a communication point. If data is
      acquired, a fixed cost (1) is added for navigation to a communication point.
    - Calibration is assumed to be needed before taking an image if the image is
      not already acquired.
    - Dropping a sample is assumed to be needed if a rover equipped for the required
      task (soil/rock) has a full store and the sample hasn't been acquired yet.
      This check is simplified: it adds the cost if *any* equipped rover has a full store.
    - Ignores specific rover assignment for tasks.

    # Heuristic Initialization
    - Extracts goal conditions.
    - Identifies lander locations and determines potential communication waypoints
      (waypoints visible from lander locations) from static facts.
    - Identifies which rovers are equipped for soil and rock analysis and maps stores to their owning rovers from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize total heuristic cost to 0.
    2. Identify the set of facts currently true in the state for quick lookup.
    3. Identify which stores are currently full.
    4. Iterate through each goal condition:
        a. If the goal is `(communicated_soil_data ?w)` and it is not yet true in the state:
            i. Add 1 to cost for the `communicate_soil_data` action.
            ii. Check if `(have_soil_analysis ?r ?w)` is true for any rover `?r`.
            iii. If no rover has the soil analysis for `?w`:
                - Add 1 to cost for the `sample_soil` action.
                - Add 2 to cost for estimated navigation (to `?w` for sampling + to a communication point).
                - Check if any rover equipped for soil analysis has a full store. If yes, add 1 to cost for the `drop` action.
            iv. If a rover *does* have the soil analysis for `?w`:
                - Add 1 to cost for estimated navigation (to a communication point).
        b. If the goal is `(communicated_rock_data ?w)` and it is not yet true in the state:
            i. Add 1 to cost for the `communicate_rock_data` action.
            ii. Check if `(have_rock_analysis ?r ?w)` is true for any rover `?r`.
            iii. If no rover has the rock analysis for `?w`:
                - Add 1 to cost for the `sample_rock` action.
                - Add 2 to cost for estimated navigation (to `?w` for sampling + to a communication point).
                - Check if any rover equipped for rock analysis has a full store. If yes, add 1 to cost for the `drop` action.
            iv. If a rover *does* have the rock analysis for `?w`:
                - Add 1 to cost for estimated navigation (to a communication point).
        c. If the goal is `(communicated_image_data ?o ?m)` and it is not yet true in the state:
            i. Add 1 to cost for the `communicate_image_data` action.
            ii. Check if `(have_image ?r ?o ?m)` is true for any rover `?r`.
            iii. If no rover has the image for `?o` and `?m`:
                - Add 1 to cost for the `take_image` action.
                - Add 1 to cost for the `calibrate` action (assuming calibration is needed before taking the image).
                - Add 3 to cost for estimated navigation (to image waypoint + to calibration waypoint + to communication point).
            iv. If a rover *does* have the image for `?o` and `?m`:
                - Add 1 to cost for estimated navigation (to a communication point).
    5. Return the total accumulated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals  # Goal conditions (frozenset of strings).
        static_facts = task.static  # Static facts (frozenset of strings).

        self.lander_locations = set()
        self.comm_waypoints = set()
        self.equipped_soil_rovers = set()
        self.equipped_rock_rovers = set()
        self.store_owner = {} # store_name -> rover_name

        # First pass: Find lander locations
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == "at_lander" and len(parts) == 3:
                # (at_lander ?l - lander ?y - waypoint)
                self.lander_locations.add(parts[2])

        # Second pass: Find communication waypoints (visible from lander locations)
        for fact in static_facts:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] == "visible" and len(parts) == 3:
                 w1, w2 = parts[1], parts[2]
                 # A waypoint w1 is a communication waypoint if it's visible from a lander location w2
                 if w2 in self.lander_locations:
                     self.comm_waypoints.add(w1)

        # Third pass: Find equipped rovers and store owners
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == "equipped_for_soil_analysis" and len(parts) == 2:
                 self.equipped_soil_rovers.add(parts[1])
            elif predicate == "equipped_for_rock_analysis" and len(parts) == 2:
                 self.equipped_rock_rovers.add(parts[1])
            elif predicate == "store_of" and len(parts) == 3:
                 self.store_owner[parts[1]] = parts[2] # store -> rover


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of strings).
        current_facts = set(state) # Convert to set for faster lookups

        total_cost = 0  # Initialize action cost counter.

        # Identify current status from state for quick lookup
        have_soil_analysis = set() # (rover, waypoint)
        have_rock_analysis = set() # (rover, waypoint)
        have_image = set()         # (rover, objective, mode)
        communicated_soil_data = set() # waypoint
        communicated_rock_data = set() # waypoint
        communicated_image_data = set() # (objective, mode)
        full_stores = set() # store name

        for fact in current_facts:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "have_soil_analysis" and len(parts) == 3:
                have_soil_analysis.add((parts[1], parts[2]))
            elif predicate == "have_rock_analysis" and len(parts) == 3:
                have_rock_analysis.add((parts[1], parts[2]))
            elif predicate == "have_image" and len(parts) == 4:
                have_image.add((parts[1], parts[2], parts[3]))
            elif predicate == "communicated_soil_data" and len(parts) == 2:
                communicated_soil_data.add(parts[1])
            elif predicate == "communicated_rock_data" and len(parts) == 2:
                communicated_rock_data.add(parts[1])
            elif predicate == "communicated_image_data" and len(parts) == 3:
                communicated_image_data.add((parts[1], parts[2]))
            elif predicate == "full" and len(parts) == 2:
                 full_stores.add(parts[1])

        # Check if any equipped rover has a full store (simplified check)
        equipped_soil_rover_has_full_store = any(
            self.store_owner.get(store) in self.equipped_soil_rovers
            for store in full_stores
        )
        equipped_rock_rover_has_full_store = any(
            self.store_owner.get(store) in self.equipped_rock_rovers
            for store in full_stores
        )

        # Process goals
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue

            predicate = parts[0]

            if predicate == "communicated_soil_data" and len(parts) == 2:
                waypoint = parts[1]
                if waypoint not in communicated_soil_data:
                    # Need to communicate soil data for this waypoint
                    total_cost += 1 # Cost for communicate_soil_data action

                    # Check if soil analysis is available for this waypoint
                    analysis_available = any(w == waypoint for r, w in have_soil_analysis)

                    if not analysis_available:
                        # Need to sample soil
                        total_cost += 1 # Cost for sample_soil action
                        total_cost += 2 # Estimated navigation: to sample waypoint + to comm waypoint
                        # Check if drop is needed before sampling (simplified)
                        if equipped_soil_rover_has_full_store:
                             total_cost += 1 # Cost for drop action
                    else:
                        # Analysis exists, only need to communicate
                        total_cost += 1 # Estimated navigation: to comm waypoint

            elif predicate == "communicated_rock_data" and len(parts) == 2:
                waypoint = parts[1]
                if waypoint not in communicated_rock_data:
                    # Need to communicate rock data for this waypoint
                    total_cost += 1 # Cost for communicate_rock_data action

                    # Check if rock analysis is available for this waypoint
                    analysis_available = any(w == waypoint for r, w in have_rock_analysis)

                    if not analysis_available:
                        # Need to sample rock
                        total_cost += 1 # Cost for sample_rock action
                        total_cost += 2 # Estimated navigation: to sample waypoint + to comm waypoint
                        # Check if drop is needed before sampling (simplified)
                        if equipped_rock_rover_has_full_store:
                             total_cost += 1 # Cost for drop action
                    else:
                        # Analysis exists, only need to communicate
                        total_cost += 1 # Estimated navigation: to comm waypoint

            elif predicate == "communicated_image_data" and len(parts) == 3:
                objective, mode = parts[1], parts[2]
                if (objective, mode) not in communicated_image_data:
                    # Need to communicate image data for this objective and mode
                    total_cost += 1 # Cost for communicate_image_data action

                    # Check if image is available for this objective and mode
                    image_available = any(o == objective and m == mode for r, o, m in have_image)

                    if not image_available:
                        # Need to take image
                        total_cost += 1 # Cost for take_image action
                        total_cost += 1 # Cost for calibrate action (assuming needed before taking image)
                        total_cost += 3 # Estimated navigation: to image waypoint + to cal waypoint + to comm waypoint
                    else:
                        # Image exists, only need to communicate
                        total_cost += 1 # Estimated navigation: to comm waypoint

        return total_cost
