# from fnmatch import fnmatch # Not used
from heuristics.heuristic_base import Heuristic # Assuming this base class exists

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by whitespace
    # Check for valid parentheses structure
    if fact[0] != '(' or fact[-1] != ')':
         # Assuming valid PDDL facts as input for this context
         return []

    return fact[1:-1].split()

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children
    specified in the goal. It counts the number of unserved children, the number
    of sandwiches that need to be made, the number of sandwiches that need to be
    moved from the kitchen onto a tray, and the number of locations with unserved
    children that need a tray moved there.

    # Assumptions
    - Each unserved child requires one suitable sandwich to be served.
    - Making a sandwich, putting a sandwich on a tray, moving a tray to a needed location,
      and serving a child each cost 1 action.
    - There are enough bread, content, and sandwich objects available to make any needed sandwiches.
    - Tray capacity is sufficient, and trays can be reused.
    - Tray movement cost is simplified to 1 action per location that needs a tray.
    - The 'at' predicate in the state refers to the location of trays.
    - Children's allergy status and initial waiting places are static.

    # Heuristic Initialization
    - Extracts the set of children that need to be served from the task goals.
    - Extracts static information about children (allergy status, waiting place)
      from the task's static facts. Stores this info for the goal children.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:
    1.  Identify the set of children that are in the goal but not yet served (`n_unserved`).
        This count is the base cost, representing the minimum number of 'serve' actions required.
        If `n_unserved` is 0, the heuristic is 0.
    2.  Among the unserved children, count how many are allergic (`n_gf_needed`) and how many are not (`n_reg_needed`).
    3.  Count the number of existing gluten-free (`avail_gf_anywhere`) and regular (`avail_reg_anywhere`) sandwiches
        currently available (in the kitchen or on trays).
    4.  Calculate the number of new gluten-free (`n_gf_to_make`) and regular (`n_reg_to_make`) sandwiches
        that must be made to satisfy the needs of unserved children (needed minus available, minimum 0).
        Add `n_gf_to_make + n_reg_to_make` to the heuristic (cost of 'make_sandwich' actions).
    5.  Count the number of sandwiches currently in the kitchen (`sandwiches_kitchen`). These sandwiches,
        plus the ones that will be made (`n_gf_to_make + n_reg_to_make`), need to be put on a tray.
        Add `len(sandwiches_kitchen) + n_gf_to_make + n_reg_to_make` to the heuristic (cost of 'put_on_tray' actions).
    6.  Identify all locations where the unserved children are waiting (`waiting_places_unserved`).
    7.  Identify all locations where trays are currently present (`tray_locations`).
    8.  Count the number of waiting locations (`waiting_places_unserved`) that do not currently have a tray
        present (`tray_locations`). Add the size of this set (`n_move_tray_needed`) to the heuristic
        (cost of 'move_tray' actions).
    9.  The total heuristic value is the sum of the counts from steps 1, 4, 5, and 8.
        h = n_unserved + (n_gf_to_make + n_reg_to_make) + n_put_on_tray_needed + n_move_tray_needed
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about children.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract the set of children that are part of the goal
        self.goal_children = set()
        for goal in self.goals:
             parts = get_parts(goal)
             if parts and parts[0] == 'served' and len(parts) == 2:
                 self.goal_children.add(parts[1])

        # Extract static information for all children mentioned in static facts
        child_is_allergic = {}
        child_initial_place = {}

        for fact in self.static:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == 'waiting' and len(parts) == 3:
                 child, place = parts[1], parts[2]
                 child_initial_place[child] = place

            elif parts[0] == 'allergic_gluten' and len(parts) == 2:
                 child = parts[1]
                 child_is_allergic[child] = True

            elif parts[0] == 'not_allergic_gluten' and len(parts) == 2:
                 child = parts[1]
                 child_is_allergic[child] = False

        # Store info only for children who are in the goal
        self.child_info = {}
        for child in self.goal_children:
             # Default to not allergic if status not explicitly stated (shouldn't happen in valid PDDL)
             is_allergic = child_is_allergic.get(child, False)
             # Initial place is needed for 'waiting' predicate, which is static
             place = child_initial_place.get(child)
             # Only include goal children for whom we know the waiting place (should be all in valid problems)
             if place is not None:
                 self.child_info[child] = {'place': place, 'is_allergic': is_allergic}
             # else: A child is in the goal but has no static waiting/allergy info? Skip or handle error.
             # Assuming valid problem instances where goal children have static info.


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # --- Collect state information ---
        served_children = set()
        sandwiches_kitchen = set()
        sandwiches_ontray = set() # Just need the set of sandwiches on trays
        tray_locations = set() # Just need the set of places with trays
        sandwich_is_gf = set() # Set of GF sandwiches (wherever they are)

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == 'served' and len(parts) == 2:
                served_children.add(parts[1])
            elif parts[0] == 'at_kitchen_sandwich' and len(parts) == 2:
                sandwiches_kitchen.add(parts[1])
            elif parts[0] == 'ontray' and len(parts) == 3:
                sandwiches_ontray.add(parts[1])
            elif parts[0] == 'at' and len(parts) == 3:
                 # Assuming 'at' predicate with a place object is always a tray location based on domain
                 tray_locations.add(parts[2]) # Add the place, not the tray name
            elif parts[0] == 'no_gluten_sandwich' and len(parts) == 2:
                sandwich_is_gf.add(parts[1])

        # --- Step 1 & 2: Count unserved children and needed sandwich types ---
        n_unserved = 0
        n_gf_needed = 0
        n_reg_needed = 0
        waiting_places_unserved = set()

        # Iterate through the children who are in the goal
        for child in self.goal_children:
            # Check if this goal child is served in the current state
            if child not in served_children:
                n_unserved += 1
                # Get the static info for this unserved child
                info = self.child_info.get(child)
                if info: # Should always exist for goal children in valid problems
                    waiting_places_unserved.add(info['place'])
                    if info['is_allergic']:
                        n_gf_needed += 1
                    else:
                        n_reg_needed += 1
                # else: This child is in the goal but has no static waiting/allergy info? Skip or handle error.
                # Assuming valid problem instances where goal children have static info.


        # If no children are unserved, the goal is reached.
        if n_unserved == 0:
            return 0

        # --- Step 3: Count available sandwiches anywhere ---
        all_available_sandwiches = sandwiches_kitchen | sandwiches_ontray
        avail_gf_anywhere = len(sandwich_is_gf.intersection(all_available_sandwiches))
        avail_reg_anywhere = len(all_available_sandwiches - sandwich_is_gf)

        # --- Step 4: Calculate sandwiches to make ---
        n_gf_to_make = max(0, n_gf_needed - avail_gf_anywhere)
        n_reg_to_make = max(0, n_reg_needed - avail_reg_anywhere)

        # --- Step 5: Calculate sandwiches needing put_on_tray ---
        # These are sandwiches currently in the kitchen plus those that will be made
        n_put_on_tray_needed = len(sandwiches_kitchen) + n_gf_to_make + n_reg_to_make

        # --- Step 6, 7 & 8: Calculate places needing tray move ---
        places_needing_tray_move = waiting_places_unserved - tray_locations
        n_move_tray_needed = len(places_needing_tray_move)

        # --- Step 9: Total heuristic ---
        # Each unserved child needs a 'serve' action (accounted by n_unserved)
        # Each sandwich to make needs a 'make' action
        # Each sandwich in kitchen or to be made needs a 'put_on_tray' action
        # Each waiting place without a tray needs a 'move_tray' action
        # Note: This sums up required actions at different stages, not a strict plan length.
        # It's a non-admissible estimate.

        heuristic_value = n_unserved + n_gf_to_make + n_reg_to_make + n_put_on_tray_needed + n_move_tray_needed

        return heuristic_value
