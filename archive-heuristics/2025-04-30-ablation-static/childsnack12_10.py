# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic class for standalone testing if needed
# This is only for testing purposes if the actual base class is not provided
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static
            self.initial_state = task.initial_state # Access initial state if needed

        def __call__(self, node):
            raise NotImplementedError


from fnmatch import fnmatch

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         # Handle potential non-fact strings if necessary, though state should only contain facts
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
    # Ensure fact has at least as many parts as args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children.
    It sums up the estimated costs for making needed sandwiches, putting them on trays,
    moving trays to children's locations, and finally serving the children.

    # Assumptions
    - Ingredients (bread and content) are assumed to be sufficient if they exist in the initial state
      to make any required sandwich type (gluten-free or regular). The heuristic only counts the 'make' action
      if the *number* of available made sandwiches is insufficient to meet the demand.
    - Sufficient 'notexist' sandwich objects are available to be made.
    - Trays can hold multiple sandwiches.
    - A single tray moved to a location can potentially serve all children waiting at that location.
    - Trays needed at children's locations are assumed to come from the kitchen if available.

    # Heuristic Initialization
    - Identify which children are allergic to gluten from the initial state. This information
      is needed to determine the type of sandwich required for each child.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic is calculated as the sum of estimated costs for four main stages:
    1. Making sandwiches that are needed but not yet made.
    2. Putting made sandwiches onto trays.
    3. Moving trays to the locations where children are waiting.
    4. Serving the children.

    Here's the detailed breakdown:

    1.  **Count Unserved Children and Their Needs:**
        - Iterate through the current state to find all children who are `waiting` but not yet `served`.
        - Count the total number of unserved children (`N_unserved`).
        - Based on the allergic status identified during initialization, count how many unserved children need gluten-free sandwiches (`N_gf_needed`) and how many need regular sandwiches (`N_reg_needed`). Note that `N_unserved = N_gf_needed + N_reg_needed`.
        - Identify the distinct places where unserved children are waiting.

    2.  **Estimate Cost for Making Sandwiches:**
        - Count the number of gluten-free and regular sandwiches that are currently made (either `at_kitchen_sandwich` or `ontray` anywhere). Let these be `N_gf_available_made` and `N_reg_available_made`.
        - The number of gluten-free sandwiches that still need to be made is `gf_to_make = max(0, N_gf_needed - N_gf_available_made)`.
        - The number of regular sandwiches that still need to be made is `reg_to_make = max(0, N_reg_needed - N_reg_available_made)`.
        - The estimated cost for making sandwiches is `gf_to_make + reg_to_make` (each `make` action creates one sandwich).

    3.  **Estimate Cost for Putting Sandwiches on Trays:**
        - Count the number of gluten-free and regular sandwiches that are currently `ontray` anywhere. Let these be `N_gf_ontray_anywhere` and `N_reg_ontray_anywhere`.
        - The total number of sandwiches that need to be put on trays is the total number of needed sandwiches (`N_unserved`) minus those already on trays.
        - The estimated cost for putting sandwiches on trays is `max(0, N_unserved - (N_gf_ontray_anywhere + N_reg_ontray_anywhere))` (each `put_on_tray` action puts one sandwich).

    4.  **Estimate Cost for Moving Trays:**
        - Identify the distinct places where unserved children are waiting (`places_with_unserved`).
        - Identify the distinct places where trays are currently located (excluding the kitchen, as trays at the kitchen are the source for moves, not destinations that already have a tray).
        - The estimated cost for moving trays is the count of places in `places_with_unserved` that do *not* currently have a tray. (Assumes one tray move per location is sufficient if a tray is needed and not present).

    5.  **Estimate Cost for Serving:**
        - Each unserved child requires one `serve` action.
        - The estimated cost for serving is `N_unserved`.

    6.  **Total Heuristic Value:**
        - Sum the estimated costs from steps 2, 3, 4, and 5.
        - `h = make_cost + put_on_tray_cost + move_tray_cost + serve_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static-like information.
        """
        self.goals = task.goals
        # Static facts are not strictly needed for this heuristic's calculation logic,
        # but the task object is required to access initial_state and goals.
        # self.static = task.static

        # Extract allergic status from the initial state
        self.allergic_children = set()
        self.not_allergic_children = set()
        for fact in task.initial_state:
            parts = get_parts(fact)
            if len(parts) == 2:
                if parts[0] == 'allergic_gluten':
                    self.allergic_children.add(parts[1])
                elif parts[0] == 'not_allergic_gluten':
                    self.not_allergic_children.add(parts[1])
            # Note: allergic status is effectively static, extracted from initial state.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Count Unserved Children and Their Needs
        unserved_children = set()
        places_with_unserved = set()
        served_children = set()

        # First, find all served children
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'served' and len(parts) == 2:
                served_children.add(parts[1])

        # Then, find all waiting children and identify unserved ones
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'waiting' and len(parts) == 3:
                child, place = parts[1], parts[2]
                if child not in served_children:
                    unserved_children.add(child)
                    places_with_unserved.add(place)

        N_unserved = len(unserved_children)

        # If no children are unserved, the goal is reached.
        if N_unserved == 0:
            return 0

        N_gf_needed = 0
        N_reg_needed = 0
        for child in unserved_children:
            if child in self.allergic_children:
                N_gf_needed += 1
            elif child in self.not_allergic_children:
                N_reg_needed += 1
            # Assume all children are either allergic or not_allergic as per domain

        # 2. Estimate Cost for Making Sandwiches
        N_gf_available_made = 0
        N_reg_available_made = 0
        sandwich_is_gf = {} # Map sandwich object to its gluten status

        # Determine gluten status and made status of sandwiches present in state
        made_sandwiches = set()
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == 'no_gluten_sandwich' and len(parts) == 2:
                 sandwich_is_gf[parts[1]] = True

            if parts[0] == 'at_kitchen_sandwich' and len(parts) == 2:
                made_sandwiches.add(parts[1])
            elif parts[0] == 'ontray' and len(parts) == 3:
                 made_sandwiches.add(parts[1])

        for s in made_sandwiches:
             if sandwich_is_gf.get(s, False):
                 N_gf_available_made += 1
             else:
                 N_reg_available_made += 1

        gf_to_make = max(0, N_gf_needed - N_gf_available_made)
        reg_to_make = max(0, N_reg_needed - N_reg_available_made)
        make_cost = gf_to_make + reg_to_make

        # 3. Estimate Cost for Putting Sandwiches on Trays
        N_gf_ontray_anywhere = 0
        N_reg_ontray_anywhere = 0
        sandwiches_ontray = set()

        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'ontray' and len(parts) == 3:
                sandwiches_ontray.add(parts[1])

        for s in sandwiches_ontray:
             if sandwich_is_gf.get(s, False):
                 N_gf_ontray_anywhere += 1
             else:
                 N_reg_ontray_anywhere += 1

        put_on_tray_cost = max(0, N_unserved - (N_gf_ontray_anywhere + N_reg_ontray_anywhere))

        # 4. Estimate Cost for Moving Trays
        places_with_trays_at_location = set()
        for fact in state:
            parts = get_parts(fact)
            # Check for facts like (at tray1 table1)
            if parts and parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('tray') and parts[2] != 'kitchen':
                 places_with_trays_at_location.add(parts[2])

        move_tray_cost = 0
        for place in places_with_unserved:
            if place not in places_with_trays_at_location:
                move_tray_cost += 1 # Need to move a tray to this location

        # 5. Estimate Cost for Serving
        serve_cost = N_unserved

        # Total Heuristic Value
        total_cost = make_cost + put_on_tray_cost + move_tray_cost + serve_cost

        return total_cost
