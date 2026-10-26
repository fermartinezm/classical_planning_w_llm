from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or non-string input defensively
    if not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by spaces
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children.
    It sums the estimated costs for:
    1. Serving each unserved child.
    2. Making necessary sandwiches (considering gluten requirements).
    3. Putting kitchen-based sandwiches onto trays.
    4. Moving trays to locations where unserved children are waiting.

    # Assumptions
    - Sufficient bread, content, and 'notexist' sandwich objects are available
      to make all required sandwiches if the problem is solvable.
    - Trays have infinite capacity for sandwiches.
    - A single 'move_tray' action is sufficient to move a tray between any two places.
    - One tray is sufficient per location where children are waiting.

    # Heuristic Initialization
    - Extracts static facts like which children are allergic to gluten,
      and which bread/content portions are gluten-free.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Identify Unserved Children and Waiting Places:**
        - Count the total number of children who are currently waiting and not yet served (`N_unserved`).
        - Identify the set of unique places where these unserved children are waiting (`waiting_places`).
        - If `N_unserved` is 0, the goal is reached, heuristic is 0.
        - `Cost_serve = N_unserved`.

    2.  **Identify Allergic Unserved Children:**
        - Count the number of unserved children who are allergic to gluten (`N_allergic_unserved`) using static facts extracted during initialization.

    3.  **Count Available Sandwiches:**
        - Count sandwiches currently in the kitchen (`CurrentAtKitchenSandwich`).
        - Count sandwiches currently on trays (`CurrentOntraySandwiches`).
        - Count total available sandwiches (`CurrentAvailableTotal`).
        - Count available gluten-free sandwiches (at kitchen or on trays) (`TotalAvailableGF`) by checking the `no_gluten_sandwich` predicate in the current state.

    4.  **Calculate Sandwiches to Make:**
        - Determine the total number of sandwiches that still need to be made to satisfy all unserved children (`SandwichesToMake`). This is the deficit between unserved children and currently available sandwiches.
        - Determine the number of gluten-free sandwiches that must be made (`MustMakeGF`). This is the deficit between allergic unserved children and available GF sandwiches.
        - Determine the number of regular sandwiches that must be made (`MustMakeRegular`). This is the remaining sandwiches to make after accounting for GF ones.
        - `Cost_make = MustMakeGF + MustMakeRegular`.

    5.  **Calculate Cost to Put on Tray:**
        - Sandwiches that are `at_kitchen_sandwich` need to be put on trays. This includes sandwiches already there and those just calculated as needing to be made.
        - `SandwichesNeedingPut = |CurrentAtKitchenSandwich| + SandwichesToMake`.
        - `Cost_put_on_tray = SandwichesNeedingPut`.

    6.  **Calculate Cost for Tray Movement:**
        - Identify all places where unserved children are waiting (`waiting_places`).
        - Identify all places where trays are currently located (`places_with_trays`).
        - Count the number of places in `waiting_places` that do not currently have a tray. Each such location requires at least one tray movement.
        - `Cost_move_tray = |{P | P in waiting_places and P not in places_with_trays}|`.

    7.  **Total Heuristic:**
        - Sum the costs from steps 1, 4, 5, and 6.
        - `h = Cost_serve + Cost_make + Cost_put_on_tray + Cost_move_tray`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts.
        """
        # The set of facts that must hold in goal states.
        self.goals = task.goals
        # Static facts are those that do not change during planning.
        static_facts = task.static

        # Extract static information about allergies and gluten-free ingredients
        self.allergic_children = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "allergic_gluten", "*")
        }
        # no_gluten_bread and no_gluten_content are also static
        self.no_gluten_bread = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_bread", "*")
        }
        self.no_gluten_content = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_content", "*")
        }
        # Note: no_gluten_sandwich is NOT static, it's an effect of make_sandwich_no_gluten

    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # --- Step 1 & 2: Identify Unserved Children and Waiting Places ---
        unserved_children = set()
        waiting_places = set()

        # Find all children who are served
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        # Identify unserved children and their places by checking waiting facts
        for fact in state:
            if match(fact, "waiting", "*", "*"):
                _, child, place = get_parts(fact)
                if child not in served_children:
                    unserved_children.add(child)
                    waiting_places.add(place)

        N_unserved = len(unserved_children)

        # If no children are unserved, the goal is reached
        if N_unserved == 0:
            return 0

        Cost_serve = N_unserved

        # Count allergic unserved children using the pre-calculated static set
        N_allergic_unserved = len(unserved_children.intersection(self.allergic_children))

        # --- Step 3: Count Available Sandwiches ---
        current_at_kitchen_sandwich = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_sandwich", "*")}
        current_ontray_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "ontray", "*", "*")}
        current_available_total = len(current_at_kitchen_sandwich.union(current_ontray_sandwiches))

        # Identify gluten-free sandwiches currently available by checking the state
        no_gluten_sandwich_facts = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}
        current_gf_at_kitchen_sandwich = current_at_kitchen_sandwich.intersection(no_gluten_sandwich_facts)
        current_gf_ontray_sandwiches = current_ontray_sandwiches.intersection(no_gluten_sandwich_facts)
        total_available_gf = len(current_gf_at_kitchen_sandwich.union(current_gf_ontray_sandwiches))

        # --- Step 4: Calculate Sandwiches to Make ---
        # Total sandwiches needed is at least N_unserved.
        # We have current_available_total. The deficit must be made.
        sandwiches_to_make = max(0, N_unserved - current_available_total)

        # We need N_allergic_unserved GF sandwiches.
        # We have total_available_gf. The deficit must be made as GF.
        must_make_gf = max(0, N_allergic_unserved - total_available_gf)

        # The remaining sandwiches to make can be regular.
        must_make_regular = max(0, sandwiches_to_make - must_make_gf)

        Cost_make = must_make_gf + must_make_regular

        # --- Step 5: Calculate Cost to Put on Tray ---
        # Sandwiches currently at kitchen need to be put on trays.
        # Sandwiches that are made will appear at the kitchen and also need to be put on trays.
        sandwiches_needing_put = len(current_at_kitchen_sandwich) + sandwiches_to_make
        Cost_put_on_tray = sandwiches_needing_put

        # --- Step 6: Calculate Cost for Tray Movement ---
        # Find all places where trays are located
        places_with_trays = set()
        for fact in state:
             if match(fact, "at", "*", "*"):
                 obj_name = get_parts(fact)[1]
                 # Assuming objects starting with 'tray' are trays
                 if obj_name.startswith("tray"):
                     place_name = get_parts(fact)[2]
                     places_with_trays.add(place_name)

        # Count waiting places that do not currently have a tray
        Cost_move_tray = 0
        for place in waiting_places:
            if place not in places_with_trays:
                 Cost_move_tray += 1

        # --- Step 7: Total Heuristic ---
        total_cost = Cost_serve + Cost_make + Cost_put_on_tray + Cost_move_tray

        return total_cost
