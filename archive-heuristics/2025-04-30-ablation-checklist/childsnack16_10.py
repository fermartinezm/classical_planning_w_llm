from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions from Logistics example, useful for parsing PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class ChildsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Childsnack domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting
    children. It counts the remaining tasks in a relaxed way:
    1. Serving each unserved child.
    2. Making sandwiches for children who need them and for whom no suitable
       sandwich is already made.
    3. Putting made sandwiches onto trays for children who need them on trays
       and for whom no suitable sandwich is already on a tray.
    4. Moving trays to locations where unserved children are waiting and no
       tray is currently present.

    # Assumptions (Relaxations)
    - Ignores resource constraints like the number of available bread/content
      portions or 'notexist' sandwich objects when counting 'make' actions.
    - Ignores resource constraints like the number of available trays when
      counting 'put_on_tray' or 'move_tray' actions.
    - Ignores the specific type of sandwich needed when counting 'put_on_tray'
      actions (assumes any made sandwich can eventually satisfy a need to be
      on a tray).
    - Ignores the specific type of sandwich on a tray when counting 'move_tray'
      actions (assumes any tray at a location is sufficient for that location's
      tray need).
    - Assumes actions can be performed in parallel if their prerequisites are
      met in the relaxed state.
    - Counts distinct action *types* needed per item/location rather than
      considering complex interactions or optimal sequencing.

    # Heuristic Initialization
    - Stores static facts (like allergy information) for quick lookup.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children who are waiting but not yet served.
    2. Count the total number of such unserved children (`N_unserved`). This is a lower bound on the number of `serve` actions needed. Add this to the total cost.
    3. Separate unserved children into gluten-allergic (`N_unserved_gf`) and non-allergic (`N_unserved_reg`).
    4. Count available sandwiches (both in the kitchen and on trays), distinguishing between gluten-free and regular.
    5. Calculate how many new gluten-free sandwiches need to be made: `max(0, N_unserved_gf - available_gf_made)`. Add this to the total cost (for `make_sandwich_no_gluten`).
    6. Calculate how many new regular sandwiches need to be made: `max(0, N_unserved_reg - available_reg_made)`. Add this to the total cost (for `make_sandwich`).
    7. Count sandwiches currently on trays (`N_ontray`).
    8. Calculate how many sandwiches still need to be put on trays: `max(0, N_unserved - N_ontray)`. Add this to the total cost (for `put_on_tray`).
    9. Identify all distinct locations where unserved children are waiting.
    10. Identify all distinct locations where trays are currently present.
    11. Count the number of locations from step 9 that are *not* in the set from step 10. This is a lower bound on the number of `move_tray` actions needed to get trays to required locations. Add this count to the total cost.
    12. Return the total accumulated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by storing static facts.
        """
        self.static = task.static # Static facts like allergy info

        # Pre-process static allergy info for faster lookup
        self._is_allergic = set()
        self._is_not_allergic = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "allergic_gluten":
                self._is_allergic.add(parts[1])
            elif parts[0] == "not_allergic_gluten":
                self._is_not_allergic.add(parts[1])


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach a goal state where all children are served.
        """
        state = node.state  # Current world state.

        cost = 0  # Initialize action cost counter.

        # --- Step 1, 2, 3: Count unserved children and their types ---
        unserved_children_gf = set()
        unserved_children_reg = set()
        locations_with_unserved = set()

        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        for fact in state:
            if match(fact, "waiting", "*", "*"):
                _, child, location = get_parts(fact)
                if child not in served_children:
                    locations_with_unserved.add(location)
                    if child in self._is_allergic:
                        unserved_children_gf.add(child)
                    elif child in self._is_not_allergic:
                        unserved_children_reg.add(child)
                    # Note: Children must be either allergic or not, based on domain.

        N_unserved_gf = len(unserved_children_gf)
        N_unserved_reg = len(unserved_children_reg)
        N_unserved = N_unserved_gf + N_unserved_reg

        # Cost for serving each child
        cost += N_unserved

        # --- Step 4, 5, 6: Count available sandwiches and needed 'make' actions ---
        available_gf_kitchen = 0
        available_reg_kitchen = 0
        available_gf_ontray = 0
        available_reg_ontray = 0

        # Track sandwiches by name to check gluten status
        sandwiches_in_kitchen = set()
        sandwiches_on_trays = set()
        gf_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}

        for fact in state:
            if match(fact, "at_kitchen_sandwich", "*"):
                sandwich = get_parts(fact)[1]
                sandwiches_in_kitchen.add(sandwich)
                if sandwich in gf_sandwiches:
                    available_gf_kitchen += 1
                else:
                    available_reg_kitchen += 1
            elif match(fact, "ontray", "*", "*"):
                sandwich = get_parts(fact)[1]
                sandwiches_on_trays.add(sandwich)
                if sandwich in gf_sandwiches:
                    available_gf_ontray += 1
                else:
                    available_reg_ontray += 1

        N_available_gf_made = available_gf_kitchen + available_gf_ontray
        N_available_reg_made = available_reg_kitchen + available_reg_ontray

        needed_to_make_gf = max(0, N_unserved_gf - N_available_gf_made)
        needed_to_make_reg = max(0, N_unserved_reg - N_available_reg_made)

        # Cost for making sandwiches
        cost += needed_to_make_gf + needed_to_make_reg

        # --- Step 7, 8: Count sandwiches on trays and needed 'put_on_tray' actions ---
        N_ontray = len(sandwiches_on_trays)

        # Number of sandwiches that need to end up on trays
        needed_put_on_tray = max(0, N_unserved - N_ontray)

        # Cost for putting sandwiches on trays
        cost += needed_put_on_tray

        # --- Step 9, 10, 11: Count tray moves needed ---
        locations_with_trays = {get_parts(fact)[2] for fact in state if match(fact, "at", "tray*", "*")}

        # Locations with unserved children that do not currently have a tray
        locations_needing_tray_moved = locations_with_unserved - locations_with_trays

        # Cost for moving trays to locations that need them
        cost += len(locations_needing_tray_moved)

        # --- Step 12: Return total cost ---
        return cost

