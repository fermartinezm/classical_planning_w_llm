from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
    - `args`: The expected pattern (strings, wildcards `*` allowed per part).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    Estimates the number of actions needed to serve all waiting children.
    The heuristic counts the steps required to:
    1. Make necessary sandwiches (considering allergy needs).
    2. Put sandwiches onto trays.
    3. Move trays to locations where children are waiting.
    4. Serve the children.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts:
        - Allergy status of children.
        - Gluten-free status of bread and content portions.
        """
        # self.goals = task.goals # Not strictly needed for this heuristic calculation
        static_facts = task.static

        self.allergic_children = set()
        self.not_allergic_children = set()
        self.gf_breads = set()
        self.gf_contents = set()

        for fact in static_facts:
            if match(fact, "allergic_gluten", "*"):
                self.allergic_children.add(get_parts(fact)[1])
            elif match(fact, "not_allergic_gluten", "*"):
                self.not_allergic_children.add(get_parts(fact)[1])
            elif match(fact, "no_gluten_bread", "*"):
                self.gf_breads.add(get_parts(fact)[1])
            elif match(fact, "no_gluten_content", "*"):
                self.gf_contents.add(get_parts(fact)[1])

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # --- Step 1: Count unserved children and their locations/needs ---
        served_children = set()
        waiting_children = {} # child -> place
        unserved_children_by_place = {} # place -> set of children
        N_unserved = 0
        N_allergic_unserved = 0
        N_non_allergic_unserved = 0

        # Identify served children and waiting children/places
        for fact in state:
            if match(fact, "served", "*"):
                served_children.add(get_parts(fact)[1])
            elif match(fact, "waiting", "*", "*"):
                child = get_parts(fact)[1]
                place = get_parts(fact)[2]
                waiting_children[child] = place

        # Count unserved children and group them by place, identify allergy status
        for child, place in waiting_children.items():
            if child not in served_children:
                N_unserved += 1
                unserved_children_by_place.setdefault(place, set()).add(child)
                if child in self.allergic_children:
                    N_allergic_unserved += 1
                elif child in self.not_allergic_children:
                    N_non_allergic_unserved += 1
                # Note: Assumes every child is either allergic or not_allergic based on static facts

        # Heuristic component: Cost for serve actions (minimum 1 action per unserved child)
        h_serve = N_unserved

        # If no children are unserved, we are at the goal.
        if N_unserved == 0:
            return 0

        # --- Step 2: Count available sandwiches by type and location ---
        sandwiches_ontray = set() # set of sandwiches ontray
        sandwiches_kitchen = set() # set of sandwiches at_kitchen_sandwich
        gf_sandwiches_in_state = set() # set of no_gluten_sandwich currently true

        # Identify sandwiches and their properties/locations
        for fact in state:
            if match(fact, "ontray", "*", "*"):
                sandwiches_ontray.add(get_parts(fact)[1])
            elif match(fact, "at_kitchen_sandwich", "*"):
                sandwiches_kitchen.add(get_parts(fact)[1])
            elif match(fact, "no_gluten_sandwich", "*"):
                gf_sandwiches_in_state.add(get_parts(fact)[1])
            # notexist facts are not needed for counting available sandwiches,
            # only for calculating how many *can* be made (which we simplify by just counting demand)

        # Count available sandwiches by type and location
        N_gf_ontray = len(sandwiches_ontray.intersection(gf_sandwiches_in_state))
        N_reg_ontray = len(sandwiches_ontray) - N_gf_ontray
        N_gf_kitchen = len(sandwiches_kitchen.intersection(gf_sandwiches_in_state))
        N_reg_kitchen = len(sandwiches_kitchen) - N_gf_kitchen

        # --- Step 3: Calculate sandwiches that need to be made ---
        N_gf_available = N_gf_ontray + N_gf_kitchen
        N_reg_available = N_reg_ontray + N_reg_kitchen

        # GF sandwiches needed specifically for allergic children
        make_gf = max(0, N_allergic_unserved - N_gf_available)

        # Sandwiches needed for non-allergic children (can be GF or regular)
        # Available for non-allergic = remaining GF + available regular
        available_for_non_allergic = max(0, N_gf_available - N_allergic_unserved) + N_reg_available
        make_reg = max(0, N_non_allergic_unserved - available_for_non_allergic)

        # Heuristic component: Cost for make actions (minimum 1 action per sandwich made)
        h_make = make_gf + make_reg

        # --- Step 4: Calculate sandwiches that need to be put on trays ---
        # We need N_unserved sandwiches on trays.
        # The remaining need to be put on trays from the kitchen pool (initial + newly made)
        h_put_on_tray = max(0, N_unserved - len(sandwiches_ontray))

        # --- Step 5: Calculate tray movements needed for locations ---
        places_with_unserved = set(unserved_children_by_place.keys())
        places_with_trays = set()
        for fact in state:
             if match(fact, "at", "*", "*"):
                 places_with_trays.add(get_parts(fact)[2])

        # Heuristic component: Cost for moving trays to new locations (minimum 1 move per new location needed)
        h_move_tray_to_place = len(places_with_unserved - places_with_trays)

        # --- Total Heuristic ---
        # Sum of estimated minimum actions for each stage
        total_heuristic = h_serve + h_make + h_put_on_tray + h_move_tray_to_place

        return total_heuristic
