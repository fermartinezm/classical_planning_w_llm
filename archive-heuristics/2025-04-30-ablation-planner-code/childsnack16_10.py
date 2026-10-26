from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the input is a string and looks like a PDDL fact
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
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


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting children.
    It counts the necessary steps: making suitable sandwiches, putting them on trays,
    moving trays to the children's locations, and finally serving the children.

    # Assumptions
    - Sufficient bread, content, and 'notexist' sandwich objects are available to make any required sandwich.
    - A tray can hold multiple sandwiches.
    - The 'kitchen' is a constant place.
    - Objects starting with 'tray' are trays.

    # Heuristic Initialization
    - Extracts the allergy status (allergic_gluten, not_allergic_gluten) for all children from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic is calculated as the sum of estimated costs for several sub-goals:

    1.  **Serving Children:** Count the number of children who are currently 'waiting' but not yet 'served'. Each such child requires a 'serve' action. Add this count to the heuristic. If this count is zero, the goal is reached, and the heuristic is 0.

    2.  **Making Sandwiches:** Determine how many new sandwiches of each type (gluten-free and non-gluten-free) need to be made to satisfy the unserved children, considering the sandwiches already available (at the kitchen or on trays).
        - Count unserved children who are allergic to gluten (`N_gf_children`).
        - Count total unserved children (`N_unserved`).
        - Count available gluten-free sandwiches (`A_gf_sandwiches`) (those currently `at_kitchen_sandwich` or `ontray` and `no_gluten_sandwich`).
        - Count available non-gluten-free sandwiches (`A_nongf_sandwiches`) (those currently `at_kitchen_sandwich` or `ontray` and *not* `no_gluten_sandwich`).
        - Allergic children *must* have gluten-free sandwiches. The number of new gluten-free sandwiches that *must* be made for allergic children is `max(0, N_gf_children - A_gf_sandwiches)`. Add this to the heuristic (cost of `make_sandwich_no_gluten`).
        - Non-allergic children can have any sandwich. After using available GF sandwiches for allergic children, the remaining available GF sandwiches and all available non-GF sandwiches can serve non-allergic children. The number of *additional* sandwiches (which can be any type) that need to be made for non-allergic children is calculated based on the total sandwiches needed vs total available, accounting for the GF sandwiches already assigned to allergic children. Add this count to the heuristic (cost of `make_sandwich`).

    3.  **Putting Sandwiches on Trays:** Count the number of sandwiches currently `at_kitchen_sandwich`. Each of these needs a 'put_on_tray' action. Add this count to the heuristic.

    4.  **Moving Trays to Kitchen:** If there are sandwiches `at_kitchen_sandwich` (or sandwiches that need to be made, as they appear at the kitchen) AND no tray is currently `at kitchen`, a tray needs to be moved to the kitchen. Add 1 to the heuristic in this case.

    5.  **Moving Trays to Children's Locations:** Identify all distinct places (other than the kitchen) where unserved children are waiting. For each such place that does *not* currently have a tray, a tray needs to be moved there. Add the count of such places to the heuristic.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts.
        """
        # The goals are not directly used in this heuristic's calculation logic,
        # but are available via task.goals if needed for other heuristics.
        # self.goals = task.goals

        self.static_facts = task.static

        # Store allergy status for quick lookup
        self.allergic_children = {get_parts(fact)[1] for fact in self.static_facts if match(fact, "allergic_gluten", "*")}
        self.not_allergic_children = {get_parts(fact)[1] for fact in self.static_facts if match(fact, "not_allergic_gluten", "*")}


    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state
        h = 0

        # 1. Count unserved children
        unserved_children = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "waiting", "*", "*") and not any(match(served_fact, "served", get_parts(fact)[1]) for served_fact in state)
        }
        num_unserved = len(unserved_children)

        # If no children are unserved, the goal is reached
        if num_unserved == 0:
            return 0

        # Add cost for serving each child
        h += num_unserved

        # 2. Making Sandwiches
        unserved_gf_count = sum(1 for c in unserved_children if c in self.allergic_children)
        unserved_nongf_count = num_unserved - unserved_gf_count

        # Count available sandwiches (at kitchen or on any tray)
        available_gf_sandwiches = {
            get_parts(s_fact)[1]
            for s_fact in state
            if match(s_fact, "no_gluten_sandwich", get_parts(s_fact)[1]) and (
                any(match(loc_fact, "at_kitchen_sandwich", get_parts(s_fact)[1]) for loc_fact in state) or
                any(match(ontray_fact, "ontray", get_parts(s_fact)[1], "*") for ontray_fact in state)
            )
        }
        available_nongf_sandwiches = {
             get_parts(s_fact)[1]
            for s_fact in state
            if not match(s_fact, "no_gluten_sandwich", get_parts(s_fact)[1]) and (
                any(match(loc_fact, "at_kitchen_sandwich", get_parts(s_fact)[1]) for loc_fact in state) or
                any(match(ontray_fact, "ontray", get_parts(s_fact)[1], "*") for ontray_fact in state)
            )
        }

        num_available_gf = len(available_gf_sandwiches)
        num_available_nongf = len(available_nongf_sandwiches)

        # Allergic children need GF sandwiches
        needed_gf_for_allergic = unserved_gf_count
        used_available_gf_for_allergic = min(needed_gf_for_allergic, num_available_gf)
        make_gf_for_allergic = max(0, needed_gf_for_allergic - used_available_gf_for_allergic)
        h += make_gf_for_allergic # Cost for make_sandwich_no_gluten

        # Remaining available GF sandwiches can serve non-allergic children
        remaining_available_gf = num_available_gf - used_available_gf_for_allergic

        # Non-allergic children need any sandwich
        needed_for_nongf = unserved_nongf_count
        available_for_nongf = num_available_nongf + remaining_available_gf
        make_any_for_nongf = max(0, needed_for_nongf - available_for_nongf)
        h += make_any_for_nongf # Cost for make_sandwich

        # 3. Putting Sandwiches on Trays
        sandwiches_at_kitchen = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "at_kitchen_sandwich", "*")
        }
        num_at_kitchen = len(sandwiches_at_kitchen)
        h += num_at_kitchen # Cost for put_on_tray for each

        # 4. Moving Trays to Kitchen (if needed for put_on_tray)
        tray_locations = {
            get_parts(fact)[2]
            for fact in state
            if match(fact, "at", "*", "*") and get_parts(fact)[1].startswith("tray") # Assuming tray objects start with 'tray'
        }
        kitchen_has_tray = "kitchen" in tray_locations

        # Need a tray at kitchen if any sandwich is at kitchen or needs to be made (as made sandwiches appear at kitchen)
        needs_tray_at_kitchen = num_at_kitchen > 0 or (make_gf_for_allergic + make_any_for_nongf) > 0

        if needs_tray_at_kitchen and not kitchen_has_tray:
             h += 1 # Move a tray to kitchen

        # 5. Moving Trays to Children's Locations (other than kitchen)
        waiting_places = {
            get_parts(fact)[2]
            for fact in state
            if match(fact, "waiting", "*", "*") and get_parts(fact)[1] in unserved_children
        }

        places_needing_tray_move = {
            p for p in waiting_places if p != "kitchen" and p not in tray_locations
        }
        h += len(places_needing_tray_move) # Move a tray to each such place

        return h
