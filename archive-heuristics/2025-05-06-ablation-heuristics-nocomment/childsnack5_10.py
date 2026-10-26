from heuristics.heuristic_base import Heuristic
import math # Although max(0, ...) is used, math is a common import for numeric heuristics

def get_parts(fact):
    """Helper function to parse a PDDL fact string into parts."""
    # Remove leading/trailing parentheses and split by space
    return fact[1:-1].split()

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Estimates the number of actions required to serve all children.
    It sums up the estimated costs for making sandwiches, putting them
    on trays, moving trays to children's locations, and serving the children.
    It considers sandwich type requirements for allergic children.
    """

    def __init__(self, task):
        """
        Initializes the heuristic with static task information.

        Heuristic Initialization:
        - Stores the set of all children who are goals (need to be served).
        - Extracts static information about which children are allergic to gluten.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store the set of all children who need to be served eventually.
        # These are the children mentioned in the goal.
        self.children_to_serve = {
            get_parts(goal)[1]
            for goal in self.goals
            if get_parts(goal)[0] == 'served'
        }

        # Extract static information about allergic children
        self.allergic_children = {
            get_parts(fact)[1]
            for fact in static_facts
            if get_parts(fact)[0] == 'allergic_gluten'
        }

    def __call__(self, node):
        """
        Computes the heuristic value for the given state.

        Step-By-Step Thinking for Computing Heuristic:
        1. Identify unserved children based on the goal and current state.
           The number of unserved children is a base cost, representing
           the minimum number of 'serve' actions required. If no children
           are unserved, the goal is reached, and the heuristic is 0.
        2. Categorize the unserved children into allergic and non-allergic
           based on static information.
        3. Count the number of available sandwiches (those currently
           'at_kitchen_sandwich' or 'ontray') distinguishing between
           gluten-free and non-gluten-free types.
        4. Estimate the number of sandwiches that need to be *made*. This is
           calculated by comparing the number of unserved children of each
           type (allergic needing GF, non-allergic needing any) against the
           available sandwiches of the corresponding type, prioritizing
           available GF for allergic children first. Add this count to the
           heuristic (cost for 'make_sandwich' actions).
        5. Count the number of sandwiches that are currently 'at_kitchen_sandwich'
           or have been estimated as needing to be made. These sandwiches
           need to be moved onto a tray. Add this count to the heuristic
           (cost for 'put_on_tray' actions).
        6. Check if any sandwiches need to be put on a tray (step 5 count > 0)
           and if there is no tray currently located at the 'kitchen'. If both
           conditions are true, add 1 to the heuristic, representing the cost
           to move a tray to the kitchen.
        7. Identify all unique locations where unserved children are waiting
           (from the current state). Count how many of these locations
           (excluding the 'kitchen', as kitchen tray needs are handled in step 6)
           currently do not have a tray. Add this count to the heuristic,
           representing the cost to move trays to these locations.
        8. The total heuristic value is the sum of costs from steps 1, 4, 5, 6, and 7.

        Assumptions:
        - Assumes sufficient bread, content, and 'notexist' sandwich objects
          are available in the initial state to make any required sandwiches.
        - Assumes trays can effectively be used to transport sandwiches to
          locations needing them (e.g., one tray can serve multiple children
          at the same location, although actions are per-sandwich/per-tray).
        - Assumes there are enough tray objects in the domain to satisfy
          the needs at different locations simultaneously.
        """
        state = node.state

        # 1. Identify unserved children
        served_children = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == 'served'}
        unserved_children = {c for c in self.children_to_serve if c not in served_children}
        N_unserved = len(unserved_children)

        if N_unserved == 0:
            return 0 # Goal state

        h = 0

        # Cost for serve actions
        h += N_unserved

        # 2. Categorize unserved children by allergy
        unserved_allergic = {c for c in unserved_children if c in self.allergic_children}
        unserved_not_allergic = {c for c in unserved_children if c not in self.allergic_children}
        N_unserved_allergic = len(unserved_allergic)
        N_unserved_not_allergic = len(unserved_not_allergic)

        # 3. Count sandwiches in different stages by type
        ontray_sandwiches = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == 'ontray'}
        at_kitchen_sandwiches = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == 'at_kitchen_sandwich'}

        # no_gluten_sandwich predicate is in the state, not static
        is_gf_sandwich = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == 'no_gluten_sandwich'}

        N_ontray_gf = len({s for s in ontray_sandwiches if s in is_gf_sandwich})
        N_ontray_nongf = len(ontray_sandwiches) - N_ontray_gf

        N_at_kitchen_gf = len({s for s in at_kitchen_sandwiches if s in is_gf_sandwich})
        N_at_kitchen_nongf = len(at_kitchen_sandwiches) - N_at_kitchen_gf

        # 4. Calculate sandwiches to make, considering type
        # Need N_unserved_allergic GF sandwiches. Have N_ontray_gf + N_at_kitchen_gf.
        required_gf_makes = max(0, N_unserved_allergic - (N_ontray_gf + N_at_kitchen_gf))

        # Need N_unserved_not_allergic non-GF sandwiches.
        # Have N_ontray_nongf + N_at_kitchen_nongf non-GF,
        # plus surplus GF: max(0, (N_ontray_gf + N_at_kitchen_gf) - N_unserved_allergic).
        available_for_nongf = N_ontray_nongf + N_at_kitchen_nongf + max(0, (N_ontray_gf + N_at_kitchen_gf) - N_unserved_allergic)
        required_nongf_makes = max(0, N_unserved_not_allergic - available_for_nongf)

        h += required_gf_makes + required_nongf_makes # Cost for make actions

        # 5. Calculate sandwiches needing put_on_tray
        # Sandwiches currently at_kitchen_sandwich need put_on_tray.
        # Sandwiches that need to be made also need put_on_tray.
        N_need_put_on_tray = len(at_kitchen_sandwiches) + required_gf_makes + required_nongf_makes

        h += N_need_put_on_tray # Cost for put_on_tray actions

        # 6. Check tray at kitchen for put_on_tray
        tray_at_kitchen = any(get_parts(fact)[0] == 'at' and get_parts(fact)[2] == 'kitchen' for fact in state)
        if N_need_put_on_tray > 0 and not tray_at_kitchen:
             h += 1 # Cost of move_tray to kitchen

        # 7. Calculate tray movements to child locations
        # Find places where unserved children are waiting (from state)
        waiting_places_in_state = {get_parts(fact)[2] for fact in state if get_parts(fact)[0] == 'waiting' and get_parts(fact)[1] in unserved_children}
        tray_locations = {get_parts(fact)[2] for fact in state if get_parts(fact)[0] == 'at'}

        # Count places needing a tray (where children wait, not kitchen, and no tray is present)
        places_needing_tray = {p for p in waiting_places_in_state if p != 'kitchen' and p not in tray_locations}
        h += len(places_needing_tray) # Cost of move_tray to child locations

        return h
