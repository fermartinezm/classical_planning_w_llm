# from fnmatch import fnmatch # Not needed with get_parts
from heuristics.heuristic_base import Heuristic # Assuming this path is correct based on examples

def get_parts(fact):
    """Helper function to split a PDDL fact string into its predicate and arguments."""
    # Remove parentheses and split by space
    return fact[1:-1].split()


class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    This heuristic estimates the cost to reach the goal (serving all children)
    by summing up the estimated number of actions required in different stages
    of the process: making sandwiches, putting them on trays, moving trays
    to children's locations, and serving the children. It prioritizes using
    existing resources (sandwiches, ingredients) and considers the specific
    needs of allergic children.

    Assumptions:
    - The state represents a solvable problem instance, unless the ingredient
      check explicitly determines otherwise (in which case infinity is returned).
    - Trays have sufficient capacity for sandwiches (not explicitly modeled in PDDL).
    - Any tray can be moved to any location.
    - Gluten-free status of initial bread, content, and sandwiches is static.
    - Waiting place and allergy status of children are static.

    Heuristic Initialization:
    In the __init__ method, the heuristic extracts static information from the task:
    - The allergy status (allergic_gluten or not_allergic_gluten) for each child.
    - The waiting place (waiting ?c ?p) for each child.
    - The gluten-free status (no_gluten_bread, no_gluten_content, no_gluten_sandwich)
      for initial bread, content, and sandwiches. This allows quickly checking
      the type of ingredients and initial sandwiches.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify unserved children: Count children for whom the '(served ?c)' fact is not true. If all are served, the heuristic is 0.
    2. Categorize unserved children by allergy status: Determine how many unserved children are allergic (need GF sandwiches) and how many are not allergic (can use any sandwich).
    3. Count available sandwiches and ingredients: Count gluten-free and regular sandwiches currently in the kitchen or on trays. Count gluten-free and regular bread and content in the kitchen.
    4. Determine sandwich needs from kitchen/making: Based on the number of unserved allergic and non-allergic children, and the available sandwiches (in kitchen or ontray), calculate how many GF and Any sandwiches are still needed. These needed sandwiches must come from the kitchen (if already made there) or be newly made.
    5. Breakdown needed sandwiches by source (kitchen vs. make): Calculate how many of the needed sandwiches are currently in the kitchen (Needed_from_kitchen) and how many must be newly made (N_sandwiches_need_make). This involves a greedy allocation strategy: prioritize using available GF sandwiches (kitchen or ontray) for allergic children first. Then, use remaining available sandwiches (GF or Reg, kitchen or ontray) for non-allergic children. The remaining required sandwiches must be made. Among those to be sourced from kitchen/make, prioritize using sandwiches already in the kitchen over making new ones.
    6. Check ingredient feasibility: Verify if there are enough gluten-free and total ingredients in the kitchen to make the required number of new sandwiches (N_sandwiches_need_make). If not, the state is likely unsolvable, and the heuristic returns infinity.
    7. Calculate cost components:
        - Serve cost: The number of unserved children (each needs one serve action).
        - Make cost: The number of sandwiches that need to be newly made (N_sandwiches_need_make).
        - Put on tray cost: The number of sandwiches that are in the kitchen (either initially or newly made) and are needed by unserved children (Needed_from_kitchen + N_sandwiches_need_make). Each needs one put_on_tray action.
        - Move tray to kitchen cost: If any sandwiches need to be put on a tray (i.e., Needed_from_kitchen + N_sandwiches_need_make > 0) and no tray is currently at the kitchen, one tray move action is needed to bring a tray to the kitchen.
        - Move tray to child place cost: For each unique location where unserved children are waiting, check if there is at least one tray currently at that location. If not, one tray move action is needed to bring a tray to that location.
    8. Sum costs: The total heuristic value is the sum of the costs from the steps above. This layered sum reflects the approximate sequence of actions required.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        self.child_allergy = {} # {child: True if allergic, False otherwise}
        self.child_waiting_place = {} # {child: place}
        self.gf_bread_static = set() # {bread}
        self.gf_content_static = set() # {content}
        self.gf_sandwich_static = set() # {sandwich} (initial)

        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == 'allergic_gluten':
                self.child_allergy[parts[1]] = True
            elif pred == 'not_allergic_gluten':
                 self.child_allergy[parts[1]] = False
            elif pred == 'waiting':
                self.child_waiting_place[parts[1]] = parts[2]
            elif pred == 'no_gluten_bread':
                self.gf_bread_static.add(parts[1])
            elif pred == 'no_gluten_content':
                self.gf_content_static.add(parts[1])
            elif pred == 'no_gluten_sandwich':
                 self.gf_sandwich_static.add(parts[1]) # Sandwiches can be GF from init state

    def __call__(self, node):
        state = node.state

        # 1. Identify unserved children
        Served_children = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == 'served'}
        Unserved_children = {c for c in self.child_waiting_place.keys() if c not in Served_children}
        N_unserved = len(Unserved_children)

        if N_unserved == 0:
            return 0 # Goal state

        # 2. Categorize unserved children by allergy status
        Allergic_unserved = {c for c in Unserved_children if self.child_allergy.get(c, False)}
        Not_allergic_unserved = Unserved_children - Allergic_unserved
        N_allergic_unserved = len(Allergic_unserved)
        N_not_allergic_unserved = len(Not_allergic_unserved)

        # 3. Count available sandwiches and ingredients
        Bread_kitchen = set()
        Content_kitchen = set()
        Sandwiches_kitchen = set()
        Sandwiches_ontray_set = set()
        Sandwiches_gf_state = set() # GF status from state facts
        Trays_at_location = {} # {tray: place}

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == 'at_kitchen_bread': Bread_kitchen.add(parts[1])
            elif pred == 'at_kitchen_content': Content_kitchen.add(parts[1])
            elif pred == 'at_kitchen_sandwich': Sandwiches_kitchen.add(parts[1])
            elif pred == 'ontray': Sandwiches_ontray_set.add(parts[1])
            elif pred == 'no_gluten_sandwich': Sandwiches_gf_state.add(parts[1])
            elif pred == 'at' and len(parts) == 3 and parts[1].startswith('tray'): # Ensure it's a tray location fact
                 Trays_at_location[parts[1]] = parts[2]

        Sandwiches_gf_all = self.gf_sandwich_static.union(Sandwiches_gf_state)

        N_sandwich_gf_kitchen = len(Sandwiches_kitchen.intersection(Sandwiches_gf_all))
        N_sandwich_reg_kitchen = len(Sandwiches_kitchen) - N_sandwich_gf_kitchen
        N_sandwich_gf_ontray = len(Sandwiches_ontray_set.intersection(Sandwiches_gf_all))
        N_sandwich_reg_ontray = len(Sandwiches_ontray_set) - N_sandwich_gf_ontray

        N_bread_gf_kitchen = len(Bread_kitchen.intersection(self.gf_bread_static))
        N_bread_reg_kitchen = len(Bread_kitchen) - N_bread_gf_kitchen
        N_content_gf_kitchen = len(Content_kitchen.intersection(self.gf_content_static))
        N_content_reg_kitchen = len(Content_kitchen) - N_content_gf_kitchen

        # 4. Determine sandwich needs from kitchen/making
        Needed_gf_sandwiches = N_allergic_unserved
        Needed_any_sandwiches = N_not_allergic_unserved

        Avail_gf_sandwiches = N_sandwich_gf_kitchen + N_sandwich_gf_ontray
        Avail_reg_sandwiches = N_sandwich_reg_kitchen + N_sandwich_reg_ontray

        # Allocate available GF sandwiches first to allergic children
        GF_used_for_allergic = min(Needed_gf_sandwiches, Avail_gf_sandwiches)
        Rem_allergic_needed = Needed_gf_sandwiches - GF_used_for_allergic
        Rem_avail_gf = Avail_gf_sandwiches - GF_used_for_allergic

        # Allocate remaining available sandwiches (GF + Reg) to non-allergic children
        Rem_not_allergic_needed = Needed_any_sandwiches
        Rem_avail_any = Rem_avail_gf + Avail_reg_sandwiches
        Any_used_for_not_allergic = min(Rem_not_allergic_needed, Rem_avail_any)
        Rem_not_allergic_needed = Rem_not_allergic_needed - Any_used_for_not_allergic # Update remaining needed

        # Total sandwiches needed from kitchen or making
        Needed_source = Rem_allergic_needed + Rem_not_allergic_needed

        # 5. Breakdown needed sandwiches by source (kitchen vs. make)
        # How many of the Needed_source can come from the kitchen?
        # Prioritize using GF kitchen sandwiches for remaining allergic needs
        Needed_gf_from_kitchen = min(Rem_allergic_needed, N_sandwich_gf_kitchen)
        Rem_allergic_needed_from_make = Rem_allergic_needed - Needed_gf_from_kitchen
        Rem_gf_kitchen = N_sandwich_gf_kitchen - Needed_gf_from_kitchen

        # Remaining needed sandwiches (any type) come from kitchen (remaining GF or Reg) or making
        Rem_any_needed_from_source = Rem_not_allergic_needed
        Avail_any_kitchen = N_sandwich_reg_kitchen + Rem_gf_kitchen
        Needed_any_from_kitchen = min(Rem_any_needed_from_source, Avail_any_kitchen)
        Rem_any_needed_from_make = Rem_any_needed_from_source - Needed_any_from_kitchen

        Needed_from_kitchen = Needed_gf_from_kitchen + Needed_any_from_kitchen
        N_sandwiches_need_make = Rem_allergic_needed_from_make + Rem_any_needed_from_make

        # 6. Check ingredient feasibility
        Needed_gf_make = Rem_allergic_needed_from_make
        Needed_any_make_after_gf = Rem_any_needed_from_make

        Ingr_bread_gf_avail = N_bread_gf_kitchen
        Ingr_content_gf_avail = N_content_gf_kitchen
        Ingr_bread_reg_avail = N_bread_reg_kitchen
        Ingr_content_reg_avail = N_content_reg_kitchen

        # Check GF ingredients for GF sandwiches needed from making
        if Ingr_bread_gf_avail < Needed_gf_make or Ingr_content_gf_avail < Needed_gf_make:
            return float('inf')

        # Remaining ingredients after notionally using GF for GF sandwiches
        Rem_ingr_bread_gf = Ingr_bread_gf_avail - Needed_gf_make
        Rem_ingr_content_gf = Ingr_content_gf_avail - Needed_gf_make
        Rem_ingr_bread_total = Rem_ingr_bread_gf + Ingr_bread_reg_avail
        Rem_ingr_content_total = Rem_ingr_content_gf + Ingr_content_reg_avail

        # Check remaining ingredients for Any sandwiches needed from making
        if Rem_ingr_bread_total < Needed_any_make_after_gf or Rem_ingr_content_total < Needed_any_make_after_gf:
             return float('inf')

        # Ingredient check passed, N_sandwiches_need_make is feasible.

        # 7. Calculate cost components
        h = N_unserved # Cost for serve actions

        h += N_sandwiches_need_make # Cost for make actions

        sandwiches_need_put_on_tray_count = Needed_from_kitchen + N_sandwiches_need_make
        h += sandwiches_need_put_on_tray_count # Cost for put_on_tray actions

        # Tray movements
        Trays_at_kitchen_set = {t for t, p in Trays_at_location.items() if p == 'kitchen'}
        N_trays_at_kitchen = len(Trays_at_kitchen_set)

        # Cost to move a tray to kitchen if needed for put_on_tray
        if sandwiches_need_put_on_tray_count > 0 and N_trays_at_kitchen == 0:
            h += 1

        # Cost to move trays to children's places
        Places_with_unserved = {self.child_waiting_place[c] for c in Unserved_children}
        Places_with_trays = set(Trays_at_location.values())
        Places_needing_tray = Places_with_unserved - Places_with_trays
        N_places_needing_tray = len(Places_needing_tray)
        h += N_places_needing_tray

        # 8. Sum costs (already summed above)
        return h
