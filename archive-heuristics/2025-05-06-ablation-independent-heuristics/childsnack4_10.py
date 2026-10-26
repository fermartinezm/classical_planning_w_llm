import re
from heuristics.heuristic_base import Heuristic
from task import Task # Assuming Task class is available

# Helper function to parse PDDL fact strings
def parse_fact(fact_str):
    """Parses a PDDL fact string into predicate and arguments."""
    # Remove surrounding brackets and split by space
    parts = fact_str[1:-1].split()
    predicate = parts[0]
    args = parts[1:]
    return predicate, args

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    Estimates the cost to reach the goal (all children served) by summing
    the estimated minimum actions required to serve each unserved child.
    It counts the number of unserved allergic (needs GF) and non-allergic
    (needs Reg/GF) children. It then counts available suitable sandwiches
    at different stages of preparation/location (on tray at destination,
    on tray elsewhere, in kitchen, makeable) and greedily assigns the
    closest available sandwiches to satisfy the needs, prioritizing GF
    needs and closer sandwiches. The cost for each served child is based
    on the stage of the sandwich used (1 for on-tray-at-dest, 2 for
    on-tray-elsewhere, 3 for kitchen, 4 for makeable). Tray availability
    in the kitchen is considered for sandwiches starting in the kitchen
    or needing to be made.

    Assumptions:
    - Children wait at fixed places (static).
    - Gluten status of children, bread, and content is static.
    - Trays can be used for any sandwich type.
    - Ingredients (bread, content) are fungible within their type (GF/Reg).
    - 'kitchen' is a defined place constant.
    - The heuristic assumes a greedy assignment of available sandwiches
      to unserved children based on increasing action cost.
    - The heuristic simplifies tray movement; it only explicitly accounts
      for trays being needed in the kitchen to pick up sandwiches.

    Heuristic Initialization:
    The constructor processes the static facts from the task description
    to identify:
    - Allergic and non-allergic children.
    - Waiting places for each child.
    - Gluten status of bread and content types.
    - All objects (children, places, sandwiches, bread, content, trays)
      by inferring types from predicate argument positions across all
      possible facts in the domain (provided via task.facts).

    Step-By-Step Thinking for Computing Heuristic:
    1. Parse the current state to identify:
       - Children already served.
       - Bread and content portions in the kitchen.
       - Sandwiches in the kitchen.
       - Sandwiches on trays and which tray they are on.
       - Gluten status of existing sandwiches.
       - Location of each tray.
       - Sandwiches that do not yet exist ('notexist').
    2. Identify unserved children, separating them into those needing
       gluten-free sandwiches (allergic) and those needing any sandwich
       (non-allergic). Count the number of unserved children of each type.
    3. If no children are unserved, the heuristic is 0 (goal state).
    4. Identify the required waiting places for the unserved children
       of each type.
    5. Count available suitable sandwiches at different "cost pools" based
       on their current state and location:
       - Pool 1 (Cost 1: Serve): Sandwiches already on a tray at a place
         where an unserved child needing that type of sandwich is waiting.
       - Pool 2 (Cost 2: Move + Serve): Sandwiches already on a tray but
         not at a required waiting place for an unserved child needing
         that type.
       - Pool 3 (Cost 3: Put + Move + Serve): Sandwiches in the kitchen.
       - Pool 4 (Cost 4: Make + Put + Move + Serve): Sandwiches that can
         be made from available ingredients and 'notexist' slots.
       Distinguish between GF and Regular sandwiches in each pool.
    6. Count available ingredients (GF/Reg bread, GF/Reg content) and
       'notexist' slots in the kitchen to determine the maximum number
       of makeable GF and Regular sandwiches (Pool 4).
    7. Count available trays in the kitchen. This limits the number of
       sandwiches from Pool 3 and Pool 4 that can be put on a tray and
       subsequently moved/served. Distribute the kitchen tray capacity
       among Pool 3 and Pool 4 sandwiches, prioritizing Pool 3 over Pool 4,
       and GF over Regular within each pool. Adjust the available counts
       for Pool 3 and 4 accordingly.
    8. Greedily assign the available sandwiches from the pools to the
       unserved children's needs (GF needs first, then Reg needs),
       starting from the lowest cost pool (Pool 1) and moving to higher
       cost pools (Pool 2, 3, 4).
    9. For each sandwich assigned, add its corresponding cost (1, 2, 3, or 4)
       to the total heuristic value.
    10. If, after exhausting all available sandwiches in all pools, there
        are still unserved children, the state is considered unsolvable
        within this relaxation, and the heuristic returns infinity.
    11. Otherwise, return the total calculated cost.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task # Store task for access to goals and all_facts

        # --- Static Information ---
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.waiting_places = {} # child -> place
        self.no_gluten_breads = set()
        self.no_gluten_contents = set()

        # --- Object Type Sets (Inferred from all facts) ---
        self.all_children = set()
        self.all_places = set()
        self.all_sandwiches = set()
        self.all_breads = set()
        self.all_contents = set()
        self.all_trays = set()

        # Mapping from predicate and arg index to object type
        # Based on domain file analysis
        pred_arg_types = {
            'at_kitchen_bread': {0: 'bread-portion'},
            'at_kitchen_content': {0: 'content-portion'},
            'at_kitchen_sandwich': {0: 'sandwich'},
            'no_gluten_bread': {0: 'bread-portion'},
            'no_gluten_content': {0: 'content-portion'},
            'ontray': {0: 'sandwich', 1: 'tray'},
            'no_gluten_sandwich': {0: 'sandwich'},
            'allergic_gluten': {0: 'child'},
            'not_allergic_gluten': {0: 'child'},
            'served': {0: 'child'},
            'waiting': {0: 'child', 1: 'place'},
            'at': {0: 'tray', 1: 'place'},
            'notexist': {0: 'sandwich'},
        }

        # Populate object type sets by parsing all possible facts
        for fact_str in task.facts:
            predicate, args = parse_fact(fact_str)
            if predicate in pred_arg_types:
                for i, obj_type in pred_arg_types[predicate].items():
                    if i < len(args):
                        obj_name = args[i]
                        if obj_type == 'child':
                            self.all_children.add(obj_name)
                        elif obj_type == 'place':
                            self.all_places.add(obj_name)
                        elif obj_type == 'sandwich':
                            self.all_sandwiches.add(obj_name)
                        elif obj_type == 'bread-portion':
                            self.all_breads.add(obj_name)
                        elif obj_type == 'content-portion':
                            self.all_contents.add(obj_name)
                        elif obj_type == 'tray':
                            self.all_trays.add(obj_name)

        # Populate static predicate sets/dicts
        for fact_str in task.static:
            predicate, args = parse_fact(fact_str)
            if predicate == 'allergic_gluten':
                self.allergic_children.add(args[0])
            elif predicate == 'not_allergic_gluten':
                self.not_allergic_children.add(args[0])
            elif predicate == 'waiting':
                self.waiting_places[args[0]] = args[1]
            elif predicate == 'no_gluten_bread':
                self.no_gluten_breads.add(args[0])
            elif predicate == 'no_gluten_content':
                self.no_gluten_contents.add(args[0])

        # Ensure 'kitchen' is in all_places if it's a constant
        # The domain file declares kitchen as a constant place.
        self.all_places.add('kitchen')


    def __call__(self, node):
        state = node.state

        # --- State Information ---
        served_children = set()
        at_kitchen_breads = set()
        at_kitchen_contents = set()
        at_kitchen_sandwiches = set()
        ontray_sandwiches = {} # sandwich -> tray
        no_gluten_sandwiches_state = set() # Sandwiches that are currently known to be GF
        tray_locations = {} # tray -> place
        notexist_sandwiches_state = set()

        for fact_str in state:
            predicate, args = parse_fact(fact_str)
            if predicate == 'served':
                served_children.add(args[0])
            elif predicate == 'at_kitchen_bread':
                at_kitchen_breads.add(args[0])
            elif predicate == 'at_kitchen_content':
                at_kitchen_contents.add(args[0])
            elif predicate == 'at_kitchen_sandwich':
                at_kitchen_sandwiches.add(args[0])
            elif predicate == 'ontray':
                ontray_sandwiches[args[0]] = args[1]
            elif predicate == 'no_gluten_sandwich':
                no_gluten_sandwiches_state.add(args[0])
            elif predicate == 'at':
                tray_locations[args[0]] = args[1]
            elif predicate == 'notexist':
                notexist_sandwiches_state.add(args[0])

        # --- Calculate Needs ---
        unserved_children = self.all_children - served_children
        num_unserved_gf = len(unserved_children.intersection(self.allergic_children))
        num_unserved_reg = len(unserved_children - self.allergic_children)

        # Goal reached
        if num_unserved_gf == 0 and num_unserved_reg == 0:
            return 0

        # --- Calculate Available Resources by Stage and Type ---

        # Helper to check if a sandwich is GF in the current state
        is_gf_sandwich = lambda s: s in no_gluten_sandwiches_state

        # Required places for unserved children
        required_places_gf = {self.waiting_places[c] for c in unserved_children.intersection(self.allergic_children)}
        required_places_reg = {self.waiting_places[c] for c in unserved_children - self.allergic_children}
        all_required_places = required_places_gf.union(required_places_reg)

        # Sandwiches on trays, categorized by location and type
        ontray_at_place = {p: {'gf': 0, 'reg': 0} for p in self.all_places}
        total_ontray_gf = 0
        total_ontray_reg = 0

        for s, t in ontray_sandwiches.items():
            p = tray_locations.get(t)
            if p: # Tray location is known
                if is_gf_sandwich(s):
                    ontray_at_place[p]['gf'] += 1
                    total_ontray_gf += 1
                else:
                    ontray_at_place[p]['reg'] += 1
                    total_ontray_reg += 1

        # Sandwiches in the kitchen, categorized by type
        kitchen_sandwich = {'gf': 0, 'reg': 0}
        for s in at_kitchen_sandwiches:
            if is_gf_sandwich(s):
                kitchen_sandwich['gf'] += 1
            else:
                kitchen_sandwich['reg'] += 1

        # Available ingredients in the kitchen
        num_gf_bread = len(at_kitchen_breads.intersection(self.no_gluten_breads))
        num_reg_bread = len(at_kitchen_breads - self.no_gluten_breads)
        num_gf_content = len(at_kitchen_contents.intersection(self.no_gluten_contents))
        num_reg_content = len(at_kitchen_contents - self.no_gluten_contents)
        num_notexist = len(notexist_sandwiches_state)

        # Makeable sandwiches (simplified ingredient logic)
        makeable_gf = min(num_gf_bread, num_gf_content, num_notexist)
        remaining_notexist = num_notexist - makeable_gf
        remaining_gf_bread = num_gf_bread - makeable_gf
        remaining_gf_content = num_gf_content - makeable_gf
        makeable_reg = min(num_reg_bread + remaining_gf_bread, num_reg_content + remaining_gf_content, remaining_notexist)

        # Available trays in the kitchen
        num_trays_kitchen = list(tray_locations.values()).count('kitchen')

        # --- Categorize Sandwiches into Cost Pools ---

        # Pool 1 (Cost 1: Serve)
        # Sandwiches on tray at a required place for *any* unserved child
        Pool1_gf = sum(ontray_at_place[p]['gf'] for p in all_required_places)
        Pool1_reg = sum(ontray_at_place[p]['reg'] for p in all_required_places)

        # Pool 2 (Cost 2: Move + Serve)
        # Sandwiches on tray elsewhere (not at any required place for an unserved child)
        Pool2_gf = total_ontray_gf - Pool1_gf
        Pool2_reg = total_ontray_reg - Pool1_reg

        # Pool 3 (Cost 3: Put + Move + Serve)
        # Sandwiches in the kitchen
        Pool3_gf = kitchen_sandwich['gf']
        Pool3_reg = kitchen_sandwich['reg']

        # Pool 4 (Cost 4: Make + Put + Move + Serve)
        # Sandwiches that can be made
        Pool4_gf = makeable_gf
        Pool4_reg = makeable_reg

        # --- Limit Pool 3 and 4 by Kitchen Tray Availability ---
        # Sandwiches from Pool 3 and 4 need a tray in the kitchen first.
        avail_P3_total = Pool3_gf + Pool3_reg
        avail_P4_total = Pool4_gf + Pool4_reg
        total_kitchen_sandwiches_potential = avail_P3_total + avail_P4_total

        num_can_put_on_tray_kitchen = min(total_kitchen_sandwiches_potential, num_trays_kitchen)

        # Distribute kitchen tray capacity, prioritizing Pool 3 GF, then Pool 3 Reg, then Pool 4 GF, then Pool 4 Reg
        avail_P3_gf_actual = min(Pool3_gf, num_can_put_on_tray_kitchen)
        num_can_put_on_tray_kitchen -= avail_P3_gf_actual

        avail_P3_reg_actual = min(Pool3_reg, num_can_put_on_tray_kitchen)
        num_can_put_on_tray_kitchen -= avail_P3_reg_actual

        avail_P4_gf_actual = min(Pool4_gf, num_can_put_on_tray_kitchen)
        num_can_put_on_tray_kitchen -= avail_P4_gf_actual

        avail_P4_reg_actual = min(Pool4_reg, num_can_put_on_tray_kitchen)
        # num_can_put_on_tray_kitchen -= avail_P4_reg_actual # Should be 0

        # --- Assign Sandwiches to Needs Greedily by Cost ---

        rem_gf = num_unserved_gf
        rem_reg = num_unserved_reg
        cost = 0

        # Available sandwiches by cost pool and type
        Avail_gf = [Pool1_gf, Pool2_gf, avail_P3_gf_actual, avail_P4_gf_actual]
        Avail_reg = [Pool1_reg, Pool2_reg, avail_P3_reg_actual, avail_P4_reg_actual]
        Costs = [1, 2, 3, 4]

        for i, c in enumerate(Costs):
            # Use GF sandwiches at this cost level for GF needs
            use_gf = min(rem_gf, Avail_gf[i])
            cost += use_gf * c
            rem_gf -= use_gf
            Avail_gf[i] -= use_gf # Consume from pool

            # Use Reg sandwiches at this cost level for Reg needs
            use_reg = min(rem_reg, Avail_reg[i])
            cost += use_reg * c
            rem_reg -= use_reg
            Avail_reg[i] -= use_reg # Consume from pool

            # Use remaining GF sandwiches at this cost level for remaining Reg needs
            use_gf_for_reg = min(rem_reg, Avail_gf[i])
            cost += use_gf_for_reg * c
            rem_reg -= use_gf_for_reg
            Avail_gf[i] -= use_gf_for_reg # Consume from pool

        # If there are still unserved children, the state is likely unsolvable
        if rem_gf > 0 or rem_reg > 0:
            return float('inf')

        return cost

