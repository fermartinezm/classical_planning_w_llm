from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
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
    It counts the number of unserved children (base cost for 'serve' actions),
    adds costs for tray movements to locations needing deliveries, and adds costs
    for making sandwiches and putting them on trays based on the deficit of
    suitable sandwiches required.

    # Assumptions
    - Each unserved child requires one suitable sandwich.
    - A gluten-free sandwich is suitable for any child. A regular sandwich is only suitable for non-allergic children.
    - Tray movements are counted per location that needs a delivery, assuming one trip can bring *a* sandwich there.
    - Making sandwiches and putting them on trays are counted based on the total deficit of suitable sandwiches,
      assuming ingredients and 'notexist' slots are available up to the counts derived from the state.
    - The cost of actions is 1.

    # Heuristic Initialization
    - Extracts static information: which children are allergic/not allergic,
      which children are waiting at which locations, which bread/content is gluten-free.
    - Identifies all children, sandwiches, bread, content, trays, and places mentioned
      in the initial state and static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all unserved children and their waiting locations.
    2. Calculate the base heuristic cost: the total number of unserved children (each needs a 'serve' action).
    3. Identify locations where unserved children are waiting. For each such location, check if *any* suitable sandwich is already on a tray present at that location. Count the number of locations that *do not* have a suitable sandwich on a tray at that location. Add this count to the heuristic (representing the cost of 'move_tray' actions to bring trays to these locations).
    4. Determine the total number of gluten-free sandwiches and regular sandwiches required (equal to the number of unserved allergic and non-allergic children, respectively).
    5. Count the available gluten-free and regular sandwiches in different states:
       - On trays anywhere.
       - In the kitchen.
       - That can be made (limited by available ingredients in the kitchen and 'notexist' sandwiches).
    6. Calculate the deficit of needed sandwiches (GF and Regular), prioritizing GF sandwiches for allergic children first, then using remaining GF and all Regular sandwiches for non-allergic children.
    7. For the sandwiches that need to be sourced from the kitchen (to meet the deficit), add 1 to the heuristic for each (cost of 'put_on_tray').
    8. For the sandwiches that need to be made (to meet the deficit), add 2 to the heuristic for each (cost of 'make_sandwich' + 'put_on_tray').
    9. If, after exhausting all available and creatable sandwiches, there is still a deficit of needed sandwiches, return a large value indicating a likely unsolvable state.
    10. The total heuristic is the sum of costs from steps 2, 3, 7, and 8.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts."""
        self.goals = task.goals
        self.static_facts = task.static

        # Extract relevant objects and static properties from static facts
        self.all_children = set()
        self.child_allergy = {} # child -> 'gluten' or 'none'
        self.child_location = {} # child -> place (from waiting)
        self.no_gluten_bread_set = set()
        self.no_gluten_content_set = set()

        # We also need to know all possible sandwiches, bread, content, trays, places
        # that might appear in the state, even if not in static facts.
        # Parsing task.facts (all ground facts) is the most robust way,
        # but iterating init + static is often sufficient for typical problems.
        # Let's iterate init + static and assume it covers relevant objects.
        all_init_static_facts = set(task.initial_state) | set(task.static)

        self.all_sandwiches = set()
        self.all_bread = set()
        self.all_content = set()
        self.all_trays = set()
        self.all_places = {'kitchen'} # kitchen is a constant

        for fact in all_init_static_facts:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'waiting':
                child, place = parts[1], parts[2]
                self.all_children.add(child)
                self.all_places.add(place)
                self.child_location[child] = place
            elif predicate == 'allergic_gluten':
                child = parts[1]
                self.all_children.add(child)
                self.child_allergy[child] = 'gluten'
            elif predicate == 'not_allergic_gluten':
                child = parts[1]
                self.all_children.add(child)
                self.child_allergy[child] = 'none'
            elif predicate == 'at_kitchen_bread':
                 self.all_bread.add(parts[1])
            elif predicate == 'at_kitchen_content':
                 self.all_content.add(parts[1])
            elif predicate == 'at_kitchen_sandwich':
                 self.all_sandwiches.add(parts[1])
            elif predicate == 'ontray':
                 self.all_sandwiches.add(parts[1])
                 self.all_trays.add(parts[2])
            elif predicate == 'no_gluten_sandwich':
                 self.all_sandwiches.add(parts[1])
            elif predicate == 'notexist':
                 self.all_sandwiches.add(parts[1])
            elif predicate == 'at':
                 self.all_trays.add(parts[1])
                 self.all_places.add(parts[2])
            elif predicate == 'served':
                 self.all_children.add(parts[1])
            elif predicate == 'no_gluten_bread':
                 self.all_bread.add(parts[1])
                 self.no_gluten_bread_set.add(parts[1])
            elif predicate == 'no_gluten_content':
                 self.all_content.add(parts[1])
                 self.no_gluten_content_set.add(parts[1])

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0

        # --- Step 1 & 2: Base cost (serve actions) ---
        unserved_children = {c for c in self.all_children if f"(served {c})" not in state}
        h += len(unserved_children)

        if not unserved_children:
            return 0 # Goal reached

        # --- Extract dynamic state information ---
        at_kitchen_bread_state = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_bread", "*")}
        at_kitchen_content_state = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_content", "*")}
        at_kitchen_sandwich_state = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_sandwich", "*")}
        ontray_map_state = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "ontray", "*", "*")}
        tray_location_map_state = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*")}
        no_gluten_sandwich_state = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}
        notexist_sandwiches_state = {get_parts(fact)[1] for fact in state if match(fact, "notexist", "*")}

        # Helper to check if a sandwich is suitable for a child
        def is_suitable(sandwich, child):
            if sandwich in no_gluten_sandwich_state:
                return True # GF is suitable for anyone
            # Not GF, only suitable for non-allergic
            return self.child_allergy.get(child) == 'none'

        # --- Step 3: Cost for tray movements to locations needing delivery ---
        locations_needing_delivery = set()
        for child in unserved_children:
            child_loc = self.child_location.get(child)
            if not child_loc: continue

            # Check if *any* suitable sandwich is already on a tray at child_loc
            suitable_at_location_exists = False
            for s, t in ontray_map_state.items():
                if tray_location_map_state.get(t) == child_loc:
                    if is_suitable(s, child):
                        suitable_at_location_exists = True
                        break
            if not suitable_at_location_exists:
                locations_needing_delivery.add(child_loc)

        h += len(locations_needing_delivery)

        # --- Step 4: Count needed sandwiches by type ---
        N_gf_needed = sum(1 for child in unserved_children if self.child_allergy.get(child) == 'gluten')
        N_reg_needed = sum(1 for child in unserved_children if self.child_allergy.get(child) == 'none')

        # --- Step 5: Count available sandwiches by type and stage ---
        # Available on trays anywhere
        gf_ontray_avail = len({s for s in ontray_map_state if s in no_gluten_sandwich_state})
        reg_ontray_avail = len({s for s in ontray_map_state if s not in no_gluten_sandwich_state})

        # Available in kitchen
        gf_kitchen_avail = len({s for s in at_kitchen_sandwich_state if s in no_gluten_sandwich_state})
        reg_kitchen_avail = len({s for s in at_kitchen_sandwich_state if s not in no_gluten_sandwich_state})

        # Available ingredients for making
        num_gf_bread_kitchen = len({b for b in at_kitchen_bread_state if b in self.no_gluten_bread_set})
        num_gf_content_kitchen = len({c for c in at_kitchen_content_state if c in self.no_gluten_content_set})
        num_any_bread_kitchen = len(at_kitchen_bread_state)
        num_any_content_kitchen = len(at_kitchen_content_state)
        num_notexist = len(notexist_sandwiches_state)

        # How many GF sandwiches can we make? Limited by GF ingredients and notexist slots.
        num_can_make_gf = min(num_gf_bread_kitchen, num_gf_content_kitchen, num_notexist)

        # How many Regular sandwiches can we make *after* making GF ones?
        # Limited by remaining any ingredients and remaining notexist slots.
        remaining_any_bread = num_any_bread_kitchen - num_can_make_gf
        remaining_any_content = num_any_content_kitchen - num_can_make_gf
        remaining_notexist = num_notexist - num_can_make_gf
        num_can_make_reg = min(remaining_any_bread, remaining_any_content, remaining_notexist)

        # --- Step 6, 7, 8: Calculate deficit and add costs for make/put ---

        # Prioritize GF sandwiches for allergic children
        use_gf_ontray_for_gf = min(N_gf_needed, gf_ontray_avail)
        N_gf_needed -= use_gf_ontray_for_gf
        gf_ontray_avail -= use_gf_ontray_for_gf

        use_gf_kitchen_for_gf = min(N_gf_needed, gf_kitchen_avail)
        N_gf_needed -= use_gf_kitchen_for_gf
        gf_kitchen_avail -= use_gf_kitchen_for_gf
        h += use_gf_kitchen_for_gf # Cost for put_on_tray

        use_gf_notexist_for_gf = min(N_gf_needed, num_can_make_gf)
        N_gf_needed -= use_gf_notexist_for_gf
        num_can_make_gf -= use_gf_notexist_for_gf
        h += use_gf_notexist_for_gf * 2 # Cost for make + put_on_tray

        # Use remaining GF sandwiches for non-allergic children
        use_gf_ontray_for_reg = min(N_reg_needed, gf_ontray_avail)
        N_reg_needed -= use_gf_ontray_for_reg
        gf_ontray_avail -= use_gf_ontray_for_reg

        use_gf_kitchen_for_reg = min(N_reg_needed, gf_kitchen_avail)
        N_reg_needed -= use_gf_kitchen_for_reg
        gf_kitchen_avail -= use_gf_kitchen_for_reg
        h += use_gf_kitchen_for_reg # Cost for put_on_tray

        use_gf_notexist_for_reg = min(N_reg_needed, num_can_make_gf) # Use remaining GF make capacity
        N_reg_needed -= use_gf_notexist_for_reg
        num_can_make_gf -= use_gf_notexist_for_reg
        h += use_gf_notexist_for_reg * 2 # Cost for make + put_on_tray

        # Use Regular sandwiches for non-allergic children
        use_reg_ontray_for_reg = min(N_reg_needed, reg_ontray_avail)
        N_reg_needed -= use_reg_ontray_for_reg
        reg_ontray_avail -= use_reg_ontray_for_reg

        use_reg_kitchen_for_reg = min(N_reg_needed, reg_kitchen_avail)
        N_reg_needed -= use_reg_kitchen_for_reg
        reg_kitchen_avail -= use_reg_kitchen_for_reg
        h += use_reg_kitchen_for_reg # Cost for put_on_tray

        use_reg_notexist_for_reg = min(N_reg_needed, num_can_make_reg) # Use remaining Reg make capacity
        N_reg_needed -= use_reg_notexist_for_reg
        num_can_make_reg -= use_reg_notexist_for_reg
        h += use_reg_notexist_for_reg * 2 # Cost for make + put_on_tray

        # --- Step 9: Check if deficit remains ---
        if N_gf_needed > 0 or N_reg_needed > 0:
            # Cannot satisfy all sandwich needs with available/creatable sandwiches
            return 1000 # Large value indicating difficulty/unsolvability

        # --- Step 10: Total heuristic ---
        return h
