import math # Import math for infinity

class childsnackHeuristic:
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    The heuristic estimates the number of actions required to reach the goal
    (all children served) by summing up the estimated costs for different
    stages of the process for all unserved children:
    1. The base cost is the number of unserved children (representing the final serve action for each).
    2. Add the number of sandwiches that need to be made.
    3. Add the number of sandwiches that are at the kitchen (either initially or just made) and need to be put on a tray.
    4. Add the number of distinct places with unserved children that do not currently have a tray (representing tray movement actions).

    Assumptions:
    - The problem instance is solvable. The heuristic includes checks for
      potential unsolvability based on available ingredients and notexist
      sandwich objects at the kitchen, returning infinity if a required
      sandwich type cannot be made from the current kitchen inventory.
    - Tray capacity is effectively infinite for heuristic calculation purposes.
    - Ingredients (bread, content) are only available at the kitchen.
    - Sandwiches are only made at the kitchen.
    - Trays can be moved between any two places.
    - The goal is exclusively to serve all children found in the problem definition.

    Heuristic Initialization:
    The constructor pre-processes the static facts from the task to store:
    - Which children are allergic or not allergic.
    - The waiting location for each child.
    - Which bread and content items are gluten-free.
    - A set of all possible place objects mentioned in the problem.
    It also identifies all potential sandwich, bread, content, and tray objects
    mentioned in the initial state or static facts.

    Step-By-Step Thinking for Computing Heuristic:
    1. Parse the current state to identify:
       - Served children.
       - Sandwiches at the kitchen (`at_kitchen_sandwich`).
       - Sandwiches on trays (`ontray`).
       - Tray locations (`at`).
       - Bread and content at the kitchen (`at_kitchen_bread`, `at_kitchen_content`).
       - Sandwiches that do not exist yet (`notexist`).
       - Sandwiches that are gluten-free (`no_gluten_sandwich` predicate in state).
    2. Calculate the base heuristic (`h_serve_base`): The total number of children who are not yet served. This is a lower bound on the number of `serve` actions.
    3. Calculate the number of sandwiches that still need to be made (`h_make`):
       - Count the total number of unserved allergic children (`num_auc`) and non-allergic children (`num_nauc`).
       - Count the total number of existing gluten-free (`avail_gf_s_total`) and regular (`avail_reg_s_total`) sandwiches across all locations and states (at kitchen, on trays).
       - Determine how many GF sandwiches are still needed: `needed_gf_to_make = max(0, num_auc - avail_gf_s_total)`.
       - Determine how many Any sandwiches (can be regular or remaining GF) are still needed: `needed_any_to_make = max(0, num_nauc - (avail_reg_s_total + max(0, avail_gf_s_total - num_auc)))`.
       - Count available ingredients (GF bread, GF content, regular bread, regular content) and `notexist` sandwich objects *at the kitchen*.
       - Check if the required number of GF and Any sandwiches can actually be made given the available ingredients and `notexist` objects at the kitchen. If not, return infinity.
       - `h_make = needed_gf_to_make + needed_any_to_make`.
    4. Calculate the number of sandwiches that need to be put on a tray (`h_put_on_tray`):
       - This includes sandwiches currently `at_kitchen_sandwich` plus the sandwiches that need to be made (`h_make`).
       - `h_put_on_tray = |{s | (at_kitchen_sandwich s) in state}| + h_make`.
    5. Calculate the number of tray movements needed (`h_move_tray`):
       - Identify all places where unserved children are waiting.
       - Identify all places where trays are currently located.
       - Count the number of places with unserved children that do *not* currently have a tray. Each such place requires at least one `move_tray` action to bring a tray there.
       - `h_move_tray = |{p | (exists c: (waiting c p) in static and (served c) not in state) and not (exists t: (at t p) in state)}|`.
    6. The total heuristic value is the sum of `h_serve_base`, `h_make`, `h_put_on_tray`, and `h_move_tray`.
    """

    def __init__(self, task):
        # Store static information
        self.child_allergy = {}  # child -> 'allergic' or 'not_allergic'
        self.child_locations = {}  # child -> place
        self.gf_bread_types = set()  # bread objects that are GF
        self.gf_content_types = set()  # content objects that are GF
        self.all_places = {'kitchen'} # kitchen is a constant place

        # Collect all objects of relevant types from initial state and static facts
        self.all_children = set()
        self.all_trays = set()
        self.all_sandwiches = set()
        self.all_bread = set()
        self.all_content = set()

        all_facts = set(task.initial_state) | set(task.static)

        for fact_str in all_facts:
            parts = fact_str.strip('()').split()
            if not parts: continue # Skip empty strings

            predicate = parts[0]
            objects = parts[1:]

            if predicate == 'allergic_gluten' and len(objects) == 1:
                child = objects[0]
                self.child_allergy[child] = 'allergic'
                self.all_children.add(child)
            elif predicate == 'not_allergic_gluten' and len(objects) == 1:
                child = objects[0]
                self.child_allergy[child] = 'not_allergic'
                self.all_children.add(child)
            elif predicate == 'waiting' and len(objects) == 2:
                child, place = objects
                self.child_locations[child] = place
                self.all_children.add(child)
                self.all_places.add(place)
            elif predicate == 'no_gluten_bread' and len(objects) == 1:
                bread = objects[0]
                self.gf_bread_types.add(bread)
                self.all_bread.add(bread)
            elif predicate == 'no_gluten_content' and len(objects) == 1:
                content = objects[0]
                self.gf_content_types.add(content)
                self.all_content.add(content)
            elif predicate == 'at' and len(objects) == 2:
                tray, place = objects
                self.all_trays.add(tray)
                self.all_places.add(place)
            elif predicate == 'at_kitchen_bread' and len(objects) == 1:
                 self.all_bread.add(objects[0])
            elif predicate == 'at_kitchen_content' and len(objects) == 1:
                 self.all_content.add(objects[0])
            elif predicate in ('at_kitchen_sandwich', 'ontray', 'notexist', 'no_gluten_sandwich') and len(objects) >= 1:
                 # ontray has 2 objects, others mentioned here have 1. We only care about the first object (sandwich).
                 self.all_sandwiches.add(objects[0])


    def __call__(self, state):
        # Check if goal is reached (all children served)
        served_children_in_state = {c for fact in state if fact.startswith('(served ') for c in [fact.strip('()').split()[1]]}
        if len(served_children_in_state) == len(self.all_children):
             return 0

        # --- Step 1: Base heuristic (unserved children) ---
        unserved_children = self.all_children - served_children_in_state
        num_unserved = len(unserved_children)

        h = num_unserved

        # --- Step 2: Sandwiches that need to be made (h_make) ---
        num_auc = len({c for c in unserved_children if self.child_allergy.get(c) == 'allergic'})
        num_nauc = len({c for c in unserved_children if self.child_allergy.get(c) == 'not_allergic'})

        # Count available sandwiches (anywhere, any state except notexist)
        avail_gf_s_total = 0
        avail_reg_s_total = 0
        notexist_s_in_state = {s for fact in state if fact.startswith('(notexist ') for s in [fact.strip('()').split()[1]]}
        gf_s_in_state = {s for fact in state if fact.startswith('(no_gluten_sandwich ') for s in [fact.strip('()').split()[1]]}

        existing_sandwiches = self.all_sandwiches - notexist_s_in_state

        for s in existing_sandwiches:
             if s in gf_s_in_state:
                 avail_gf_s_total += 1
             else:
                 avail_reg_s_total += 1

        # Calculate needed sandwiches to make
        needed_gf_to_make = max(0, num_auc - avail_gf_s_total)
        rem_avail_gf = max(0, avail_gf_s_total - num_auc) # GF sandwiches left after potentially serving allergic
        needed_any_to_make = max(0, num_nauc - (avail_reg_s_total + rem_avail_gf))

        # Check if making is possible given ingredients and notexist objects at kitchen
        at_kitchen_b = {b for fact in state if fact.startswith('(at_kitchen_bread ') for b in [fact.strip('()').split()[1]]}
        at_kitchen_c = {c for fact in state if fact.startswith('(at_kitchen_content ') for c in [fact.strip('()').split()[1]]}
        avail_ne_s = len(notexist_s_in_state)

        avail_gf_b_k = len({b for b in at_kitchen_b if b in self.gf_bread_types})
        avail_reg_b_k = len({b for b in at_kitchen_b if b not in self.gf_bread_types})
        avail_gf_c_k = len({c for c in at_kitchen_c if c in self.gf_content_types})
        avail_reg_c_k = len({c for c in at_kitchen_c if c not in self.gf_content_types})

        can_make_gf = min(avail_gf_b_k, avail_gf_c_k, avail_ne_s)
        if needed_gf_to_make > can_make_gf:
             # Cannot make enough GF sandwiches from kitchen inventory
             return math.inf # Problem likely unsolvable from this state

        remaining_ne_s_after_gf = avail_ne_s - needed_gf_to_make
        can_make_reg = min(avail_reg_b_k, avail_reg_c_k, remaining_ne_s_after_gf)
        if needed_any_to_make > can_make_reg:
             # Cannot make enough Regular sandwiches from kitchen inventory
             return math.inf # Problem likely unsolvable from this state

        h_make = needed_gf_to_make + needed_any_to_make
        h += h_make

        # --- Step 3: Sandwiches at kitchen needing put_on_tray (h_put_on_tray) ---
        at_kitchen_s = {s for fact in state if fact.startswith('(at_kitchen_sandwich ') for s in [fact.strip('()').split()[1]]}
        h_put_on_tray = len(at_kitchen_s) + h_make
        h += h_put_on_tray

        # --- Step 4: Places needing tray movement (h_move_tray) ---
        places_with_unserved = {self.child_locations[c] for c in unserved_children}
        tray_locations = {t: p for fact in state if fact.startswith('(at ') for t, p in [fact.strip('()').split()[1:]]}
        places_with_tray = set(tray_locations.values())

        places_needing_tray_move = places_with_unserved - places_with_tray
        h_move_tray = len(places_needing_tray_move)
        h += h_move_tray

        return h
