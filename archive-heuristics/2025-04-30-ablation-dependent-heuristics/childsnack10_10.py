from collections import defaultdict

def _parse_fact(fact_string):
    """Parses a PDDL fact string into predicate and arguments."""
    # Example: '(at tray1 kitchen)' -> ['at', 'tray1', 'kitchen']
    parts = fact_string.strip('()').split()
    if not parts: # Handle empty string or just '()'
        return None, []
    return parts[0], parts[1:]

class childsnackHeuristic:
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
        This heuristic estimates the number of actions required to serve all
        unserved children. It does this by summing up the estimated minimum
        actions needed to get a suitable sandwich to each unserved child and
        perform the serving action. It prioritizes satisfying demand for
        gluten-free sandwiches first, and for both types of demand, it
        prioritizes using sandwiches that are closer to the child's waiting
        place (already at the place, on a tray elsewhere, in the kitchen,
        or makable from ingredients).

    Assumptions:
        - Tray capacity is effectively unlimited for the purpose of putting
          sandwiches on them.
        - Trays are always located at some place.
        - Ingredients (bread and content) are fungible within their type
          (gluten-free or regular).
        - Making a sandwich requires one bread, one content, and one
          non-existent sandwich object.
        - The cost of moving a tray is always 1, regardless of distance or
          current location (as long as it's not already at the target place).
        - The heuristic does not explicitly model potential conflicts over
          trays or ingredients beyond simple counts.
        - If a child cannot be served due to lack of resources (even makable),
          a large penalty is added.
        - The goal is assumed to be serving all children present in the initial
          state's 'waiting' facts.

    Heuristic Initialization:
        The constructor parses the static facts and initial state facts from
        the task definition to extract static information and identify all
        relevant objects (children, trays, sandwiches, places).
        It identifies:
        - Which children are allergic to gluten (`allergic_children`).
        - The waiting place for each child (`child_place_map`).
        - Which bread portions are gluten-free (`gf_bread`).
        - Which content portions are gluten-free (`gf_content`).
        - Sets of all children, trays, sandwiches, and places mentioned in the
          initial state and static facts, to be able to iterate over them
          during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
        1. Check if the goal is reached (all children are served). If so, return 0.
        2. Identify all unserved children in the current state.
        3. Categorize unserved children into those needing gluten-free
           sandwiches (allergic) and those needing any sandwich (not allergic).
           Count the total demand for each type (`demand_gf`, `demand_reg`).
        4. Count the available sandwiches in the current state, categorized by
           type (GF/Regular) and location/state:
           - On trays at places where unserved children are waiting.
           - On trays at other places.
           - In the kitchen.
        5. Count the available ingredients (GF bread, GF content, Regular bread,
           Regular content) and non-existent sandwich objects in the kitchen.
        6. Calculate the maximum number of GF sandwiches that can be made from
           the current available GF ingredients and non-existent sandwich objects.
        7. Initialize the heuristic value `h` to 0.
        8. Greedily satisfy the `demand_gf` (allergic children) by using available
           GF sandwiches in increasing order of estimated cost:
           - Cost 1: On trays at their waiting places.
           - Cost 2: On trays elsewhere (requires move_tray).
           - Cost 3: In the kitchen (requires put_on_tray + move_tray).
           - Cost 4: Makable GF sandwiches (requires make_sandwich_no_gluten + put_on_tray + move_tray).
           Update `h` and remaining `demand_gf`, and consume used resources
           (sandwiches, ingredients, notexist objects).
        9. If `demand_gf` is still greater than 0 after using all potential GF
           resources, add a large penalty to `h` for each unsatisfied GF demand.
        10. Greedily satisfy the `demand_reg` (non-allergic children) by using
            *any* remaining available sandwiches (GF or Regular) in increasing
            order of estimated cost:
            - Cost 1: On trays at their waiting places.
            - Cost 2: On trays elsewhere.
            - Cost 3: In the kitchen.
            - Cost 4: Makable Regular sandwiches (requires make_sandwich + put_on_tray + move_tray), using remaining ingredients (GF or Regular) and notexist objects.
            Update `h` and remaining `demand_reg`, and consume used resources.
        11. If `demand_reg` is still greater than 0 after using all potential
            resources, add a large penalty to `h` for each unsatisfied Regular demand.
        12. The final value of `h` is the heuristic estimate.
    """
    def __init__(self, task):
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.child_place_map = {}
        self.gf_bread = set()
        self.gf_content = set()
        self.all_children = set()
        self.all_trays = set()
        self.all_sandwiches = set()
        self.all_places = set()

        # Parse static facts and initial state to collect objects and static info
        # We iterate through both as objects might appear in either.
        for fact_string in task.static | task.initial_state:
            pred, args = _parse_fact(fact_string)
            if pred is None: continue # Skip invalid facts

            if pred == 'allergic_gluten':
                self.allergic_children.add(args[0])
                self.all_children.add(args[0])
            elif pred == 'not_allergic_gluten':
                self.not_allergic_children.add(args[0])
                self.all_children.add(args[0])
            elif pred == 'waiting':
                child, place = args
                self.child_place_map[child] = place
                self.all_children.add(child)
                self.all_places.add(place)
            elif pred == 'no_gluten_bread':
                self.gf_bread.add(args[0])
            elif pred == 'no_gluten_content':
                self.gf_content.add(args[0])
            # Collect objects by type regardless of predicate using naming convention
            for arg in args:
                 if arg.startswith('child'): self.all_children.add(arg)
                 elif arg.startswith('tray'): self.all_trays.add(arg)
                 elif arg.startswith('sandw'): self.all_sandwiches.add(arg)
                 elif arg.startswith('table') or arg == 'kitchen': self.all_places.add(arg)
                 # Bread and content objects are collected via gf_bread/content sets if they are GF,
                 # or implicitly when counting kitchen ingredients. We don't need a set of *all* bread/content objects.


        # Ensure kitchen is in places if it's a constant
        self.all_places.add('kitchen')

        # Ensure all children mentioned in goals are in self.all_children
        # (Although the problem description implies goal is all children from initial waiting)
        for goal_fact in task.goals:
             pred, args = _parse_fact(goal_fact)
             if pred == 'served':
                  self.all_children.add(args[0])


    def __call__(self, state):
        # If goal is reached, heuristic is 0
        # The goal is that all children in the goal set are served.
        # We assume the goal set is exactly self.all_children collected in __init__.
        served_children = set(args[0] for fact in state if _parse_fact(fact)[0] == 'served')
        unserved_children = self.all_children - served_children

        if not unserved_children:
            return 0

        # --- Extract state-dependent information ---
        ontray_map = defaultdict(set)
        tray_location_map = {}
        kitchen_sandwiches = set()
        notexist_sandwiches = set()
        gf_sandwiches_in_state = set()
        kitchen_bread = set()
        kitchen_content = set()

        for fact_string in state:
            pred, args = _parse_fact(fact_string)
            if pred is None: continue # Skip invalid facts

            if pred == 'ontray':
                sandwich, tray = args
                ontray_map[tray].add(sandwich)
            elif pred == 'at':
                tray, place = args
                tray_location_map[tray] = place
            elif pred == 'at_kitchen_sandwich':
                kitchen_sandwiches.add(args[0])
            elif pred == 'notexist':
                notexist_sandwiches.add(args[0])
            elif pred == 'no_gluten_sandwich':
                gf_sandwiches_in_state.add(args[0])
            elif pred == 'at_kitchen_bread':
                kitchen_bread.add(args[0])
            elif pred == 'at_kitchen_content':
                kitchen_content.add(args[0])

        # --- Calculate demands ---
        demand_gf = len(unserved_children.intersection(self.allergic_children))
        demand_reg = len(unserved_children - self.allergic_children)

        # --- Calculate available sandwiches by location/state ---
        waiting_places = set(self.child_place_map[c] for c in unserved_children if c in self.child_place_map) # Only consider places for waiting unserved children

        available_gf_sandwiches_at_waiting_place = {p: 0 for p in waiting_places}
        available_reg_sandwiches_at_waiting_place = {p: 0 for p in waiting_places}
        available_gf_sandwiches_on_trays_elsewhere = 0
        available_reg_sandwiches_on_trays_elsewhere = 0
        available_gf_sandwiches_kitchen = 0
        available_reg_sandwiches_kitchen = 0

        # Sandwiches on trays
        for tray in self.all_trays: # Iterate through all known trays
            t_loc = tray_location_map.get(tray)
            if t_loc is None: # Tray not located, ignore or penalize? Assume ignore for simplicity.
                continue
            sandwiches_on_t = ontray_map.get(tray, set())
            for s in sandwiches_on_t:
                is_gf = (s in gf_sandwiches_in_state)
                if is_gf:
                    if t_loc in waiting_places:
                        available_gf_sandwiches_at_waiting_place[t_loc] += 1
                    else:
                        available_gf_sandwiches_on_trays_elsewhere += 1
                else: # Regular sandwich
                    if t_loc in waiting_places:
                        available_reg_sandwiches_at_waiting_place[t_loc] += 1
                    else:
                        available_reg_sandwiches_on_trays_elsewhere += 1

        # Sandwiches in kitchen
        for s in self.all_sandwiches: # Iterate through all known sandwiches
             if s in kitchen_sandwiches:
                 is_gf = (s in gf_sandwiches_in_state)
                 if is_gf:
                     available_gf_sandwiches_kitchen += 1
                 else:
                     available_reg_sandwiches_kitchen += 1

        # --- Calculate makable sandwiches ---
        num_gf_bread_kitchen_current = len(kitchen_bread.intersection(self.gf_bread))
        num_reg_bread_kitchen_current = len(kitchen_bread - self.gf_bread)
        num_gf_content_kitchen_current = len(kitchen_content.intersection(self.gf_content))
        num_reg_content_kitchen_current = len(kitchen_content - self.gf_content)
        num_notexist_sandwich_current = len(notexist_sandwiches)

        # Potential makable counts based on current ingredients/notexist
        gf_makable_potential = min(num_gf_bread_kitchen_current, num_gf_content_kitchen_current, num_notexist_sandwich_current)

        # --- Greedy assignment and cost calculation ---
        h = 0

        # Satisfy GF demand (allergic children)
        # 1. From trays at waiting places (Cost 1)
        for p in waiting_places:
            can_use = min(demand_gf, available_gf_sandwiches_at_waiting_place[p])
            h += can_use * 1
            demand_gf -= can_use
            available_gf_sandwiches_at_waiting_place[p] -= can_use # Consume

        # 2. From trays elsewhere (Cost 2)
        can_use = min(demand_gf, available_gf_sandwiches_on_trays_elsewhere)
        h += can_use * 2
        demand_gf -= can_use
        available_gf_sandwiches_on_trays_elsewhere -= can_use # Consume

        # 3. From kitchen (Cost 3)
        can_use = min(demand_gf, available_gf_sandwiches_kitchen)
        h += can_use * 3
        demand_gf -= can_use
        available_gf_sandwiches_kitchen -= can_use # Consume

        # 4. From makable GF (Cost 4)
        can_use = min(demand_gf, gf_makable_potential)
        h += can_use * 4
        demand_gf -= can_use
        # Consume ingredients/notexist for GF sandwiches from current counts
        used_notexist_for_gf = can_use
        num_notexist_sandwich_current -= used_notexist_for_gf
        num_gf_bread_kitchen_current -= used_notexist_for_gf # Assuming 1 bread per sandwich
        num_gf_content_kitchen_current -= used_notexist_for_gf # Assuming 1 content per sandwich


        # Penalty for unmet GF demand
        if demand_gf > 0:
             h += demand_gf * 1000 # Large penalty

        # Satisfy Regular demand (non-allergic children)
        # 1. From remaining GF sandwiches at waiting places (Cost 1)
        for p in waiting_places:
            can_use = min(demand_reg, available_gf_sandwiches_at_waiting_place[p])
            h += can_use * 1
            demand_reg -= can_use
            available_gf_sandwiches_at_waiting_place[p] -= can_use # Consume

        # 2. From Regular sandwiches at waiting places (Cost 1)
        for p in waiting_places:
            can_use = min(demand_reg, available_reg_sandwiches_at_waiting_place[p])
            h += can_use * 1
            demand_reg -= can_use
            available_reg_sandwiches_at_waiting_place[p] -= can_use # Consume

        # 3. From remaining GF sandwiches on trays elsewhere (Cost 2)
        can_use = min(demand_reg, available_gf_sandwiches_on_trays_elsewhere)
        h += can_use * 2
        demand_reg -= can_use
        available_gf_sandwiches_on_trays_elsewhere -= can_use # Consume

        # 4. From Regular sandwiches on trays elsewhere (Cost 2)
        can_use = min(demand_reg, available_reg_sandwiches_on_trays_elsewhere)
        h += can_use * 2
        demand_reg -= can_use
        available_reg_sandwiches_on_trays_elsewhere -= can_use # Consume

        # 5. From remaining GF sandwiches in kitchen (Cost 3)
        can_use = min(demand_reg, available_gf_sandwiches_kitchen)
        h += can_use * 3
        demand_reg -= can_use
        available_gf_sandwiches_kitchen -= can_use # Consume

        # 6. From Regular sandwiches in kitchen (Cost 3)
        can_use = min(demand_reg, available_reg_sandwiches_kitchen)
        h += can_use * 3
        demand_reg -= can_use
        available_reg_sandwiches_kitchen -= can_use # Consume

        # 7. From remaining makable GF (Cost 4) - already accounted for in GF demand step, no need to use again

        # 8. From makable Regular (Cost 4) - use remaining ingredients/notexist
        # Recalculate available ingredients after GF sandwich making
        # num_gf_bread_kitchen_current and num_gf_content_kitchen_current were updated
        # num_reg_bread_kitchen_current and num_reg_content_kitchen_current were NOT updated
        rem_any_bread_kitchen = num_gf_bread_kitchen_current + num_reg_bread_kitchen_current
        rem_any_content_kitchen = num_gf_content_kitchen_current + num_reg_content_kitchen_current

        reg_makable_remaining = min(rem_any_bread_kitchen, rem_any_content_kitchen, num_notexist_sandwich_current)

        can_use = min(demand_reg, reg_makable_remaining)
        h += can_use * 4
        demand_reg -= can_use
        # Consume ingredients/notexist for Reg sandwiches (not strictly needed for heuristic value)
        # used_notexist_for_reg = can_use
        # num_notexist_sandwich_current -= used_notexist_for_reg
        # ... ingredients consumption ...


        # Penalty for unmet Regular demand
        if demand_reg > 0:
            h += demand_reg * 1000 # Large penalty


        return h
