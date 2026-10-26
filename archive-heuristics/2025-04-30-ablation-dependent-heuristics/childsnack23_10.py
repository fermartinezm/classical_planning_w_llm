import math

class childsnackHeuristic:
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    The heuristic estimates the number of actions required to reach a goal state
    by summing the number of unserved children (representing the final 'serve' action)
    and the estimated minimum number of actions (make, put-on-tray, move-tray)
    needed to get enough suitable sandwiches onto trays and delivered to the
    locations where unserved children are waiting. It prioritizes using sandwiches
    that are closer to being served (already on trays at other locations, then
    in the kitchen, then needing to be made from ingredients). If the total
    available sandwiches/ingredients are insufficient to meet the needs of
    all unserved children, the heuristic returns infinity.

    Assumptions:
    - The planning task is solvable (i.e., there are enough total ingredients
      and sandwiches across the entire problem to serve all children). The
      heuristic specifically checks if enough ingredients are in the kitchen
      or can be moved there (implicitly via sandwiches/trays), returning
      infinity if not.
    - The heuristic is not admissible; it is designed for greedy best-first search
      to minimize expanded nodes.
    - The cost of moving a tray is 1, putting a sandwich on a tray is 1,
      making a sandwich is 1, and serving a sandwich is 1. These costs are
      reflected in the heuristic calculation stages.
    - All relevant objects (children, places, sandwiches, trays, bread, content)
      are mentioned in the static facts or the initial state facts.

    Heuristic Initialization:
    The constructor processes the static facts and initial state facts from the task
    to build lookup structures for:
    - Identifying allergic and non-allergic children.
    - Mapping children to their waiting places.
    - Identifying gluten-free bread and content portions.
    - Identifying all children, places, and sandwiches defined in the problem.
    - Identifying the goal children.
    - Identifying all trays defined in the problem.

    Step-By-Step Thinking for Computing Heuristic:
    1.  **Identify Unserved Children:** Determine which goal children are not yet served in the current state. Count the total number of unserved children (`N_unserved`). This contributes `N_unserved` to the heuristic (representing the final 'serve' action for each).
    2.  **Group Needs by Location and Allergy:** For each unserved child, identify their waiting place and allergy status (determining the required sandwich type: gluten-free or regular). Count how many children need a specific sandwich type at each specific location (`needed_at_place[place][allergy]`).
    3.  **Count Available Sandwiches on Trays at Locations:** For each place and allergy type, count how many suitable sandwiches are currently on trays located at that place (`available_ontray_at_place[place][allergy]`).
    4.  **Calculate Deficit at Locations:** For each place and allergy type, calculate the number of suitable sandwiches that are still needed on trays at that location (`deficit_at_place[place][allergy] = max(0, needed_at_place[place][allergy] - available_ontray_at_place[place][allergy])`). Sum these deficits to get `total_deficit_gf` and `total_deficit_reg`.
    5.  **Count Available Supply:** Count suitable sandwiches available at different stages *beyond* those already satisfying needs at deficit locations:
        - On trays at places where *no* unserved child is waiting (`S_ontray_elsewhere`). These can be moved (cost 1).
        - In the kitchen (`S_kitchen`). These need putting on a tray and moving (cost 1 + 1 = 2).
        - Makable from ingredients in the kitchen (`S_ingredients`). These need making, putting on a tray, and moving (cost 1 + 1 + 1 = 3). Ingredient availability is calculated considering GF ingredients are prioritized for GF sandwiches.
    6.  **Satisfy Deficit Greedily:** Attempt to satisfy the `total_deficit_gf` and `total_deficit_reg` using the available supply, starting with the cheapest options (cost 1), then cost 2, then cost 3.
    7.  **Calculate Supply Cost:** Sum the costs incurred in step 6. If the total available supply (from all stages) is less than the total deficit for either GF or regular sandwiches, it's impossible to satisfy all needs from current kitchen ingredients and existing sandwiches/trays, so the heuristic is infinity.
    8.  **Total Heuristic:** The total heuristic value is `N_unserved` + `cost_gf` + `cost_reg`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic with static information from the task.
        """
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.waiting_place = {} # child_name -> place_name
        self.no_gluten_bread = set()
        self.no_gluten_content = set()
        self.all_places = set()
        self.all_children = set()
        self.all_sandwiches = set()
        self.all_trays = set()
        self.goal_children = set()

        # Process static facts
        for fact in task.static:
            parts = fact.strip('()').split()
            predicate = parts[0]
            if predicate == 'allergic_gluten':
                self.allergic_children.add(parts[1])
            elif predicate == 'not_allergic_gluten':
                self.not_allergic_children.add(parts[1])
            elif predicate == 'waiting':
                self.all_children.add(parts[1])
                self.all_places.add(parts[2])
                self.waiting_place[parts[1]] = parts[2]
            elif predicate == 'no_gluten_bread':
                self.no_gluten_bread.add(parts[1])
            elif predicate == 'no_gluten_content':
                self.no_gluten_content.add(parts[1])

        # Process initial state facts to find all sandwiches and trays
        for fact in task.initial_state:
             parts = fact.strip('()').split()
             predicate = parts[0]
             if predicate == 'at': # (at tray place)
                 self.all_trays.add(parts[1])
                 self.all_places.add(parts[2])
             elif predicate == 'ontray': # (ontray sandwich tray)
                 self.all_sandwiches.add(parts[1])
                 self.all_trays.add(parts[2])
             elif predicate == 'at_kitchen_sandwich': # (at_kitchen_sandwich sandwich)
                 self.all_sandwiches.add(parts[1])
             elif predicate == 'notexist': # (notexist sandwich)
                 self.all_sandwiches.add(parts[1]) # Sandwiches are defined by notexist in init

        self.all_places.add('kitchen') # Ensure kitchen is always a place

        # Extract goal children
        if isinstance(task.goals, frozenset):
             for goal_fact in task.goals:
                 parts = goal_fact.strip('()').split()
                 if parts[0] == 'served':
                     self.goal_children.add(parts[1])

    def __call__(self, state):
        """
        Computes the domain-dependent heuristic value for the given state.
        """
        INF = float('inf')

        # --- Parse State Facts ---
        served_children = set()
        kitchen_bread = set()
        kitchen_content = set()
        no_gluten_sandwiches = set()
        kitchen_sandwiches = set() # Sandwiches in kitchen
        ontray_map = {} # sandwich_name -> tray_name
        tray_location = {} # tray_name -> place_name

        # Determine existing sandwiches by checking for the absence of 'notexist'
        notexist_sandwiches_in_state = set()
        for fact in state:
             parts = fact.strip('()').split()
             if parts[0] == 'notexist':
                 notexist_sandwiches_in_state.add(parts[1])

        existing_sandwiches = self.all_sandwiches - notexist_sandwiches_in_state

        for fact in state:
            parts = fact.strip('()').split()
            predicate = parts[0]
            if predicate == 'served':
                served_children.add(parts[1])
            elif predicate == 'at_kitchen_bread':
                kitchen_bread.add(parts[1])
            elif predicate == 'at_kitchen_content':
                kitchen_content.add(parts[1])
            elif predicate == 'at_kitchen_sandwich':
                 kitchen_sandwiches.add(parts[1])
            elif predicate == 'ontray':
                 ontray_map[parts[1]] = parts[2]
            elif predicate == 'no_gluten_sandwich':
                 no_gluten_sandwiches.add(parts[1])
            elif predicate == 'at':
                 tray_location[parts[1]] = parts[2]

        # --- Heuristic Calculation ---

        # 1. Unserved Children
        unserved_children = self.goal_children - served_children
        N_unserved = len(unserved_children)

        # If no unserved children, goal is reached
        if N_unserved == 0:
            return 0

        # 2. Group Needs by Location and Allergy
        needed_at_place_gf = {} # place -> count
        needed_at_place_reg = {} # place -> count
        places_with_unserved_children = set()

        for child in unserved_children:
            place = self.waiting_place.get(child)
            if place is None:
                 # Goal child has no waiting place defined in static facts - unsolvable
                 return INF

            places_with_unserved_children.add(place)

            if child in self.allergic_children:
                needed_at_place_gf[place] = needed_at_place_gf.get(place, 0) + 1
            elif child in self.not_allergic_children:
                needed_at_place_reg[place] = needed_at_place_reg.get(place, 0) + 1
            else:
                 # Goal child is neither allergic nor not_allergic - unsolvable
                 return INF

        # 3. Count Available Sandwiches on Trays at Locations
        available_ontray_at_place_gf = {} # place -> count
        available_ontray_at_place_reg = {} # place -> count
        sandwiches_on_tray = {} # tray_name -> list of sandwich_names

        for s, t in ontray_map.items():
            sandwiches_on_tray.setdefault(t, []).append(s)

        for tray, place in tray_location.items():
            if tray in sandwiches_on_tray:
                for s in sandwiches_on_tray[tray]:
                    if s in no_gluten_sandwiches:
                        available_ontray_at_place_gf[place] = available_ontray_at_place_gf.get(place, 0) + 1
                    else:
                        available_ontray_at_place_reg[place] = available_ontray_at_place_reg.get(place, 0) + 1

        # 4. Calculate Deficit at Locations
        deficit_at_place_gf = {}
        deficit_at_place_reg = {}
        total_deficit_gf = 0
        total_deficit_reg = 0

        for place, count in needed_at_place_gf.items():
            available = available_ontray_at_place_gf.get(place, 0)
            deficit = max(0, count - available)
            deficit_at_place_gf[place] = deficit
            total_deficit_gf += deficit

        for place, count in needed_at_place_reg.items():
            available = available_ontray_at_place_reg.get(place, 0)
            deficit = max(0, count - available)
            deficit_at_place_reg[place] = deficit
            total_deficit_reg += deficit

        # If no deficit, all needed sandwiches are already on trays at the right places
        if total_deficit_gf == 0 and total_deficit_reg == 0:
             return N_unserved # Only serve actions needed

        # 5. Count Available Supply (beyond what's already at needed locations)

        # Sandwiches on trays elsewhere (cost 1 to move)
        # These are suitable sandwiches on trays located at places where *no* unserved child is waiting.
        S_ontray_elsewhere_gf = 0
        S_ontray_elsewhere_reg = 0

        for tray, place in tray_location.items():
             if place not in places_with_unserved_children and place != 'kitchen':
                 if tray in sandwiches_on_tray:
                     for s in sandwiches_on_tray[tray]:
                         if s in no_gluten_sandwiches:
                             S_ontray_elsewhere_gf += 1
                         else:
                             S_ontray_elsewhere_reg += 1

        # Sandwiches in kitchen (cost 2 to put + move)
        S_kitchen_gf = len([s for s in kitchen_sandwiches if s in no_gluten_sandwiches])
        S_kitchen_reg = len([s for s in kitchen_sandwiches if s not in no_gluten_sandwiches])

        # Sandwiches makable from ingredients in kitchen (cost 3 to make + put + move)
        gf_bread_kit = len([b for b in kitchen_bread if b in self.no_gluten_bread])
        reg_bread_kit = len([b for b in kitchen_bread if b not in self.no_gluten_bread])
        gf_content_kit = len([c for c in kitchen_content if c in self.no_gluten_content])
        reg_content_kit = len([c for c in kitchen_content if c not in self.no_gluten_content])
        total_bread_kit = gf_bread_kit + reg_bread_kit
        total_content_kit = gf_content_kit + reg_content_kit

        S_ingredients_gf = min(gf_bread_kit, gf_content_kit)

        # 6. Satisfy Deficit Greedily and 7. Calculate Supply Cost

        cost_gf = 0
        rem_deficit_gf = total_deficit_gf

        # Cost 1: Sandwiches on trays elsewhere
        use = min(rem_deficit_gf, S_ontray_elsewhere_gf)
        cost_gf += use * 1
        rem_deficit_gf -= use

        # Cost 2: Sandwiches in kitchen
        use = min(rem_deficit_gf, S_kitchen_gf)
        cost_gf += use * 2
        rem_deficit_gf -= use

        # Cost 3: Sandwiches from ingredients
        use = min(rem_deficit_gf, S_ingredients_gf)
        cost_gf += use * 3
        rem_deficit_gf -= use

        # Check if GF deficit can be met
        if rem_deficit_gf > 0:
            return INF

        # Calculate ingredients used for GF (needed for regular calculation)
        # This is the number of GF sandwiches we *had* to make from ingredients
        # to satisfy the GF deficit after using cheaper options.
        gf_ing_used_from_supply = max(0, total_deficit_gf - S_ontray_elsewhere_gf - S_kitchen_gf)
        gf_ing_used_from_supply = min(gf_ing_used_from_supply, S_ingredients_gf) # Cannot use more ingredients than available

        # Recalculate available ingredients for regular sandwiches
        remaining_total_bread_after_gf = total_bread_kit - gf_ing_used_from_supply
        remaining_total_content_after_gf = total_content_kit - gf_ing_used_from_supply
        S_ingredients_reg = min(remaining_total_bread_after_gf, remaining_total_content_after_gf)

        cost_reg = 0
        rem_deficit_reg = total_deficit_reg

        # Cost 1: Sandwiches on trays elsewhere
        use = min(rem_deficit_reg, S_ontray_elsewhere_reg)
        cost_reg += use * 1
        rem_deficit_reg -= use

        # Cost 2: Sandwiches in kitchen
        use = min(rem_deficit_reg, S_kitchen_reg)
        cost_reg += use * 2
        rem_deficit_reg -= use

        # Cost 3: Sandwiches from ingredients
        use = min(rem_deficit_reg, S_ingredients_reg)
        cost_reg += use * 3
        rem_deficit_reg -= use

        # Check if Reg deficit can be met
        if rem_deficit_reg > 0:
            return INF

        # 8. Total Heuristic
        return N_unserved + cost_gf + cost_reg
