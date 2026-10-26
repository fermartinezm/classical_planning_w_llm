from heuristics.heuristic_base import Heuristic
# Assuming Task class is available in the environment where this heuristic runs
# from task import Task # Not needed for import, Task object is passed

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    Estimates the number of actions required to reach the goal state (all children served)
    by summing the number of unserved children (representing the final 'serve' action)
    and the estimated number of sandwiches that need to pass through earlier stages
    of the preparation and delivery pipeline (make, put_on_tray, move_tray).
    The heuristic counts the deficit of suitable sandwiches at each stage, working
    backwards from the goal state, and sums these deficits.

    Assumptions:
    - All actions have a unit cost of 1.
    - There are enough trays available at the kitchen when needed for 'put_on_tray'
      and subsequent 'move_tray' actions originating from the kitchen. The heuristic
      does not explicitly model tray availability at the kitchen beyond counting
      sandwiches that need to pass through the 'put_on_tray' stage.
    - There are enough 'notexist' sandwich objects available to make all needed
      sandwiches, provided ingredients are available. This is checked, and infinity
      is returned if not enough can be made.
    - The heuristic prioritizes using Gluten-Free sandwiches for allergic children
      first, then uses remaining GF sandwiches for non-allergic children, and finally
      uses Regular sandwiches for non-allergic children.

    Heuristic Initialization:
    The constructor processes the static facts from the task description to identify:
    - Which children are allergic to gluten.
    - The waiting location for each child.
    - The gluten status of bread and content portions.
    It also collects the names of all objects (children, trays, sandwiches, bread, content, places)
    present in the initial state and static facts, which are assumed to cover all relevant objects.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Identify all children who are waiting but not yet served. Count the total number
        of unserved children (`N_unserved`), the number of unserved allergic children
        (`Needed_GF`), and the number of unserved non-allergic children (`Needed_Reg`).
        If `N_unserved` is 0, the goal is reached, and the heuristic is 0.
        The base heuristic value starts at `N_unserved` (representing the final 'serve' action for each).
    2.  Parse the current state to determine the location of each tray, the location
        of each sandwich (at kitchen or on a specific tray), the gluten status of
        sandwiches, and the availability of bread and content portions in the kitchen,
        as well as 'notexist' sandwich objects.
    3.  Count the number of suitable sandwiches available at different stages of the
        preparation and delivery pipeline:
        -   Stage 4 (On tray at correct location): Sandwiches on trays located at a place
            where an unserved child is waiting, and suitable for that child (GF for allergic,
            any for non-allergic). Count GF and Regular ones separately (`Avail_ontray_at_loc_gf`,
            `Avail_ontray_at_loc_reg`).
        -   Stage 3 (On tray at wrong location): Sandwiches on trays not at the kitchen
            and not at a location where an unserved child is waiting, but suitable for
            some unserved child. Count GF and Regular ones (`Avail_GF_ontray_wrong_loc`,
            `Avail_Reg_ontray_wrong_loc`).
        -   Stage 2 (At kitchen): Sandwiches located at the kitchen, suitable for some
            unserved child. Count GF and Regular ones (`Avail_kitchen_gf`, `Avail_kitchen_reg`).
        -   Stage 1 (Can be made): Suitable sandwiches that can be created from available
            'notexist' objects and ingredients in the kitchen. Count GF and Regular ones
            (`Can_Make_GF`, `Can_Make_Reg`), accounting for ingredient sharing and GF
            ingredients usable for Regular sandwiches.
    4.  Calculate the number of sandwiches that must come from earlier stages (deficits),
        working backwards from the goal (`N_unserved` sandwiches needed at Stage 4).
        -   `D1`: Number of sandwiches needed that are not already at Stage 4. Calculated
            by subtracting the total needs met at Stage 4 (prioritizing GF for allergic,
            then GF for non-allergic, then Reg for non-allergic) from `N_unserved`.
            These `D1` sandwiches must come from Stage 3 or earlier. Add `D1` to the heuristic.
        -   `D2`: Number of sandwiches needed that are not already at Stage 4 or Stage 3.
            Calculated by subtracting the total needs met at Stage 3 (from `D1` remaining needs,
            prioritizing similarly) from `D1`. These `D2` sandwiches must come from Stage 2
            or earlier. Add `D2` to the heuristic.
        -   `D3`: Number of sandwiches needed that are not already at Stage 4, 3, or 2.
            Calculated by subtracting the total needs met at Stage 2 (from `D2` remaining needs,
            prioritizing similarly) from `D2`. These `D3` sandwiches must come from Stage 1.
            Add `D3` to the heuristic.
    5.  Check if the number of sandwiches needed from Stage 1 (`D3`) exceeds the number
        that can actually be made (`Can_Make_GF + Can_Make_Reg`). If so, the goal is
        unreachable from this state, and the heuristic returns infinity.
    6.  The final heuristic value is the sum of `N_unserved + D1 + D2 + D3`.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals
        self.static = task.static

        # Pre-process static information
        self.allergic_children = set()
        self.waiting_children_loc = {}  # child -> place
        self.bread_is_gf = {}  # bread -> bool
        self.content_is_gf = {}  # content -> bool

        # Collect all object names from static and initial state for identification
        self.all_children = set()
        self.all_trays = set()
        self.all_sandwiches = set()
        self.all_bread = set()
        self.all_content = set()
        self.all_places = {'kitchen'} # kitchen is a constant place

        # Helper to parse fact strings like '(predicate arg1 arg2)'
        def parse_fact(fact_string):
            parts = fact_string.strip('()').split()
            return parts[0], parts[1:]

        # Process static facts
        for fact_string in self.static:
            pred, args = parse_fact(fact_string)
            if pred == 'allergic_gluten':
                if len(args) == 1: self.allergic_children.add(args[0])
                if len(args) >= 1: self.all_children.update(args)
            elif pred == 'not_allergic_gluten':
                 if len(args) >= 1: self.all_children.update(args)
            elif pred == 'waiting':
                if len(args) == 2: self.waiting_children_loc[args[0]] = args[1]
                if len(args) >= 1: self.all_children.update([args[0]])
                if len(args) >= 2: self.all_places.update([args[1]])
            elif pred == 'no_gluten_bread':
                if len(args) == 1: self.bread_is_gf[args[0]] = True
                if len(args) >= 1: self.all_bread.update(args)
            elif pred == 'no_gluten_content':
                if len(args) == 1: self.content_is_gf[args[0]] = True
                if len(args) >= 1: self.all_content.update(args)
            # Collect other objects from static facts
            elif pred == 'at': # (at tray place)
                 if len(args) == 2:
                    self.all_trays.add(args[0])
                    self.all_places.add(args[1])
            elif pred == 'ontray': # (ontray sandwich tray)
                 if len(args) == 2:
                    self.all_sandwiches.add(args[0])
                    self.all_trays.add(args[1])
            elif pred == 'at_kitchen_sandwich': # (at_kitchen_sandwich sandwich)
                 if len(args) == 1: self.all_sandwiches.add(args[0])
            elif pred == 'at_kitchen_bread': # (at_kitchen_bread bread)
                 if len(args) == 1: self.all_bread.add(args[0])
            elif pred == 'at_kitchen_content': # (at_kitchen_content content)
                 if len(args) == 1: self.all_content.add(args[0])
            elif pred == 'notexist': # (notexist sandwich)
                 if len(args) == 1: self.all_sandwiches.add(args[0])
            elif pred == 'no_gluten_sandwich': # (no_gluten_sandwich sandwich)
                 if len(args) == 1: self.all_sandwiches.add(args[0])


        # Process initial state facts to ensure all objects are collected
        # (Static facts might not list all objects if they don't appear in static predicates)
        for fact_string in task.initial_state:
             pred, args = parse_fact(fact_string)
             if pred in ['allergic_gluten', 'not_allergic_gluten', 'served', 'waiting']:
                 if len(args) >= 1: self.all_children.update(args)
             elif pred in ['ontray', 'at']:
                 if len(args) >= 1:
                     if args[0].startswith('tray'): self.all_trays.add(args[0])
                     if len(args) > 1 and args[1] != 'kitchen' and not args[1].startswith('child') and not args[1].startswith('tray') and not args[1].startswith('sandw') and not args[1].startswith('bread') and not args[1].startswith('content'): self.all_places.add(args[1])
             elif pred in ['at_kitchen_sandwich', 'notexist', 'no_gluten_sandwich']:
                 if len(args) >= 1: self.all_sandwiches.update(args)
             elif pred in ['at_kitchen_bread', 'no_gluten_bread']:
                 if len(args) >= 1: self.all_bread.update(args)
             elif pred in ['at_kitchen_content', 'no_gluten_content']:
                 if len(args) >= 1: self.all_content.update(args)


    def __call__(self, node):
        state = node.state

        # Helper to parse fact strings like '(predicate arg1 arg2)'
        def parse_fact(fact_string):
            parts = fact_string.strip('()').split()
            return parts[0], parts[1:]

        # --- Step 1: Identify unserved children ---
        served_children = set()
        for fact_string in state:
            pred, args = parse_fact(fact_string)
            if pred == 'served':
                if len(args) >= 1: served_children.add(args[0])

        unserved_children = {c for c in self.waiting_children_loc if c not in served_children}
        N_unserved = len(unserved_children)

        if N_unserved == 0:
            return 0 # Goal reached

        h = N_unserved # Base cost: one 'serve' action per unserved child

        Needed_GF = len([c for c in unserved_children if c in self.allergic_children])
        Needed_Reg = len([c for c in unserved_children if c not in self.allergic_children])
        Total_Needed = Needed_GF + Needed_Reg # Should be equal to N_unserved

        # --- Step 2: Parse current state for object locations and properties ---
        tray_loc = {'kitchen': 'kitchen'} # Initialize kitchen as a place
        sandwich_is_gf = {}
        sandwich_loc = {} # sandwich -> 'kitchen' or tray_name
        kitchen_bread_avail = set()
        kitchen_content_avail = set()

        for fact_string in state:
            pred, args = parse_fact(fact_string)
            if pred == 'at': # (at tray place)
                if len(args) == 2: tray_loc[args[0]] = args[1]
            elif pred == 'no_gluten_sandwich': # (no_gluten_sandwich sandwich)
                if len(args) == 1: sandwich_is_gf[args[0]] = True
            elif pred == 'at_kitchen_sandwich': # (at_kitchen_sandwich sandwich)
                if len(args) == 1: sandwich_loc[args[0]] = 'kitchen'
            elif pred == 'ontray': # (ontray sandwich tray)
                if len(args) == 2: sandwich_loc[args[0]] = args[1] # args[1] is tray name
            elif pred == 'at_kitchen_bread': # (at_kitchen_bread bread)
                if len(args) == 1: kitchen_bread_avail.add(args[0])
            elif pred == 'at_kitchen_content': # (at_kitchen_content content)
                if len(args) == 1: kitchen_content_avail.add(args[0])
            # 'notexist' facts are not needed to determine existing sandwiches

        # Sandwiches that are not located are considered 'notexist'
        located_sandwiches = set(sandwich_loc.keys())
        notexist_sandwiches_avail = self.all_sandwiches - located_sandwiches


        # --- Step 3: Count available suitable sandwiches at different stages ---

        # Stage 4 (On tray at correct location)
        Avail_ontray_at_loc_gf = 0
        Avail_ontray_at_loc_reg = 0 # Regular sandwiches (not GF) at correct loc

        locations_with_unserved_allergic = {self.waiting_children_loc[c] for c in unserved_children if c in self.allergic_children}
        locations_with_unserved_nonallergic = {self.waiting_children_loc[c] for c in unserved_children if c not in self.allergic_children}

        for sandwich, tray in sandwich_loc.items():
            if tray == 'kitchen': continue # Kitchen sandwiches are Stage 2
            if tray not in tray_loc: continue # Tray location unknown

            curr_loc = tray_loc[tray]
            is_gf = sandwich_is_gf.get(sandwich, False)

            is_at_allergic_loc = curr_loc in locations_with_unserved_allergic
            is_at_nonallergic_loc = curr_loc in locations_with_unserved_nonallergic

            if is_at_allergic_loc or is_at_nonallergic_loc:
                # This sandwich is on a tray at a location with unserved children.
                # Is it suitable for *any* unserved child at this location?
                is_suitable_at_loc = False
                if is_at_allergic_loc and is_gf: is_suitable_at_loc = True # GF needed by allergic
                if is_at_nonallergic_loc: is_suitable_at_loc = True # Any needed by non-allergic

                if is_suitable_at_loc:
                    if is_gf: Avail_ontray_at_loc_gf += 1
                    else: Avail_ontray_at_loc_reg += 1

        # Stage 3 (On tray wrong location)
        Avail_GF_ontray_wrong_loc = 0
        Avail_Reg_ontray_wrong_loc = 0

        for sandwich, tray in sandwich_loc.items():
            if tray == 'kitchen': continue # Kitchen sandwiches are Stage 2
            if tray not in tray_loc: continue # Tray location unknown

            curr_loc = tray_loc[tray]
            is_gf = sandwich_is_gf.get(sandwich, False)

            is_at_allergic_loc = curr_loc in locations_with_unserved_allergic
            is_at_nonallergic_loc = curr_loc in locations_with_unserved_nonallergic

            # If it's not at a needed location and not at the kitchen, it's at a wrong location
            if not (is_at_allergic_loc or is_at_nonallergic_loc) and curr_loc != 'kitchen':
                 # Is it suitable for *any* unserved child anywhere?
                 is_suitable_anywhere = False
                 if Needed_GF > 0 and is_gf: is_suitable_anywhere = True # GF needed by allergic anywhere
                 if Needed_Reg > 0: is_suitable_anywhere = True # Any needed by non-allergic anywhere

                 if is_suitable_anywhere:
                    if is_gf: Avail_GF_ontray_wrong_loc += 1
                    else: Avail_Reg_ontray_wrong_loc += 1


        # Stage 2 (At kitchen)
        Avail_kitchen_gf = 0
        Avail_kitchen_reg = 0
        for sandwich, loc in sandwich_loc.items():
            if loc == 'kitchen':
                is_gf = sandwich_is_gf.get(sandwich, False)
                # Is it suitable for *any* unserved child anywhere?
                is_suitable_anywhere = False
                if Needed_GF > 0 and is_gf: is_suitable_anywhere = True # GF needed by allergic anywhere
                if Needed_Reg > 0: is_suitable_anywhere = True # Any needed by non-allergic anywhere

                if is_suitable_anywhere:
                    if is_gf: Avail_kitchen_gf += 1
                    else: Avail_kitchen_reg += 1

        # Stage 1 (Can be made)
        N_notexist = len(notexist_sandwiches_avail)
        N_gf_bread_kitchen = len([b for b in kitchen_bread_avail if self.bread_is_gf.get(b, False)])
        N_reg_bread_kitchen = len(kitchen_bread_avail) - N_gf_bread_kitchen
        N_gf_content_kitchen = len([c for c in kitchen_content_avail if self.content_is_gf.get(c, False)])
        N_reg_content_kitchen = len(kitchen_content_avail) - N_gf_content_kitchen

        # Calculate how many GF sandwiches can be made
        Can_Make_GF = min(N_notexist, N_gf_bread_kitchen, N_gf_content_kitchen)

        # Calculate remaining resources after making GF sandwiches
        Rem_notexist = N_notexist - Can_Make_GF
        Rem_gf_bread = N_gf_bread_kitchen - Can_Make_GF
        Rem_gf_content = N_gf_content_kitchen - Can_Make_GF

        # Available ingredients for regular sandwiches (regular + remaining GF)
        Avail_bread_for_reg = N_reg_bread_kitchen + Rem_gf_bread
        Avail_content_for_reg = N_reg_content_kitchen + Rem_gf_content

        # Calculate how many Regular sandwiches can be made
        Can_Make_Reg = min(Rem_notexist, Avail_bread_for_reg, Avail_content_for_reg)

        # --- Step 4 & 5: Calculate deficits and check reachability ---

        # Needs remaining after Stage 4 (on tray at correct location)
        met_gf_4 = min(Needed_GF, Avail_ontray_at_loc_gf)
        rem_gf_needed_4 = Needed_GF - met_gf_4

        met_reg_4_reg = min(Needed_Reg, Avail_ontray_at_loc_reg)
        rem_reg_needed_4 = Needed_Reg - met_reg_4_reg

        met_reg_4_gf = min(rem_reg_needed_4, Avail_ontray_at_loc_gf - met_gf_4)
        rem_reg_needed_4_after_gf = rem_reg_needed_4 - met_reg_4_gf

        # D1: Number of sandwiches needing to come from Stage 3 or earlier
        D1 = rem_gf_needed_4 + rem_reg_needed_4_after_gf
        h += D1 # Cost for delivery (move_tray or more)

        # Needs remaining after Stage 3 (on tray wrong location)
        met_gf_3 = min(rem_gf_needed_4, Avail_GF_ontray_wrong_loc)
        rem_gf_needed_3 = rem_gf_needed_4 - met_gf_3

        met_reg_3_reg = min(rem_reg_needed_4_after_gf, Avail_Reg_ontray_wrong_loc)
        rem_reg_needed_3 = rem_reg_needed_4_after_gf - met_reg_3_reg

        met_reg_3_gf = min(rem_reg_needed_3, Avail_GF_ontray_wrong_loc - met_gf_3)
        rem_reg_needed_3_after_gf = rem_reg_needed_3 - met_reg_3_gf

        # D2: Number of sandwiches needing to come from Stage 2 or earlier
        D2 = rem_gf_needed_3 + rem_reg_needed_3_after_gf
        h += D2 # Cost for put_on_tray (or make + put_on_tray)

        # Needs remaining after Stage 2 (at kitchen)
        met_gf_2 = min(rem_gf_needed_3, Avail_kitchen_gf)
        rem_gf_needed_2 = rem_gf_needed_3 - met_gf_2

        met_reg_2_reg = min(rem_reg_needed_3_after_gf, Avail_kitchen_reg)
        rem_reg_needed_2 = rem_reg_needed_3_after_gf - met_reg_2_reg

        met_reg_2_gf = min(rem_reg_needed_2, Avail_kitchen_gf - met_gf_2)
        rem_reg_needed_2_after_gf = rem_reg_needed_2 - met_reg_2_gf

        # D3: Number of sandwiches needing to come from Stage 1 (can make)
        D3 = rem_gf_needed_2 + rem_reg_needed_2_after_gf
        h += D3 # Cost for make_sandwich

        # Needs remaining after Stage 1 (can make)
        met_gf_1 = min(rem_gf_needed_2, Can_Make_GF)
        rem_gf_needed_1 = rem_gf_needed_2 - met_gf_1

        met_reg_1_reg = min(rem_reg_needed_2_after_gf, Can_Make_Reg)
        rem_reg_needed_1 = rem_reg_needed_2_after_gf - met_reg_1_reg

        met_reg_1_gf = min(rem_reg_needed_1, Can_Make_GF - met_gf_1)
        rem_reg_needed_1_after_gf = rem_reg_needed_1 - met_reg_1_gf

        # D4: Number of sandwiches needing to come from Stage 0 (impossible)
        D4 = rem_gf_needed_1 + rem_reg_needed_1_after_gf

        if D4 > 0:
            return float('inf') # Cannot make enough sandwiches

        # --- Step 6: Return total heuristic value ---
        return h
