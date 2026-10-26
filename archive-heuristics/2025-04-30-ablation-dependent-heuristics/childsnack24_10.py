import re

class childsnackHeuristic:
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    The heuristic estimates the cost to reach the goal (all children served)
    by summing up the estimated costs of the necessary steps in the snack
    preparation and delivery pipeline:
    1. Making sandwiches (if needed, considering gluten constraints).
    2. Putting sandwiches onto trays (if needed).
    3. Moving trays to the kitchen (if needed for step 2).
    4. Moving trays to the locations where children are waiting (if needed).
    5. Moving trays that are currently at irrelevant locations.
    6. Serving the children (one action per unserved child).

    It calculates the deficit of required items (sandwiches, sandwiches on trays,
    trays at locations) and estimates the actions needed to fulfill these deficits
    based on available resources (ingredients, notexist sandwich objects, trays,
    sandwich locations).

    Assumptions:
    - The domain uses the predicates and action structure described in the PDDL file.
    - Object names can be reliably extracted from facts using simple string splitting
      based on the expected PDDL fact format `(predicate obj1 obj2 ...)`.
    - Tray capacity is effectively 1 for heuristic calculation purposes (each served
      child requires a distinct sandwich, and sandwiches are served from trays).
    - The problem is solvable within the available objects (e.g., enough total trays,
      enough total sandwich objects). The heuristic does not explicitly check for
      unsolvability due to insufficient objects, but resource counts cap the
      estimated 'make' actions.
    - The cost of moving a tray between any two places is 1 action.

    Heuristic Initialization:
    The constructor processes the static facts from the task:
    - Identifies allergic and non-allergic children.
    - Records the waiting location for each child.
    - Identifies no-gluten bread and content portions.
    - Identifies all tray objects present in the initial state (assuming all trays
      are mentioned in `(at ...)` or `(ontray ...)` facts in the initial state).

    Step-By-Step Thinking for Computing Heuristic:
    1.  **Identify Unserved Children:** Determine which children are not yet served
        by checking for the `(served ?c)` fact in the current state. Count the total
        number of unserved children (`N_unserved`), and separate them into allergic
        (`N_allergic_unserved`) and non-allergic (`N_non_allergic_unserved`). If
        `N_unserved` is 0, the goal is reached, and the heuristic is 0.

    2.  **Count Available Resources:**
        -   Count available bread and content portions in the kitchen, distinguishing
            between no-gluten and gluten types.
        -   Count available sandwich objects that `notexist`.
        -   Count available sandwiches (already made), distinguishing between those
            `at_kitchen_sandwich` and those `ontray`, and also by gluten status.
        -   Record the current location of each tray. Count trays `at kitchen`.

    3.  **Estimate Sandwiches to Make:**
        -   Calculate the deficit of no-gluten sandwiches needed (`N_allergic_unserved`)
            versus those available (`at_kitchen_sandwich` or `ontray`).
        -   Calculate the deficit of total sandwiches needed (`N_unserved`) versus
            those available.
        -   Estimate the number of no-gluten sandwiches that *must* be made (`NG_to_make`),
            limited by the no-gluten ingredient pairs and `notexist` objects available
            in the kitchen. Add `NG_to_make` to the heuristic.
        -   Estimate the number of additional sandwiches (any type) that need to be
            made (`Any_to_make`) to meet the total deficit, limited by remaining
            ingredient pairs and `notexist` objects. Add `Any_to_make` to the heuristic.
        -   The total sandwiches made is `N_sandwiches_made = NG_to_make + Any_to_make`.

    4.  **Estimate Sandwiches to Put on Tray:**
        -   Calculate the number of sandwiches needed on trays (`N_unserved`) versus
            those already on trays (`N_ontray_initial`).
        -   The deficit must be filled by sandwiches currently `at_kitchen_sandwich`
            (either initially or just made).
        -   Estimate the number of sandwiches to move from kitchen to tray (`N_put_on_tray`),
            limited by the deficit and the number of sandwiches available in the kitchen.
            Add `N_put_on_tray` to the heuristic.

    5.  **Estimate Trays to Move to Kitchen:**
        -   The `put_on_tray` action requires trays to be `at kitchen`.
        -   Estimate the number of trays that need to be moved *to* the kitchen
            (`N_move_trays_to_kitchen`) to support the `N_put_on_tray` actions,
            limited by the number of trays not already at the kitchen. Add
            `N_move_trays_to_kitchen` to the heuristic.

    6.  **Estimate Trays to Move to Children's Locations:**
        -   Identify the distinct places where unserved children are waiting.
        -   Count the distinct places where unserved children are waiting but where
            no tray is currently located. Each such place needs a tray moved *to* it.
            Add this count to the heuristic.
        -   Count trays that are currently at locations that are *not* the kitchen
            and *not* one of the places where unserved children are waiting. These
            'misplaced' trays might need one move to become useful (either go to
            kitchen or a child's place). Add this count to the heuristic.

    7.  **Estimate Serving Actions:**
        -   Each unserved child requires one `serve` action. Add `N_unserved` to the
            heuristic. This also serves as a base cost and ensures the heuristic is
            non-zero for non-goal states.

    The total heuristic value is the sum of the costs estimated in steps 3, 4, 5, 6, and 7.
    """
    def __init__(self, task):
        """
        Initializes the heuristic with static information from the task.

        Args:
            task: The planning task object.
        """
        self.task = task
        self.static_facts = task.static

        self.allergic_children = set()
        self.not_allergic_children = set()
        self.waiting_children_locations = {} # child -> place
        self.no_gluten_bread = set()
        self.no_gluten_content = set()
        self.total_trays = set() # To count total trays

        # Extract static information
        for fact in self.static_facts:
            if fact.startswith('(allergic_gluten '):
                child = fact.split(' ')[1][:-1]
                self.allergic_children.add(child)
            elif fact.startswith('(not_allergic_gluten '):
                child = fact.split(' ')[1][:-1]
                self.not_allergic_children.add(child)
            elif fact.startswith('(waiting '):
                parts = fact.split(' ')
                if len(parts) == 3:
                    child = parts[1]
                    place = parts[2][:-1]
                    self.waiting_children_locations[child] = place
            elif fact.startswith('(no_gluten_bread '):
                bread = fact.split(' ')[1][:-1]
                self.no_gluten_bread.add(bread)
            elif fact.startswith('(no_gluten_content '):
                content = fact.split(' ')[1][:-1]
                self.no_gluten_content.add(content)

        self.all_children = self.allergic_children | self.not_allergic_children

        # Extract total trays from initial state facts that mention trays
        # This assumes all trays are mentioned in initial state facts like (at ...) or (ontray ...)
        for fact in task.initial_state:
             if fact.startswith('(at '):
                 parts = fact.split(' ')
                 # Fact format: (at tray1 kitchen)
                 if len(parts) == 3:
                     tray = parts[1]
                     self.total_trays.add(tray)
             elif fact.startswith('(ontray '):
                 parts = fact.split(' ')
                 # Fact format: (ontray sandw1 tray1)
                 if len(parts) == 3:
                     tray = parts[2][:-1] # Remove ')'
                     self.total_trays.add(tray)


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        Args:
            state: The current state (frozenset of strings).

        Returns:
            An integer representing the estimated cost to reach the goal.
        """
        # --- Step 1: Count unserved children ---
        served_children = {fact.split(' ')[1][:-1] for fact in state if fact.startswith('(served ')}
        unserved_children = self.all_children - served_children

        if not unserved_children:
            return 0 # Goal reached

        N_unserved = len(unserved_children)

        # --- Step 2: Count needed sandwiches by type ---
        N_allergic_unserved = len([c for c in unserved_children if c in self.allergic_children])
        # N_non_allergic_unserved = N_unserved - N_allergic_unserved # Not explicitly needed for deficit calculation below

        # --- Step 3: Count available sandwiches and tray locations ---
        N_no_gluten_sandwich_kitchen = 0
        N_gluten_sandwich_kitchen = 0
        N_no_gluten_sandwich_ontray = 0
        N_gluten_sandwich_ontray = 0
        sandwiches_ontray = {} # {sandwich: tray}
        trays_at_location = {} # {tray: place}
        sandwich_is_no_gluten = set()

        for fact in state:
            if fact.startswith('(no_gluten_sandwich '):
                 sandwich = fact.split(' ')[1][:-1]
                 sandwich_is_no_gluten.add(sandwich)

        for fact in state:
             if fact.startswith('(at_kitchen_sandwich '):
                 sandwich = fact.split(' ')[1][:-1]
                 if sandwich in sandwich_is_no_gluten:
                     N_no_gluten_sandwich_kitchen += 1
                 else:
                     N_gluten_sandwich_kitchen += 1
             elif fact.startswith('(ontray '):
                 parts = fact.split(' ')
                 # Fact format: (ontray sandw1 tray1)
                 if len(parts) == 3:
                     sandwich = parts[1]
                     tray = parts[2][:-1]
                     sandwiches_ontray[sandwich] = tray
                     if sandwich in sandwich_is_no_gluten:
                         N_no_gluten_sandwich_ontray += 1
                     else:
                         N_gluten_sandwich_ontray += 1
             elif fact.startswith('(at '):
                 parts = fact.split(' ')
                 # Fact format: (at tray1 kitchen)
                 if len(parts) == 3:
                     tray = parts[1]
                     place = parts[2][:-1]
                     trays_at_location[tray] = place

        N_no_gluten_available_anywhere = N_no_gluten_sandwich_kitchen + N_no_gluten_sandwich_ontray
        N_gluten_available_anywhere = N_gluten_sandwich_kitchen + N_gluten_sandwich_ontray
        N_sandwich_available_anywhere = N_no_gluten_available_anywhere + N_gluten_available_anywhere
        N_ontray_initial = N_no_gluten_sandwich_ontray + N_gluten_sandwich_ontray
        N_kitchen_initial = N_no_gluten_sandwich_kitchen + N_gluten_sandwich_kitchen
        N_trays_at_kitchen = len([tray for tray, place in trays_at_location.items() if place == 'kitchen'])
        N_total_trays = len(self.total_trays)


        # --- Step 4: Count available ingredients and notexist objects ---
        N_no_gluten_bread_kitchen = 0
        N_gluten_bread_kitchen = 0
        N_no_gluten_content_kitchen = 0
        N_gluten_content_kitchen = 0
        N_notexist_sandwich = 0

        for fact in state:
            if fact.startswith('(at_kitchen_bread '):
                bread = fact.split(' ')[1][:-1]
                if bread in self.no_gluten_bread:
                    N_no_gluten_bread_kitchen += 1
                else:
                    N_gluten_bread_kitchen += 1
            elif fact.startswith('(at_kitchen_content '):
                content = fact.split(' ')[1][:-1]
                if content in self.no_gluten_content:
                    N_no_gluten_content_kitchen += 1
                else:
                    N_gluten_content_kitchen += 1
            elif fact.startswith('(notexist '):
                N_notexist_sandwich += 1

        # --- Step 7: Calculate cost components ---
        heuristic_value = 0

        # 7a: Sandwiches to make
        # Need N_allergic_unserved no-gluten sandwiches total.
        # Have N_no_gluten_available_anywhere.
        NG_deficit = max(0, N_allergic_unserved - N_no_gluten_available_anywhere)

        # Need N_unserved total sandwiches.
        # Have N_sandwich_available_anywhere.
        Total_deficit = max(0, N_unserved - N_sandwich_available_anywhere)

        # Number of no-gluten sandwiches we must make
        available_no_gluten_pairs = min(N_no_gluten_bread_kitchen, N_no_gluten_content_kitchen)
        NG_to_make = min(NG_deficit, available_no_gluten_pairs, N_notexist_sandwich)
        heuristic_value += NG_to_make

        # Number of additional sandwiches (any type) to make
        remaining_notexist = N_notexist_sandwich - NG_to_make
        remaining_ng_bread = N_no_gluten_bread_kitchen - NG_to_make
        remaining_ng_content = N_no_gluten_content_kitchen - NG_to_make
        remaining_g_bread = N_gluten_bread_kitchen
        remaining_g_content = N_gluten_content_kitchen

        available_any_pairs = min(remaining_ng_bread + remaining_g_bread, remaining_ng_content + remaining_g_content)

        Any_to_make = min(max(0, Total_deficit - NG_to_make), available_any_pairs, remaining_notexist)
        heuristic_value += Any_to_make

        N_sandwiches_made = NG_to_make + Any_to_make

        # 7b: Sandwiches to put on tray
        # Total sandwiches needed on trays is N_unserved.
        # Sandwiches already on trays: N_ontray_initial
        # Sandwiches available in kitchen (initial + just made)
        N_kitchen_available_now = N_kitchen_initial + N_sandwiches_made

        # Number of sandwiches that need to move from kitchen to tray
        N_put_on_tray = min(max(0, N_unserved - N_ontray_initial), N_kitchen_available_now)
        heuristic_value += N_put_on_tray

        # 7c: Trays to move to kitchen (for put_on_tray)
        # Need N_put_on_tray trays at kitchen.
        # Have N_trays_at_kitchen.
        # Number of trays that are NOT at kitchen: N_total_trays - N_trays_at_kitchen
        N_move_trays_to_kitchen = min(max(0, N_put_on_tray - N_trays_at_kitchen), N_total_trays - N_trays_at_kitchen)
        heuristic_value += N_move_trays_to_kitchen

        # 7d: Trays to move to children's locations
        # Identify places where unserved children are waiting
        places_with_unserved = set()
        for child in unserved_children:
            # Should always be true based on domain, but check defensively
            if child in self.waiting_children_locations:
                places_with_unserved.add(self.waiting_children_locations[child])

        # Count places that need a tray moved there (places with unserved children but no tray)
        places_without_tray_and_unserved = set()
        for place in places_with_unserved:
             has_tray_at_place = False
             for tray, loc in trays_at_location.items():
                 if loc == place:
                     has_tray_at_place = True
                     break
             if not has_tray_at_place:
                 places_without_tray_and_unserved.add(place)

        # Each such place needs a tray moved there. This costs 1 move per place.
        heuristic_value += len(places_without_tray_and_unserved)

        # Also, consider trays that are at locations that are NOT kitchen and NOT a place with unserved children.
        # These trays are currently useless and might need one move to become useful (either go to kitchen or a child's place).
        # This counts trays that are 'out of the way'.
        N_trays_at_other_places = 0
        for tray, place in trays_at_location.items():
             if place != 'kitchen' and place not in places_with_unserved:
                 N_trays_at_other_places += 1
        heuristic_value += N_trays_at_other_places # Each might need one move to a useful location


        # 7e: Serving actions
        # Each unserved child needs one serve action. This is the final step.
        heuristic_value += N_unserved

        return heuristic_value
