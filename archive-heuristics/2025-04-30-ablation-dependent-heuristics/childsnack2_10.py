import re

class childsnackHeuristic:
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    The heuristic estimates the cost to reach the goal (all children served)
    by summing up the estimated steps required for each child that is not yet served.
    For each unserved child, it finds the "most advanced" state of a suitable
    sandwich intended for them (e.g., sandwich exists at kitchen, on a tray
    elsewhere, or on a tray at their location) and assigns a cost based on
    the remaining steps from that state to serving the child. If no suitable
    sandwich exists, it assumes one needs to be made.

    Assumptions:
    - The heuristic is non-admissible and designed for greedy best-first search.
    - It assumes that necessary resources (ingredients, trays) will eventually
      become available to perform the required actions, even if they are not
      immediately available in the current state (this is a relaxation).
    - It assumes that for each unserved child, we can dedicate one suitable
      sandwich and potentially one tray and associated actions (make, put-on-tray,
      move-tray, serve) independently of other children.

    Heuristic Initialization:
    The constructor processes the static facts from the task definition.
    It identifies:
    - All child objects and their allergy status (allergic or not_allergic).
    - The waiting place for each child.
    - All potential sandwich, tray, place, bread, and content objects defined in the problem
      by inspecting all possible ground facts (`task.facts`) and initial state facts.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Check if the state is a goal state (all children served). If yes, return 0.
    2.  Identify all children that are not yet served in the current state.
    3.  For each unserved child `c` waiting at place `p_c`:
        a.  Determine if the child is allergic or not using pre-calculated static info.
        b.  Initialize the estimated cost for this child to 4 (representing the need to make a sandwich, put it on a tray, move the tray, and serve). This is the maximum cost if no suitable sandwich currently exists.
        c.  Identify all sandwiches `s` that *exist* in the current state (i.e., `(notexist s)` is NOT in the state).
        d.  For each existing sandwich `s`:
            i.  Check if `s` is a no-gluten sandwich (`(no_gluten_sandwich s)` in state).
            ii. Check if `s` is suitable for child `c`:
                - If `c` is allergic, `s` is suitable only if it is a no-gluten sandwich.
                - If `c` is not allergic, `s` is always suitable.
            iii. If `s` is suitable for `c`:
                - Find the location of `s`:
                    - If `(at_kitchen_sandwich s)` is in state, location is 'kitchen_sandwich'.
                    - If `(ontray s t)` is in state for some tray `t`, find the location of `t` (`(at t p_t)` in state). The location of `s` is `p_t`.
                - Based on the location of the suitable sandwich `s`, update the estimated cost for child `c`:
                    - If location is `p_c` (child's waiting place): The sandwich is ready to be served. Estimated remaining steps: 1 (serve). Update child's cost to `min(current_cost, 1)`.
                    - If location is a place `p_t` where `p_t != p_c`: The sandwich is on a tray elsewhere. Estimated remaining steps: 1 (move tray) + 1 (serve) = 2. Update child's cost to `min(current_cost, 2)`.
                    - If location is 'kitchen_sandwich': The sandwich is in the kitchen, not on a tray. Estimated remaining steps: 1 (put on tray) + 1 (move tray) + 1 (serve) = 3. Update child's cost to `min(current_cost, 3)`.
        e.  After checking all existing sandwiches, the child's cost will be the minimum steps found (1, 2, 3) or remain 4 if no suitable sandwich exists.
    4.  The total heuristic value is the sum of the estimated costs for all unserved children.
    """

    def __init__(self, task):
        # Helper function to parse PDDL facts
        def parse_fact(fact_str):
            # Removes surrounding parentheses and splits into predicate and arguments
            parts = fact_str[1:-1].split()
            if not parts: # Handle empty fact string, though unlikely in valid PDDL
                return None, []
            return parts[0], parts[1:]

        self.task = task
        self.goals = task.goals
        self.static_info = task.static

        # Extract static information
        self.child_allergy = {} # child_name -> 'allergic' or 'not_allergic'
        self.child_waiting_place = {} # child_name -> place_name

        # Extract all possible objects from task.facts (all ground facts)
        # This is a robust way to get all objects mentioned in the domain/problem
        self.all_children = set()
        self.all_sandwiches = set()
        self.all_trays = set()
        self.all_places = set()
        self.all_bread = set()
        self.all_content = set()

        for fact_str in task.facts:
             pred, args = parse_fact(fact_str)
             if pred is None: continue # Skip invalid facts

             # Infer object types based on predicates they appear in
             if pred in ['allergic_gluten', 'not_allergic_gluten', 'served', 'waiting']:
                 if args: self.all_children.add(args[0])
                 if pred == 'waiting' and len(args) > 1: self.all_places.add(args[1])
             elif pred in ['at_kitchen_sandwich', 'ontray', 'no_gluten_sandwich', 'notexist']:
                 if args: self.all_sandwiches.add(args[0])
                 if pred == 'ontray' and len(args) > 1: self.all_trays.add(args[1])
             elif pred in ['at_kitchen_bread', 'no_gluten_bread']:
                 if args: self.all_bread.add(args[0])
             elif pred in ['at_kitchen_content', 'no_gluten_content']:
                 if args: self.all_content.add(args[0])
             elif pred == 'at' and len(args) > 1:
                 # The second argument of 'at' is always a place
                 self.all_places.add(args[1])
                 # The first argument could be a tray or other movable object.
                 # We add it to trays if it appears in ontray facts later, or assume based on naming.
                 # For this domain, it's primarily trays. We'll rely on ontray facts to confirm trays.


        # Add 'kitchen' constant explicitly as it's a place
        self.all_places.add('kitchen')

        # Populate static info dictionaries
        for fact_str in self.static_info:
            pred, args = parse_fact(fact_str)
            if pred == 'allergic_gluten':
                self.child_allergy[args[0]] = 'allergic'
            elif pred == 'not_allergic_gluten':
                self.child_allergy[args[0]] = 'not_allergic'
            elif pred == 'waiting':
                self.child_waiting_place[args[0]] = args[1]

        # Ensure all children mentioned in static facts are in our set
        self.all_children.update(self.child_allergy.keys())
        self.all_children.update(self.child_waiting_place.keys())


    def __call__(self, state):
        # Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        # Helper function to parse PDDL facts from state
        def parse_state_facts(state_frozenset):
            parsed = {}
            for fact_str in state_frozenset:
                pred, args = parse_fact(fact_str)
                if pred is None: continue # Skip invalid facts
                if pred not in parsed:
                    parsed[pred] = []
                parsed[pred].append(args)
            return parsed

        state_facts = parse_state_facts(state)

        # Identify served children
        served_children = set()
        if 'served' in state_facts:
            for args in state_facts['served']:
                served_children.add(args[0])

        # Identify unserved children
        unserved_children = [c for c in self.all_children if c not in served_children]

        # If no children are unserved, and goal wasn't reached, something is inconsistent.
        # Based on domain, goal is only about serving children.
        if not unserved_children:
             return 0 # Should be caught by task.goal_reached, but safe fallback.


        # Identify existing sandwiches (those not marked as notexist)
        not_exist_sandwiches = set()
        if 'notexist' in state_facts:
             not_exist_sandwiches = {args[0] for args in state_facts['notexist']}

        existing_sandwiches = {s for s in self.all_sandwiches if s not in not_exist_sandwiches}

        # Identify sandwich properties (no-gluten status)
        sandwich_is_ng = {} # sandwich_name -> True/False
        if 'no_gluten_sandwich' in state_facts:
            for args in state_facts['no_gluten_sandwich']:
                sandwich_is_ng[args[0]] = True

        # Identify sandwich locations ('kitchen_sandwich' or tray_name)
        sandwich_location = {} # sandwich_name -> 'kitchen_sandwich' or tray_name
        if 'at_kitchen_sandwich' in state_facts:
            for args in state_facts['at_kitchen_sandwich']:
                sandwich_location[args[0]] = 'kitchen_sandwich'
        if 'ontray' in state_facts:
             for args in state_facts['ontray']:
                 sandwich_location[args[0]] = args[1] # Store tray name

        # Identify tray locations
        tray_location = {} # tray_name -> place_name
        if 'at' in state_facts:
            for args in state_facts['at']:
                # Check if the first argument is a known tray object
                if args[0] in self.all_trays:
                     tray_location[args[0]] = args[1] # Store place name


        total_heuristic_cost = 0

        # Calculate cost for each unserved child
        for child in unserved_children:
            child_place = self.child_waiting_place.get(child)
            child_allergy_status = self.child_allergy.get(child)

            # Skip if static info is missing for a child (should not happen in valid problems)
            if child_place is None or child_allergy_status is None:
                 continue

            best_child_cost = 4 # Default: need to make sandwich, put on tray, move, serve

            # Find the best suitable sandwich situation for this child among existing sandwiches
            for s_obj in existing_sandwiches:
                is_ng = sandwich_is_ng.get(s_obj, False)

                # Check if sandwich is suitable
                is_suitable = False
                if child_allergy_status == 'allergic' and is_ng:
                    is_suitable = True
                elif child_allergy_status == 'not_allergic':
                    is_suitable = True

                if is_suitable:
                    s_loc = sandwich_location.get(s_obj)

                    if s_loc == 'kitchen_sandwich':
                        # Suitable sandwich exists at kitchen, not on tray
                        best_child_cost = min(best_child_cost, 3) # put on tray + move + serve
                    elif s_loc in self.all_trays: # s_loc is a tray name
                        tray = s_loc
                        t_loc = tray_location.get(tray)
                        if t_loc == child_place:
                            # Suitable sandwich is on a tray at the child's location
                            best_child_cost = min(best_child_cost, 1) # serve
                        elif t_loc is not None: # Suitable sandwich is on a tray elsewhere
                             best_child_cost = min(best_child_cost, 2) # move + serve
                        # If t_loc is None, the tray location is unknown, ignore this sandwich for now
                    # If s_loc is None, the sandwich exists but its location is unknown, ignore this sandwich for now
                    # (This case should ideally not happen based on domain actions)


            total_heuristic_cost += best_child_cost

        return total_heuristic_cost
