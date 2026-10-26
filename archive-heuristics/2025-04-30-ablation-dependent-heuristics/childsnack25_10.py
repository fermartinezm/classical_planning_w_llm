def parse_fact(fact_string):
    """Parses a PDDL fact string into predicate and arguments."""
    # Removes surrounding brackets and splits by space
    parts = fact_string.strip("()").split()
    predicate = parts[0]
    args = parts[1:]
    return predicate, args

class childsnackHeuristic:
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    The heuristic estimates the number of actions required to reach the goal
    state (all children served). It sums up the estimated number of actions
    of each type (make_sandwich, put_on_tray, move_tray, serve_sandwich)
    needed to satisfy the requirements of all unserved children. It attempts
    to account for shared resources (sandwiches, trays) and dependencies
    between actions.

    Assumptions:
    - The heuristic assumes that necessary bread and content portions are
      available in the kitchen if a sandwich needs to be made.
    - It assumes trays are available when needed for put_on_tray actions.
    - It assumes that any sandwich can be put on any tray.
    - It assumes that tray movements are independent for different needed locations.
    - The heuristic is not admissible; it is designed for greedy best-first search.
    - All relevant objects (children, trays, places) are mentioned in the
      initial state or static facts.

    Heuristic Initialization:
    The constructor processes the static facts and initial state facts from
    the task definition to identify all relevant objects (children, trays, places)
    and store static information like child allergies and waiting places.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state, the heuristic calculates the sum of estimated costs for
    different types of actions needed to reach the goal:

    1.  **Serve Cost:** Count the number of children who are not yet served. Each unserved child requires at least one 'serve_sandwich' action. Add this count to the heuristic.

    2.  **Make Sandwich Cost:** Determine the number of gluten-free and non-gluten-free sandwiches needed by unserved children that are not currently available (either at the kitchen or on a tray). Calculate the minimum number of 'make_sandwich' actions required to produce these needed sandwiches. Add this count to the heuristic.

    3.  **Put On Tray Cost:** Count the number of distinct sandwiches that are currently 'at_kitchen_sandwich' and are suitable for at least one unserved child. Also, count the number of sandwiches that need to be made (from step 2). Each of these sandwiches will require a 'put_on_tray' action to be moved from the kitchen onto a tray. Sum these counts and add to the heuristic.

    4.  **Move Tray Cost:** Estimate the number of 'move_tray' actions needed. This is approximated by counting the number of distinct locations where unserved children are waiting, and subtracting the number of distinct locations that already have a tray present. This counts the minimum number of locations that a tray needs to be moved to. Add this value (or 0 if negative) to the heuristic.

    The total heuristic value is the sum of costs from steps 1, 2, 3, and 4.
    """
    def __init__(self, task):
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.child_waiting_place = {} # child -> place
        self.all_children = set()
        self.all_trays = set()
        self.all_places = set()
        self.kitchen_place = 'kitchen' # Assuming constant name

        # Mapping from predicate to argument types (simplified for object collection)
        arg_types = {
            'allergic_gluten': ['child'],
            'not_allergic_gluten': ['child'],
            'served': ['child'],
            'waiting': ['child', 'place'],
            'at': ['tray', 'place'],
            'ontray': ['sandwich', 'tray'],
            'at_kitchen_bread': ['bread-portion'],
            'at_kitchen_content': ['content-portion'],
            'at_kitchen_sandwich': ['sandwich'],
            'no_gluten_bread': ['bread-portion'],
            'no_gluten_content': ['content-portion'],
            'no_gluten_sandwich': ['sandwich'],
            'notexist': ['sandwich'],
        }


        # Collect objects and static info from static and initial state facts
        all_facts = task.static | task.initial_state
        for fact_string in all_facts:
            predicate, args = parse_fact(fact_string)

            if predicate in arg_types:
                types = arg_types[predicate]
                for i, arg in enumerate(args):
                    if i < len(types): # Ensure index is within bounds
                        arg_type = types[i]
                        if arg_type == 'child':
                            self.all_children.add(arg)
                        elif arg_type == 'tray':
                            self.all_trays.add(arg)
                        elif arg_type == 'place':
                            self.all_places.add(arg)

            # Store specific static info
            if predicate == 'allergic_gluten':
                self.allergic_children.add(args[0])
            elif predicate == 'not_allergic_gluten':
                self.not_allergic_children.add(args[0])
            elif predicate == 'waiting':
                if len(args) == 2:
                    self.child_waiting_place[args[0]] = args[1]

        # Ensure kitchen is in all_places
        self.all_places.add(self.kitchen_place)

        # Ensure all children mentioned in waiting are in all_children
        self.all_children.update(self.child_waiting_place.keys())


    def __call__(self, state):
        # Parse state facts
        served_children = set()
        at_kitchen_sandwich = set()
        ontray_sandwiches = set() # set of (sandwich, tray)
        tray_locations = {} # tray -> place
        no_gluten_sandwich = set()

        for fact_string in state:
            predicate, args = parse_fact(fact_string)
            if predicate == 'served':
                served_children.add(args[0])
            elif predicate == 'at_kitchen_sandwich':
                at_kitchen_sandwich.add(args[0])
            elif predicate == 'ontray':
                if len(args) == 2:
                    ontray_sandwiches.add((args[0], args[1]))
            elif predicate == 'at':
                if len(args) == 2:
                    tray, place = args
                    tray_locations[tray] = place
            elif predicate == 'no_gluten_sandwich':
                no_gluten_sandwich.add(args[0])

        # --- Heuristic Calculation ---
        h = 0

        # 1. Identify unserved children
        unserved_children = self.all_children - served_children

        # If goal reached, heuristic is 0
        if not unserved_children:
            return 0

        # 2. Serve Cost: Each unserved child needs a 'serve_sandwich' action
        h += len(unserved_children)

        # 3. Categorize unserved children by allergy
        unserved_gf = unserved_children & self.allergic_children
        unserved_nongf = unserved_children & self.not_allergic_children

        # 4. Categorize available sandwiches by type and location
        S_kitchen_gf = {s for s in at_kitchen_sandwich if s in no_gluten_sandwich}
        S_kitchen_nongf = {s for s in at_kitchen_sandwich if s not in no_gluten_sandwich}
        S_ontray_gf = {s for (s, t) in ontray_sandwiches if s in no_gluten_sandwich}
        S_ontray_nongf = {s for (s, t) in ontray_sandwiches if s not in no_gluten_sandwich}

        avail_gf_total = len(S_kitchen_gf) + len(S_ontray_gf)
        avail_nongf_total = len(S_kitchen_nongf) + len(S_ontray_nongf)

        # 5. Calculate 'make_sandwich' cost
        # Children needing GF *must* use GF sandwiches.
        # Children needing non-GF *can* use remaining GF or non-GF sandwiches.
        needed_gf_to_make = max(0, len(unserved_gf) - avail_gf_total)

        # Available for non-GF children = remaining GF + available non-GF
        avail_for_nongf = max(0, avail_gf_total - len(unserved_gf)) + avail_nongf_total
        needed_nongf_to_make = max(0, len(unserved_nongf) - avail_for_nongf)

        h += needed_gf_to_make + needed_nongf_to_make

        # 6. Calculate 'put_on_tray' cost
        # Count distinct kitchen sandwiches needed by unserved children
        kitchen_sandwiches_potentially_needed = set()
        if unserved_children: # If any child is unserved, GF kitchen sandwiches are potentially needed
             kitchen_sandwiches_potentially_needed.update(S_kitchen_gf)
        if unserved_nongf: # If any non-allergic child is unserved, non-GF kitchen sandwiches are potentially needed
             kitchen_sandwiches_potentially_needed.update(S_kitchen_nongf)

        # Sandwiches currently at kitchen that are needed
        h += len(kitchen_sandwiches_potentially_needed)
        # Sandwiches that need to be made (they will end up at kitchen)
        h += needed_gf_to_make + needed_nongf_to_make


        # 7. Calculate 'move_tray' cost
        # Estimate based on locations needing service vs locations with trays
        needed_places = {self.child_waiting_place[c] for c in unserved_children if c in self.child_waiting_place} # Ensure child has a waiting place
        trays_at_needed_places = {t for t, p in tray_locations.items() if p in needed_places}
        places_with_trays = {tray_locations[t] for t in trays_at_needed_places}

        h += max(0, len(needed_places) - len(places_with_trays))

        return h
