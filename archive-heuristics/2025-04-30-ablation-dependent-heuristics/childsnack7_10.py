import sys

class childsnackHeuristic:
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    The heuristic estimates the cost to reach the goal (all children served)
    by summing the estimated costs for each unserved child. It calculates
    the deficit of required sandwiches (no-gluten for allergic, any for
    non-allergic) at the children's locations and adds the estimated cost
    to bring or make these sandwiches from available resources (sandwiches
    on trays at kitchen, sandwiches at kitchen, ingredients). The cost
    depends on the "stage" of the sandwich (ingredients, kitchen, on tray
    at kitchen, on tray at child location).

    Assumptions:
    - The heuristic assumes that trays are sufficiently available to move
      sandwiches once they are on a tray. It does not explicitly model
      tray conflicts or limited tray availability beyond what is needed
      for the current set of sandwiches on trays.
    - It assumes that any bread/content pair can be used to make a regular
      sandwich, and any no-gluten bread/content pair can be used to make
      a no-gluten sandwich, limited only by the count of available
      ingredients and 'notexist' sandwich objects.
    - The costs assigned to different stages (1, 2, 3, 4 actions) are
      approximations based on the sequence of actions required (serve,
      move+serve, put+move+serve, make+put+move+serve).

    Heuristic Initialization:
    The constructor processes the static facts from the task description
    to identify which bread and content portions are no-gluten. It also
    parses all possible ground facts from the task to identify the universe
    of objects (children, places, trays, sandwiches, bread, content) present
    in the problem instance. This allows the heuristic to know the total
    number of children that need to be served.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state is a goal state by verifying if all children
       are marked as 'served'. If yes, return 0.
    2. Identify all unserved children based on the state and the known set of
       all children.
    3. Count the number of unserved allergic children (`N_allergic_unserved`)
       and unserved non-allergic children (`N_non_allergic_unserved`) based
       on the allergy status facts in the current state.
    4. If no children are unserved, return 0 (redundant check, but safe).
    5. Identify the places where unserved children are waiting based on the
       'waiting' facts in the current state.
    6. Count available sandwiches in the current state, categorized by type
       (no-gluten vs any) and "stage":
       - Stage 3 (`S_any_ontray_at_child_loc`, `S_ng_ontray_at_child_loc`):
         On a tray that is currently located at a place where an unserved
         child is waiting.
       - Stage 2 (`S_any_ontray_at_kitchen`, `S_ng_ontray_at_kitchen`):
         On a tray that is currently located at the kitchen.
       - Stage 1 (`S_any_at_kitchen`, `S_ng_at_kitchen`):
         At the kitchen, but not currently on a tray.
       - Stage 0 (`N_makeable_any`, `N_makeable_ng`):
         Potential sandwiches that can be made from available ingredients
         ('at_kitchen_bread', 'at_kitchen_content') and 'notexist' sandwich
         objects in the current state, considering no-gluten requirements.
    7. Calculate the total number of no-gluten sandwiches needed (`Total_Needs_ng`)
       and any sandwiches needed (`Total_Needs_any`), which are equal to
       `N_allergic_unserved` and `N_non_allergic_unserved` respectively.
    8. The base cost starts with the estimated cost to serve sandwiches already
       at Stage 3 (`S_any_ontray_at_child_loc * 1`).
    9. Calculate the deficit of no-gluten sandwiches (`Needed_from_elsewhere_ng`)
       and any sandwiches (`Needed_from_elsewhere_any`) that are not yet at
       Stage 3.
    10. Greedily fulfill `Needed_from_elsewhere_ng` using available global
        no-gluten resources from Stage 2 (cost 2), then Stage 1 (cost 3),
        then Stage 0 (cost 4). Add these costs to the total. Consume the used
        no-gluten resources from the available pools.
    11. Greedily fulfill `Needed_from_elsewhere_any` using remaining available
        global any resources (including any remaining no-gluten ones) from
        Stage 2 (cost 2), then Stage 1 (cost 3), then Stage 0 (cost 4). Add
        these costs to the total. Consume the used any resources from the
        available pools.
    12. If, after using all available resources from Stage 0, 1, and 2, there
        are still needed sandwiches (`needed_ng_rem > 0` or `needed_any_rem > 0`),
        it implies the state might be unsolvable with the current resources.
        Return a large number (e.g., 1000000) to represent a very high cost.
    13. Otherwise, return the total calculated cost.
    """

    def __init__(self, task):
        # Extract static information from the task
        # Static predicates based on domain file analysis:
        # no_gluten_bread, no_gluten_content
        self.ng_bread_types = set()
        self.ng_content_types = set()

        for fact in task.static:
             parts = fact.strip("()").split()
             if not parts: continue
             predicate = parts[0]
             if predicate == 'no_gluten_bread' and len(parts) == 2:
                 self.ng_bread_types.add(parts[1])
             elif predicate == 'no_gluten_content' and len(parts) == 2:
                 self.ng_content_types.add(parts[1])

        # Extract all object names from task.facts to know the universe of objects
        # This is needed to identify all children, sandwiches, etc.
        self.all_children = set()
        self.all_places = set()
        self.all_trays = set()
        self.all_sandwiches = set()
        self.all_bread = set()
        self.all_content = set()

        for fact in task.facts:
             parts = fact.strip("()").split()
             if len(parts) > 1:
                 predicate = parts[0]
                 # Infer object types from predicates
                 if predicate == 'at' and len(parts) == 3: # (at ?t ?p)
                     self.all_trays.add(parts[1])
                     self.all_places.add(parts[2])
                 elif predicate == 'at_kitchen_bread' and len(parts) == 2: # (at_kitchen_bread ?b)
                     self.all_bread.add(parts[1])
                 elif predicate == 'at_kitchen_content' and len(parts) == 2: # (at_kitchen_content ?c)
                     self.all_content.add(parts[1])
                 elif predicate == 'at_kitchen_sandwich' and len(parts) == 2: # (at_kitchen_sandwich ?s)
                     self.all_sandwiches.add(parts[1])
                 elif predicate == 'ontray' and len(parts) == 3: # (ontray ?s ?t)
                     self.all_sandwiches.add(parts[1])
                     self.all_trays.add(parts[2])
                 elif predicate == 'no_gluten_sandwich' and len(parts) == 2: # (no_gluten_sandwich ?s)
                     self.all_sandwiches.add(parts[1])
                 elif predicate == 'allergic_gluten' and len(parts) == 2: # (allergic_gluten ?c)
                     self.all_children.add(parts[1])
                 elif predicate == 'not_allergic_gluten' and len(parts) == 2: # (not_allergic_gluten ?c)
                     self.all_children.add(parts[1])
                 elif predicate == 'served' and len(parts) == 2: # (served ?c)
                     self.all_children.add(parts[1])
                 elif predicate == 'waiting' and len(parts) == 3: # (waiting ?c ?p)
                     self.all_children.add(parts[1])
                     self.all_places.add(parts[2])
                 elif predicate == 'notexist' and len(parts) == 2: # (notexist ?s)
                     self.all_sandwiches.add(parts[1])
                 # Add other predicates if they introduce new object types

    def __call__(self, state):
        # Check for goal state
        all_children_served = True
        for child in self.all_children:
            if '(served ' + child + ')' not in state:
                all_children_served = False
                break
        if all_children_served:
            return 0

        # --- Parse state to get dynamic info ---
        allergic_children_state = set()
        not_allergic_children_state = set()
        child_waiting_place_state = {} # {child_name: place_name}
        served_children_state = set()
        at_kitchen_bread_state = set()
        at_kitchen_content_state = set()
        at_kitchen_sandwich_state = set()
        ontray_state = set() # Store as tuple (sandwich, tray)
        no_gluten_sandwich_state = set()
        at_tray_place_state = {} # {tray_name: place_name}
        notexist_sandwich_state = set()

        for fact in state:
            parts = fact.strip("()").split()
            if not parts: continue # Skip empty strings from malformed facts
            predicate = parts[0]
            if predicate == 'allergic_gluten' and len(parts) == 2:
                allergic_children_state.add(parts[1])
            elif predicate == 'not_allergic_gluten' and len(parts) == 2:
                not_allergic_children_state.add(parts[1])
            elif predicate == 'waiting' and len(parts) == 3:
                child, place = parts[1], parts[2]
                child_waiting_place_state[child] = place
            elif predicate == 'served' and len(parts) == 2:
                served_children_state.add(parts[1])
            elif predicate == 'at_kitchen_bread' and len(parts) == 2:
                at_kitchen_bread_state.add(parts[1])
            elif predicate == 'at_kitchen_content' and len(parts) == 2:
                at_kitchen_content_state.add(parts[1])
            elif predicate == 'at_kitchen_sandwich' and len(parts) == 2:
                at_kitchen_sandwich_state.add(parts[1])
            elif predicate == 'ontray' and len(parts) == 3:
                ontray_state.add((parts[1], parts[2])) # store (sandwich, tray)
            elif predicate == 'no_gluten_sandwich' and len(parts) == 2:
                no_gluten_sandwich_state.add(parts[1])
            elif predicate == 'at' and len(parts) == 3: # (at ?t ?p)
                 tray, place = parts[1], parts[2]
                 at_tray_place_state[tray] = place
            elif predicate == 'notexist' and len(parts) == 2:
                 notexist_sandwich_state.add(parts[1])

        # --- Calculate needs ---
        unserved_children = self.all_children - served_children_state
        N_allergic_unserved = len(unserved_children.intersection(allergic_children_state))
        N_non_allergic_unserved = len(unserved_children.intersection(not_allergic_children_state))

        if N_allergic_unserved == 0 and N_non_allergic_unserved == 0:
             return 0 # Should be caught by goal check, but safety first

        Total_Needs_ng = N_allergic_unserved
        Total_Needs_any = N_non_allergic_unserved

        # --- Calculate available sandwiches by stage and type ---

        # Stage 3: On tray at a location with an unserved child
        waiting_places_state = {child_waiting_place_state[c] for c in unserved_children if c in child_waiting_place_state}

        S_ng_ontray_at_child_loc = 0
        S_any_ontray_at_child_loc = 0

        for s, t in ontray_state: # Unpack tuple (sandwich, tray)
            if t in at_tray_place_state:
                tray_place = at_tray_place_state[t]
                if tray_place in waiting_places_state:
                    if s in no_gluten_sandwich_state:
                        S_ng_ontray_at_child_loc += 1
                    S_any_ontray_at_child_loc += 1 # Includes ng

        # Stage 2: On tray at kitchen
        S_ng_ontray_at_kitchen = 0
        S_any_ontray_at_kitchen = 0

        for s, t in ontray_state: # Unpack tuple (sandwich, tray)
             if t in at_tray_place_state and at_tray_place_state[t] == 'kitchen':
                 if s in no_gluten_sandwich_state:
                     S_ng_ontray_at_kitchen += 1
                 S_any_ontray_at_kitchen += 1 # Includes ng

        # Stage 1: At kitchen (not on tray)
        # Need to exclude sandwiches that are already on trays anywhere
        sandwiches_on_trays_anywhere = {s for s, t in ontray_state}

        S_ng_at_kitchen = 0
        S_any_at_kitchen = 0
        for s in at_kitchen_sandwich_state:
            if s not in sandwiches_on_trays_anywhere: # Only count if not already on a tray
                if s in no_gluten_sandwich_state:
                    S_ng_at_kitchen += 1
                S_any_at_kitchen += 1 # Includes ng


        # Stage 0: Potential from ingredients
        N_ng_bread_avail = len([b for b in at_kitchen_bread_state if b in self.ng_bread_types])
        N_any_bread_avail = len(at_kitchen_bread_state)
        N_ng_content_avail = len([c for c in at_kitchen_content_state if c in self.ng_content_types])
        N_any_content_avail = len(at_kitchen_content_state)
        N_notexist_sandwich = len(notexist_sandwich_state)

        # Need a notexist object to make a sandwich
        N_makeable_ng = min(N_ng_bread_avail, N_ng_content_avail, N_notexist_sandwich) if N_notexist_sandwich > 0 else 0
        N_makeable_any = min(N_any_bread_avail, N_any_content_avail, N_notexist_sandwich) if N_notexist_sandwich > 0 else 0


        # --- Calculate cost ---
        cost = 0

        # Cost for sandwiches already at Stage 3 (on tray at child location)
        # These cost 1 action (serve)
        cost += S_any_ontray_at_child_loc * 1

        # Calculate deficit needed from elsewhere
        Needed_from_elsewhere_ng = max(0, Total_Needs_ng - S_ng_ontray_at_child_loc)
        Needed_from_elsewhere_any = max(0, Total_Needs_any - S_any_ontray_at_child_loc)

        # Available global resources (S2, S1, S0) - Use copies as they are consumed
        Avail_ng_S2 = S_ng_ontray_at_kitchen
        Avail_any_S2 = S_any_ontray_at_kitchen # Includes NG
        Avail_ng_S1 = S_ng_at_kitchen
        Avail_any_S1 = S_any_at_kitchen # Includes NG
        Avail_ng_S0 = N_makeable_ng
        Avail_any_S0 = N_makeable_any # Includes NG potential

        # Store original counts to calculate consumed resources later
        Orig_Avail_ng_S2 = Avail_ng_S2
        Orig_Avail_ng_S1 = Avail_ng_S1
        Orig_Avail_ng_S0 = Avail_ng_S0

        # Fulfill Needed_from_elsewhere_ng using global NG resources
        needed_ng_rem = Needed_from_elsewhere_ng

        use = min(needed_ng_rem, Avail_ng_S2)
        cost += use * 2 # Cost 2: move + serve
        needed_ng_rem -= use
        Avail_ng_S2 -= use # Consume resource

        use = min(needed_ng_rem, Avail_ng_S1)
        cost += use * 3 # Cost 3: put + move + serve
        needed_ng_rem -= use
        Avail_ng_S1 -= use # Consume resource

        use = min(needed_ng_rem, Avail_ng_S0)
        cost += use * 4 # Cost 4: make + put + move + serve
        needed_ng_rem -= use
        Avail_ng_S0 -= use # Consume resource

        # Fulfill Needed_from_elsewhere_any using remaining global ANY resources
        needed_any_rem = Needed_from_elsewhere_any

        # Remaining ANY resources at each stage after NG fulfillment
        # S2: Total ANY at S2 minus those NG ones used for NG need
        Rem_Avail_any_S2 = Avail_any_S2 - (Orig_Avail_ng_S2 - Avail_ng_S2)
        # S1: Total ANY at S1 minus those NG ones used for NG need
        Rem_Avail_any_S1 = Avail_any_S1 - (Orig_Avail_ng_S1 - Avail_ng_S1)
        # S0: Total ANY at S0 minus those NG ones used for NG need
        # The number of NG sandwiches taken from S0 is (Orig_Avail_ng_S0 - Avail_ng_S0).
        # Each such sandwich consumes one unit from the total makeable pool (N_makeable_any).
        Rem_Avail_any_S0 = Avail_any_S0 - (Orig_Avail_ng_S0 - Avail_ng_S0)

        # Ensure counts are not negative (can happen due to approximations)
        Rem_Avail_any_S2 = max(0, Rem_Avail_any_S2)
        Rem_Avail_any_S1 = max(0, Rem_Avail_any_S1)
        Rem_Avail_any_S0 = max(0, Rem_Avail_any_S0)


        use = min(needed_any_rem, Rem_Avail_any_S2)
        cost += use * 2
        needed_any_rem -= use
        Rem_Avail_any_S2 -= use

        use = min(needed_any_rem, Rem_Avail_any_S1)
        cost += use * 3
        needed_any_rem -= use
        Rem_Avail_any_S1 -= use

        use = min(needed_any_rem, Rem_Avail_any_S0)
        cost += use * 4
        needed_any_rem -= use
        Rem_Avail_any_S0 -= use


        # If we still need sandwiches after using all resources
        if needed_ng_rem > 0 or needed_any_rem > 0:
             # This state is likely unsolvable with current resources
             # Or requires actions not modeled (e.g., getting more ingredients)
             # Return a large value
             return 1000000 # A large number representing infinity

        return cost
