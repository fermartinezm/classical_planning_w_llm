import math

def parse_fact(fact_str):
    """Parses a PDDL fact string into a tuple (predicate, arg1, arg2, ...)."""
    # Remove leading/trailing parentheses and split by space
    parts = fact_str[1:-1].split()
    return tuple(parts)

class childsnackHeuristic:
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    The heuristic estimates the cost to reach the goal (all children served)
    by summing the estimated costs to provide a suitable sandwich to each
    unserved child. It prioritizes using sandwiches that are closer to being
    served, assigning costs based on the actions required:
    1. Sandwich is on a tray at a child's waiting location (cost 1: serve).
    2. Sandwich is on a tray in the kitchen (cost 2: move tray + serve).
    3. Sandwich is in the kitchen (cost 3: put on tray + move tray + serve).
    4. Ingredients are available to make the sandwich (cost 4: make + put + move + serve).
    An additional cost of 1 is added once if any sandwich from stage 3 or 4
    is needed and no tray is currently in the kitchen (representing the cost
    to move a tray to the kitchen). The heuristic returns infinity if the
    required sandwiches cannot be made due to insufficient ingredients or
    sandwich objects.

    Assumptions:
    - The problem is solvable from the initial state (sufficient total resources exist).
    - The heuristic assumes trays can be moved freely between locations.
    - The heuristic simplifies tray usage: it only adds a cost for moving a tray
      *to* the kitchen if needed for making/putting sandwiches, but doesn't
      explicitly model tray capacity or availability at child locations
      beyond checking for sandwiches already present there.
    - The heuristic assumes any available sandwich object can be used to make
      any type of sandwich (gluten-free or regular), limited only by ingredients.

    Heuristic Initialization:
    The constructor pre-processes the static facts from the task definition
    and initial state to store information about children's allergies and
    waiting locations, and the gluten status of bread and content portions.
    This information remains constant throughout the search.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Parse the current state facts to identify the status and location
        of all relevant objects (children, sandwiches, bread, content, trays).
    2.  Identify all children who are not yet served.
    3.  Categorize unserved children by their allergy status (allergic/non-allergic).
    4.  Count the total number of unserved allergic children (`N_allergic_unserved`)
        and unserved non-allergic children (`N_non_allergic_unserved`). These
        represent the total demand for gluten-free and any sandwiches, respectively.
    5.  Count the available sandwiches and ingredients, categorized by type (GF/Any)
        and "stage" of readiness:
        - Stage 0 (S0): On a tray at a child's waiting location.
        - Stage 1 (S1): On a tray in the kitchen.
        - Stage 2 (S2): As a sandwich in the kitchen (`at_kitchen_sandwich`).
        - Stage 3 (S3): Potential sandwiches that can be made from ingredients
          (`at_kitchen_bread`, `at_kitchen_content`) and available sandwich objects (`notexist`).
        Ensure sandwiches are counted in only one stage (the lowest applicable one).
    6.  Initialize the heuristic value `h` to 0.
    7.  Iterate through the stages (0 to 3) for both gluten-free and any sandwiches,
        prioritizing gluten-free needs first, and within each type, prioritizing
        cheaper stages (lower stage number).
    8.  For each stage and type, calculate how many sandwiches can be used from
        the available supply to meet the remaining demand. Add the cost per
        sandwich for that stage to `h` and reduce the remaining demand.
    9.  Specifically, for stages 2 and 3 (which require putting a sandwich on a tray
        in the kitchen), check if a tray is available in the kitchen. If not, and
        if any sandwiches are needed from stages 2 or 3, add a cost of 1 once
        to `h` to represent moving a tray to the kitchen. This cost is added
        before calculating the per-sandwich costs for S2 and S3.
    10. If, after exhausting all available resources across all stages, there is
        still remaining demand (`needed_GF > 0` or `needed_Any > 0`), return
        `float('inf')` as the state is likely unsolvable.
    11. Otherwise, return the calculated heuristic value `h`.
    12. The heuristic is 0 if and only if all children are served (goal state),
        as the initial demands (`N_allergic_unserved`, `N_non_allergic_unserved`)
        will be zero, resulting in `h=0`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic with static information from the task.

        @param task: The planning task object.
        """
        self.task = task
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.child_locations = {} # child -> place
        self.no_gluten_bread_types = set() # Store the names of GF bread objects
        self.no_gluten_content_types = set() # Store the names of GF content objects
        self.all_children = set() # Keep track of all children mentioned

        # Extract static information from static facts and initial state
        for fact_str in task.static | task.initial_state:
            fact = parse_fact(fact_str)
            if fact[0] == 'allergic_gluten':
                child = fact[1]
                self.allergic_children.add(child)
                self.all_children.add(child)
            elif fact[0] == 'not_allergic_gluten':
                child = fact[1]
                self.not_allergic_children.add(child)
                self.all_children.add(child)
            elif fact[0] == 'waiting':
                child, place = fact[1], fact[2]
                self.child_locations[child] = place
            elif fact[0] == 'no_gluten_bread':
                self.no_gluten_bread_types.add(fact[1])
            elif fact[0] == 'no_gluten_content':
                self.no_gluten_content_types.add(fact[1])

        # Ensure all children mentioned in goals are also in self.all_children
        # This handles cases where a child might only appear in the goal and not static/init
        for goal_fact_str in task.goals:
             goal_fact = parse_fact(goal_fact_str)
             if goal_fact[0] == 'served':
                 child = goal_fact[1]
                 self.all_children.add(child)
                 # We don't know allergy/location from goal, assume they were in static/init


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state: The current state (frozenset of fact strings).
        @return: The estimated cost to reach the goal, or float('inf').
        """
        # Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        # --- Parse State Facts ---
        served_children = set()
        at_kitchen_bread_objects = set()
        at_kitchen_content_objects = set()
        at_kitchen_sandwich_objects = set()
        ontray_facts = set() # Store (sandwich, tray) tuples
        no_gluten_sandwich_objects = set() # Store names of GF sandwiches
        tray_locations = {} # tray -> place
        not_exist_sandwich_objects = set()

        for fact_str in state:
            fact = parse_fact(fact_str)
            if fact[0] == 'served':
                served_children.add(fact[1])
            elif fact[0] == 'at_kitchen_bread':
                at_kitchen_bread_objects.add(fact[1])
            elif fact[0] == 'at_kitchen_content':
                at_kitchen_content_objects.add(fact[1])
            elif fact[0] == 'at_kitchen_sandwich':
                at_kitchen_sandwich_objects.add(fact[1])
            elif fact[0] == 'ontray':
                ontray_facts.add((fact[1], fact[2]))
            elif fact[0] == 'no_gluten_sandwich':
                no_gluten_sandwich_objects.add(fact[1])
            elif fact[0] == 'at':
                tray_locations[fact[1]] = fact[2]
            elif fact[0] == 'notexist':
                not_exist_sandwich_objects.add(fact[1])

        # --- Calculate Needs ---
        unserved_allergic = [c for c in self.allergic_children if c not in served_children]
        unserved_not_allergic = [c for c in self.not_allergic_children if c not in served_children]

        needed_GF = len(unserved_allergic)
        needed_Any = len(unserved_not_allergic)

        # If no children need serving, but goal not reached, something is wrong (should be caught by goal_reached)
        if needed_GF == 0 and needed_Any == 0:
             return float('inf') # Should not happen if goal_reached is False

        # --- Count Available Resources by Stage ---

        Avail_GF_S0 = 0 # On tray at child's location
        Avail_Any_S0 = 0 # On tray at child's location (includes GF)
        Avail_GF_S1 = 0 # On tray in kitchen
        Avail_Any_S1 = 0 # On tray in kitchen (includes GF)
        Avail_GF_S2 = 0 # Sandwich in kitchen
        Avail_Any_S2 = 0 # Sandwich in kitchen (includes GF)

        sandwiches_accounted_for = set() # Ensure each sandwich object is counted once

        # Stage 0: On tray at child's location (Cost 1: serve)
        child_waiting_locations = set(self.child_locations.get(c) for c in unserved_allergic + unserved_not_allergic if c in self.child_locations)
        trays_at_child_loc = {t for t, loc in tray_locations.items() if loc in child_waiting_locations}

        for s, t in ontray_facts:
             if t in trays_at_child_loc:
                 if s not in sandwiches_accounted_for:
                     sandwiches_accounted_for.add(s)
                     is_gf = s in no_gluten_sandwich_objects
                     if is_gf:
                         Avail_GF_S0 += 1
                         Avail_Any_S0 += 1
                     else:
                         Avail_Any_S0 += 1

        # Stage 1: On tray in kitchen (Cost 2: move + serve)
        trays_at_kitchen = {t for t, loc in tray_locations.items() if loc == 'kitchen'}

        for s, t in ontray_facts:
             if t in trays_at_kitchen:
                 if s not in sandwiches_accounted_for:
                     sandwiches_accounted_for.add(s)
                     is_gf = s in no_gluten_sandwich_objects
                     if is_gf:
                         Avail_GF_S1 += 1
                         Avail_Any_S1 += 1
                     else:
                         Avail_Any_S1 += 1

        # Stage 2: Sandwich in kitchen (Cost 3: put + move + serve)
        for s in at_kitchen_sandwich_objects:
             if s not in sandwiches_accounted_for:
                 sandwiches_accounted_for.add(s)
                 is_gf = s in no_gluten_sandwich_objects
                 if is_gf:
                     Avail_GF_S2 += 1
                     Avail_Any_S2 += 1
                 else:
                     Avail_Any_S2 += 1

        # Stage 3: Ingredients available (Cost 4: make + put + move + serve)
        GF_bread_kitchen = len([b for b in at_kitchen_bread_objects if b in self.no_gluten_bread_types])
        GF_content_kitchen = len([c for c in at_kitchen_content_objects if c in self.no_gluten_content_types])
        Any_bread_kitchen = len(at_kitchen_bread_objects)
        Any_content_kitchen = len(at_kitchen_content_objects)
        Avail_Sandwich_Objects = len(not_exist_sandwich_objects)

        Avail_GF_S3_potential = min(GF_bread_kitchen, GF_content_kitchen)
        # Avail_Any_S3_potential is calculated dynamically below


        Avail_Trays_Kitchen_Count = len(trays_at_kitchen)
        total_trays = len(tray_locations)


        # --- Calculate Heuristic Cost ---
        h = 0

        # Stage 0 (Cost 1)
        use_GF_S0 = min(needed_GF, Avail_GF_S0)
        h += use_GF_S0 * 1
        needed_GF -= use_GF_S0

        use_Any_S0 = min(needed_Any, Avail_Any_S0 - use_GF_S0) # Use non-GF S0 for Any need
        h += use_Any_S0 * 1
        needed_Any -= use_Any_S0

        # Stage 1 (Cost 2)
        use_GF_S1 = min(needed_GF, Avail_GF_S1)
        h += use_GF_S1 * 2
        needed_GF -= use_GF_S1

        use_Any_S1 = min(needed_Any, Avail_Any_S1 - use_GF_S1)
        h += use_Any_S1 * 2
        needed_Any -= use_Any_S1

        # Determine if a tray move to kitchen is needed for S2/S3 sandwiches.
        total_needed_S2_S3 = needed_GF + needed_Any
        tray_move_to_kitchen_cost = 0
        if Avail_Trays_Kitchen_Count == 0 and total_needed_S2_S3 > 0:
             if total_trays > 0:
                 tray_move_to_kitchen_cost = 1
             else:
                 # No trays exist at all. Cannot put sandwiches on trays. Unsolvable.
                 return float('inf')

        # Stage 2 (Cost 3)
        cost_S2 = 3
        use_GF_S2 = min(needed_GF, Avail_GF_S2)
        h += use_GF_S2 * cost_S2
        needed_GF -= use_GF_S2

        use_Any_S2 = min(needed_Any, Avail_Any_S2 - use_GF_S2)
        h += use_Any_S2 * cost_S2
        needed_Any -= use_Any_S2

        # Stage 3 (Cost 4)
        cost_S3 = 4

        # GF sandwiches from ingredients
        can_make_GF = min(needed_GF, Avail_GF_S3_potential, Avail_Sandwich_Objects)
        h += can_make_GF * cost_S3
        needed_GF -= can_make_GF
        Avail_Sandwich_Objects -= can_make_GF # Consume sandwich objects

        # Remaining GF ingredients can be used for Any sandwiches
        rem_GF_bread = max(0, GF_bread_kitchen - can_make_GF)
        rem_GF_content = max(0, GF_content_kitchen - can_make_GF)

        # Any sandwiches from ingredients (using non-GF + remaining GF ingredients)
        # Total bread objects available for Any = All bread - GF bread used for GF sandwiches
        avail_Any_bread_total = Any_bread_kitchen - (GF_bread_kitchen - rem_GF_bread)
        avail_Any_content_total = Any_content_kitchen - (GF_content_kitchen - rem_GF_content)

        avail_Any_make_potential_total = min(avail_Any_bread_total, avail_Any_content_total)

        avail_Any_make_potential_final = min(needed_Any, avail_Any_make_potential_total, Avail_Sandwich_Objects)
        use_Any_S3 = avail_Any_make_potential_final
        h += use_Any_S3 * cost_S3
        needed_Any -= use_Any_S3
        # Avail_Sandwich_Objects -= use_Any_S3 # Consume sandwich objects (already accounted for in potential)


        # Add the cost for moving a tray to the kitchen if it was necessary.
        h += tray_move_to_kitchen_cost

        # If we still need sandwiches, it's unsolvable from here (insufficient ingredients/objects).
        if needed_GF > 0 or needed_Any > 0:
            return float('inf')

        return h
