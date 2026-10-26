from heuristics.heuristic_base import Heuristic
# No fnmatch needed

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty fact string or malformed facts gracefully, though PDDL states are structured.
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
         return []
    return fact[1:-1].split()

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all unserved children.
    It calculates the minimum steps needed for each unserved child to receive a suitable
    sandwich on a tray at their location, and sums these minimum steps across all children.
    The cost for a sandwich depends on its current state (ready to serve, on tray wrong place,
    at kitchen, needs to be made). Gluten-free needs are prioritized for gluten-free sandwiches.

    # Assumptions
    - Each unserved child requires one suitable sandwich.
    - Actions have a cost of 1.
    - The heuristic estimates steps per sandwich/child independently after the initial
      assignment based on availability, potentially overcounting shared actions like
      moving a tray with multiple sandwiches or making sandwiches using shared ingredients.
      This makes the heuristic non-admissible but potentially more informative.
    - The problem is solvable, meaning enough total resources (ingredients, sandwich slots, trays) exist across the problem instance to serve all children.

    # Heuristic Initialization
    - Extracts the set of all children that need to be served from the task goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children that need to be served (from task goals).
    2. Iterate through the current state to determine which of these children are not yet served, their waiting location, and allergy status.
    3. Parse the state to gather information about:
       - Sandwich types (GF/Reg) and their current status (at kitchen, on tray, notexist).
       - Tray locations.
       - Available ingredients (GF/Reg bread/content) in the kitchen.
       - Available 'notexist' sandwich slots.
    4. Calculate the maximum number of gluten-free and regular sandwiches that can be made from available ingredients and slots.
    5. Count available sandwiches (existing and creatable) by type (GF/Reg) and their current location/status.
    6. Greedily assign available sandwiches to unserved children based on estimated cost and type preference (GF for allergic):
       - Prioritize sandwiches already on trays at the child's location (Cost 1: Serve).
       - Next, prioritize sandwiches on trays at other locations (Cost 2: Move Tray + Serve).
       - Next, prioritize sandwiches at the kitchen (Cost 3: Put on Tray + Move Tray + Serve).
       - Finally, use creatable sandwiches (Cost 4: Make + Put on Tray + Move Tray + Serve).
    7. Sum the costs for each assigned sandwich. This sum is the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal children."""
        self.goals = task.goals
        # Extract the set of children that need to be served from the goals
        self.goal_children = {get_parts(g)[1] for g in self.goals if get_parts(g)[0] == 'served'}

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- State Parsing ---

        unserved_children = set(self.goal_children)
        child_place = {} # {child: place}
        child_allergy = {} # {child: bool (True for allergic)}
        sandwich_types = {} # {sandwich: 'gf'/'reg'}
        sandwich_status = {} # {sandwich: 'kitchen'/'ontray'/None}
        sandwich_tray = {} # {sandwich: tray or None}
        tray_location = {} # {tray: place}
        notexist_sandwiches = set()
        bread_types = {} # {bread: 'gf'/'reg'}
        content_types = {} # {content: 'gf'/'reg'}
        gf_bread_kitchen = 0
        reg_bread_kitchen = 0
        gf_content_kitchen = 0
        reg_content_kitchen = 0

        # First pass to get types (allergy, no_gluten) and initial existence (notexist)
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            if pred == 'served':
                unserved_children.discard(parts[1])
            elif pred == 'waiting':
                child_place[parts[1]] = parts[2]
            elif pred == 'allergic_gluten':
                child_allergy[parts[1]] = True
            elif pred == 'not_allergic_gluten':
                child_allergy[parts[1]] = False
            elif pred == 'no_gluten_sandwich':
                sandwich_types[parts[1]] = 'gf'
            elif pred == 'no_gluten_bread':
                bread_types[parts[1]] = 'gf'
            elif pred == 'no_gluten_content':
                content_types[parts[1]] = 'gf'
            elif pred == 'notexist':
                notexist_sandwiches.add(parts[1])

        # Second pass for locations and status
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            if pred == 'at_kitchen_sandwich':
                s = parts[1]
                sandwich_status[s] = 'kitchen'
            elif pred == 'ontray':
                s, t = parts[1], parts[2]
                sandwich_status[s] = 'ontray'
                sandwich_tray[s] = t
            elif pred == 'at':
                tray_location[parts[1]] = parts[2]
            elif pred == 'at_kitchen_bread':
                b = parts[1]
                if bread_types.get(b, 'reg') == 'gf': gf_bread_kitchen += 1
                else: reg_bread_kitchen += 1
            elif pred == 'at_kitchen_content':
                c = parts[1]
                if content_types.get(c, 'reg') == 'gf': gf_content_kitchen += 1
                else: reg_content_kitchen += 1

        # --- Heuristic Calculation ---

        N_allergic_unserved = sum(1 for c in unserved_children if child_allergy.get(c, False))
        N_regular_unserved = len(unserved_children) - N_allergic_unserved

        if len(unserved_children) == 0:
            return 0 # Goal state

        # 1. Count available sandwiches by type and location/status
        gf_at_kitchen = 0
        reg_at_kitchen = 0
        gf_ontray_at_place = {} # {place: count}
        reg_ontray_at_place = {} # {place: count}

        for s in sandwich_status:
            s_type = sandwich_types.get(s, 'reg')
            status = sandwich_status[s]

            if status == 'kitchen':
                if s_type == 'gf': gf_at_kitchen += 1
                else: reg_at_kitchen += 1
            elif status == 'ontray':
                t = sandwich_tray.get(s)
                p_tray = tray_location.get(t)
                if p_tray: # Only count if tray location is known
                    if s_type == 'gf':
                        gf_ontray_at_place[p_tray] = gf_ontray_at_place.get(p_tray, 0) + 1
                    else:
                        reg_ontray_at_place[p_tray] = reg_ontray_at_place.get(p_tray, 0) + 1

        # 2. Calculate creatable sandwiches (Cost 4 source)
        can_make_gf = min(gf_bread_kitchen, gf_content_kitchen, len(notexist_sandwiches))
        remaining_notexist = len(notexist_sandwiches) - can_make_gf
        remaining_gf_bread = gf_bread_kitchen - can_make_gf
        remaining_gf_content = gf_content_kitchen - can_make_gf
        can_make_reg = min(reg_bread_kitchen + remaining_gf_bread, reg_content_kitchen + remaining_gf_content, remaining_notexist)

        # 3. Assign sandwiches greedily by cost and type
        h = 0
        needed_gf = N_allergic_unserved
        needed_reg = N_regular_unserved

        places_with_unserved_allergic = {child_place[c] for c in unserved_children if child_allergy.get(c, False)}
        places_with_unserved_regular = {child_place[c] for c in unserved_children if not child_allergy.get(c, False)}

        # Cost 1 (Serve)
        # GF needs from GF ontray at allergic places
        for p in places_with_unserved_allergic:
            available = gf_ontray_at_place.get(p, 0)
            use = min(needed_gf, available)
            h += use * 1
            needed_gf -= use
            gf_ontray_at_place[p] -= use # Consume

        # Reg needs from Reg ontray at regular places
        for p in places_with_unserved_regular:
            available = reg_ontray_at_place.get(p, 0)
            use = min(needed_reg, available)
            h += use * 1
            needed_reg -= use
            reg_ontray_at_place[p] -= use # Consume

        # Reg needs from remaining GF ontray at regular places
        for p in places_with_unserved_regular:
            available = gf_ontray_at_place.get(p, 0) # Use remaining GF sandwiches at this location
            use = min(needed_reg, available)
            h += use * 1
            needed_reg -= use
            gf_ontray_at_place[p] -= use # Consume


        # Cost 2 (Move + Serve)
        # Count remaining on-tray sandwiches not used for cost 1
        remaining_gf_ontray = sum(gf_ontray_at_place.values())
        remaining_reg_ontray = sum(reg_ontray_at_place.values())

        # GF needs from remaining GF ontray
        use = min(needed_gf, remaining_gf_ontray)
        h += use * 2
        needed_gf -= use
        remaining_gf_ontray -= use # Consume

        # Reg needs from remaining Reg ontray
        use = min(needed_reg, remaining_reg_ontray)
        h += use * 2
        needed_reg -= use
        remaining_reg_ontray -= use # Consume

        # Reg needs from remaining GF ontray
        use = min(needed_reg, remaining_gf_ontray)
        h += use * 2
        needed_reg -= use
        remaining_gf_ontray -= use # Consume


        # Cost 3 (Put + Move + Serve)
        # GF needs from GF at kitchen
        use = min(needed_gf, gf_at_kitchen)
        h += use * 3
        needed_gf -= use
        gf_at_kitchen -= use # Consume

        # Reg needs from Reg at kitchen
        use = min(needed_reg, reg_at_kitchen)
        h += use * 3
        needed_reg -= use
        reg_at_kitchen -= use # Consume

        # Reg needs from remaining GF at kitchen
        use = min(needed_reg, gf_at_kitchen)
        h += use * 3
        needed_reg -= use
        gf_at_kitchen -= use # Consume


        # Cost 4 (Make + Put + Move + Serve)
        # GF needs from creatable GF
        use = min(needed_gf, can_make_gf)
        h += use * 4
        needed_gf -= use
        can_make_gf -= use # Consume

        # Reg needs from creatable Reg
        use = min(needed_reg, can_make_reg)
        h += use * 4
        needed_reg -= use
        can_make_reg -= use # Consume

        # Reg needs from remaining creatable GF
        use = min(needed_reg, can_make_gf)
        h += use * 4
        needed_reg -= use
        can_make_gf -= use # Consume

        # If needed_gf > 0 or needed_reg > 0 here, it implies the problem is unsolvable
        # from this state with the available resources. Assuming solvable problems,
        # this shouldn't happen. We can assert or return a large value if needed.
        # assert needed_gf == 0 and needed_reg == 0, "Heuristic calculation error: Not enough resources?"

        return h
