from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting
    children. It does this by counting the number of waiting children who need
    a sandwich, categorized by their allergy status (gluten or not), and then
    assigning costs based on the current "stage" of the most readily available
    suitable sandwiches. Sandwiches closer to being served (e.g., already on a
    tray at the table) contribute less to the heuristic than those further away
    (e.g., still needing to be made from ingredients). The heuristic prioritizes
    satisfying gluten-allergic children's needs with gluten-free sandwiches.

    # Assumptions
    - The goal is to serve all children specified in the task goals.
    - A child is served by providing them with a suitable sandwich on a tray
      at their waiting location.
    - Gluten-allergic children require gluten-free sandwiches. Non-allergic
      children can have any sandwich.
    - Sandwiches progress through stages: notexist -> at_kitchen_sandwich
      -> ontray at kitchen -> ontray at table -> served.
    - Making a sandwich requires one bread and one content portion in the kitchen.
    - A gluten-free sandwich requires gluten-free bread and gluten-free content.
    - There are enough sandwich objects defined in the problem to serve all children.
    - Tray availability is not a bottleneck considered by this heuristic for
      moving sandwiches from kitchen to table or putting sandwiches on trays.
      Ingredient availability for making sandwiches *is* considered.

    # Heuristic Initialization
    - Extracts static information about which children are allergic to gluten,
      and which bread and content portions are gluten-free.
    - Extracts the set of children who are the goal of the task (i.e., need to be served).

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic calculates the cost based on the number of waiting children
    and the state of available suitable sandwiches.

    1.  Identify all children currently waiting and their locations from the state.
    2.  Categorize waiting children into those needing gluten-free sandwiches
        (allergic) and those needing any sandwich (non-allergic). Count these:
        `N_wait_gf` and `N_wait_any`. If both counts are zero, the goal is reached,
        and the heuristic is 0.
    3.  Identify all sandwich objects mentioned in the state and determine their
        current state: `notexist`, `at_kitchen_sandwich`, or `ontray` (and the
        location of the tray). Also, determine which sandwiches are currently
        marked as gluten-free (`no_gluten_sandwich`).
    4.  Count available sandwiches by type (gluten-free or any/non-gluten-free)
        and by "stage", representing how close they are to being served:
        - Stage 3 (Cost 1): On a tray at a table where a child is waiting.
        - Stage 2 (Cost 2): On a tray at the kitchen.
        - Stage 1 (Cost 3): `at_kitchen_sandwich`.
        - Stage 0 (Cost 4): `notexist`.
        Let these counts be `N_s3_gf`, `N_s3_any`, `N_s2_gf`, `N_s2_any`, etc.
        A sandwich is "any" (non-GF) if it's not marked as gluten-free.
    5.  Count available ingredients in the kitchen: gluten-free bread/content
        pairs (`N_gf_ing`) and non-gluten-free bread/content pairs (`N_any_ing`).
    6.  Calculate the heuristic cost by greedily satisfying the needs of waiting
        children using the available sandwiches, prioritizing:
        - Gluten-free needs first.
        - Sandwiches from later stages (lower cost) first.
        - Gluten-free sandwiches for gluten-free needs, then for any needs.
        - Non-gluten-free sandwiches only for any needs.
    7.  Iterate through stages from 3 down to 0:
        - For Stage 3 (Cost 1): Use available S3 GF sandwiches for remaining GF needs,
          then available S3 Any sandwiches for remaining Any needs, then remaining
          S3 GF sandwiches for remaining Any needs. Add `count * 1` to heuristic.
        - For Stage 2 (Cost 2): Similar process using S2 sandwiches. Add `count * 2`.
        - For Stage 1 (Cost 3): Similar process using S1 sandwiches. Add `count * 3`.
        - For Stage 0 (Cost 4): Similar process using S0 (`notexist`) sandwiches.
          This stage is limited by the number of `notexist` sandwich objects and
          the availability of ingredients (`N_gf_ing`, `N_any_ing`). Calculate how
          many GF and Any sandwiches can actually be made from S0 objects and
          available ingredients, and use these to satisfy remaining needs. Add
          `count * 4`.
    8.  If, after exhausting all available sandwiches and makeable sandwiches
        from S0, there are still remaining waiting children (needs not met),
        return a large value (e.g., 1000) indicating a likely unsolvable state
        or a state far from the goal. Otherwise, return the calculated total cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts and goal information.
        """
        self.goals = task.goals

        # Extract static information
        self.allergic_children_static = set()
        self.gf_bread_static = set()
        self.gf_content_static = set()

        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == 'allergic_gluten' and len(parts) > 1:
                self.allergic_children_static.add(parts[1])
            elif predicate == 'no_gluten_bread' and len(parts) > 1:
                self.gf_bread_static.add(parts[1])
            elif predicate == 'no_gluten_content' and len(parts) > 1:
                self.gf_content_static.add(parts[1])

        # Extract goal children (not strictly needed for this heuristic which focuses on waiting children)
        # self.goal_served_children = set()
        # for goal in self.goals:
        #     parts = get_parts(goal)
        #     if not parts: continue
        #     predicate = parts[0]
        #     if predicate == 'served' and len(parts) > 1:
        #         self.goal_served_children.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Parse State ---
        waiting_children_state = {} # {child: place}
        tray_location = {}          # {tray: place}
        bread_kitchen = set()       # {bread}
        content_kitchen = set()     # {content}
        sandwich_state = {}         # {sandwich: state_info ('notexist',), ('kitchen',), ('ontray', tray_obj)}
        sandwich_is_gf_state = set() # {sandwich}

        # Collect all sandwich objects mentioned in the state
        all_sandwiches_in_state = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]

            if predicate == 'waiting' and len(parts) > 2:
                waiting_children_state[parts[1]] = parts[2]
            elif predicate == 'at' and len(parts) > 2:
                # Assuming 'tray' is the object type for movable containers
                if parts[1].startswith('tray'):
                     tray_location[parts[1]] = parts[2]
            elif predicate == 'at_kitchen_bread' and len(parts) > 1:
                bread_kitchen.add(parts[1])
            elif predicate == 'at_kitchen_content' and len(parts) > 1:
                content_kitchen.add(parts[1])
            elif predicate == 'notexist' and len(parts) > 1:
                s_obj = parts[1]
                sandwich_state[s_obj] = ('notexist',)
                all_sandwiches_in_state.add(s_obj)
            elif predicate == 'at_kitchen_sandwich' and len(parts) > 1:
                 s_obj = parts[1]
                 sandwich_state[s_obj] = ('kitchen',)
                 all_sandwiches_in_state.add(s_obj)
            elif predicate == 'ontray' and len(parts) > 2:
                 s_obj = parts[1]
                 tray_obj = parts[2]
                 sandwich_state[s_obj] = ('ontray', tray_obj)
                 all_sandwiches_in_state.add(s_obj)
            elif predicate == 'no_gluten_sandwich' and len(parts) > 1:
                 s_obj = parts[1]
                 sandwich_is_gf_state.add(s_obj)
                 all_sandwiches_in_state.add(s_obj)

        # --- Identify Needs ---
        waiting_gf_children = {c for c in waiting_children_state if c in self.allergic_children_static}
        waiting_any_children = {c for c in waiting_children_state if c not in self.allergic_children_static}

        rem_gf = len(waiting_gf_children)
        rem_any = len(waiting_any_children)

        # Goal reached if no children are waiting
        if rem_gf == 0 and rem_any == 0:
            return 0

        # --- Count Available Suitable Sandwiches by Stage ---
        # Stage 3 (Cost 1): On tray at table
        # Stage 2 (Cost 2): On tray at kitchen
        # Stage 1 (Cost 3): at_kitchen_sandwich
        # Stage 0 (Cost 4): notexist

        s_by_stage_gf = {1: [], 2: [], 3: []} # List of sandwich objects for stages 1, 2, 3
        s_by_stage_any = {1: [], 2: [], 3: []} # List of sandwich objects (non-GF) for stages 1, 2, 3

        N_s0_total = 0 # Count for stage 0 (notexist)

        for s_obj, state_info in sandwich_state.items():
            if state_info[0] == 'notexist':
                N_s0_total += 1
                continue # Handle S0 separately based on ingredients

            is_gf = s_obj in sandwich_is_gf_state

            if state_info[0] == 'ontray':
                tray_obj = state_info[1]
                if tray_obj in tray_location:
                    loc = tray_location[tray_obj]
                    # Check if the tray is at a table where a child is waiting
                    # This is a simplification; ideally, it should be the *correct* table for a waiting child
                    is_at_waiting_table = loc in waiting_children_state.values()
                    if is_at_waiting_table:
                        if is_gf: s_by_stage_gf[3].append(s_obj)
                        else: s_by_stage_any[3].append(s_obj)
                    elif loc == 'kitchen':
                        if is_gf: s_by_stage_gf[2].append(s_obj)
                        else: s_by_stage_any[2].append(s_obj)
            elif state_info[0] == 'kitchen':
                if is_gf: s_by_stage_gf[1].append(s_obj)
                else: s_by_stage_any[1].append(s_obj)


        # --- Count Available Ingredients ---
        gf_bread_kitchen = bread_kitchen.intersection(self.gf_bread_static)
        gf_content_kitchen = content_kitchen.intersection(self.gf_content_static)
        N_gf_ing = min(len(gf_bread_kitchen), len(gf_content_kitchen))

        any_bread_kitchen = bread_kitchen - gf_bread_kitchen
        any_content_kitchen = content_kitchen - gf_content_kitchen
        N_any_ing = min(len(any_bread_kitchen), len(any_content_kitchen)) # Non-GF ingredients

        # --- Calculate Heuristic Cost ---
        h = 0

        # Stage 3 (Cost 1: serve)
        use_s3_gf_for_gf = min(rem_gf, len(s_by_stage_gf[3]))
        h += use_s3_gf_for_gf * 1
        rem_gf -= use_s3_gf_for_gf
        N_s3_gf_rem = len(s_by_stage_gf[3]) - use_s3_gf_for_gf # Remaining S3 GF

        use_s3_any_for_any = min(rem_any, len(s_by_stage_any[3]))
        h += use_s3_any_for_any * 1
        rem_any -= use_s3_any_for_any
        N_s3_any_rem = len(s_by_stage_any[3]) - use_s3_any_for_any # Remaining S3 Any

        use_s3_gf_for_any = min(rem_any, N_s3_gf_rem)
        h += use_s3_gf_for_any * 1
        rem_any -= use_s3_gf_for_any
        # N_s3_gf_rem -= use_s3_gf_for_any # Not needed for calculation

        # Stage 2 (Cost 2: move + serve)
        use_s2_gf_for_gf = min(rem_gf, len(s_by_stage_gf[2]))
        h += use_s2_gf_for_gf * 2
        rem_gf -= use_s2_gf_for_gf
        N_s2_gf_rem = len(s_by_stage_gf[2]) - use_s2_gf_for_gf

        use_s2_any_for_any = min(rem_any, len(s_by_stage_any[2]))
        h += use_s2_any_for_any * 2
        rem_any -= use_s2_any_for_any
        N_s2_any_rem = len(s_by_stage_any[2]) - use_s2_any_for_any

        use_s2_gf_for_any = min(rem_any, N_s2_gf_rem)
        h += use_s2_gf_for_any * 2
        rem_any -= use_s2_gf_for_any
        # N_s2_gf_rem -= use_s2_gf_for_any

        # Stage 1 (Cost 3: put + move + serve)
        use_s1_gf_for_gf = min(rem_gf, len(s_by_stage_gf[1]))
        h += use_s1_gf_for_gf * 3
        rem_gf -= use_s1_gf_for_gf
        N_s1_gf_rem = len(s_by_stage_gf[1]) - use_s1_gf_for_gf

        use_s1_any_for_any = min(rem_any, len(s_by_stage_any[1]))
        h += use_s1_any_for_any * 3
        rem_any -= use_s1_any_for_any
        N_s1_any_rem = len(s_by_stage_any[1]) - use_s1_any_rem

        use_s1_gf_for_any = min(rem_any, N_s1_gf_rem)
        h += use_s1_gf_for_any * 3
        rem_any -= use_s1_gf_for_any
        # N_s1_gf_rem -= use_s1_gf_for_any

        # Stage 0 (Cost 4: make + put + move + serve)
        # Needs rem_gf GF sandwiches and rem_any Any sandwiches from S0.
        # Limited by N_s0_total, N_gf_ing, N_any_ing.

        # How many GF sandwiches can we make from S0? Limited by rem_gf needed, N_s0_total objects, N_gf_ing.
        make_s0_gf = min(rem_gf, N_s0_total, N_gf_ing)
        h += make_s0_gf * 4
        rem_gf -= make_s0_gf
        N_s0_total -= make_s0_gf
        N_gf_ing -= make_s0_gf

        # How many Any sandwiches (non-GF ingredients) can we make from remaining S0?
        # Limited by rem_any needed, remaining N_s0_total objects, N_any_ing.
        make_s0_any_nongf_ing = min(rem_any, N_s0_total, N_any_ing)
        h += make_s0_any_nongf_ing * 4
        rem_any -= make_s0_any_nongf_ing
        N_s0_total -= make_s0_any_nongf_ing
        N_any_ing -= make_s0_any_nongf_ing

        # How many Any sandwiches (GF ingredients) can we make from remaining S0?
        # Limited by rem_any needed, remaining N_s0_total objects, remaining N_gf_ing.
        make_s0_any_gf_ing = min(rem_any, N_s0_total, N_gf_ing)
        h += make_s0_any_gf_ing * 4
        rem_any -= make_s0_any_gf_ing
        N_s0_total -= make_s0_any_gf_ing
        N_gf_ing -= make_s0_any_gf_ing


        # If any needs remain, the state is likely unsolvable or very far.
        if rem_gf > 0 or rem_any > 0:
             # Return a large value to guide search away from such states.
             return 1000 # Use a large constant instead of inf for robustness

        return h
