from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if the actual one is not provided
# This allows the code structure to be correct for the problem description.
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        raise NotImplementedError("Subclasses must implement __call__")

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or invalid formats gracefully
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj1 loc1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting
    children. It counts the number of sandwiches needed at different stages of
    preparation and delivery (from ingredients to on-tray at table), weighted
    by the estimated number of actions remaining from that stage to reach the
    final 'served' state. It prioritizes using items that are closer to being served
    and accounts for gluten allergies.

    # Assumptions
    - Each unserved child requires one suitable sandwich delivery.
    - Gluten-allergic children require gluten-free sandwiches. Non-allergic children
      can accept any sandwich.
    - The stages of sandwich preparation and delivery are sequential:
      Ingredients -> Made -> At_kitchen_sandwich -> Ontray in Kitchen -> Ontray at Table -> Served.
    - Each transition between these stages costs approximately one action.
    - The heuristic sums the estimated costs for the *minimum number of sandwiches*
      needed at each stage to satisfy all unserved children, prioritizing items
      further along the pipeline.
    - Tray capacity and the ability to carry multiple sandwiches on one tray
      or serve multiple children at the same table with one tray are simplified
      or ignored in the counting for efficiency.

    # Heuristic Initialization
    - Extracts static information about which children are allergic to gluten
      and which bread/content portions are gluten-free.
    - Stores the goal conditions (which children need to be served).

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  **Identify Unserved Children:** Determine which children are in the goal
        conditions but are not currently in the 'served' state. Separate them
        into those needing gluten-free sandwiches (allergic) and those who don't.
        Count the total number of unserved children (`N_unserved`), and counts
        for GF-allergic (`N_gf_unserved`) and non-allergic (`N_reg_unserved`).

    2.  **Identify Available Sandwiches and Ingredients:** Scan the current state
        to find existing sandwiches and ingredients.
        - Determine the gluten status of made sandwiches based on the ingredients
          used (if known) or explicit predicates like `(no_gluten_sandwich S)`.
        - Track sandwiches that are `(ontray S T)`, their tray's location `(at T ?loc)`,
          and their gluten status.
        - Track sandwiches that are `(at_kitchen_sandwich S)` and their gluten status.
        - Track sandwiches that are `(made S B C)` but not yet `at_kitchen_sandwich`
          or on a tray, and their gluten status.
        - Count available gluten-free and regular bread and content portions
          in the kitchen `(at_kitchen_bread B)`, `(at_kitchen_content C)`.

    3.  **Categorize Available Items by Stage and Gluten Status:**
        Group the available sandwiches and ingredients based on their current
        "stage" towards being served and their gluten status (GF or Regular).
        - Stage 4 (Ontray at Table): Sandwiches on trays located at a table.
        - Stage 3 (Ontray in Kitchen): Sandwiches on trays located in the kitchen.
        - Stage 2 (At_kitchen_sandwich): Sandwiches in the kitchen, not on a tray.
        - Stage 1 (Made): Sandwiches that are made but not yet in the kitchen_sandwich state or on a tray.
        - Stage 0 (Ingredients): Available ingredient pairs that can be used to make a sandwich.

    4.  **Estimate Costs for Each Stage:** Assign a cost to bring an item from
        each stage to the final 'served' state:
        - Stage 4 (Ontray at Table): 1 action (serve)
        - Stage 3 (Ontray in Kitchen): 2 actions (move tray, serve)
        - Stage 2 (At_kitchen_sandwich): 3 actions (put on tray, move tray, serve)
        - Stage 1 (Made): 4 actions (put in kitchen, put on tray, move tray, serve)
        - Stage 0 (Ingredients): 5 actions (make, put in kitchen, put on tray, move tray, serve)

    5.  **Greedily Satisfy Needs:** Iterate through the unserved children's needs,
        prioritizing gluten-free needs first, and for each need type, prioritize
        using available items from the highest stage (closest to goal) downwards.
        - For GF needs: Use available GF items from Stage 4, then Stage 3, etc.,
          until all GF needs are met or GF items are exhausted. Add the corresponding
          cost for each item used.
        - For Regular needs: Use available Regular items from Stage 4, then Stage 3, etc.
          If Regular items are exhausted, use remaining available GF items from
          Stage 4, then Stage 3, etc. Add the corresponding cost for each item used.

    6.  **Sum Costs:** The total heuristic value is the sum of the costs accumulated
        while satisfying the needs of all unserved children. This sum represents
        an estimate of the total actions required, counting the "work" needed
        for each required sandwich delivery based on its current state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Extract child allergy status
        self.child_needs_gf = {} # Map child name to True if allergic, False otherwise
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'allergic_gluten':
                self.child_needs_gf[parts[1]] = True
            elif parts and parts[0] == 'not_allergic_gluten':
                 self.child_needs_gf[parts[1]] = False # Default to False if not explicitly listed? No, check all children later.

        # Extract ingredient gluten status
        self.no_gluten_bread_names = set()
        self.no_gluten_content_names = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'no_gluten_bread':
                self.no_gluten_bread_names.add(parts[1])
            elif parts and parts[0] == 'no_gluten_content':
                self.no_gluten_content_names.add(parts[1])

        # Identify all children from goals to ensure all are covered
        all_children_in_goals = set()
        for goal in self.goals:
             parts = get_parts(goal)
             if parts and parts[0] == 'served':
                 all_children_in_goals.add(parts[1])

        # Ensure all children mentioned in goals have an allergy status recorded
        # Default to not allergic if not specified in static facts
        for child in all_children_in_goals:
             if child not in self.child_needs_gf:
                 self.child_needs_gf[child] = False


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify Unserved Children
        unserved_children = set()
        N_gf_unserved = 0
        N_reg_unserved = 0

        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'served':
                child = parts[1]
                if goal not in state:
                    unserved_children.add(child)
                    if self.child_needs_gf.get(child, False): # Default to False if child not in static (shouldn't happen with init logic)
                        N_gf_unserved += 1
                    else:
                        N_reg_unserved += 1

        if not unserved_children:
            return 0 # Goal reached

        # 2. & 3. Identify Available Sandwiches and Ingredients by Stage and Gluten Status
        ontray_sandwiches_map = {}  # sandwich -> tray
        kitchen_sandwiches_set = set() # set of sandwiches at_kitchen_sandwich
        made_sandwiches_map = {}    # sandwich -> (bread, content)
        sandwich_gluten_status = {} # sandwich -> is_gf (True/False)
        tray_locations = {}         # tray -> location
        available_bread = set()     # set of bread names in kitchen
        available_content = set()   # set of content names in kitchen

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == 'ontray':
                ontray_sandwiches_map[parts[1]] = parts[2]
            elif parts[0] == 'at_kitchen_sandwich':
                kitchen_sandwiches_set.add(parts[1])
            elif parts[0] == 'made':
                made_sandwiches_map[parts[1]] = (parts[2], parts[3])
            elif parts[0] == 'no_gluten_sandwich':
                sandwich_gluten_status[parts[1]] = True
            elif parts[0] == 'has_gluten':
                sandwich_gluten_status[parts[1]] = False
            elif parts[0] == 'at' and len(parts) == 3: # Check for (at ?obj ?loc)
                 # Need to identify trays specifically. Assuming tray objects start with 'tray'
                 if parts[1].startswith('tray'):
                    tray_locations[parts[1]] = parts[2]
            elif parts[0] == 'at_kitchen_bread':
                available_bread.add(parts[1])
            elif parts[0] == 'at_kitchen_content':
                available_content.add(parts[1])

        # Infer gluten status for made sandwiches if not explicitly stated
        for s, (b, c) in made_sandwiches_map.items():
            if s not in sandwich_gluten_status:
                is_gf = (b in self.no_gluten_bread_names) and (c in self.no_gluten_content_names)
                sandwich_gluten_status[s] = is_gf

        # Count available sandwiches by stage and gluten status
        Avail_s4_gf = 0 # Ontray at Table
        Avail_s4_reg = 0
        Avail_s3_gf = 0 # Ontray in Kitchen
        Avail_s3_reg = 0
        Avail_s2_gf = 0 # At_kitchen_sandwich
        Avail_s2_reg = 0
        Avail_s1_gf = 0 # Made

        for s, t in ontray_sandwiches_map.items():
            loc = tray_locations.get(t)
            is_gf = sandwich_gluten_status.get(s, False) # Default to False if status unknown
            if loc is not None:
                if loc != 'kitchen': # Assume any non-kitchen location is a table
                    if is_gf: Avail_s4_gf += 1
                    else: Avail_s4_reg += 1
                else: # In kitchen
                    if is_gf: Avail_s3_gf += 1
                    else: Avail_s3_reg += 1

        for s in kitchen_sandwiches_set:
            if s not in ontray_sandwiches_map: # Count only if not already on a tray
                is_gf = sandwich_gluten_status.get(s, False)
                if is_gf: Avail_s2_gf += 1
                else: Avail_s2_reg += 1

        # Count 'made' sandwiches that are not yet at_kitchen_sandwich or ontray
        # We only need the count of GF 'made' sandwiches, as regular ones can be
        # made from any ingredients including GF ones. The cost is the same.
        # Let's just count the total number of 'made' sandwiches not yet further along.
        Avail_s1_total = 0
        for s in made_sandwiches_map.keys():
             if s not in ontray_sandwiches_map and s not in kitchen_sandwiches_set:
                 Avail_s1_total += 1
        # We don't strictly need Avail_s1_gf/reg separately if the cost is the same.
        # But for clarity in satisfying needs, let's track GF status.
        Avail_s1_gf = 0
        Avail_s1_reg = 0
        for s in made_sandwiches_map.keys():
             if s not in ontray_sandwiches_map and s not in kitchen_sandwiches_set:
                 is_gf = sandwich_gluten_status.get(s, False)
                 if is_gf: Avail_s1_gf += 1
                 else: Avail_s1_reg += 1


        # Count available ingredients for making sandwiches
        gf_bread_count = len([b for b in available_bread if b in self.no_gluten_bread_names])
        reg_bread_count = len([b for b in available_bread if b not in self.no_gluten_bread_names])
        gf_content_count = len([c for c in available_content if c in self.no_gluten_content_names])
        reg_content_count = len([c for c in available_content if c not in self.no_gluten_content_names])

        Avail_s0_gf = min(gf_bread_count, gf_content_count)
        # Regular sandwiches can be made from any bread/content.
        # The number of regular sandwiches we can make is limited by the total available ingredients
        # minus those used for GF sandwiches.
        # This is a simplification. A more accurate count would consider pairs.
        # Let's assume we can make a regular sandwich from any bread + any content.
        # Total available bread = gf_bread_count + reg_bread_count
        # Total available content = gf_content_count + reg_content_count
        # Ingredients used for Avail_s0_gf: Avail_s0_gf of GF bread, Avail_s0_gf of GF content.
        # Remaining GF bread = gf_bread_count - Avail_s0_gf
        # Remaining GF content = gf_content_count - Avail_s0_gf
        # Remaining Reg bread = reg_bread_count
        # Remaining Reg content = reg_content_count
        # Total remaining bread = (gf_bread_count - Avail_s0_gf) + reg_bread_count
        # Total remaining content = (gf_content_count - Avail_s0_gf) + reg_content_count
        # Avail_s0_reg = min(Total remaining bread, Total remaining content)
        # This is still complex. Let's simplify: assume we can make N sandwiches if we have N bread and N content.
        # Avail_s0_total_ingredients = min(len(available_bread), len(available_content))
        # Avail_s0_reg = Avail_s0_total_ingredients - Avail_s0_gf # This assumes GF ingredients are a subset of total.

        # Simpler ingredient counting: Count how many GF sandwiches can be made. Count how many non-GF sandwiches can be made.
        # Non-GF sandwich needs non-GF bread AND non-GF content.
        Avail_s0_reg = min(reg_bread_count, reg_content_count)


        # 4. Estimate Costs for Each Stage (already defined in thought process)
        # Cost_s4 = 1 (serve)
        # Cost_s3 = 2 (move, serve)
        # Cost_s2 = 3 (put, move, serve)
        # Cost_s1 = 4 (put_in_kitchen, put, move, serve)
        # Cost_s0 = 5 (make, put_in_kitchen, put, move, serve)

        Heuristic = 0

        # 5. Greedily Satisfy Needs

        # Satisfy GF needs (N_gf_unserved)
        use = min(N_gf_unserved, Avail_s4_gf)
        Heuristic += use * 1
        N_gf_unserved -= use
        Avail_s4_gf -= use

        use = min(N_gf_unserved, Avail_s3_gf)
        Heuristic += use * 2
        N_gf_unserved -= use
        Avail_s3_gf -= use

        use = min(N_gf_unserved, Avail_s2_gf)
        Heuristic += use * 3
        N_gf_unserved -= use
        Avail_s2_gf -= use

        use = min(N_gf_unserved, Avail_s1_gf)
        Heuristic += use * 4
        N_gf_unserved -= use
        Avail_s1_gf -= use

        use = min(N_gf_unserved, Avail_s0_gf)
        Heuristic += use * 5
        N_gf_unserved -= use
        Avail_s0_gf -= use

        # Satisfy Regular needs (N_reg_unserved) - Can use Reg or GF items

        # Use Stage 4 items (Reg then GF)
        use_reg_s4 = min(N_reg_unserved, Avail_s4_reg)
        Heuristic += use_reg_s4 * 1
        N_reg_unserved -= use_reg_s4
        Avail_s4_reg -= use_reg_s4

        use_gf_s4 = min(N_reg_unserved, Avail_s4_gf)
        Heuristic += use_gf_s4 * 1
        N_reg_unserved -= use_gf_s4
        Avail_s4_gf -= use_gf_s4

        # Use Stage 3 items (Reg then GF)
        use_reg_s3 = min(N_reg_unserved, Avail_s3_reg)
        Heuristic += use_reg_s3 * 2
        N_reg_unserved -= use_reg_s3
        Avail_s3_reg -= use_reg_s3

        use_gf_s3 = min(N_reg_unserved, Avail_s3_gf)
        Heuristic += use_gf_s3 * 2
        N_reg_unserved -= use_gf_s3
        Avail_s3_gf -= use_gf_s3

        # Use Stage 2 items (Reg then GF)
        use_reg_s2 = min(N_reg_unserved, Avail_s2_reg)
        Heuristic += use_reg_s2 * 3
        N_reg_unserved -= use_reg_s2
        Avail_s2_reg -= use_reg_s2

        use_gf_s2 = min(N_reg_unserved, Avail_s2_gf)
        Heuristic += use_gf_s2 * 3
        N_reg_unserved -= use_gf_s2
        Avail_s2_gf -= use_gf_s2

        # Use Stage 1 items (Reg then GF)
        # Need to count Avail_s1_reg correctly. A 'made' sandwich is reg if it's not GF.
        # Avail_s1_reg was calculated earlier.
        use_reg_s1 = min(N_reg_unserved, Avail_s1_reg)
        Heuristic += use_reg_s1 * 4
        N_reg_unserved -= use_reg_s1
        Avail_s1_reg -= use_reg_s1

        use_gf_s1 = min(N_reg_unserved, Avail_s1_gf)
        Heuristic += use_gf_s1 * 4
        N_reg_unserved -= use_gf_s1
        Avail_s1_gf -= use_gf_s1


        # Use Stage 0 ingredients (Reg then GF)
        # Need to count Avail_s0_reg correctly. It's min(reg_bread, reg_content).
        # GF ingredients can also make regular sandwiches if needed.
        # Total ingredients available for regular sandwiches = (remaining gf bread + reg bread) and (remaining gf content + reg content)
        # Remaining GF bread = gf_bread_count - (Avail_s0_gf used for GF needs) - (Avail_s0_gf used for Reg needs from GF pool)
        # Let's re-calculate Avail_s0_gf and Avail_s0_reg based on initial counts
        initial_Avail_s0_gf = min(gf_bread_count, gf_content_count)
        initial_Avail_s0_reg = min(reg_bread_count, reg_content_count) # Non-GF ingredients only

        # Ingredients available for ANY sandwich (total bread/content pairs)
        Avail_s0_any = min(gf_bread_count + reg_bread_count, gf_content_count + reg_content_count)

        # How many ingredient pairs were used for GF needs?
        used_s0_gf_for_gf_needs = min(N_gf_unserved, initial_Avail_s0_gf)

        # Remaining ingredients for any sandwich
        remaining_s0_any = Avail_s0_any - used_s0_gf_for_gf_needs

        # Remaining non-GF ingredients for regular sandwiches
        remaining_s0_reg = initial_Avail_s0_reg # Non-GF ingredients only

        # Now satisfy remaining Regular needs using remaining ingredients
        # Prioritize using non-GF ingredients first for regular sandwiches
        use_reg_s0 = min(N_reg_unserved, remaining_s0_reg)
        Heuristic += use_reg_s0 * 5
        N_reg_unserved -= use_reg_s0
        remaining_s0_any -= use_reg_s0 # These ingredients are now used

        # Use remaining ANY ingredients (could be GF or Reg) for remaining regular needs
        use_any_s0 = min(N_reg_unserved, remaining_s0_any)
        Heuristic += use_any_s0 * 5
        N_reg_unserved -= use_any_s0
        remaining_s0_any -= use_any_s0


        # If N_gf_unserved > 0 or N_reg_unserved > 0 here, it means there aren't enough
        # total ingredients/sandwiches in the state to satisfy all children.
        # For a solvable problem, this shouldn't happen if the initial state is solvable.
        # If it can happen in unsolvable states, we might return infinity.
        # Assuming solvable states always have enough resources eventually,
        # the remaining N_gf_unserved and N_reg_unserved should be 0.

        return Heuristic

