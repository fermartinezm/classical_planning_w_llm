from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or invalid format defensively
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
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
    This heuristic estimates the minimum number of actions required to serve all
    children specified in the goal. It considers the state of sandwiches and
    ingredients, and the location of trays, assigning available resources
    greedily to unserved children based on the estimated cost to get a suitable
    sandwich to their table.

    # Assumptions
    - Each child needs exactly one sandwich.
    - All children specified in the goal must be served.
    - All actions have a cost of 1.
    - Resources (sandwiches, trays, ingredients) are consumed when used to serve a child.
    - The heuristic assumes solvable problems have sufficient resources eventually.
    - The heuristic assigns resources greedily based on the "stage" of readiness,
      prioritizing cheaper stages and No-Gluten resources for allergic children:
        1. Suitable sandwiches already on trays located at the child's table (cost 1: serve).
        2. Suitable sandwiches on trays elsewhere (cost 2: move tray + serve).
        3. Suitable sandwiches in the kitchen (`at_kitchen_sandwich`) (cost 3: put on tray + move tray + serve).
        4. Suitable sandwiches that need to be made from ingredients (cost 4: make + put on tray + move tray + serve).

    # Heuristic Initialization
    The heuristic extracts the following static information from the task:
    - The set of children that need to be served (from the goal).
    - The allergy status (allergic_gluten or not_allergic_gluten) for each child.
    - The table where each child is waiting.
    - The gluten status (no_gluten_bread, no_gluten_content) for bread and content portions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the set of children who are not yet served. This is the set of goal children
       minus those for whom `(served child)` is true in the current state.
    2. If no children are unserved, the goal is reached, return 0.
    3. Separate the unserved children into those who are allergic to gluten and those who are not.
    4. Initialize the heuristic value to the total number of unserved children (representing the minimum
       cost of the final `serve` action for each).
    5. Count available resources at different stages of readiness:
       - Stage 1: Suitable sandwiches already on trays located at the tables of unserved children.
       - Stage 2: Suitable sandwiches on trays located elsewhere (not at a table of an unserved child).
       - Stage 3: Suitable sandwiches located in the kitchen (`at_kitchen_sandwich`).
       - Stage 4: Available ingredients in the kitchen (`at_kitchen_bread`, `at_kitchen_content`)
         that can be used to make suitable sandwiches.
       - Available trays in the kitchen (`at tray kitchen`).
       Keep track of specific resources (sandwiches, trays) used in Stages 1 and 2 to avoid double-counting.
    6. Greedily assign resources from the cheapest stages first to satisfy the needs of the
       remaining unserved children. Prioritize assigning No-Gluten resources to allergic children.
       - For each child satisfied by a Stage 2 resource, add 1 to the heuristic (cost of move).
       - For each child satisfied by a Stage 3 resource, add 2 to the heuristic (cost of put + move).
       - For each child satisfied by a Stage 4 resource, add 3 to the heuristic (cost of make + put + move).
       Update the counts of remaining children needing service and available resources after each assignment stage.
    7. If, after considering all available resources, there are still unserved children, the state
       is likely unsolvable with the available initial resources. Return infinity in this case.
    8. The final heuristic value is the sum of the base cost (number of unserved children) and the
       additional costs calculated in step 6.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.goal_children = set()
        for goal in self.goals:
            # Goal is typically (served ?child)
            parts = get_parts(goal)
            if parts and parts[0] == "served":
                self.goal_children.add(parts[1])

        self.child_allergy = {} # child -> bool (True if allergic)
        self.child_to_table = {} # child -> table
        self.is_nogluten_bread = {} # bread -> bool (True if no-gluten)
        self.is_nogluten_content = {} # content -> bool (True if no-gluten)
        self.table_to_children = {} # table -> list of children waiting there (static)

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == "allergic_gluten":
                self.child_allergy[parts[1]] = True
            elif parts[0] == "not_allergic_gluten":
                self.child_allergy[parts[1]] = False
            elif parts[0] == "waiting":
                child, table = parts[1], parts[2]
                self.child_to_table[child] = table
                self.table_to_children.setdefault(table, []).append(child)
            elif parts[0] == "no_gluten_bread":
                self.is_nogluten_bread[parts[1]] = True
            elif parts[0] == "no_gluten_content":
                self.is_nogluten_content[parts[1]] = True


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unserved children
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "?c")}
        unserved_children = self.goal_children - served_children

        # 2. If goal reached, return 0
        if not unserved_children:
            return 0

        # 3. Separate unserved children by allergy status
        unserved_ng = {c for c in unserved_children if self.child_allergy.get(c, False)}
        unserved_g = unserved_children - unserved_ng

        # 4. Initialize heuristic (cost of serve action for each)
        h = len(unserved_children)

        # Track children who still need a sandwich delivered
        children_needing_delivery_ng = set(unserved_ng)
        children_needing_delivery_g = set(unserved_g)

        # Track used resources to avoid double counting
        used_s_t = set() # (sandwich, tray) pairs


        # --- Stage 1: Ready at Table (Cost +0) ---
        # Count suitable sandwiches already on trays at tables of unserved children
        # Prioritize NG for NG children
        available_s_t_at_table_ng = [] # List of (s, t) pairs
        available_s_t_at_table_g = [] # List of (s, t) pairs

        for fact in state:
            if match(fact, "ontray", "?s", "?t"):
                s = get_parts(fact)[1]
                t = get_parts(fact)[2]
                tray_loc = None
                for fact2 in state:
                    if match(fact2, "at", t, "?loc"):
                        tray_loc = get_parts(fact2)[2]
                        break
                # Check if this tray is at a table where *any* unserved child is waiting
                if tray_loc in self.table_to_children:
                     is_ng_sandwich = "(no_gluten_sandwich " + s + ")" in state
                     if is_ng_sandwich:
                         available_s_t_at_table_ng.append((s, t))
                     else:
                         available_s_t_at_table_g.append((s, t))

        # Assign NG at table to NG children needing delivery
        k = min(len(children_needing_delivery_ng), len(available_s_t_at_table_ng))
        for i in range(k):
            child = list(children_needing_delivery_ng)[0]
            children_needing_delivery_ng.remove(child)
            used_s_t.add(available_s_t_at_table_ng.pop(0))

        # Assign G at table (and remaining NG) to G children needing delivery
        available_s_t_at_table_any = available_s_t_at_table_ng + available_s_t_at_table_g
        k = min(len(children_needing_delivery_g), len(available_s_t_at_table_any))
        for i in range(k):
            child = list(children_needing_delivery_g)[0]
            children_needing_delivery_g.remove(child)
            used_s_t.add(available_s_t_at_table_any.pop(0))


        # --- Stage 2: On Tray Elsewhere (Cost +1) ---
        available_s_t_elsewhere_ng = []
        available_s_t_elsewhere_g = []

        for fact in state:
            if match(fact, "ontray", "?s", "?t"):
                s = get_parts(fact)[1]
                t = get_parts(fact)[2]
                if (s, t) not in used_s_t:
                    is_ng_sandwich = "(no_gluten_sandwich " + s + ")" in state
                    if is_ng_sandwich:
                        available_s_t_elsewhere_ng.append((s, t))
                    else:
                        available_s_t_elsewhere_g.append((s, t))


        # Assign NG elsewhere to remaining NG children
        k = min(len(children_needing_delivery_ng), len(available_s_t_elsewhere_ng))
        h += k * 1 # 1 move action
        for i in range(k):
            child = list(children_needing_delivery_ng)[0]
            children_needing_delivery_ng.remove(child)
            used_s_t.add(available_s_t_elsewhere_ng.pop(0))

        # Assign G elsewhere (and remaining NG) to remaining G children
        available_s_t_elsewhere_any = available_s_t_elsewhere_ng + available_s_t_elsewhere_g
        k = min(len(children_needing_delivery_g), len(available_s_t_elsewhere_any))
        h += k * 1 # 1 move action
        for i in range(k):
            child = list(children_needing_delivery_g)[0]
            children_needing_delivery_g.remove(child)
            used_s_t.add(available_s_t_elsewhere_any.pop(0))


        # --- Stage 3: Kitchen Sandwich (Cost +2) ---
        available_s_kitchen_ng = [get_parts(f)[1] for f in state if match(f, "at_kitchen_sandwich", "?s") and "(no_gluten_sandwich " + get_parts(f)[1] + ")" in state]
        available_s_kitchen_g = [get_parts(f)[1] for f in state if match(f, "at_kitchen_sandwich", "?s") and "(no_gluten_sandwich " + get_parts(f)[1] + ")" not in state]
        available_t_kitchen = [get_parts(f)[1] for f in state if match(f, "at", "?t", "kitchen")]

        # Assign NG kitchen to remaining NG children
        k = min(len(children_needing_delivery_ng), len(available_s_kitchen_ng), len(available_t_kitchen))
        h += k * 2 # 1 put + 1 move
        for i in range(k):
            child = list(children_needing_delivery_ng)[0]
            children_needing_delivery_ng.remove(child)
            available_s_kitchen_ng.pop(0) # Consume sandwich
            available_t_kitchen.pop(0) # Consume tray

        # Assign G kitchen (and remaining NG) to remaining G children
        available_s_kitchen_any = available_s_kitchen_ng + available_s_kitchen_g
        k = min(len(children_needing_delivery_g), len(available_s_kitchen_any), len(available_t_kitchen))
        h += k * 2 # 1 put + 1 move
        for i in range(k):
            child = list(children_needing_delivery_g)[0]
            children_needing_delivery_g.remove(child)
            available_s_kitchen_any.pop(0) # Consume sandwich
            available_t_kitchen.pop(0) # Consume tray


        # --- Stage 4: Ingredients (Cost +3) ---
        # Count available ingredients
        available_b_kitchen_ng = len([get_parts(f)[1] for f in state if match(f, "at_kitchen_bread", "?b") and "(no_gluten_bread " + get_parts(f)[1] + ")" in state])
        available_c_kitchen_ng = len([get_parts(f)[1] for f in state if match(f, "at_kitchen_content", "?c") and "(no_gluten_content " + get_parts(f)[1] + ")" in state])
        available_b_kitchen_g = len([get_parts(f)[1] for f in state if match(f, "at_kitchen_bread", "?b") and "(no_gluten_bread " + get_parts(f)[1] + ")" not in state])
        available_c_kitchen_g = len([get_parts(f)[1] for f in state if match(f, "at_kitchen_content", "?c") and "(no_gluten_content " + get_parts(f)[1] + ")" not in state])

        makeable_ng_total = min(available_b_kitchen_ng, available_c_kitchen_ng)

        # Ingredients remaining after considering makeable NG
        b_rem_ng = available_b_kitchen_ng - makeable_ng_total
        c_rem_ng = available_c_kitchen_ng - makeable_ng_total

        makeable_g_bread_avail = available_b_kitchen_g + b_rem_ng
        makeable_g_content_avail = available_c_kitchen_g + c_rem_ng
        makeable_g_total = min(makeable_g_bread_avail, makeable_g_content_avail)

        # Assign makeable NG to remaining NG children
        k = min(len(children_needing_delivery_ng), makeable_ng_total, len(available_t_kitchen))
        h += k * 3 # 1 make + 1 put + 1 move
        for i in range(k):
             child = list(children_needing_delivery_ng)[0]
             children_needing_delivery_ng.remove(child)
             # Assume ingredients are used
             available_t_kitchen.pop(0) # Consume a tray

        # Assign makeable G (and remaining makeable NG) to remaining G children
        # Remaining makeable NG can be used for G children
        makeable_g_total_with_rem_ng = makeable_g_total + (makeable_ng_total - k) # Remaining NG makeable after NG children used some

        k = min(len(children_needing_delivery_g), makeable_g_total_with_rem_ng, len(available_t_kitchen))
        h += k * 3 # 1 make + 1 put + 1 move
        for i in range(k):
             child = list(children_needing_delivery_g)[0]
             children_needing_delivery_g.remove(child)
             # Assume ingredients are used
             available_t_kitchen.pop(0) # Consume a tray


        # 7. Check if all children are covered
        if len(children_needing_delivery_ng) > 0 or len(children_needing_delivery_g) > 0:
            # This indicates an unsolvable state or insufficient resources in the model
            # For greedy search, a large value is appropriate.
            return float('inf')

        return h
