from fnmatch import fnmatch
# Assuming Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic

# If the Heuristic base class is not provided in the execution environment,
# you might need a simple definition like this:
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or non-string input defensively
    if not isinstance(fact, str) or len(fact) < 2:
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
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
    This heuristic estimates the number of actions required to serve all waiting children.
    It counts the number of unserved children of each type (allergic/non-allergic)
    and estimates the cost to deliver a suitable sandwich to them based on the
    sandwich's current location/state (on tray at child's location, on tray elsewhere,
    in kitchen, needs making). It prioritizes using sandwiches that are closer
    to being served.

    # Assumptions
    - Each unserved child requires exactly one sandwich of the appropriate type.
    - Any suitable sandwich can be used for any child of the corresponding type.
    - The cost is estimated by summing the minimum actions needed for each
      sandwich requirement, assuming resources (trays, ingredients, notexist
      sandwich slots) are shared optimally but actions are counted per requirement.
    - Action costs are uniform (cost 1 per make, put, move, serve).
    - If ingredients/notexist slots are insufficient to make needed sandwiches,
      a large penalty is added, implying the state is likely unsolvable.

    # Heuristic Initialization
    - Extracts static information: which children are allergic/not allergic,
      which bread/content is gluten-free, and which children are initially
      waiting and at which locations. This helps identify the total number
      of children that need serving and their types.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children who are initially waiting but are not yet served
       in the current state.
    2. Count the number of unserved allergic children (`N_gf_needed`) and
       unserved non-allergic children (`N_reg_needed`). If zero, the heuristic is 0.
    3. Count available sandwiches of each type (GF/Regular) based on their
       current state/location, categorized by the minimum number of actions
       needed to get them to a child's location and served:
       - Stage 1 (Cost 1: serve): Suitable sandwiches on trays at locations
         where a child needing that type is waiting.
       - Stage 2 (Cost 2: move + serve): Suitable sandwiches on trays elsewhere.
       - Stage 3 (Cost 3: put + move + serve): Suitable sandwiches in the kitchen.
       - Stage 4 (Cost 4: make + put + move + serve): Sandwiches that can be
         made from available ingredients and `notexist` slots.
    4. Count available ingredients in the kitchen: total bread, total content,
       GF bread, GF content, and available `notexist` sandwich slots.
    5. Initialize total estimated cost to 0.
    6. Greedily satisfy the total needs (`N_gf_needed`, `N_reg_needed`) using
       sandwiches from the cheapest stages first (Stage 1, then 2, then 3, then 4).
       For each sandwich used from a stage, add the corresponding cost (1, 2, 3, or 4)
       to the total cost and decrement the remaining need.
    7. When using Stage 4 (making sandwiches), consider the ingredient constraints.
       Prioritize making GF sandwiches first if needed, using GF ingredients and
       `notexist` slots. Then, make Regular sandwiches from remaining ingredients
       (including any unused GF ingredients) and `notexist` slots.
    8. If, after exhausting all available sandwiches and makeable capacity,
       there are still unserved children (remaining needed > 0), add a large
       penalty to the cost, indicating that the current state likely cannot
       reach the goal with available resources.
    9. Return the total estimated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.
        """
        self.goals = task.goals
        static_facts = task.static

        self.allergic_children = set()
        self.not_allergic_children = set()
        self.gf_bread_types = set()
        self.gf_content_types = set()
        self.initial_waiting_children = {} # child -> place

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "allergic_gluten":
                self.allergic_children.add(parts[1])
            elif predicate == "not_allergic_gluten":
                self.not_allergic_children.add(parts[1])
            elif predicate == "no_gluten_bread":
                self.gf_bread_types.add(parts[1])
            elif predicate == "no_gluten_content":
                self.gf_content_types.add(parts[1])
            elif predicate == "waiting":
                 # Store initial waiting children and their locations
                 child, place = parts[1], parts[2]
                 self.initial_waiting_children[child] = place

        # Identify the set of children who must be served (those initially waiting)
        self.children_to_serve = set(self.initial_waiting_children.keys())


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unserved children
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children = self.children_to_serve - served_children

        if not unserved_children:
            return 0

        # 2. Count needed sandwiches by type
        N_gf_needed = sum(1 for child in unserved_children if child in self.allergic_children)
        N_reg_needed = sum(1 for child in unserved_children if child in self.not_allergic_children)

        # 3. Count available sandwiches by type and stage

        # Identify locations where unserved children are waiting
        gf_needed_locations = {self.initial_waiting_children[child] for child in unserved_children if child in self.allergic_children}
        reg_needed_locations = {self.initial_waiting_children[child] for child in unserved_children if child in self.not_allergic_children}

        # Track trays and their locations
        tray_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3 and parts[1].startswith("tray"):
                     tray_locations[parts[1]] = parts[2]

        # Count sandwiches on trays by type and location stage
        GF_stage1 = 0 # GF on tray at GF needed location
        Reg_stage1 = 0 # Reg on tray at Reg needed location
        GF_ontray_total = 0 # Total GF on trays
        Reg_ontray_total = 0 # Total Reg on trays

        sandwiches_on_trays = set() # Keep track of sandwiches on trays

        for fact in state:
             if match(fact, "ontray", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) == 3 and parts[1].startswith("sandw") and parts[2].startswith("tray"):
                     sandwich = parts[1]
                     tray = parts[2]
                     sandwiches_on_trays.add(sandwich)

                     is_gf_sandwich = False
                     # Check if the corresponding (no_gluten_sandwich ?) fact exists in the state
                     if f"(no_gluten_sandwich {sandwich})" in state:
                          is_gf_sandwich = True

                     tray_loc = tray_locations.get(tray)

                     if tray_loc: # Only count if tray location is known
                         if is_gf_sandwich:
                             GF_ontray_total += 1
                             if tray_loc in gf_needed_locations:
                                 GF_stage1 += 1
                         else:
                             Reg_ontray_total += 1
                             if tray_loc in reg_needed_locations:
                                 Reg_stage1 += 1

        # Stage 2: On tray elsewhere = Total on tray - Stage 1
        GF_stage2 = GF_ontray_total - GF_stage1
        Reg_stage2 = Reg_ontray_total - Reg_stage1

        # Stage 3: In kitchen
        GF_stage3 = 0
        Reg_stage3 = 0
        for fact in state:
            if match(fact, "at_kitchen_sandwich", "*"):
                parts = get_parts(fact)
                if len(parts) == 2 and parts[1].startswith("sandw"):
                    sandwich = parts[1]
                    # Only count if it's not already counted as on a tray
                    if sandwich not in sandwiches_on_trays:
                        is_gf_sandwich = False
                        # Check if the corresponding (no_gluten_sandwich ?) fact exists in the state
                        if f"(no_gluten_sandwich {sandwich})" in state:
                             is_gf_sandwich = True

                        if is_gf_sandwich:
                            GF_stage3 += 1
                        else:
                            Reg_stage3 += 1

        # 4. Count available ingredients and notexist slots
        B_total = 0
        C_total = 0
        NE_total = 0
        B_gf = 0
        C_gf = 0

        for fact in state:
            if match(fact, "at_kitchen_bread", "*"):
                B_total += 1
                bread = get_parts(fact)[1]
                if bread in self.gf_bread_types:
                    B_gf += 1
            elif match(fact, "at_kitchen_content", "*"):
                C_total += 1
                content = get_parts(fact)[1]
                if content in self.gf_content_types:
                    C_gf += 1
            elif match(fact, "notexist", "*"):
                NE_total += 1

        # 5. Initialize cost
        TotalCost = 0

        # 6. Calculate remaining needs
        rem_gf_needed = N_gf_needed
        rem_reg_needed = N_reg_needed

        # 7. Use Stage 1 (cost 1: serve)
        use_gf_s1 = min(rem_gf_needed, GF_stage1)
        TotalCost += use_gf_s1 * 1
        rem_gf_needed -= use_gf_s1

        use_reg_s1 = min(rem_reg_needed, Reg_stage1)
        TotalCost += use_reg_s1 * 1
        rem_reg_needed -= use_reg_s1

        # 8. Use Stage 2 (cost 2: move + serve)
        use_gf_s2 = min(rem_gf_needed, GF_stage2)
        TotalCost += use_gf_s2 * 2
        rem_gf_needed -= use_gf_s2

        use_reg_s2 = min(rem_reg_needed, Reg_stage2)
        TotalCost += use_reg_s2 * 2
        rem_reg_needed -= use_reg_s2

        # 9. Use Stage 3 (cost 3: put + move + serve)
        use_gf_s3 = min(rem_gf_needed, GF_stage3)
        TotalCost += use_gf_s3 * 3
        rem_gf_needed -= use_gf_s3

        use_reg_s3 = min(rem_reg_needed, Reg_stage3)
        TotalCost += use_reg_s3 * 3
        rem_reg_needed -= use_reg_s3

        # 10. Use ingredients to make sandwiches (cost 4: make + put + move + serve)
        # Prioritize making GF sandwiches if needed
        can_make_gf = min(rem_gf_needed, B_gf, C_gf, NE_total)
        use_make_gf = can_make_gf
        TotalCost += use_make_gf * 4
        rem_gf_needed -= use_make_gf

        # Update available ingredients/notexist after making GF
        B_gf_used_for_make = use_make_gf
        C_gf_used_for_make = use_make_gf
        NE_remaining_after_gf = NE_total - use_make_gf

        B_reg_avail_for_make = B_total - B_gf_used_for_make
        C_reg_avail_for_make = C_total - C_gf_used_for_make
        NE_reg_avail_for_make = NE_remaining_after_gf # Use the remaining NE

        # Calculate how many Regular sandwiches can be made for the remaining need
        can_make_reg = min(rem_reg_needed, B_reg_avail_for_make, C_reg_avail_for_make, NE_reg_avail_for_make)
        use_make_reg = can_make_reg
        TotalCost += use_make_reg * 4
        rem_reg_needed -= use_make_reg

        # 11. Penalty for unmet needs
        Penalty = (rem_gf_needed + rem_reg_needed) * 100 # Large penalty

        return TotalCost + Penalty
