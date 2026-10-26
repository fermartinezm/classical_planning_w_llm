from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting children.
    It calculates the cost for each unserved child independently, based on the state
    and type (gluten-free or regular) of the most readily available suitable sandwich.
    The total heuristic is the sum of these individual costs.

    # Assumptions
    - There are always enough ingredients (bread and content) in the kitchen to make any required sandwich.
    - There are always enough 'notexist' sandwich objects to create new sandwiches if needed.
    - Tray capacity and the number of available trays are not bottlenecks (i.e., we don't explicitly model waiting for a tray or limited items per tray trip).
    - The robot is implicitly located with the tray it is using.
    - The cost of actions is uniform (1 per action). The heuristic uses derived costs for sandwich states.

    # Heuristic Initialization
    - Extracts the allergy status (allergic_gluten or not_allergic_gluten) for each child from static facts.
    - Identifies all children present in the problem from the goal conditions or static facts.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic estimates the cost for each unserved child by finding the minimum actions
    needed to get a suitable sandwich to them, considering the sandwich's current state.
    The cost is calculated as follows:

    1.  **Identify Unserved Children**: Determine which children are waiting and not yet served. Count how many are allergic to gluten and how many are not.
    2.  **Categorize Sandwiches**: Identify all sandwich objects in the state and classify them based on:
        -   **State**:
            -   State 0: `(notexist ?sandw)` (needs making)
            -   State 1: `(at_kitchen_sandwich ?sandw)` (made, in kitchen)
            -   State 2: `(ontray ?sandw ?tray)` where `(at ?tray kitchen)` (on tray, in kitchen)
            -   State 3: `(ontray ?sandw ?tray)` where `(at ?tray ?table)` and `?table` is not the kitchen (on tray, at a table)
        -   **Type**: `(no_gluten_sandwich ?sandw)` is true (GF) or false (Regular).
    3.  **Greedy Sandwich Assignment and Costing**: Iterate through the unserved children. Prioritize allergic children. For each child, find the "most ready" suitable sandwich (GF for allergic, any for regular) from the available pools (State 3, then State 2, then State 1, then State 0).
        -   Assigning a State 3 sandwich costs 1 action (Serve).
        -   Assigning a State 2 sandwich costs 2 actions (Move tray + Serve).
        -   Assigning a State 1 sandwich costs 3 actions (Pick sandwich + Move tray + Serve).
        -   Assigning a State 0 sandwich (needs making) costs 6 actions (Pick bread + Pick content + Make sandwich + Pick sandwich + Move tray + Serve).
        Decrement the counts of available sandwiches/objects as they are "used" by the heuristic for a child.
    4.  **Sum Costs**: The total heuristic value is the sum of the costs calculated for each unserved child.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting child allergy information."""
        self.goals = task.goals
        self.static = task.static

        # Extract static info about child allergies
        self.child_allergies = {} # child -> is_allergic_gluten (bool)
        for fact in self.static:
            parts = get_parts(fact)
            if len(parts) == 2 and parts[0] == "allergic_gluten":
                self.child_allergies[parts[1]] = True
            elif len(parts) == 2 and parts[0] == "not_allergic_gluten":
                self.child_allergies[parts[1]] = False

        # Identify all children in the problem (from goals or static facts)
        self.all_children = set()
        for goal in self.goals:
             if match(goal, "served", "*"):
                 self.all_children.add(get_parts(goal)[1])
        # Also add children from static allergy facts if not already included
        self.all_children.update(self.child_allergies.keys())


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unserved children and their needs
        unserved_children = set()
        unserved_allergic_count = 0
        unserved_regular_count = 0

        for child in self.all_children:
            if f"(served {child})" not in state:
                unserved_children.add(child)
                # Default to not allergic if allergy status is not explicitly stated
                if self.child_allergies.get(child, False):
                    unserved_allergic_count += 1
                else:
                    unserved_regular_count += 1

        # If no children are unserved, goal is reached, heuristic is 0.
        if not unserved_children:
            return 0

        # 2. Categorize sandwiches by state and type
        sandwiches_s0 = [] # notexist
        sandwiches_s1_gf = [] # at_kitchen_sandwich, GF
        sandwiches_s1_reg = [] # at_kitchen_sandwich, Reg
        sandwiches_s2_gf = [] # ontray at kitchen, GF
        sandwiches_s2_reg = [] # ontray at kitchen, Reg
        sandwiches_s3_gf = [] # ontray at table, GF
        sandwiches_s3_reg = [] # ontray at table, Reg

        all_sandwich_objects = set()
        tray_locations = {} # tray -> place
        ontray_map = {} # sandwich -> tray

        # Build necessary maps from state facts
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 2 and parts[0] == "notexist":
                all_sandwich_objects.add(parts[1])
            elif len(parts) == 2 and parts[0] == "at_kitchen_sandwich":
                 all_sandwich_objects.add(parts[1])
            elif len(parts) == 3 and parts[0] == "ontray":
                 all_sandwich_objects.add(parts[1])
                 ontray_map[parts[1]] = parts[2]
            elif len(parts) == 3 and parts[0] == "at" and parts[1].startswith("tray"): # Assuming tray objects start with "tray"
                 tray_locations[parts[1]] = parts[2]

        # Now categorize the sandwiches
        for sandw in all_sandwich_objects:
            is_gf = f"(no_gluten_sandwich {sandw})" in state

            if f"(notexist {sandw})" in state:
                sandwiches_s0.append(sandw)
            elif f"(at_kitchen_sandwich {sandw})" in state:
                if is_gf:
                    sandwiches_s1_gf.append(sandw)
                else:
                    sandwiches_s1_reg.append(sandw)
            else: # Must be ontray somewhere
                current_tray = ontray_map.get(sandw)

                if current_tray and current_tray in tray_locations:
                    tray_loc = tray_locations[current_tray]
                    if tray_loc == "kitchen":
                        if is_gf:
                            sandwiches_s2_gf.append(sandw)
                        else:
                            sandwiches_s2_reg.append(sandw)
                    else: # Must be at a table
                         if is_gf:
                            sandwiches_s3_gf.append(sandw)
                         else:
                            sandwiches_s3_reg.append(sandw)
                # else: sandwich is on a tray whose location is unknown or tray doesn't exist? Ignore for heuristic.


        # 3. Calculate heuristic cost using greedy assignment
        cost = 0

        # Costs based on sandwich state:
        # S3 (ontray at table): 1 (Serve)
        # S2 (ontray at kitchen): 2 (Move tray + Serve)
        # S1 (at_kitchen_sandwich): 3 (Pick + Move tray + Serve)
        # S0 (notexist): 6 (Pick B + Pick Co + Make + Pick + Move tray + Serve)

        # Prioritize serving allergic children with GF sandwiches
        served_gf_s3 = min(unserved_allergic_count, len(sandwiches_s3_gf))
        cost += served_gf_s3 * 1
        unserved_allergic_count -= served_gf_s3

        served_gf_s2 = min(unserved_allergic_count, len(sandwiches_s2_gf))
        cost += served_gf_s2 * 2
        unserved_allergic_count -= served_gf_s2

        served_gf_s1 = min(unserved_allergic_count, len(sandwiches_s1_gf))
        cost += served_gf_s1 * 3
        unserved_allergic_count -= served_gf_s1

        # Need to make remaining GF sandwiches from S0 objects
        can_make_gf_s0 = min(unserved_allergic_count, len(sandwiches_s0))
        cost += can_make_gf_s0 * 6
        unserved_allergic_count -= can_make_gf_s0
        sandwiches_s0_used_for_gf = can_make_gf_s0 # Track used S0 objects

        # Prioritize serving regular children with remaining sandwiches (Reg or GF)
        remaining_s3 = (len(sandwiches_s3_gf) - served_gf_s3) + len(sandwiches_s3_reg)
        served_reg_s3 = min(unserved_regular_count, remaining_s3)
        cost += served_reg_s3 * 1
        unserved_regular_count -= served_reg_s3

        remaining_s2 = (len(sandwiches_s2_gf) - served_gf_s2) + len(sandwiches_s2_reg)
        served_reg_s2 = min(unserved_regular_count, remaining_s2)
        cost += served_reg_s2 * 2
        unserved_regular_count -= served_reg_s2

        remaining_s1 = (len(sandwiches_s1_gf) - served_gf_s1) + len(sandwiches_s1_reg)
        served_reg_s1 = min(unserved_regular_count, remaining_s1)
        cost += served_reg_s1 * 3
        unserved_regular_count -= served_reg_s1

        # Need to make remaining regular sandwiches from S0 objects
        remaining_s0_objects = len(sandwiches_s0) - sandwiches_s0_used_for_gf
        can_make_reg_s0 = min(unserved_regular_count, remaining_s0_objects)
        cost += can_make_reg_s0 * 6
        unserved_regular_count -= can_make_reg_s0

        # If unserved_allergic_count > 0 or unserved_regular_count > 0 here,
        # it means there weren't enough sandwich objects in the problem.
        # Assuming solvable problems have enough objects, this shouldn't happen.
        # If it could happen and we needed to signal unsolvability, returning float('inf')
        # would be an option, but the problem implies solvable instances.

        return cost
