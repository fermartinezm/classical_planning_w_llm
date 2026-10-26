from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper function to split a PDDL fact string into predicate and arguments."""
    # Remove surrounding parentheses and split by spaces
    return fact[1:-1].split()

def match(fact, *args):
    """Helper function to check if a fact matches a pattern using fnmatch."""
    parts = get_parts(fact)
    # Check if the number of parts matches the number of args, and if each part matches the corresponding arg pattern
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    This heuristic estimates the cost to reach the goal (serving all children)
    by summing the estimated costs for delivering a suitable sandwich to each
    unserved child. The cost for delivering a single sandwich is estimated
    based on its current status:
    - Cost 1: If the sandwich is already on a tray at a waiting location (serve action).
    - Cost 2: If the sandwich is on a tray elsewhere (move tray + serve actions).
    - Cost 3: If the sandwich is in the kitchen (put on tray + move tray + serve actions).
    - Cost 4: If the sandwich needs to be made (make + put on tray + move tray + serve actions).
    The heuristic greedily satisfies the need for sandwiches (GF first, then regular)
    by using the cheapest available sandwiches first. Surplus GF sandwiches can
    be used to satisfy regular needs at the same cost level.

    Assumptions:
    - The problem is solvable from the initial state. If not enough ingredients
      or sandwich objects exist to make the required sandwiches, the heuristic
      returns a large value (1000000) indicating a likely dead end.
    - Trays are available for necessary put_on_tray and move_tray actions. The
      heuristic does not explicitly model tray availability or capacity beyond
      assuming that if a sandwich needs moving/putting, a tray can eventually
      be made available for it with the estimated cost.
    - Each sandwich serves exactly one child.

    Heuristic Initialization:
    The constructor extracts static information from the task:
    - Which children are allergic to gluten.
    - Which bread types are gluten-free.
    - Which content types are gluten-free.
    This information is stored in sets for efficient lookup during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Identify all children who have not yet been served.
    2.  Group the unserved children by their waiting location and allergy status
        (gluten-allergic or not). Count the total number of GF and regular
        sandwiches needed. Also, identify all distinct places where unserved
        children are waiting.
    3.  Identify and count the available sandwiches in the current state,
        categorizing them by type (GF/regular) and status/location:
        - On a tray at *any* waiting location (Cost 1 pool).
        - On a tray elsewhere (not at a waiting location) (Cost 2 pool).
        - In the kitchen (`at_kitchen_sandwich`) (Cost 3 pool).
        - Potential sandwiches that can be made from available ingredients
          (`at_kitchen_bread`, `at_kitchen_content`) and `notexist` sandwich objects
          in the kitchen (Cost 4 pool). Account for shared `notexist` objects.
    4.  Initialize the total heuristic cost to 0.
    5.  Satisfy the total GF sandwich need by drawing from the GF supply pools
        in increasing order of cost (Cost 1, 2, 3, 4). Add the corresponding cost
        to the total. Track remaining needed GF sandwiches and remaining supply
        in each GF pool. If need > supply, return 1000000.
    6.  Satisfy the total regular sandwich need by drawing from the available
        supply pools (both remaining GF and regular) in increasing order of cost
        (Cost 1, 2, 3, 4). Add the corresponding cost. Prioritize using the
        cheapest available sandwiches regardless of initial type (GF surplus
        can cover regular needs). If need > supply, return 1000000.
    7.  Return the total calculated cost.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # Extract static information
        self.allergic_children = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "allergic_gluten", "*")
        }
        self.no_gluten_bread_types = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_bread", "*")
        }
        self.no_gluten_content_types = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_content", "*")
        }

    def __call__(self, node):
        state = node.state

        # 1. Identify unserved children
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        waiting_children_locations = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "waiting", "*", "*")}

        unserved_children_count = 0
        total_needed_gf = 0
        total_needed_reg = 0
        waiting_places = set()

        for child, place in waiting_children_locations.items():
            if child not in served_children:
                unserved_children_count += 1
                waiting_places.add(place)
                if child in self.allergic_children:
                    total_needed_gf += 1
                else:
                    total_needed_reg += 1

        # 2. If no unserved children, goal is reached
        if unserved_children_count == 0:
            return 0

        # 3. Count available sandwiches by type and status/location
        sandwich_is_gf = {get_parts(fact)[1]: True for fact in state if match(fact, "no_gluten_sandwich", "*")}
        tray_location = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*")}

        # Available sandwiches categorized by cost pool
        pool_gf_1 = 0 # On tray at a waiting place (Cost 1)
        pool_reg_1 = 0 # On tray at a waiting place (Cost 1)
        pool_gf_2 = 0 # On tray elsewhere (Cost 2)
        pool_reg_2 = 0 # On tray elsewhere (Cost 2)
        pool_gf_3 = 0 # In kitchen (Cost 3)
        pool_reg_3 = 0 # In kitchen (Cost 3)

        available_gf_bread = 0
        available_gf_content = 0
        available_reg_bread = 0
        available_reg_content = 0
        available_notexist = 0

        # Collect sandwiches on trays first to determine their location pool
        sandwiches_on_trays = {} # {sandwich: tray}
        for fact in state:
             if match(fact, "ontray", "*", "*"):
                 s, t = get_parts(fact)[1:]
                 sandwiches_on_trays[s] = t

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]

            if predicate == "at_kitchen_bread":
                bread = parts[1]
                if bread in self.no_gluten_bread_types:
                    available_gf_bread += 1
                else:
                    available_reg_bread += 1
            elif predicate == "at_kitchen_content":
                content = parts[1]
                if content in self.no_gluten_content_types:
                    available_gf_content += 1
                else:
                    available_reg_content += 1
            elif predicate == "at_kitchen_sandwich":
                sandwich = parts[1]
                if sandwich_is_gf.get(sandwich, False):
                    pool_gf_3 += 1
                else:
                    pool_reg_3 += 1
            elif predicate == "ontray":
                # Processed below using sandwiches_on_trays
                pass
            elif predicate == "notexist":
                available_notexist += 1

        # Process sandwiches on trays based on their location
        for sandwich, tray in sandwiches_on_trays.items():
            p_tray = tray_location.get(tray)
            is_gf = sandwich_is_gf.get(sandwich, False)

            if p_tray in waiting_places:
                 if is_gf:
                     pool_gf_1 += 1
                 else:
                     pool_reg_1 += 1
            else:
                 if is_gf:
                     pool_gf_2 += 1
                 else:
                     pool_reg_2 += 1

        # Calculate sandwiches that can be made (Cost 4 pool)
        can_make_gf = min(available_gf_bread, available_gf_content, available_notexist)
        notexist_after_gf = max(0, available_notexist - can_make_gf)
        can_make_reg = min(available_reg_bread, available_reg_content, notexist_after_gf)

        pool_gf_4 = can_make_gf
        pool_reg_4 = can_make_reg

        # 5. Calculate cost by satisfying needs from cheapest pools first

        total_cost = 0
        needed_gf = total_needed_gf
        needed_reg = total_needed_reg

        # Satisfy GF needs
        use = min(needed_gf, pool_gf_1)
        total_cost += use * 1
        needed_gf -= use
        pool_gf_1 -= use # Consume from pool

        use = min(needed_gf, pool_gf_2)
        total_cost += use * 2
        needed_gf -= use
        pool_gf_2 -= use # Consume from pool

        use = min(needed_gf, pool_gf_3)
        total_cost += use * 3
        needed_gf -= use
        pool_gf_3 -= use # Consume from pool

        use = min(needed_gf, pool_gf_4)
        total_cost += use * 4
        needed_gf -= use
        pool_gf_4 -= use # Consume from pool

        # If still needed, problem is likely unsolvable
        if needed_gf > 0:
            return 1000000 # Large value for unsolvable

        # Satisfy Regular needs (can use remaining GF or Regular sandwiches)
        # Pool 1: Reg_1 + remaining GF_1
        use = min(needed_reg, pool_reg_1 + pool_gf_1)
        total_cost += use * 1
        needed_reg -= use
        # Consume from pools (prioritize Reg_1 then GF_1)
        consume_reg = min(use, pool_reg_1)
        pool_reg_1 -= consume_reg
        use -= consume_reg
        pool_gf_1 -= use

        # Pool 2: Reg_2 + remaining GF_2
        use = min(needed_reg, pool_reg_2 + pool_gf_2)
        total_cost += use * 2
        needed_reg -= use
        # Consume from pools (prioritize Reg_2 then GF_2)
        consume_reg = min(use, pool_reg_2)
        pool_reg_2 -= consume_reg
        use -= consume_reg
        pool_gf_2 -= use

        # Pool 3: Reg_3 + remaining GF_3
        use = min(needed_reg, pool_reg_3 + pool_gf_3)
        total_cost += use * 3
        needed_reg -= use
        # Consume from pools (prioritize Reg_3 then GF_3)
        consume_reg = min(use, pool_reg_3)
        pool_reg_3 -= consume_reg
        use -= consume_reg
        pool_gf_3 -= use

        # Pool 4: Reg_4 + remaining GF_4
        use = min(needed_reg, pool_reg_4 + pool_gf_4)
        total_cost += use * 4
        needed_reg -= use
        # Consume from pools (prioritize Reg_4 then GF_4)
        consume_reg = min(use, pool_reg_4)
        pool_reg_4 -= consume_reg
        use -= consume_reg
        pool_gf_4 -= use

        # If still needed, problem is likely unsolvable
        if needed_reg > 0:
            return 1000000 # Large value for unsolvable

        return total_cost
