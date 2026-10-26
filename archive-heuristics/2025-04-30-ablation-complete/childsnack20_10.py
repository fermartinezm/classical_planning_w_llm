from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact string matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args) and '*' not in args:
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all unserved children.
    It counts the final 'serve' action for each unserved child and adds the estimated
    cost to get a suitable sandwich ready for serving. The cost is estimated based on
    the current "stage" of the sandwich: on a tray (cost 1 for movement), in the kitchen
    (cost 1 for put_on_tray + 1 for movement = 2), or needing to be made (cost 1 for make
    + 1 for put_on_tray + 1 for movement = 3). Gluten-free requirements are prioritized.

    # Assumptions
    - Each unserved child requires one suitable sandwich.
    - A gluten-free sandwich can satisfy both allergic and non-allergic children.
    - A regular sandwich can only satisfy non-allergic children.
    - The cost to get a sandwich ready for serving depends only on its current state
      (on tray anywhere, in kitchen, needs making), not its specific location on a tray
      or tray availability beyond the counts.
    - Ingredients and 'notexist' sandwich slots are sufficient to make all needed sandwiches
      up to the counts calculated.
    - Trays are available for 'put_on_tray' actions and for moving sandwiches.

    # Heuristic Initialization
    - Extracts static information: which children are allergic/not allergic,
      where each child is waiting, which bread/content items are gluten-free,
      and the total list of possible sandwich, bread, and content objects.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all unserved children and count them (`N_unserved`). This count is the base
       heuristic value (representing the final 'serve' action for each).
    2. Count the total number of unserved allergic children (`N_allergic_unserved`) and
       non-allergic children (`N_regular_unserved`).
    3. Count available sandwiches by type (GF/Regular) and "stage" in the current state:
       - Stage 2 (On tray, any loc): Count sandwiches `(ontray s t)`.
       - Stage 1 (In kitchen): Count sandwiches `(at_kitchen_sandwich s)`.
       - Stage 0 (Can be made): Count available ingredients `(at_kitchen_bread b)`,
         `(at_kitchen_content c)` and `(notexist s)` slots to determine the maximum
         number of GF and Regular sandwiches that can be made.
    4. Calculate the cost to source the required GF sandwiches (`N_allergic_unserved`).
       Greedily use available GF sandwiches from the highest stage (cheapest cost) down:
       - Use GF sandwiches on trays (Cost 1 per sandwich).
       - Use GF sandwiches in the kitchen (Cost 2 per sandwich).
       - Use makeable GF sandwiches (Cost 3 per sandwich).
       Update remaining needed GF sandwiches and available sandwich counts after each step.
    5. Calculate the cost to source the required Regular sandwiches (`N_regular_unserved`).
       Greedily use remaining available sandwiches (both GF and Regular) from the highest
       stage (cheapest cost) down:
       - Use remaining sandwiches on trays (Cost 1 per sandwich).
       - Use remaining sandwiches in the kitchen (Cost 2 per sandwich).
       - Use remaining makeable sandwiches (Cost 3 per sandwich).
       Add the corresponding costs to the heuristic.
    6. The total heuristic value is the sum of the base count (serve actions) and the costs
       calculated in steps 4 and 5.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract static information
        self.allergic_children = {get_parts(fact)[1] for fact in static_facts if match(fact, "allergic_gluten", "*")}
        self.waiting_places = {get_parts(fact)[1]: get_parts(fact)[2] for fact in static_facts if match(fact, "waiting", "*", "*")}
        self.no_gluten_bread_set = {get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_bread", "*")}
        self.no_gluten_content_set = {get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_content", "*")}

        # Extract all objects of relevant types (needed for counting notexist, bread, content)
        # Assuming objects are listed in the initial state or task facts
        # The Task object provides task.facts which seems to be (object, type) pairs
        self.all_sandwiches = {obj for obj, obj_type in task.facts if obj_type == 'sandwich'}
        self.all_bread = {obj for obj, obj_type in task.facts if obj_type == 'bread-portion'}
        self.all_content = {obj for obj, obj_type in task.facts if obj_type == 'content-portion'}

        # Identify all children from goals (assuming all children in goals need serving)
        self.all_children = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        cost = 0

        # 1. Identify unserved children
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children = self.all_children - served_children

        # Base cost: 1 action for each serve action
        cost += len(unserved_children)

        if not unserved_children:
            return 0 # Goal state

        # 2. Count unserved children by type
        num_allergic_unserved = sum(1 for child in unserved_children if child in self.allergic_children)
        num_regular_unserved = len(unserved_children) - num_allergic_unserved

        # 3. Count available sandwiches by type and stage
        num_ontray_gf = 0
        num_ontray_reg = 0
        num_kitchen_gf = 0
        num_kitchen_reg = 0
        avail_bread_gf = 0
        avail_content_gf = 0
        avail_bread_reg = 0
        avail_content_reg = 0
        avail_notexist = 0

        for fact in state:
            if match(fact, "ontray", "*", "*"):
                s = get_parts(fact)[1]
                # Check if the corresponding no_gluten_sandwich fact exists in the state
                if f"(no_gluten_sandwich {s})" in state:
                    num_ontray_gf += 1
                else:
                    num_ontray_reg += 1
            elif match(fact, "at_kitchen_sandwich", "*"):
                s = get_parts(fact)[1]
                 # Check if the corresponding no_gluten_sandwich fact exists in the state
                if f"(no_gluten_sandwich {s})" in state:
                    num_kitchen_gf += 1
                else:
                    num_kitchen_reg += 1
            elif match(fact, "at_kitchen_bread", "*"):
                b = get_parts(fact)[1]
                if b in self.no_gluten_bread_set:
                    avail_bread_gf += 1
                else:
                    avail_bread_reg += 1
            elif match(fact, "at_kitchen_content", "*"):
                c = get_parts(fact)[1]
                if c in self.no_gluten_content_set:
                    avail_content_gf += 1
                else:
                    avail_content_reg += 1
            elif match(fact, "notexist", "*"):
                 avail_notexist += 1

        # Calculate makeable sandwiches
        can_make_gf = min(avail_bread_gf, avail_content_gf, avail_notexist)
        notexist_rem = avail_notexist - can_make_gf
        bread_gf_rem = avail_bread_gf - can_make_gf
        content_gf_rem = avail_content_gf - can_make_gf
        can_make_reg = min(avail_bread_reg + bread_gf_rem, avail_content_reg + content_gf_rem, notexist_rem)


        # 4. Source GF needs (cost 1, 2, 3)
        needed_gf = num_allergic_unserved

        # Cost 1: On tray
        use = min(needed_gf, num_ontray_gf)
        cost += use * 1
        needed_gf -= use
        num_ontray_gf -= use # Update remaining available GF on tray

        # Cost 2: In kitchen
        use = min(needed_gf, num_kitchen_gf)
        cost += use * 2
        needed_gf -= use
        num_kitchen_gf -= use # Update remaining available GF in kitchen

        # Cost 3: Can make
        use = min(needed_gf, can_make_gf)
        cost += use * 3
        needed_gf -= use
        can_make_gf -= use # Update remaining makeable GF

        # 5. Source Reg needs (cost 1, 2, 3) - can use remaining GF or Reg
        needed_reg = num_regular_unserved

        # Cost 1: On tray (remaining GF and Reg)
        avail_c1 = num_ontray_gf + num_ontray_reg # Use updated num_ontray_gf
        use = min(needed_reg, avail_c1)
        cost += use * 1
        needed_reg -= use
        # No need to update avail_c1 counts further as they are consumed here

        # Cost 2: In kitchen (remaining GF and Reg)
        avail_c2 = num_kitchen_gf + num_kitchen_reg # Use updated num_kitchen_gf
        use = min(needed_reg, avail_c2)
        cost += use * 2
        needed_reg -= use
        # No need to update avail_c2 counts further

        # Cost 3: Can make (remaining GF and Reg)
        avail_c3 = can_make_gf + can_make_reg # Use updated can_make_gf
        use = min(needed_reg, avail_c3)
        cost += use * 3
        needed_reg -= use
        # No need to update avail_c3 counts further

        return cost
