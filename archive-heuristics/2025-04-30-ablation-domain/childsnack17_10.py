from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
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
    This heuristic estimates the number of actions required to serve all waiting children.
    It counts the number of unserved children (base cost for 'serve' actions) and adds
    an estimated cost to get a suitable sandwich on a tray to each child's table.
    The delivery cost is estimated based on the current state of available sandwiches
    (where they are, if they exist, if ingredients are available) and the availability
    of trays in the kitchen needed to prepare sandwiches for delivery.

    # Assumptions
    - The goal is to serve all children who are initially waiting.
    - Children may have gluten allergies, requiring gluten-free sandwiches.
    - Sandwiches are made from bread and content in the kitchen.
    - A sandwich is gluten-free if and only if both its bread and content are gluten-free.
    - Sandwiches must be on a tray to be moved to a table and served.
    - Trays are needed in the kitchen to put sandwiches onto them.
    - Trays become available again after a sandwich is served from them.
    - The primary bottleneck for delivery from the kitchen/creation is the number of trays in the kitchen.
    - The heuristic assumes solvable instances have enough total ingredients and sandwich objects defined.
    - Costs are simplified: make=1, put_on_tray=1, move_tray=1, serve=1.

    # Heuristic Initialization
    - Extracts static facts about child allergies (`allergic_gluten`, `not_allergic_gluten`).
    - Extracts static facts about ingredient types (`no_gluten_bread`, `no_gluten_content`).
    - Identifies all sandwich objects defined in the problem.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children who are currently waiting but not yet served.
    2. Categorize unserved children into those needing gluten-free sandwiches (allergic) and those who don't. Count `num_unserved_gf` and `num_unserved_non_gf`. The base heuristic cost is `num_unserved = num_unserved_gf + num_unserved_non_gf` (representing the 'serve' action for each).
    3. Count available ingredients in the kitchen, distinguishing gluten-free and gluten types.
    4. Count available trays in the kitchen.
    5. Categorize all sandwich objects based on their current state and gluten status:
       - Cost 0 delivery: On a tray at any table (`on_tray_at_table`).
       - Cost 1 delivery: On a tray in the kitchen (`on_tray_kitchen`).
       - Cost 2 delivery: In the kitchen, not on a tray (`in_kitchen`). Needs a tray in the kitchen.
       - Cost 3 delivery: Does not exist (`notexist`). Needs ingredients and a tray in the kitchen to be made and put on a tray.
       Count how many GF and Gluten sandwiches fall into each category (c0, c1, c2, c3).
    6. Calculate the number of GF and Gluten sandwiches that can be made from available ingredients and `notexist` objects (`gf_pool3_makeable`, `gluten_pool3_makeable`).
    7. Calculate the delivery cost (`delivery_cost`) and the number of 'put_on_tray' actions required (`trays_needed_for_put`) by greedily assigning sandwiches to satisfy needs:
       - First, satisfy `num_unserved_gf` needs using available GF sandwiches from the cheapest pools (c0, c1, c2, c3) in order. Track trays/ingredients used for c2/c3.
       - Second, satisfy `num_unserved_non_gf` needs using remaining GF or any available Gluten sandwiches from the cheapest pools (c0, c1, c2, c3) in order. Track trays/ingredients used for c2/c3.
    8. Calculate the tray penalty: `max(0, trays_needed_for_put - num_trays_kitchen)`. This estimates the delay caused by insufficient trays for simultaneous 'put_on_tray' actions.
    9. The total heuristic value is the sum of the base cost (`num_unserved`), the estimated delivery cost (`delivery_cost`), and the tray penalty (`tray_penalty`).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts and object information.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract static information
        self.allergic_children = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "allergic_gluten", "*")
        }
        self.not_allergic_children = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "not_allergic_gluten", "*")
        }
        self.gf_bread_types = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_bread", "*")
        }
        self.gf_content_types = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_content", "*")
        }
        # Store initial waiting locations for children (not strictly needed for this heuristic, but good practice)
        self.initial_waiting = {
             get_parts(fact)[1]: get_parts(fact)[2] for fact in static_facts if match(fact, "waiting", "*", "*")
        }

        # Identify all sandwich objects
        self.all_sandwiches = {
            obj for obj_type, obj_list in task.objects.items()
            for obj in obj_list if obj_type == 'sandwich'
        }

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unserved children and their allergy status
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children = {
            child for child in self.initial_waiting.keys() if child not in served_children
        }

        num_unserved_gf = len([
            child for child in unserved_children if child in self.allergic_children
        ])
        num_unserved_non_gf = len([
            child for child in unserved_children if child in self.not_allergic_children
        ])
        num_unserved = num_unserved_gf + num_unserved_non_gf

        # If no children are unserved, the goal is reached.
        if num_unserved == 0:
            return 0

        # Base heuristic cost is the number of serve actions needed
        h = num_unserved

        # 2. Count available ingredients in kitchen
        kitchen_bread = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_bread", "*")}
        kitchen_content = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_content", "*")}

        num_gf_bread_kitchen = len(kitchen_bread.intersection(self.gf_bread_types))
        num_gluten_bread_kitchen = len(kitchen_bread) - num_gf_bread_kitchen

        num_gf_content_kitchen = len(kitchen_content.intersection(self.gf_content_types))
        num_gluten_content_kitchen = len(kitchen_content) - num_gf_content_kitchen

        # 3. Count available trays in kitchen
        num_trays_kitchen = sum(1 for fact in state if match(fact, "at", "*", "kitchen") and "tray" in fact)

        # 4. Categorize sandwiches by state and gluten status
        sandwiches_on_tray = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "ontray", "*", "*")}
        trays_location = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*") and "tray" in fact}
        sandwiches_in_kitchen = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_sandwich", "*")}
        sandwiches_notexist = {get_parts(fact)[1] for fact in state if match(fact, "notexist", "*")}
        sandwiches_served_set = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")} # Sandwiches used to serve children

        # Determine gluten status of existing sandwiches (only possible if (no_gluten_sandwich S) is true)
        # We assume a sandwich is gluten unless (no_gluten_sandwich S) is explicitly true.
        gf_sandwiches_made = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}

        # Pool counts by delivery cost (0, 1, 2, 3) and gluten status
        # Cost 0: On tray at table
        # Cost 1: On tray in kitchen
        # Cost 2: In kitchen, not on tray
        # Cost 3: Does not exist (makeable) - handled via makeable counts below

        gf_c0, gf_c1, gf_c2 = 0, 0, 0
        gluten_c0, gluten_c1, gluten_c2 = 0, 0, 0

        for s in self.all_sandwiches:
            # A sandwich is considered available if it's not already served
            if s in sandwiches_served_set:
                continue

            is_gf_s = s in gf_sandwiches_made
            is_gluten_s = not is_gf_s # Assumption: if not GF, it's Gluten

            if s in sandwiches_on_tray:
                tray = sandwiches_on_tray[s]
                if tray in trays_location:
                    location = trays_location[tray]
                    if location != 'kitchen': # Assume any non-kitchen location is a table
                        # Cost 0: On tray at table
                        if is_gf_s: gf_c0 += 1
                        else: gluten_c0 += 1
                    else:
                        # Cost 1: On tray in kitchen
                        if is_gf_s: gf_c1 += 1
                        else: gluten_c1 += 1
            elif s in sandwiches_in_kitchen:
                # Cost 2: In kitchen, not on tray
                if is_gf_s: gf_c2 += 1
                else: gluten_c2 += 1
            # Sandwiches in sandwiches_notexist are handled by makeable counts

        # 6. Calculate makeable counts from notexist + ingredients
        num_notexist_objects = len(sandwiches_notexist)
        # We can make a GF sandwich if we have a notexist object AND GF ingredients
        gf_pool3_makeable = min(num_notexist_objects, num_gf_bread_kitchen, num_gf_content_kitchen)
        # We can make a Gluten sandwich if we have a notexist object AND Gluten ingredients
        gluten_pool3_makeable = min(num_notexist_objects, num_gluten_bread_kitchen, num_gluten_content_kitchen)
        # Note: A single notexist object can only be made into *one* sandwich (either GF or Gluten).
        # The min(num_notexist_objects, ...) correctly models this.

        # 7. Calculate delivery cost and trays needed for 'put_on_tray'
        needed_gf = num_unserved_gf
        needed_non_gf = num_unserved_non_gf
        delivery_cost = 0
        trays_needed_for_put = 0
        gf_makeable_used = 0
        gluten_makeable_used = 0

        # Satisfy GF needs first (must use GF sandwiches)
        # Use GF from c0 (cost 0)
        take = min(needed_gf, gf_c0)
        needed_gf -= take
        gf_c0 -= take # Consume from pool count

        # Use GF from c1 (cost 1)
        take = min(needed_gf, gf_c1)
        delivery_cost += take * 1
        needed_gf -= take
        gf_c1 -= take # Consume from pool count

        # Use GF from c2 (cost 2, needs tray)
        take = min(needed_gf, gf_c2)
        delivery_cost += take * 2
        needed_gf -= take
        trays_needed_for_put += take
        gf_c2 -= take # Consume from pool count

        # Use GF from c3 (cost 3, needs ingredients, needs tray)
        take = min(needed_gf, gf_pool3_makeable - gf_makeable_used) # Use from makeable count
        delivery_cost += take * 3
        needed_gf -= take
        trays_needed_for_put += take
        gf_makeable_used += take

        # Now satisfy non-GF needs (can use remaining GF or any Gluten)
        needed = needed_non_gf

        # Use remaining GF from c0 (cost 0)
        take = min(needed, gf_c0)
        needed -= take
        gf_c0 -= take

        # Use Gluten from c0 (cost 0)
        take = min(needed, gluten_c0)
        needed -= take
        gluten_c0 -= take

        # Use remaining GF from c1 (cost 1)
        take = min(needed, gf_c1)
        delivery_cost += take * 1
        needed -= take
        gf_c1 -= take

        # Use Gluten from c1 (cost 1)
        take = min(needed, gluten_c1)
        delivery_cost += take * 1
        needed -= take
        gluten_c1 -= take

        # Use remaining GF from c2 (cost 2, needs tray)
        take = min(needed, gf_c2)
        delivery_cost += take * 2
        needed -= take
        trays_needed_for_put += take
        gf_c2 -= take

        # Use Gluten from c2 (cost 2, needs tray)
        take = min(needed, gluten_c2)
        delivery_cost += take * 2
        needed -= take
        trays_needed_for_put += take
        gluten_c2 -= take

        # Use remaining GF from c3 (cost 3, needs ingredients, needs tray)
        take = min(needed, gf_pool3_makeable - gf_makeable_used)
        delivery_cost += take * 3
        needed -= take
        trays_needed_for_put += take
        gf_makeable_used += take

        # Use remaining Gluten from c3 (cost 3, needs ingredients, needs tray)
        take = min(needed, gluten_pool3_makeable - gluten_makeable_used)
        delivery_cost += take * 3
        needed -= take
        trays_needed_for_put += take
        gluten_makeable_used += take

        # 8. Calculate tray penalty
        # The number of 'put_on_tray' actions that can happen simultaneously is limited by trays in kitchen.
        # If we need more 'put_on_tray' actions than we have trays, each extra needed action
        # implies waiting for a tray or an alternative plan (which this heuristic doesn't model).
        # We penalize based on the deficit of trays vs. required 'put_on_tray' actions.
        tray_penalty = max(0, trays_needed_for_put - num_trays_kitchen)

        # 9. Total heuristic
        return h + delivery_cost + tray_penalty
