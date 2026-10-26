from fnmatch import fnmatch
from collections import defaultdict
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # The number of parts must match the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the total number of actions required to serve all
    waiting children. It calculates the number of children needing gluten-free
    and regular sandwiches, and then counts how many of these needs can be met
    by sandwiches available in different stages of preparation (already on tray
    at the child's location, on tray elsewhere, at kitchen, or needing to be made).
    It assigns costs based on the most expensive stage required for each needed
    sandwich, prioritizing the use of sandwiches that are closer to being served.

    # Assumptions
    - Each unserved child requires exactly one sandwich of the correct type (gluten-free for allergic, any for non-allergic).
    - Sandwiches must be on a tray at the child's waiting location to be served.
    - Trays can be moved between any two places.
    - Ingredients and 'notexist' sandwich objects are consumed when making sandwiches.
    - The problem is solvable, meaning sufficient resources (ingredients, notexist objects, trays) exist in total to make all necessary sandwiches and move them.
    - Action costs are uniform (implicitly 1). The heuristic sums the number of required actions per sandwich based on its starting stage.

    # Heuristic Initialization
    - Extracts static information about child allergies and waiting places.
    - Extracts static information about gluten-free ingredients (bread and content).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all unserved children by checking which children in the static `waiting` list are not in the current state's `served` facts.
    2. For each unserved child, determine their sandwich need: gluten-free if allergic, regular otherwise. Also, note their waiting place.
    3. Count the total number of gluten-free and regular sandwiches required across all unserved children.
    4. Count the number of available gluten-free and regular sandwiches currently in different "preparation stages" based on the current state:
       - Stage 1 (Cost 1: Serve): On a tray *at* a location where a child needing that type of sandwich is waiting. Count how many *needs* can be met by sandwiches already at the correct location.
       - Stage 2 (Cost 2: Move + Serve): On a tray *elsewhere* (sandwiches on trays not used to meet needs at their current location).
       - Stage 3 (Cost 3: Put + Move + Serve): `at_kitchen_sandwich`.
       - Stage 4 (Cost 4: Make + Put + Move + Serve): Can be *made* (limited by available ingredients at kitchen and 'notexist' sandwich objects). Calculate the maximum number of GF and regular sandwiches that can be made.
    5. Greedily satisfy the total sandwich needs (calculated in step 3) using the available sandwiches from the cheapest stages first (Stage 1, then 2, then 3, then 4).
       - Assign as many needs as possible to Stage 1 sandwiches.
       - Assign remaining needs to Stage 2 sandwiches.
       - Assign further remaining needs to Stage 3 sandwiches.
       - Assign the final remaining needs to Stage 4 (makeable) sandwiches, respecting ingredient and 'notexist' limits, prioritizing GF if needed.
    6. Sum the costs based on how many needed sandwiches were sourced from each stage. The total heuristic value is the sum of (number of sandwiches sourced from Stage 1) * 1 + (number from Stage 2) * 2 + (number from Stage 3) * 3 + (number from Stage 4) * 4.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts.
        """
        # self.goals = task.goals # Not strictly needed for this heuristic, but available
        static_facts = task.static

        # Map children to their allergy status
        self.allergic_children = {get_parts(fact)[1] for fact in static_facts if match(fact, "allergic_gluten", "*")}
        # self.not_allergic_children = {get_parts(fact)[1] for fact in static_facts if match(fact, "not_allergic_gluten", "*")} # Not strictly needed

        # Map children to their waiting places
        self.waiting_places = {get_parts(fact)[1]: get_parts(fact)[2] for fact in static_facts if match(fact, "waiting", "*", "*")}

        # Identify static gluten-free ingredients (needed for make action)
        self.static_gf_bread = {get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_bread", "*")}
        self.static_gf_content = {get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_content", "*")}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unserved children and their needs
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children = {c for c in self.waiting_places if c not in served_children}

        if not unserved_children:
            return 0 # Goal state reached

        # Count needed sandwiches by type and location
        needs_at_loc = defaultdict(lambda: {'gf': 0, 'reg': 0})
        total_needed_gf = 0
        total_needed_reg = 0
        for child in unserved_children:
            p = self.waiting_places[child]
            is_allergic = child in self.allergic_children
            needed_type = 'gf' if is_allergic else 'reg'
            needs_at_loc[p][needed_type] += 1
            if needed_type == 'gf':
                total_needed_gf += 1
            else:
                total_needed_reg += 1

        # 2. & 3. Count available sandwiches by type and stage

        # Identify sandwiches by type
        gf_sandwiches_in_state = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}

        # Identify sandwich and tray locations
        sandwiches_on_trays = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, 'ontray', '*', '*')}
        # Assuming 'at' predicate is only for trays and kitchen constant.
        tray_locations = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, 'at', '*', '*') and 'tray' in get_parts(fact)[0]}


        # Count sandwiches on trays at each location
        sandwiches_on_tray_at_loc = defaultdict(lambda: {'gf': 0, 'reg': 0})
        for s, t in sandwiches_on_trays.items():
            if t in tray_locations:
                p = tray_locations[t]
                s_is_gf = s in gf_sandwiches_in_state
                sandwich_type = 'gf' if s_is_gf else 'reg'
                sandwiches_on_tray_at_loc[p][sandwich_type] += 1

        # Count how many needs can be met by sandwiches already at the location (Stage 1)
        met_at_loc_gf = 0
        met_at_loc_reg = 0
        available_on_tray_elsewhere_gf_count = 0
        available_on_tray_elsewhere_reg_count = 0

        all_locations_with_trays = set(tray_locations.values())
        all_locations_with_needs = set(needs_at_loc.keys())
        relevant_locations = all_locations_with_trays.union(all_locations_with_needs)

        for p in relevant_locations:
            num_gf_needed_here = needs_at_loc[p]['gf']
            num_reg_needed_here = needs_at_loc[p]['reg']
            num_gf_available_here = sandwiches_on_tray_at_loc[p]['gf']
            num_reg_available_here = sandwiches_on_tray_at_loc[p]['reg']

            num_gf_met_here = min(num_gf_needed_here, num_gf_available_here)
            num_reg_met_here = min(num_reg_needed_here, num_reg_available_here)

            met_at_loc_gf += num_gf_met_here
            met_at_loc_reg += num_reg_met_here

            # Remaining sandwiches on trays at this location are available to be moved elsewhere (contribute to Stage 2)
            available_on_tray_elsewhere_gf_count += num_gf_available_here - num_gf_met_here
            available_on_tray_elsewhere_reg_count += num_reg_available_here - num_reg_met_here


        # Count sandwiches at kitchen (Contribute to Stage 3)
        kitchen_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, 'at_kitchen_sandwich', '*')}
        kitchen_gf_count = len(kitchen_sandwiches.intersection(gf_sandwiches_in_state))
        kitchen_reg_count = len(kitchen_sandwiches) - kitchen_gf_count

        # Count sandwiches that can be made (Contribute to Stage 4)
        notexist_count = len({get_parts(fact)[1] for fact in state if match(fact, 'notexist', '*')})
        kitchen_bread = {get_parts(fact)[1] for fact in state if match(fact, 'at_kitchen_bread', '*')}
        kitchen_content = {get_parts(fact)[1] for fact in state if match(fact, 'at_kitchen_content', '*')}

        gf_bread_kitchen_count = len(kitchen_bread.intersection(self.static_gf_bread))
        gf_content_kitchen_count = len(kitchen_content.intersection(self.static_gf_content))
        reg_bread_kitchen_count = len(kitchen_bread) - gf_bread_kitchen_count # Any bread not GF can be used for Reg
        reg_content_kitchen_count = len(kitchen_content) - gf_content_kitchen_count # Any content not GF can be used for Reg

        # Max number of GF sandwiches we can make (limited by notexist, GF bread, GF content)
        gf_can_make_count = min(notexist_count, gf_bread_kitchen_count, gf_content_kitchen_count)
        # Max number of Reg sandwiches we can make using remaining notexist objects and regular ingredients
        reg_can_make_count = min(notexist_count - gf_can_make_count, reg_bread_kitchen_count, reg_content_kitchen_count)


        # 4. & 5. Assign available sandwiches (by stage) to needs and sum costs
        h = 0

        # Needs met by sandwiches already at location (Cost 1: Serve)
        used_at_loc_gf = min(total_needed_gf, met_at_loc_gf)
        used_at_loc_reg = min(total_needed_reg, met_at_loc_reg)
        h += (used_at_loc_gf + used_at_loc_reg) * 1

        rem_needed_gf = total_needed_gf - used_at_loc_gf
        rem_needed_reg = total_needed_reg - used_at_loc_reg

        # Needs met by sandwiches on trays elsewhere (Cost 2: Move + Serve)
        used_ontray_elsewhere_gf = min(rem_needed_gf, available_on_tray_elsewhere_gf_count)
        rem_needed_gf -= used_ontray_elsewhere_gf
        used_ontray_elsewhere_reg = min(rem_needed_reg, available_on_tray_elsewhere_reg_count)
        rem_needed_reg -= used_ontray_elsewhere_reg
        h += (used_ontray_elsewhere_gf + used_ontray_elsewhere_reg) * 2

        # Needs met by sandwiches at kitchen (Cost 3: Put + Move + Serve)
        used_kitchen_gf = min(rem_needed_gf, kitchen_gf_count)
        rem_needed_gf -= used_kitchen_gf
        used_kitchen_reg = min(rem_needed_reg, kitchen_reg_count)
        rem_needed_reg -= used_kitchen_reg
        h += (used_kitchen_gf + used_kitchen_reg) * 3

        # Needs met by making new sandwiches (Cost 4: Make + Put + Move + Serve)
        # Prioritize making GF if needed, as they are more restrictive and use specific ingredients
        can_make_now_gf = min(rem_needed_gf, gf_can_make_count)
        rem_needed_gf -= can_make_now_gf

        # Update available notexist objects after potentially making GF
        notexist_after_gf = notexist_count - can_make_now_gf

        # Max number of Reg sandwiches we can make using remaining notexist objects and original reg ingredients
        reg_can_make_now = min(rem_needed_reg, notexist_after_gf, reg_bread_kitchen_count, reg_content_kitchen_count)
        rem_needed_reg -= reg_can_make_now

        used_make_gf = can_make_now_gf
        used_make_reg = reg_can_make_now

        h += (used_make_gf + used_make_reg) * 4

        # If after exhausting all available sandwiches and makeable sandwiches,
        # there are still remaining needs, it implies the problem is unsolvable
        # from this state (e.g., not enough total ingredients/notexist objects).
        # For solvable instances, rem_needed_gf and rem_needed_reg should be 0 here.
        # If rem_needed_gf > 0 or rem_needed_reg > 0:
        #     # This state is likely unsolvable or requires resources not accounted for
        #     # (e.g., ingredients not in kitchen). Return a large value.
        #     return float('inf') # Or a large constant

        return h
