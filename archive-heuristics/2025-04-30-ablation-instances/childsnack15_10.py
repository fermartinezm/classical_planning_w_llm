from fnmatch import fnmatch
# Assuming heuristic_base is available in the environment path
from heuristics.heuristic_base import Heuristic

# Helper functions
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
    # Ensure the number of parts in the fact matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# The heuristic class
class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children.
    It counts the number of unserved children and estimates the cost to get a
    suitable sandwich (considering gluten allergies) to each child's location
    based on the current state of available sandwiches and ingredients. The
    heuristic prioritizes using sandwiches that are closer to being served.

    # Assumptions
    - The goal is to serve all children.
    - Each child requires exactly one sandwich.
    - Gluten-allergic children require gluten-free sandwiches.
    - Non-allergic children can eat any sandwich.
    - Sandwiches can be made from bread and content in the kitchen.
    - Sandwiches must be on a tray to be moved and served.
    - Trays can be moved between the kitchen and child waiting places.
    - The costs for actions are simplified to 1 per logical step (make, put, move, serve).
    - The heuristic assumes a greedy strategy of using the most "ready" sandwiches first.
    - It simplifies tray handling for phases 3 and 4 by only considering trays starting in the kitchen pool for these steps.

    # Heuristic Initialization
    - Identify all children and their allergy status (allergic or not) from static facts.
    - Identify the waiting place for each child from static facts.
    - Identify all gluten-free bread and content types from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children and their properties (allergy, waiting place) from static facts.
    2. Identify which children are already served in the current state.
    3. Count the number of unserved children, separating them into allergic and non-allergic. If none, heuristic is 0.
    4. Count available sandwiches and resources in the current state, categorized by:
       - Gluten status (GF or Regular).
       - Location/State:
         - On tray at a child's waiting location (Cost 1: serve).
         - On tray at the kitchen or another location (Cost 2: move_tray, serve).
         - In the kitchen, but not on a tray (Cost 3: put_on_tray, move_tray, serve; requires tray at kitchen).
         - Not yet existing (`notexist`) (Cost 4: make, put_on_tray, move_tray, serve; requires ingredients, notexist sandwich object, tray at kitchen).
       - Available bread and content in the kitchen (GF and Regular).
       - Available trays in the kitchen (needed for phases 3 and 4).
    5. Calculate the heuristic cost by greedily "serving" the unserved children using the most readily available suitable sandwiches first, based on estimated minimum actions. Resource counts (sandwiches, ingredients, trays) are decremented as they are used in the calculation pipeline.
       - Prioritize serving allergic children with GF sandwiches.
       - Prioritize using sandwiches that require fewer actions (Cost 1 < Cost 2 < Cost 3 < Cost 4).
       - When making regular sandwiches (Cost 4), prioritize using regular ingredients before GF ingredients.
    6. Sum the costs for all children served in this greedy fashion. The total cost is the heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about children,
        allergies, waiting places, and gluten-free ingredients.
        """
        self.goals = task.goals
        static_facts = task.static

        self.children = set()
        self.allergic_children = set()
        self.non_allergic_children = set()
        self.child_waiting_place = {} # Map child -> place

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'allergic_gluten':
                child = parts[1]
                self.children.add(child)
                self.allergic_children.add(child)
            elif parts[0] == 'not_allergic_gluten':
                child = parts[1]
                self.children.add(child)
                self.non_allergic_children.add(child)
            elif parts[0] == 'waiting':
                child, place = parts[1], parts[2]
                self.child_waiting_place[child] = place

        self.gf_bread_types = {get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_bread", "*")}
        self.gf_content_types = {get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_content", "*")}

        # Get the set of all child waiting places
        self.child_places = set(self.child_waiting_place.values())


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify served children
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        # 2. Count unserved children
        unserved_children = self.children - served_children
        current_needed_gf = len(unserved_children.intersection(self.allergic_children))
        current_needed_reg = len(unserved_children.intersection(self.non_allergic_children))

        # If all children are served, the heuristic is 0.
        if current_needed_gf == 0 and current_needed_reg == 0:
            return 0

        # 3. Count available sandwiches and resources
        # Track GF status of existing sandwiches
        gf_sandwiches_in_state = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}

        # Map trays to locations
        tray_locations = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*")}

        # Track which sandwiches are on trays
        sandwiches_on_trays_set = {get_parts(fact)[1] for fact in state if match(fact, "ontray", "*", "*")}

        # Count sandwiches by state/location and type (GF/Reg)
        avail_s_ontray_at_child_loc_gf = 0 # on tray at child's loc, GF
        avail_s_ontray_at_child_loc_reg = 0 # on tray at child's loc, Reg
        avail_s_ontray_at_kitchen_gf = 0 # on tray at kitchen, GF
        avail_s_ontray_at_kitchen_reg = 0 # on tray at kitchen, Reg
        avail_s_ontray_at_other_gf = 0 # on tray elsewhere (not kitchen, not child loc), GF
        avail_s_ontray_at_other_reg = 0 # on tray elsewhere (not kitchen, not child loc), Reg
        avail_s_kitchen_gf = 0 # in kitchen, not on tray, GF
        avail_s_kitchen_reg = 0 # in kitchen, not on tray, Reg

        # Count sandwiches on trays based on tray location
        for fact in state:
             if match(fact, "ontray", "*", "*"):
                 s, t = get_parts(fact)[1], get_parts(fact)[2]
                 is_gf = s in gf_sandwiches_in_state
                 tray_loc = tray_locations.get(t)

                 if tray_loc == 'kitchen':
                     if is_gf:
                         avail_s_ontray_at_kitchen_gf += 1
                     else:
                         avail_s_ontray_at_kitchen_reg += 1
                 elif tray_loc in self.child_places:
                      if is_gf:
                          avail_s_ontray_at_child_loc_gf += 1
                      else:
                          avail_s_ontray_at_child_loc_reg += 1
                 elif tray_loc is not None: # Tray is somewhere else
                      if is_gf:
                          avail_s_ontray_at_other_gf += 1
                      else:
                          avail_s_ontray_at_other_reg += 1
                 # else: tray_loc is None, ignore this sandwich/tray state

        # Count sandwiches in kitchen (not on trays)
        kitchen_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_sandwich", "*")}
        for s in kitchen_sandwiches:
             if s not in sandwiches_on_trays_set: # Only count if not already on a tray
                 is_gf = s in gf_sandwiches_in_state
                 if is_gf:
                     avail_s_kitchen_gf += 1
                 else:
                     avail_s_kitchen_reg += 1

        # Count ingredients and notexist sandwiches
        avail_s_notexist = sum(1 for fact in state if match(fact, "notexist", "*"))
        avail_bread_gf = sum(1 for fact in state if match(fact, "at_kitchen_bread", "*") and get_parts(fact)[1] in self.gf_bread_types)
        avail_bread_reg = sum(1 for fact in state if match(fact, "at_kitchen_bread", "*") and get_parts(fact)[1] not in self.gf_bread_types)
        avail_content_gf = sum(1 for fact in state if match(fact, "at_kitchen_content", "*") and get_parts(fact)[1] in self.gf_content_types)
        avail_content_reg = sum(1 for fact in state if match(fact, "at_kitchen_content", "*") and get_parts(fact)[1] not in self.gf_content_types)

        # Count trays in kitchen (needed for phases 3 and 4)
        avail_trays_kitchen = sum(1 for fact in state if match(fact, "at", "*", "kitchen"))

        # --- Heuristic Calculation ---
        cost = 0

        # Phase 1: Sandwiches on tray at child's location (Cost 1: serve)
        # Prioritize GF for allergic, then Reg for non-allergic, then GF for non-allergic
        num_to_serve = min(current_needed_gf, avail_s_ontray_at_child_loc_gf)
        cost += num_to_serve * 1
        current_needed_gf -= num_to_serve
        avail_s_ontray_at_child_loc_gf -= num_to_serve

        num_to_serve = min(current_needed_reg, avail_s_ontray_at_child_loc_reg)
        cost += num_to_serve * 1
        current_needed_reg -= num_to_serve
        avail_s_ontray_at_child_loc_reg -= num_to_serve

        num_to_serve = min(current_needed_reg, avail_s_ontray_at_child_loc_gf) # Use remaining GF for regular if needed
        cost += num_to_serve * 1
        current_needed_reg -= num_to_serve
        avail_s_ontray_at_child_loc_gf -= num_to_serve


        # Phase 2: Sandwiches on tray at kitchen or elsewhere (Cost 2: move + serve)
        # Combine counts from kitchen and other locations
        avail_s_ontray_elsewhere_gf = avail_s_ontray_at_kitchen_gf + avail_s_ontray_at_other_gf
        avail_s_ontray_elsewhere_reg = avail_s_ontray_at_kitchen_reg + avail_s_ontray_at_other_reg

        num_to_serve = min(current_needed_gf, avail_s_ontray_elsewhere_gf)
        cost += num_to_serve * 2
        current_needed_gf -= num_to_serve
        avail_s_ontray_elsewhere_gf -= num_to_serve

        num_to_serve = min(current_needed_reg, avail_s_ontray_elsewhere_reg)
        cost += num_to_serve * 2
        current_needed_reg -= num_to_serve
        avail_s_ontray_elsewhere_reg -= num_to_serve

        num_to_serve = min(current_needed_reg, avail_s_ontray_elsewhere_gf) # Use remaining GF for regular
        cost += num_to_serve * 2
        current_needed_reg -= num_to_serve
        avail_s_ontray_elsewhere_gf -= num_to_serve


        # Phase 3: Sandwiches in the kitchen (not on trays) (Cost 3: put + move + serve)
        # Requires a tray at the kitchen.
        # Serve GF first
        num_to_serve_gf = min(current_needed_gf, avail_s_kitchen_gf, avail_trays_kitchen)
        cost += num_to_serve_gf * 3
        current_needed_gf -= num_to_serve_gf
        avail_s_kitchen_gf -= num_to_serve_gf
        avail_trays_kitchen -= num_to_serve_gf

        # Serve Reg
        num_to_serve_reg = min(current_needed_reg, avail_s_kitchen_reg, avail_trays_kitchen)
        cost += num_to_serve_reg * 3
        current_needed_reg -= num_to_serve_reg
        avail_s_kitchen_reg -= num_to_serve_reg
        avail_trays_kitchen -= num_to_serve_reg

        # Serve remaining Reg with remaining GF
        num_to_serve_reg_with_gf = min(current_needed_reg, avail_s_kitchen_gf, avail_trays_kitchen)
        cost += num_to_serve_reg_with_gf * 3
        current_needed_reg -= num_to_serve_reg_with_gf
        avail_s_kitchen_gf -= num_to_serve_reg_with_gf
        avail_trays_kitchen -= num_to_serve_reg_with_gf


        # Phase 4: Make new sandwiches (Cost 4: make + put + move + serve)
        # Requires notexist sandwich object, ingredients, and a tray at kitchen.
        # Prioritize making GF for allergic children
        num_to_make_gf = min(current_needed_gf, avail_s_notexist, avail_bread_gf, avail_content_gf, avail_trays_kitchen)
        cost += num_to_make_gf * 4
        current_needed_gf -= num_to_make_gf
        avail_s_notexist -= num_to_make_gf
        avail_bread_gf -= num_to_make_gf
        avail_content_gf -= num_to_make_gf
        avail_trays_kitchen -= num_to_make_gf

        # Track remaining resources after making GF sandwiches
        rem_bread_gf = avail_bread_gf
        rem_content_gf = avail_content_gf
        rem_bread_reg = avail_bread_reg
        rem_content_reg = avail_content_reg
        rem_s_notexist = avail_s_notexist
        rem_trays_kitchen = avail_trays_kitchen

        # Then make regular for non-allergic
        num_to_make_reg = current_needed_reg # How many regular sandwiches we still need

        # Total available bread and content for regular sandwiches (can use GF ingredients)
        total_avail_bread_for_reg = rem_bread_reg + rem_bread_gf
        total_avail_content_for_reg = rem_content_reg + rem_content_gf

        # Number of regular sandwiches we can make is limited by notexist, ingredients, and trays
        num_can_make_reg = min(num_to_make_reg, rem_s_notexist, total_avail_bread_for_reg, total_avail_content_for_reg, rem_trays_kitchen)

        cost += num_can_make_reg * 4
        current_needed_reg -= num_can_make_reg
        rem_s_notexist -= num_can_make_reg
        rem_trays_kitchen -= num_can_make_reg

        # Consuming ingredients for regular sandwiches:
        # We need num_can_make_reg bread and num_can_make_reg content.
        # Use regular ingredients first.
        used_reg_bread = min(num_can_make_reg, rem_bread_reg)
        rem_bread_reg -= used_reg_bread
        needed_bread_from_gf = num_can_make_reg - used_reg_bread
        rem_bread_gf -= needed_bread_from_gf # Use GF bread if needed

        used_reg_content = min(num_can_make_reg, rem_content_reg)
        rem_content_reg -= used_reg_content
        needed_content_from_gf = num_can_make_reg - used_reg_content
        rem_content_gf -= needed_content_from_gf # Use GF content if needed

        # The heuristic value is the total cost calculated.
        return cost
