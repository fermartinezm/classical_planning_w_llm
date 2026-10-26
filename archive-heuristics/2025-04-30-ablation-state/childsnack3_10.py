from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not fact.strip() or fact[0] != '(' or fact[-1] != ')':
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
    This heuristic estimates the number of actions required to serve all waiting
    children. It counts the final 'serve' action for each unserved child and
    adds estimates for the necessary 'make_sandwich', 'put_on_tray', and
    'move_tray' actions required to get suitable sandwiches onto trays and
    delivered to the children's locations.

    # Assumptions
    - Each unserved child requires one suitable sandwich.
    - Gluten-allergic children require gluten-free sandwiches. Non-allergic
      children can accept any sandwich.
    - Sandwiches can be made if 'notexist' slots and ingredients are available
      (availability of ingredients is simplified in the heuristic calculation).
    - Sandwiches must be on a tray to be served.
    - Trays must be at the child's location to serve.
    - A tray can hold multiple sandwiches (simplification for move cost).
    - Any sandwich currently in the kitchen or on a tray (anywhere) is
      potentially available for a child needing delivery, unless it's already
      positioned to serve a child at their location.

    # Heuristic Initialization
    - Extracts static facts about child allergies.
    - Extracts goal facts about which children need to be served.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children who need to be served (from goals) and are not yet served (from state).
    2. For each unserved child, determine their waiting location and allergy status (using static facts).
    3. Count the total number of unserved children (`N_unserved`). This contributes `N_unserved` to the heuristic (representing the final 'serve' action for each).
    4. Identify which unserved children are 'ready-to-serve': those for whom a suitable sandwich is already on a tray at their waiting location. Count allergic and non-allergic ready children (`N_ready_to_serve_GF`, `N_ready_to_serve_Reg`).
    5. Identify children who 'need-delivery': the remaining unserved children. Count allergic and non-allergic needing delivery (`N_needs_delivery_GF`, `N_needs_delivery_Reg`). Total `N_needs_delivery`.
    6. If `N_needs_delivery > 0`, calculate additional costs:
        a.  **Make Cost:** Estimate how many sandwiches need to be made for the children needing delivery. This is the number of needed sandwiches (`N_needs_delivery_GF` GF, `N_needs_delivery_Reg` Reg) minus the number of suitable sandwiches already available (in kitchen or on any tray). Prioritize using available GF sandwiches for GF needs. Add this count to the heuristic.
        b.  **Put on Tray Cost:** Estimate how many sandwiches need to be moved from the kitchen (or just made) onto a tray for the children needing delivery. This is the number of needed sandwiches (`N_needs_delivery`) minus those already on trays. Add this count to the heuristic.
        c.  **Move Tray Cost:** Estimate how many tray movements are needed. Identify the distinct locations where children needing delivery are waiting (`Delivery_Locations`). Count how many trays are already at these locations. The number of moves needed is the number of locations minus the number of trays already there (minimum 0). Add this count to the heuristic.
    7. The total heuristic value is the sum of costs from steps 3 and 6.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting child allergy status and goal children.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Extract allergy status from static facts
        self.allergic_children = {get_parts(fact)[1] for fact in self.static_facts if match(fact, "allergic_gluten", "*")}
        self.not_allergic_children = {get_parts(fact)[1] for fact in self.static_facts if match(fact, "not_allergic_gluten", "*")}

        # Extract children who need to be served from goals
        self.children_to_serve = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0

        # --- Step 1 & 2: Identify unserved children, their needs, and locations ---
        unserved_children_info = [] # List of (child, place, is_allergic)
        waiting_places_in_state = {} # Map child -> place from current state
        sandwich_is_gf = {} # Map sandwich -> is_gluten_free from current state

        # Pre-process state facts for quick lookups
        state_facts_set = set(state) # Convert frozenset to set for faster lookups
        at_facts = {} # Map object -> location (primarily trays)
        ontray_facts = {} # Map sandwich -> tray
        at_kitchen_sandwich_facts = set() # Set of sandwiches at kitchen
        # at_kitchen_bread_facts = set() # Not directly needed for heuristic calculation logic
        # at_kitchen_content_facts = set() # Not directly needed for heuristic calculation logic
        # notexist_sandwiches = set() # Not directly needed for heuristic calculation logic

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "waiting" and len(parts) == 3:
                child, place = parts[1], parts[2]
                waiting_places_in_state[child] = place
            elif predicate == "at" and len(parts) == 3:
                 obj, place = parts[1], parts[2]
                 # Only track tray locations for move cost calculation
                 if obj.startswith("tray"):
                    at_facts[obj] = place
            elif predicate == "ontray" and len(parts) == 3:
                 sandwich, tray = parts[1], parts[2]
                 ontray_facts[sandwich] = tray
            elif predicate == "at_kitchen_sandwich" and len(parts) == 2:
                 sandwich = parts[1]
                 at_kitchen_sandwich_facts.add(sandwich)
            elif predicate == "no_gluten_sandwich" and len(parts) == 2:
                 sandwich = parts[1]
                 sandwich_is_gf[sandwich] = True
            # notexist facts are implicitly handled by counting existing sandwiches vs needed

        # Determine gluten status for all known sandwiches (in kitchen or on tray)
        all_known_sandwiches = set(ontray_facts.keys()) | at_kitchen_sandwich_facts
        for s in all_known_sandwiches:
             if s not in sandwich_is_gf:
                 sandwich_is_gf[s] = False # Assume not gluten-free unless stated

        N_allergic_unserved = 0
        N_non_allergic_unserved = 0

        for child in self.children_to_serve:
            if f"(served {child})" not in state_facts_set:
                # Child is unserved
                place = waiting_places_in_state.get(child)
                if place is None:
                    # Should not happen in valid problems, but handle defensively
                    continue

                is_allergic = child in self.allergic_children
                unserved_children_info.append((child, place, is_allergic))
                if is_allergic:
                    N_allergic_unserved += 1
                else:
                    N_non_allergic_unserved += 1

        N_unserved = len(unserved_children_info)

        # --- Step 3: Base cost for serving ---
        h += N_unserved # Each needs a 'serve' action

        # --- Step 4 & 5: Identify ready-to-serve vs needing delivery ---
        N_ready_to_serve_GF = 0
        N_ready_to_serve_Reg = 0
        delivery_locations = set() # Places where children needing delivery are waiting

        for child, place, is_allergic in unserved_children_info:
            is_ready = False
            # Check if a suitable sandwich is on a tray at the child's location
            for sandwich, tray in ontray_facts.items():
                if at_facts.get(tray) == place:
                    # Tray is at the child's location
                    is_gf_s = sandwich_is_gf.get(sandwich, False)
                    if is_allergic and is_gf_s:
                        is_ready = True
                        break
                    elif not is_allergic:
                        is_ready = True
                        break

            if is_ready:
                if is_allergic:
                    N_ready_to_serve_GF += 1
                else:
                    N_ready_to_serve_Reg += 1
            else:
                # Child needs delivery
                delivery_locations.add(place)

        N_needs_delivery_GF = N_allergic_unserved - N_ready_to_serve_GF
        N_needs_delivery_Reg = N_non_allergic_unserved - N_ready_to_serve_Reg
        N_needs_delivery = N_needs_delivery_GF + N_needs_delivery_Reg

        # --- Step 6: Calculate additional costs if deliveries are needed ---
        if N_needs_delivery > 0:
            # a. Make Cost
            # Count available sandwiches (in kitchen or on any tray)
            avail_gf_s = sum(1 for s in all_known_sandwiches if sandwich_is_gf.get(s, False))
            avail_reg_s = sum(1 for s in all_known_sandwiches if not sandwich_is_gf.get(s, False))

            needed_gf_delivery = N_needs_delivery_GF
            needed_reg_delivery = N_needs_delivery_Reg

            make_gf = max(0, needed_gf_delivery - avail_gf_s)
            avail_gf_s_after_gf_needs = max(0, avail_gf_s - needed_gf_delivery)
            make_reg = max(0, needed_reg_delivery - (avail_reg_s + avail_gf_s_after_gf_needs))
            n_sandwiches_to_make = make_gf + make_reg

            h += n_sandwiches_to_make # Cost for 'make_sandwich' actions

            # b. Put on Tray Cost
            n_ontray = len(ontray_facts)
            # Number of sandwiches that need to transition to 'ontray' for delivery
            # This is the number of deliveries needed minus those already on trays
            h += max(0, N_needs_delivery - n_ontray) # Cost for 'put_on_tray' actions

            # c. Move Tray Cost
            n_delivery_locations = len(delivery_locations)
            # Count trays already at the locations where deliveries are needed
            n_trays_at_delivery_locations = sum(1 for tray, place in at_facts.items() if place in delivery_locations)
            h += max(0, n_delivery_locations - n_trays_at_delivery_locations) # Cost for 'move_tray' actions

        return h

