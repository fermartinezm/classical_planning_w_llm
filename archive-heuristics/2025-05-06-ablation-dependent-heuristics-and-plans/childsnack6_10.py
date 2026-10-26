from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and strip whitespace
    fact_str = str(fact).strip()
    # Check if it looks like a PDDL fact (starts with '(' and ends with ')')
    if len(fact_str) > 1 and fact_str[0] == '(' and fact_str[-1] == ')':
        # Remove parentheses and split by whitespace
        return fact_str[1:-1].split()
    else:
        # Handle cases that don't match the expected format
        # This should ideally not happen with valid state/static representations
        # Return an empty list to indicate parsing failed for this item
        return []

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting children.
    It breaks down the problem into sequential stages: making necessary sandwiches,
    putting them on trays, moving trays to the children's locations, and finally serving
    the children. The heuristic sums the estimated minimum actions for each stage,
    ignoring some dependencies and resource constraints for computational efficiency
    and non-admissibility.

    # Assumptions
    - Each waiting child requires one suitable sandwich.
    - Sandwiches must typically be made in the kitchen, put on a tray in the kitchen, and then
      the tray moved to the child's location.
    - A tray can carry multiple sandwiches.
    - A tray movement action costs 1, regardless of distance or contents.
    - Making a sandwich costs 1 action, putting on a tray costs 1 action, serving costs 1 action.
    - Sufficient ingredients and sandwich objects are available to make needed sandwiches
      (this is a simplification for non-admissibility).
    - A location needing sandwich deliveries requires at least one tray movement to it.

    # Heuristic Initialization
    - Extracts the allergy status for each child from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic is calculated as the sum of estimated minimum actions for four main stages:
    1.  **Making Sandwiches:** Estimate the number of sandwiches (gluten-free and regular) that still need to be made to satisfy the demand from waiting children, based on currently available made sandwiches. This is `max(0, needed_gf - available_gf_made) + max(0, needed_reg - available_reg_made)`.
    2.  **Putting Sandwiches on Trays:** Estimate the number of sandwiches that need to be placed on trays to satisfy the demand from waiting children, based on sandwiches already on trays. This is `max(0, total_needed_sandwiches - total_sandwiches_on_trays)`.
    3.  **Moving Trays:** Estimate the number of tray movements required to bring needed sandwiches to the locations where children are waiting. This is estimated by counting the number of waiting locations that do not currently have enough suitable sandwiches on trays present at that location. Each such location is assumed to require one tray movement to deliver the needed items.
    4.  **Serving Children:** Count the number of children who are still waiting to be served. Each waiting child requires one serving action. This is simply the count of waiting children.

    The total heuristic value is the sum of the estimated costs for these four stages.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about child allergies.
        """
        # Map child to their allergy status (True if allergic, False otherwise)
        self.child_allergy = {}
        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue # Skip if parsing failed
            if parts[0] == 'allergic_gluten':
                self.child_allergy[parts[1]] = True
            elif parts[0] == 'not_allergic_gluten':
                 self.child_allergy[parts[1]] = False
            # Children not listed in static facts are not handled, assuming all children
            # are explicitly marked as allergic or not_allergic in the static facts.


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to serve all children.
        """
        state = node.state

        # --- Data structures to hold state information ---
        waiting_children_details = {} # {child: {'place': p, 'allergic': bool}}
        served_children = set()
        sandwiches_ontray = {} # {sandwich: {'tray': t, 'gluten_free': bool}}
        sandwiches_kitchen = {} # {sandwich: {'gluten_free': bool}}
        sandwiches_made = set() # Set of sandwich objects that exist
        trays_at_place = {} # {tray: place}
        sandwich_gluten_status = {} # {sandwich: bool} True if GF

        # --- Populate data structures from state facts ---
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip if parsing failed

            predicate = parts[0]
            if predicate == 'served':
                served_children.add(parts[1])
            elif predicate == 'waiting':
                 child, place = parts[1], parts[2]
                 # Add to waiting_children_details later, after checking served status
                 waiting_children_details[child] = {'place': place, 'allergic': self.child_allergy.get(child, False)} # Lookup allergy status
            elif predicate == 'ontray':
                sandwich, tray = parts[1], parts[2]
                sandwiches_ontray[sandwich] = {'tray': tray}
                sandwiches_made.add(sandwich)
            elif predicate == 'at_kitchen_sandwich':
                 sandwich = parts[1]
                 sandwiches_kitchen[sandwich] = {}
                 sandwiches_made.add(sandwich)
            elif predicate == 'no_gluten_sandwich':
                 sandwich_gluten_status[parts[1]] = True
            elif predicate == 'at' and parts[1].startswith('tray'): # Assuming objects starting with 'tray' are trays
                  tray, place = parts[1], parts[2]
                  trays_at_place[tray] = place

        # Filter out children who are already served
        waiting_children_details = {
            child: details for child, details in waiting_children_details.items()
            if child not in served_children
        }

        N_waiting_total = len(waiting_children_details)

        # If no children are waiting, the goal is reached.
        if N_waiting_total == 0:
            return 0

        # --- Determine gluten status for made sandwiches ---
        for s in sandwiches_ontray:
             sandwiches_ontray[s]['gluten_free'] = sandwich_gluten_status.get(s, False)
        for s in sandwiches_kitchen:
             sandwiches_kitchen[s]['gluten_free'] = sandwich_gluten_status.get(s, False)

        # --- Count available sandwiches by type and location ---
        S_gf_ontray = sum(1 for details in sandwiches_ontray.values() if details['gluten_free'])
        S_reg_ontray = len(sandwiches_ontray) - S_gf_ontray

        S_gf_kitchen = sum(1 for details in sandwiches_kitchen.values() if details['gluten_free'])
        S_reg_kitchen = len(sandwiches_kitchen) - S_gf_kitchen

        S_gf_made = S_gf_ontray + S_gf_kitchen
        S_reg_made = S_reg_ontray + S_reg_kitchen

        N_allergic_waiting = sum(1 for details in waiting_children_details.values() if details['allergic'])
        N_non_allergic_waiting = N_waiting_total - N_allergic_waiting

        # --- Calculate cost components ---

        # Cost 1: Making sandwiches
        # Estimate the number of sandwiches that still need to be made.
        # Assumes ingredients and sandwich objects are available.
        cost_make = max(0, N_allergic_waiting - S_gf_made) + max(0, N_non_allergic_waiting - S_reg_made)

        # Cost 2: Putting sandwiches on trays
        # Estimate the number of sandwiches that need to be put on trays.
        # This is the total number of sandwiches needed minus those already on trays.
        cost_put_on_tray = max(0, N_waiting_total - (S_gf_ontray + S_reg_ontray))

        # Cost 3: Moving trays
        # Estimate the number of tray movements needed to bring sandwiches to locations.
        # Count locations where children are waiting and need more suitable sandwiches.
        waiting_places = set(details['place'] for details in waiting_children_details.values())

        sandwiches_ontray_at_place = {p: {'gf': 0, 'reg': 0} for p in waiting_places}
        for s, details in sandwiches_ontray.items():
             t = details['tray']
             if t in trays_at_place:
                  p_t = trays_at_place[t]
                  if p_t in waiting_places:
                       if details['gluten_free']:
                            sandwiches_ontray_at_place[p_t]['gf'] += 1
                       else:
                            sandwiches_ontray_at_place[p_t]['reg'] += 1

        cost_move_tray = 0
        for p in waiting_places:
             N_allergic_at_p = sum(1 for c, details in waiting_children_details.items() if details['place'] == p and details['allergic'])
             N_non_allergic_at_p = sum(1 for c, details in waiting_children_details.items() if details['place'] == p and not details['allergic'])
             S_gf_ontray_at_p = sandwiches_ontray_at_place.get(p, {}).get('gf', 0)
             S_reg_ontray_at_p = sandwiches_ontray_at_place.get(p, {}).get('reg', 0)

             # Total suitable sandwiches needed at this location
             # A child needs a GF sandwich if allergic, or a REG sandwich if not.
             # We count how many children of each type are waiting at 'p'
             # and subtract how many suitable sandwiches are already on trays at 'p'.
             needed_at_p = max(0, N_allergic_at_p - S_gf_ontray_at_p) + max(0, N_non_allergic_at_p - S_reg_ontray_at_p)

             if needed_at_p > 0:
                  # This location needs sandwiches delivered. Assume at least one tray move is needed to bring them.
                  cost_move_tray += 1

        # Cost 4: Serving children
        # Each waiting child needs one serve action.
        cost_serve = N_waiting_total

        # Total heuristic is the sum of costs from different stages
        total_cost = cost_make + cost_put_on_tray + cost_move_tray + cost_serve

        return total_cost
