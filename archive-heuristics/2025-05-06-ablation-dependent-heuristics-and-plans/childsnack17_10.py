from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle facts like '(at tray1 kitchen)'
    if fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    # Handle facts that might not be wrapped in parentheses (though PDDL facts usually are)
    return fact.split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we have enough parts to match the pattern
    if len(parts) < len(args):
         return False
    # Check if each part matches the corresponding argument pattern
    # Use zip which stops at the shortest sequence, handling cases where
    # the fact might have more parts than the pattern args (though less likely in PDDL).
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children.
    It counts the number of unserved children and adds estimated costs for:
    1. Making necessary sandwiches.
    2. Putting necessary sandwiches on trays.
    3. Moving trays to the locations where children are waiting.
    Each required action (make, put_on_tray, move_tray, serve) is estimated to cost 1.

    # Assumptions
    - Each unserved child requires one suitable sandwich.
    - A gluten-free sandwich can satisfy a non-allergic child, but not vice-versa.
    - A tray is needed at each distinct location where unserved children are waiting.
    - The heuristic does not explicitly check for the availability of bread, content,
      sandwich objects (`notexist`), or trays in the kitchen for `put_on_tray` actions,
      assuming they are sufficient if a state is reachable.
    - Tray objects can be identified by matching the pattern "tray*" in the 'at' facts.
    - All relevant children and their allergy status are present in static facts.

    # Heuristic Initialization
    - Identifies all children and their allergy status from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. **Count Unserved Children:** Identify all children who have not yet been served.
       The number of unserved children is the base estimate, representing the minimum
       number of 'serve' actions required.
    2. **Determine Sandwich Needs:** For the unserved children, count how many require
       gluten-free sandwiches (allergic children) and how many require regular sandwiches
       (non-allergic children). Gluten-free sandwiches can satisfy both needs.
    3. **Estimate Make Actions:** Count the number of suitable sandwiches that have
       already been made (either `at_kitchen_sandwich` or `ontray`). Calculate how many
       more sandwiches are needed to meet the total demand of unserved children,
       considering sandwich types. This difference, if positive, is the estimated
       number of 'make_sandwich' actions needed.
    4. **Estimate Put-on-Tray Actions:** Count the number of suitable sandwiches that
       are currently `ontray`. Calculate how many sandwiches needed for unserved children
       are not yet on trays. This difference, if positive, is the estimated number of
       'put_on_tray' actions needed.
    5. **Estimate Move-Tray Actions:** Identify the distinct locations where unserved
       children are waiting. Identify the distinct locations where trays are currently
       present. Count the number of waiting locations that do not currently have a tray.
       This count is the estimated number of 'move_tray' actions needed to get trays
       to the required locations.
    6. **Sum Costs:** The total heuristic value is the sum of the estimated costs
       from steps 1, 3, 4, and 5.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about children.
        Assumes all relevant children and their allergy status are in static facts.
        """
        self.all_children = set()
        self.allergic_children = set()
        self.not_allergic_children = set()

        # Extract children and allergy status from static facts
        # Example static fact: '(allergic_gluten child4)'
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == 'allergic_gluten':
                if len(parts) > 1:
                    child = parts[1]
                    self.allergic_children.add(child)
                    self.all_children.add(child)
            elif parts and parts[0] == 'not_allergic_gluten':
                 if len(parts) > 1:
                    child = parts[1]
                    self.not_allergic_children.add(child)
                    self.all_children.add(child)

        # Note: If children are not listed in static facts (e.g., only in :objects
        # and :init waiting facts without allergy status), this heuristic might
        # not identify all children or their types correctly. It relies on the
        # pattern seen in the provided examples where allergy status is in :init.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # --- Step 1: Count unserved children ---
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children = self.all_children - served_children
        num_unserved = len(unserved_children)

        # If no children are unserved, the goal is reached.
        if num_unserved == 0:
            return 0

        # Initial heuristic estimate is the number of serve actions needed
        total_cost = num_unserved

        # --- Step 2: Determine sandwich needs ---
        # Note: This relies on self.allergic_children being correctly populated in __init__.
        # If not, all children are treated as non-allergic by default calculation below.
        gf_unserved = self.allergic_children.intersection(unserved_children)
        num_gf_unserved = len(gf_unserved)
        num_reg_unserved = num_unserved - num_gf_unserved # Non-allergic unserved children

        total_sandwiches_needed = num_unserved # Each unserved child needs one sandwich

        # --- Step 3: Estimate make_sandwich actions ---
        at_kitchen_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_sandwich", "*")}
        ontray_sandwiches_set = {get_parts(fact)[1] for fact in state if match(fact, "ontray", "*", "*")}
        no_gluten_sandwiches_set = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}

        # Count existing suitable sandwiches (made)
        # A GF sandwich is suitable for both allergic and non-allergic.
        # A regular sandwich is suitable only for non-allergic.
        existing_gf_sandwiches = {s for s in (at_kitchen_sandwiches | ontray_sandwiches_set) if s in no_gluten_sandwiches_set}
        existing_reg_sandwiches = {s for s in (at_kitchen_sandwiches | ontray_sandwiches_set) if s not in no_gluten_sandwiches_set}

        # Calculate how many needed sandwiches can be satisfied by existing ones
        # Prioritize using existing GF for GF needs
        satisfied_gf_by_existing_gf = min(num_gf_unserved, len(existing_gf_sandwiches))
        remaining_gf_need = num_gf_unserved - satisfied_gf_by_existing_gf
        surplus_gf_avail = len(existing_gf_sandwiches) - satisfied_gf_by_existing_gf

        # Use surplus GF for Reg needs
        satisfied_reg_by_existing_gf = min(num_reg_unserved, surplus_gf_avail)
        remaining_reg_need = num_reg_unserved - satisfied_reg_by_existing_gf

        # Use existing Reg for remaining Reg needs
        satisfied_reg_by_existing_reg = min(remaining_reg_need, len(existing_reg_sandwiches))
        # remaining_reg_need_final = remaining_reg_need - satisfied_reg_by_existing_reg

        total_satisfied_by_existing = satisfied_gf_by_existing_gf + satisfied_reg_by_existing_gf + satisfied_reg_by_existing_reg

        # Sandwiches that still need to be made
        needed_to_make_total = max(0, total_sandwiches_needed - total_satisfied_by_existing)
        total_cost += needed_to_make_total # Add cost for make actions

        # --- Step 4: Estimate put_on_tray actions ---
        # Count sandwiches already on trays
        num_ontray = len(ontray_sandwiches_set)

        # Sandwiches that need to be put on trays
        # This is the total needed minus those already on trays.
        needed_to_put_on_tray = max(0, total_sandwiches_needed - num_ontray)
        total_cost += needed_to_put_on_tray # Add cost for put_on_tray actions

        # --- Step 5: Estimate move_tray actions ---
        # Find distinct locations where unserved children are waiting
        waiting_places = {get_parts(fact)[2] for fact in state if match(fact, "waiting", "*", "*") and get_parts(fact)[1] in unserved_children}

        # Find distinct locations where trays are currently present
        # Assuming tray objects match the pattern "tray*"
        tray_locations = {get_parts(fact)[2] for fact in state if match(fact, "at", "tray*", "*")}

        # Count locations needing a tray that don't have one
        required_tray_locations = waiting_places
        num_locations_needing_move = len(required_tray_locations - tray_locations)

        total_cost += num_locations_needing_move # Add cost for move_tray actions

        return total_cost
