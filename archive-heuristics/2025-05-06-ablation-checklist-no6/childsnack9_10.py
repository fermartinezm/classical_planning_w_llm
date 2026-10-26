from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # This case should ideally not happen with valid PDDL states from the planner
        # but adding a safeguard.
        return []
    return fact[1:-1].split()

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    Estimates the number of actions needed to serve all waiting children.
    The heuristic sums up the estimated costs for:
    1. Serving each unserved child.
    2. Making sandwiches that are needed but not yet made.
    3. Putting needed sandwiches onto trays if they are not already.
    4. Moving trays to locations where unserved children are waiting if no tray is present there.

    This heuristic is non-admissible and does not account for complex resource contention
    (e.g., limited ingredients beyond total count, limited trays, trays being in the wrong place for put_on_tray).
    It assumes ingredients are sufficient to make needed sandwiches based on total counts.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts.
        Static facts include child allergies and gluten-free properties of ingredients/sandwiches.
        """
        self.static_facts = set(task.static) # Convert frozenset to set for faster lookups

        # Pre-process static facts for quick lookup
        self._allergic_gluten = {get_parts(fact)[1] for fact in self.static_facts if get_parts(fact) and get_parts(fact)[0] == 'allergic_gluten'}
        self._not_allergic_gluten = {get_parts(fact)[1] for fact in self.static_facts if get_parts(fact) and get_parts(fact)[0] == 'not_allergic_gluten'}
        # Note: no_gluten_bread/content are not directly used in the final heuristic logic,
        # as we assume ingredients are sufficient to make the needed sandwiches based on total counts.


    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        state_facts = set(state) # Convert frozenset to set for faster lookups

        h = 0

        # Identify unserved children and their waiting locations
        unserved_children = {} # {child_id: waiting_place}
        for fact in state_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'waiting' and len(parts) == 3:
                child, place = parts[1], parts[2]
                if f'(served {child})' not in state_facts:
                    unserved_children[child] = place

        # If no unserved children, goal is reached
        if not unserved_children:
            return 0

        # 1. Count `serve` actions needed
        # Each unserved child needs one serve action
        h += len(unserved_children)

        # 2. Count sandwiches that need to be made
        # Count needed sandwiches by type based on unserved children
        needed_gf = sum(1 for child in unserved_children if child in self._allergic_gluten)
        needed_reg = sum(1 for child in unserved_children if child in self._not_allergic_gluten)

        # Count existing suitable sandwiches (made, either in kitchen or on tray)
        existing_gf_made = 0
        existing_reg_made = 0
        existing_sandwiches_set = set() # Keep track of sandwiches counted to avoid double counting

        for fact in state_facts:
             parts = get_parts(fact)
             if parts and parts[0] in ['at_kitchen_sandwich', 'ontray'] and len(parts) >= 2:
                 s = parts[1]
                 if s in existing_sandwiches_set:
                     continue # Already counted this sandwich
                 existing_sandwiches_set.add(s)

                 is_gf_sandwich = f'(no_gluten_sandwich {s})' in state_facts

                 # A GF sandwich can serve an allergic or non-allergic child
                 if is_gf_sandwich:
                     existing_gf_made += 1
                 # A regular sandwich can only serve a non-allergic child
                 else:
                     existing_reg_made += 1

        # Calculate how many more sandwiches of each type are needed (deficit)
        # Prioritize using existing GF for GF demand.
        gf_made_for_gf = min(needed_gf, existing_gf_made)
        remaining_existing_gf_made = existing_gf_made - gf_made_for_gf

        still_need_make_gf = max(0, needed_gf - existing_gf_made)
        still_need_make_reg = max(0, needed_reg - (existing_reg_made + remaining_existing_gf_made))

        h += still_need_make_gf # make_sandwich_no_gluten actions
        h += still_need_make_reg # make_sandwich actions


        # 3. Count sandwiches that need to be put on a tray
        # These are the sandwiches needed by unserved children that are currently *not* on a tray.
        # Count existing suitable sandwiches that are currently on a tray
        existing_gf_on_tray = 0
        existing_reg_on_tray = 0
        existing_sandwiches_on_tray = set()

        for fact in state_facts:
             parts = get_parts(fact)
             if parts and parts[0] == 'ontray' and len(parts) == 3:
                 s = parts[1]
                 if s in existing_sandwiches_on_tray:
                     continue
                 existing_sandwiches_on_tray.add(s)

                 is_gf_sandwich = f'(no_gluten_sandwich {s})' in state_facts

                 if is_gf_sandwich:
                     existing_gf_on_tray += 1
                 else:
                     existing_reg_on_tray += 1

        # How many needed GF sandwiches are not on a tray?
        # Prioritize using existing GF on tray for GF demand.
        gf_on_tray_for_gf = min(needed_gf, existing_gf_on_tray)
        remaining_existing_gf_on_tray = existing_gf_on_tray - gf_on_tray_for_gf

        still_need_put_gf = max(0, needed_gf - existing_gf_on_tray)

        # How many needed regular sandwiches are not on a tray?
        # Use remaining existing GF on tray + existing regular on tray for regular demand.
        reg_on_tray_for_reg = min(needed_reg, existing_reg_on_tray)
        # Total on tray covering regular demand (regular + remaining GF)
        total_on_tray_covering_reg_demand = reg_on_tray_for_reg + min(remaining_existing_gf_on_tray, max(0, needed_reg - reg_on_tray_for_reg))

        still_need_put_reg = max(0, needed_reg - total_on_tray_covering_reg_demand)

        h += still_need_put_gf # put_on_tray actions for needed GF sandwiches not on tray
        h += still_need_put_reg # put_on_tray actions for needed regular sandwiches not on tray


        # 4. Count trays that need to be moved
        # For each location 'p' where unserved children are waiting, check if *any* tray is currently at 'p'.
        # If not, we need to move a tray there.
        waiting_locations = set(unserved_children.values())
        trays_at_locations = {get_parts(fact)[2] for fact in state_facts if get_parts(fact) and get_parts(fact)[0] == 'at' and len(get_parts(fact)) == 3 and get_parts(fact)[1].startswith('tray')}

        for p in waiting_locations:
            # Children waiting at the kitchen need a tray at the kitchen.
            # Children waiting at other places need a tray at that place.
            # If the waiting location 'p' does not have a tray, we need a move.
            if p not in trays_at_locations:
                 h += 1 # move_tray action needed to get *a* tray to location 'p'


        return h
