# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
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
    This heuristic estimates the total number of actions required to serve all
    children. It calculates the minimum number of steps needed for each unserved
    child independently and sums these minimums.

    # Assumptions
    - The heuristic is additive: it sums the minimum costs for each unserved child,
      ignoring potential resource contention (e.g., multiple children needing the
      same tray or ingredient simultaneously).
    - It assumes that if ingredients and sandwich slots exist globally, a new
      sandwich can be made when needed, without tracking specific resource
      consumption for each child in the heuristic calculation.
    - Assumes solvable instances where sufficient global resources exist to
      eventually serve all children.

    # Heuristic Initialization
    - Extracts static information about children: allergy status and waiting places.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is the sum of costs for each unserved child. For an
    unserved child waiting at a specific place, the minimum cost is estimated as:

    1.  **Cost for the final 'serve' action:** +1
    2.  **Minimum transport/preparation cost** to get a suitable sandwich onto a tray
        and delivered to the child's waiting place. This minimum is found by
        checking the "closest" available suitable sandwich:
        -   If a suitable sandwich is already on a tray at the child's location: +0
        -   If a suitable sandwich is on a tray at a different location: +1 (for 'move_tray')
        -   If a suitable sandwich is in the kitchen (not on a tray): +2 (for 'put_on_tray' + 'move_tray')
        -   If no suitable sandwich exists but can be made (ingredients and slot available): +3 (for 'make_sandwich' + 'put_on_tray' + 'move_tray')

    The heuristic sums these costs (1 + minimum transport/preparation cost) for every
    child that is not yet marked as 'served'. If a child cannot be served due to
    lack of any suitable sandwich source (existing or makable), the heuristic
    returns infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts.
        """
        super().__init__(task)
        self.allergy_status = {}
        self.waiting_places = {}

        for fact in self.static:
            if match(fact, "allergic_gluten", "?c"):
                child = get_parts(fact)[1]
                self.allergy_status[child] = True
            elif match(fact, "not_allergic_gluten", "?c"):
                child = get_parts(fact)[1]
                self.allergy_status[child] = False
            elif match(fact, "waiting", "?c", "?p"):
                child = get_parts(fact)[1]
                place = get_parts(fact)[2]
                self.waiting_places[child] = place

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # 1. Identify unserved children and their needs/locations.
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "?c")}
        unserved_children_list = []
        # Iterate through all children identified in static facts
        for child in self.allergy_status:
            if child not in served_children:
                place = self.waiting_places.get(child)
                is_allergic = self.allergy_status.get(child, False)
                # Ensure child has a waiting place defined in static facts
                if place:
                    unserved_children_list.append((child, place, is_allergic))
                # else: Child exists but no waiting place? This shouldn't happen in valid problems.

        if not unserved_children_list:
            # All children are served, goal reached.
            return 0

        # 2. Identify available sandwiches and their properties/locations.
        sandwich_details = {}
        for fact in state:
            if match(fact, "at_kitchen_sandwich", "?s"):
                s = get_parts(fact)[1]
                sandwich_details[s] = {'location': 'kitchen', 'is_gf': False, 'tray': None}
            elif match(fact, "ontray", "?s", "?t"):
                s = get_parts(fact)[1]
                t = get_parts(fact)[2]
                sandwich_details[s] = {'location': 'ontray', 'is_gf': False, 'tray': t}
            # Sandwiches that don't exist or are served are not available

        # Determine GF status for sandwiches found
        for s in list(sandwich_details.keys()): # Iterate over a copy in case we modify
             if "(no_gluten_sandwich {})".format(s) in state:
                 sandwich_details[s]['is_gf'] = True
             # If a sandwich exists but its GF status is not specified and it wasn't made with make_sandwich_no_gluten,
             # it's implicitly not GF. The current logic handles this by initializing is_gf to False.

        # 3. Identify tray locations.
        tray_locations = {}
        for fact in state:
            if match(fact, "at", "?t", "?p"):
                t = get_parts(fact)[1]
                p = get_parts(fact)[2]
                tray_locations[t] = p

        # 4. Identify available ingredients and sandwich slots.
        num_any_bread = sum(1 for fact in state if match(fact, "at_kitchen_bread", "*"))
        num_gf_bread = sum(1 for fact in state if match(fact, "at_kitchen_bread", "?b") and "(no_gluten_bread {})".format(get_parts(fact)[1]) in state)
        num_any_content = sum(1 for fact in state if match(fact, "at_kitchen_content", "*"))
        num_gf_content = sum(1 for fact in state if match(fact, "at_kitchen_content", "?c") and "(no_gluten_content {})".format(get_parts(fact)[1]) in state)
        num_notexist_sandwiches = sum(1 for fact in state if match(fact, "notexist", "*"))

        # 5. Calculate heuristic sum.
        h = 0

        for child, place, is_allergic in unserved_children_list:
            # Cost for the final 'serve' action
            h += 1

            # Minimum transport/preparation cost
            min_transport_cost = float('inf')

            # Check existing sandwiches
            for s, details in sandwich_details.items():
                if is_allergic and not details['is_gf']:
                    continue # Not suitable

                if details['location'] == 'ontray':
                    tray = details['tray']
                    tray_loc = tray_locations.get(tray)
                    if tray_loc == place:
                        min_transport_cost = min(min_transport_cost, 0) # Already on tray at location
                    elif tray_loc is not None: # On tray elsewhere
                        min_transport_cost = min(min_transport_cost, 1) # Need 1 move_tray
                    # else: tray location unknown? Should not happen in valid states.
                elif details['location'] == 'kitchen':
                    min_transport_cost = min(min_transport_cost, 2) # Need 1 put_on_tray + 1 move_tray

            # Check if making a new sandwich is an option
            can_make_suitable = False
            if num_notexist_sandwiches > 0:
                if is_allergic:
                    # Need GF ingredients
                    if num_gf_bread > 0 and num_gf_content > 0:
                        can_make_suitable = True
                else: # Need Any sandwich (GF or non-GF)
                    # Can make GF?
                    can_make_gf = num_gf_bread > 0 and num_gf_content > 0
                    # Can make non-GF? Need non-GF bread AND non-GF content.
                    # Total bread - GF bread = non-GF bread count.
                    # Total content - GF content = non-GF content count.
                    num_nongf_bread = num_any_bread - num_gf_bread
                    num_nongf_content = num_any_content - num_gf_content
                    can_make_nongf = num_nongf_bread > 0 and num_nongf_content > 0

                    if can_make_gf or can_make_nongf:
                         can_make_suitable = True

            if can_make_suitable:
                min_transport_cost = min(min_transport_cost, 3) # make + put_on_tray + move_tray

            # Add the minimum transport cost for this child
            if min_transport_cost == float('inf'):
                 # This child cannot be served with current resources (no suitable sandwich source).
                 # Problem is likely unsolvable from this state.
                 return float('inf') # Return infinity or a very large number

            h += min_transport_cost

        return h
