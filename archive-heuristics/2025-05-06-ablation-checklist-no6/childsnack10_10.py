from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or empty fact strings
    fact = fact.strip()
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return [] # Return empty list for invalid format
    return fact[1:-1].split()


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    Estimates the number of actions needed to serve all children.
    The heuristic is a sum of estimated costs for sequential stages:
    1. Making necessary sandwiches.
    2. Putting necessary sandwiches on trays.
    3. Moving trays to children's locations.
    4. Serving the children.

    It counts the number of items/tasks needed in each stage based on the
    number of unserved children and the current state, without considering
    resource limits (like bread/content availability beyond initial counts,
    or tray capacity) or complex action interactions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about children.
        """
        # Extract child information: waiting place and allergy status
        self.child_info = {} # Map child_name -> {'place': place, 'allergic': bool}
        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or invalid facts

            predicate = parts[0]
            if predicate == 'waiting':
                # Ensure fact has expected number of parts
                if len(parts) == 3:
                    child, place = parts[1], parts[2]
                    if child not in self.child_info:
                        self.child_info[child] = {'place': place, 'allergic': False} # Default to not allergic
                    else:
                         self.child_info[child]['place'] = place
            elif predicate == 'allergic_gluten':
                 # Ensure fact has expected number of parts
                if len(parts) == 2:
                    child = parts[1]
                    if child not in self.child_info:
                         self.child_info[child] = {'place': None, 'allergic': True}
                    else:
                         self.child_info[child]['allergic'] = True
            # not_allergic_gluten is implicitly handled by the default False

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # 1. Identify unserved children and their needs
        served_children = set()
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'served' and len(parts) == 2:
                served_children.add(parts[1])

        unserved_children_details = [] # List of (child, place, is_allergic)
        for child, info in self.child_info.items():
            if child not in served_children:
                # Only consider children who are actually waiting somewhere
                if info['place'] is not None:
                    unserved_children_details.append((child, info['place'], info['allergic']))

        num_unserved = len(unserved_children_details)

        # Heuristic is 0 if all children are served (goal state)
        if num_unserved == 0:
            return 0

        # Count needed sandwiches by type based on unserved children
        gf_needed = sum(1 for _, _, allergic in unserved_children_details if allergic)
        reg_needed = sum(1 for _, _, allergic in unserved_children_details if not allergic)

        # Identify unique places where unserved children are waiting
        places_needed = {place for _, place, _ in unserved_children_details}

        # 2. Count relevant items in the current state
        kitchen_sandwiches = set()
        ontray_sandwiches = set()
        gf_sandwiches_in_state = set() # Sandwiches explicitly marked gluten-free
        tray_locations = {} # Map tray -> place

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or invalid facts

            predicate = parts[0]
            if predicate == 'at_kitchen_sandwich' and len(parts) == 2:
                kitchen_sandwiches.add(parts[1])
            elif predicate == 'ontray' and len(parts) == 3:
                ontray_sandwiches.add(parts[1])
            elif predicate == 'no_gluten_sandwich' and len(parts) == 2:
                gf_sandwiches_in_state.add(parts[1])
            elif predicate == 'at' and len(parts) == 3 and parts[1].startswith('tray'): # Assuming 'at' with 3 parts and 2nd part is a tray
                 tray_locations[parts[1]] = parts[2]

        # Classify sandwiches that are 'made' (either in kitchen or on tray)
        made_sandwiches = kitchen_sandwiches | ontray_sandwiches
        gf_made = len({s for s in made_sandwiches if s in gf_sandwiches_in_state})
        # Assumes any made sandwich not explicitly marked GF is regular
        reg_made = len({s for s in made_sandwiches if s not in gf_sandwiches_in_state})

        # Classify sandwiches that are 'ontray'
        gf_ontray = len({s for s in ontray_sandwiches if s in gf_sandwiches_in_state})
        reg_ontray = len({s for s in ontray_sandwiches if s not in gf_sandwiches_in_state})

        # Identify places where trays are currently located
        places_with_trays = set(tray_locations.values())

        # 3. Calculate heuristic components (estimated actions)

        # Cost to make sandwiches: Number of needed sandwiches not yet made.
        # This is a lower bound assuming sufficient bread/content/notexist objects.
        cost_make = max(0, gf_needed - gf_made) + max(0, reg_needed - reg_made)

        # Cost to put on trays: Number of needed sandwiches not yet on trays.
        # This is a lower bound assuming trays are available in the kitchen or can be moved there.
        cost_put = max(0, gf_needed - gf_ontray) + max(0, reg_needed - reg_ontray)

        # Cost to move trays: Number of unique places needing a tray that don't currently have one.
        # This is a lower bound assuming sufficient trays in total.
        cost_move = len(places_needed - places_with_trays)

        # Cost to serve: Each unserved child needs one serve action.
        cost_serve = num_unserved

        # Total heuristic is the sum of these estimated costs
        total_cost = cost_make + cost_put + cost_move + cost_serve

        return total_cost
