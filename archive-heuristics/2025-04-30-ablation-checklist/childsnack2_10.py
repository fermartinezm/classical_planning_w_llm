from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by spaces
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    Estimates the number of actions needed to serve all waiting children.
    This is an additive heuristic where the cost for each unserved child
    is estimated independently based on the state of the closest suitable
    sandwich/tray.

    The heuristic cost for a single unserved child is determined by the
    "most expensive" remaining step required to get a suitable sandwich
    to them:
    - 1: If a suitable sandwich is already on a tray at the child's location. (Needs only the 'serve' action)
    - 2: If a suitable sandwich is on a tray elsewhere. (Needs 'move_tray' + 'serve')
    - 3: If a suitable sandwich is in the kitchen (at_kitchen_sandwich). (Needs 'put_on_tray' + 'move_tray' + 'serve')
    - 4: If a suitable sandwich needs to be made (exists as 'notexist'). (Needs 'make_sandwich' + 'put_on_tray' + 'move_tray' + 'serve')
    - 1000: If no suitable sandwich object exists at all (neither made, in kitchen, on tray, nor 'notexist'). This indicates an likely unsolvable state for this child.

    The total heuristic is the sum of these minimum costs for all unserved children.

    Assumptions:
    - Sufficient ingredients are available in the kitchen to make any 'notexist' sandwich.
    - Trays are available in the kitchen if needed for 'put_on_tray'.
    - Any sandwich object defined in the problem can potentially be made into either a standard or GF sandwich, provided suitable ingredients are available (which is typical in these problems). The heuristic checks for the *existence* of *any* suitable sandwich object in a given stage.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about children
        (allergy status, waiting location) and identifying all possible sandwich
        and tray objects defined in the domain/problem.
        """
        # Store goals if needed, though not directly used in this heuristic calculation
        self.goals = task.goals
        self.static_facts = task.static

        # Extract child allergy status and waiting locations from static facts
        self.child_info = {}
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts:
                predicate = parts[0]
                if predicate == "allergic_gluten" and len(parts) == 2:
                    child = parts[1]
                    if child not in self.child_info:
                        self.child_info[child] = {}
                    self.child_info[child]['allergy'] = 'allergic_gluten'
                elif predicate == "not_allergic_gluten" and len(parts) == 2:
                    child = parts[1]
                    if child not in self.child_info:
                        self.child_info[child] = {}
                    self.child_info[child]['allergy'] = 'not_allergic_gluten'
                elif predicate == "waiting" and len(parts) == 3:
                    child = parts[1]
                    place = parts[2]
                    if child not in self.child_info:
                         self.child_info[child] = {}
                    self.child_info[child]['location'] = place

        # Extract all possible sandwich and tray objects from task.facts.
        # task.facts contains all ground facts that can ever be true in the domain.
        self.all_sandwiches = set()
        self.all_trays = set()
        for fact in task.facts:
             parts = get_parts(fact)
             if len(parts) > 1:
                 predicate = parts[0]
                 # Sandwiches appear as the first argument in these predicates
                 if predicate in ["at_kitchen_sandwich", "ontray", "no_gluten_sandwich", "notexist"] and len(parts) >= 2:
                     self.all_sandwiches.add(parts[1])
                 # Trays appear as the second argument in 'ontray' and first in 'at'
                 if predicate == "ontray" and len(parts) == 3:
                     self.all_trays.add(parts[2])
                 elif predicate == "at" and len(parts) >= 2:
                     # Assuming the first argument of 'at' (if not kitchen constant) is a tray
                     obj = parts[1]
                     if obj != 'kitchen': # 'kitchen' is a place constant, not a movable object
                         self.all_trays.add(obj)


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state
        total_cost = 0

        # Build efficient lookups for the current state
        served_children = set()
        # Status of each sandwich: 'kitchen', 'ontray_T', 'notexist', or 'consumed' (if no longer in state)
        # We initialize based on known sandwiches and update from state.
        sandwich_status = {s: 'consumed' for s in self.all_sandwiches} # Default status if not found
        sandwich_on_tray = {} # {sandwich: tray}
        sandwich_is_gf = set() # {sandwich} if it's GF
        tray_at_place = {} # {tray: place}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts

            predicate = parts[0]

            if predicate == "served" and len(parts) == 2:
                served_children.add(parts[1])
            elif predicate == "at_kitchen_sandwich" and len(parts) == 2:
                sandwich = parts[1]
                if sandwich in self.all_sandwiches:
                    sandwich_status[sandwich] = 'kitchen'
            elif predicate == "ontray" and len(parts) == 3:
                sandwich, tray = parts[1], parts[2]
                if sandwich in self.all_sandwiches:
                    sandwich_status[sandwich] = 'ontray_T'
                    sandwich_on_tray[sandwich] = tray
            elif predicate == "no_gluten_sandwich" and len(parts) == 2:
                 sandwich = parts[1]
                 if sandwich in self.all_sandwiches:
                    sandwich_is_gf.add(sandwich)
            elif predicate == "at" and len(parts) == 3:
                 obj, place = parts[1], parts[2]
                 if obj in self.all_trays: # Check if the object is a known tray
                     tray_at_place[obj] = place
            elif predicate == "notexist" and len(parts) == 2:
                 sandwich = parts[1]
                 if sandwich in self.all_sandwiches:
                    sandwich_status[sandwich] = 'notexist' # Overwrite 'consumed' if it exists as notexist

        # Now calculate cost for each unserved child
        for child, info in self.child_info.items():
            if child in served_children:
                continue # Child is already served

            child_place = info['location']
            needs_gf = (info['allergy'] == 'allergic_gluten')

            # Determine the minimum cost to serve this child based on available sandwiches
            child_cost = 1000 # Default to a high cost (unsolvable for this child)

            # Flags to track if a suitable sandwich exists in a certain stage
            found_at_place = False
            found_on_tray_elsewhere = False
            found_in_kitchen = False
            can_be_made = False

            # Iterate through all possible sandwich objects to find a suitable one
            # in the "highest" possible stage (closest to being served).
            for sandwich in self.all_sandwiches:
                is_gf = sandwich in sandwich_is_gf
                is_suitable = (needs_gf and is_gf) or (not needs_gf)

                if is_suitable:
                    status = sandwich_status.get(sandwich, 'consumed') # Get status, default to consumed

                    if status == 'ontray_T':
                        tray = sandwich_on_tray.get(sandwich)
                        if tray: # Ensure the sandwich is mapped to a tray
                            place = tray_at_place.get(tray)
                            if place: # Ensure the tray is at a place
                                if place == child_place:
                                    found_at_place = True
                                else:
                                    found_on_tray_elsewhere = True
                    elif status == 'kitchen':
                        found_in_kitchen = True
                    elif status == 'notexist':
                        can_be_made = True

            # Determine the minimum cost for this child based on the flags,
            # prioritizing the stages closest to the goal.
            if found_at_place:
                child_cost = 1 # Needs only 'serve'
            elif found_on_tray_elsewhere:
                child_cost = 2 # Needs 'move_tray' + 'serve'
            elif found_in_kitchen:
                child_cost = 3 # Needs 'put_on_tray' + 'move_tray' + 'serve'
            elif can_be_made:
                child_cost = 4 # Needs 'make_sandwich' + 'put_on_tray' + 'move_tray' + 'serve'
            # If none of the above flags are true, child_cost remains 1000,
            # indicating no suitable sandwich exists or can be made.

            total_cost += child_cost

        return total_cost

