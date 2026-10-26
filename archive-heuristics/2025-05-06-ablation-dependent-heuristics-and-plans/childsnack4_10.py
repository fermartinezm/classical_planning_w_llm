from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         # Handle unexpected format, return empty list
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
    children. It sums up the estimated number of 'make_sandwich', 'put_on_tray',
    'move_tray', and 'serve_sandwich' actions needed based on the current state
    of children, sandwiches, and trays.

    # Assumptions
    - Sufficient bread, content, and sandwich objects exist in the kitchen
      to make any required sandwiches.
    - Sufficient trays exist to put sandwiches on.
    - Children do not change their waiting location or allergy status.
    - The cost of each action type is 1.
    - The heuristic is non-admissible and aims to guide a greedy best-first search.

    # Heuristic Initialization
    The heuristic extracts static information from the task:
    - Which children are waiting and at which place.
    - Which children are allergic to gluten.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:
    1. Identify all children who are waiting but have not yet been served. If all are served, the heuristic is 0.
    2. Initialize the heuristic value with the number of unserved children. This accounts for the final 'serve' action needed for each.
    3. Determine the type of sandwich (gluten-free or regular) needed for each unserved child based on their allergy status.
    4. Count the total number of sandwiches of each type (GF and regular) required to serve all unserved children.
    5. Count the number of sandwiches of each type that are currently 'made' (either at the kitchen or on a tray).
    6. Calculate the deficit of made sandwiches for each type (needed - available). Add this deficit to the heuristic value. This estimates the number of 'make_sandwich' actions needed.
    7. Count the number of sandwiches of each type that are currently 'ontray' (anywhere).
    8. Calculate the deficit of sandwiches on trays for each type (needed - available on tray). Add this deficit to the heuristic value. This estimates the number of 'put_on_tray' actions needed.
    9. Identify all places where unserved children are waiting.
    10. For each such place, determine the types of sandwiches needed by children waiting there.
    11. For each place and needed sandwich type, check if there is *any* suitable sandwich currently on a tray that is *at that specific place*.
    12. Count the number of *distinct places* where, for at least one needed sandwich type, no suitable sandwich is currently available on a tray at that place. Add this count to the heuristic value. This estimates the number of 'move_tray' actions needed to bring trays with required sandwiches to the correct locations.
    13. The total heuristic value is the sum of costs from steps 2, 6, 8, and 12.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts:
        - Waiting children and their locations.
        - Allergic children.
        """
        super().__init__(task) # Call base class constructor

        self.waiting_children = {} # {child: place}
        self.allergic_children = set() # {child}

        for fact in self.static:
            if match(fact, "waiting", "*", "*"):
                parts = get_parts(fact)
                child, place = parts[1], parts[2]
                self.waiting_children[child] = place
            elif match(fact, "allergic_gluten", "*"):
                parts = get_parts(fact)
                child = parts[1]
                self.allergic_children.add(child)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
             return 0

        # --- Parse State Facts ---
        served_children = set()
        sandwich_locations = {} # {sandwich: 'kitchen' or tray_object}
        sandwich_ontray = {} # {sandwich: tray_object}
        all_sandwiches_in_state = set()
        tray_locations = {} # {tray: place}
        notexist_sandwiches = set()
        gluten_free_sandwiches_in_state = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            predicate = parts[0]

            if predicate == "served":
                served_children.add(parts[1])
            elif predicate == "at_kitchen_sandwich":
                s = parts[1]
                sandwich_locations[s] = 'kitchen'
                all_sandwiches_in_state.add(s)
            elif predicate == "ontray":
                s, t = parts[1], parts[2]
                sandwich_locations[s] = t
                sandwich_ontray[s] = t
                all_sandwiches_in_state.add(s)
            elif predicate == "at":
                 # This predicate is used for trays and robot location (not in this domain)
                 # Assuming it's only for trays based on domain file
                 t, p = parts[1], parts[2]
                 tray_locations[t] = p
            elif predicate == "notexist":
                s = parts[1]
                notexist_sandwiches.add(s)
                all_sandwiches_in_state.add(s) # Add to know all possible sandwich objects
            elif predicate == "no_gluten_sandwich":
                 s = parts[1]
                 gluten_free_sandwiches_in_state.add(s)
                 # Add to all_sandwiches_in_state just in case it's not in other facts yet
                 all_sandwiches_in_state.add(s)


        # Determine sandwich types for all known sandwich objects
        sandwich_types = {} # {sandwich: 'gf' or 'reg'}
        for s in all_sandwiches_in_state:
             sandwich_types[s] = 'gf' if s in gluten_free_sandwiches_in_state else 'reg'


        # --- Heuristic Calculation ---

        # 1. Identify unserved children and their needs
        unserved_children = {c for c, p in self.waiting_children.items() if c not in served_children}
        num_unserved = len(unserved_children)

        # If num_unserved == 0, goal is reached (already checked at the start, but good practice)
        if num_unserved == 0:
             return 0

        child_needs = {} # {child: (place, type)}
        for c in unserved_children:
            p = self.waiting_children[c]
            st = 'gf' if c in self.allergic_children else 'reg'
            child_needs[c] = (p, st)

        # Initialize heuristic with cost for serve actions
        h = num_unserved

        # 2. Cost for make_sandwich actions
        needed_sandwiches_count = {'gf': 0, 'reg': 0}
        for c, (p, st) in child_needs.items():
            needed_sandwiches_count[st] += 1

        available_made_count = {'gf': 0, 'reg': 0}
        for s in all_sandwiches_in_state:
            # A sandwich is 'made' if it exists and is not 'notexist'
            if s not in notexist_sandwiches:
                 st = sandwich_types.get(s)
                 if st: # Ensure type is known
                     available_made_count[st] += 1

        to_make_count = {'gf': 0, 'reg': 0}
        for st in ['gf', 'reg']:
            to_make_count[st] = max(0, needed_sandwiches_count[st] - available_made_count[st])
            h += to_make_count[st]

        # 3. Cost for put_on_tray actions
        available_ontray_count = {'gf': 0, 'reg': 0}
        for s in sandwich_ontray:
            st = sandwich_types.get(s)
            if st: # Ensure type is known
                available_ontray_count[st] += 1

        to_put_on_tray_count = {'gf': 0, 'reg': 0}
        for st in ['gf', 'reg']:
            # We need needed_sandwiches_count[st] sandwiches on trays in total.
            # available_ontray_count[st] are already on trays.
            # The difference needs to be put on trays.
            to_put_on_tray_count[st] = max(0, needed_sandwiches_count[st] - available_ontray_count[st])
            h += to_put_on_tray_count[st]


        # 4. Cost for move_tray actions
        places_with_unserved = {p for c, p in child_needs.items()}
        places_needing_delivery_move = set()

        for p in places_with_unserved:
            types_needed_at_p = {st for c, (place, st) in child_needs.items() if place == p}
            delivery_needed_to_p = False

            for st in types_needed_at_p:
                # Check if any suitable sandwich is already on a tray at this place p
                suitable_sandwich_at_p_ontray = False
                for s, t in sandwich_ontray.items():
                     if tray_locations.get(t) == p: # Tray is at the child's place
                        s_type = sandwich_types.get(s)
                        if s_type == st: # Sandwich is suitable type
                            suitable_sandwich_at_p_ontray = True
                            break # Found one suitable sandwich of this type at the right place

                if not suitable_sandwich_at_p_ontray:
                    # If for this type 'st' at place 'p', no suitable sandwich is on a tray at 'p',
                    # then a delivery is needed to place 'p' for type 'st'.
                    # We only need *one* move action to bring *a* tray to 'p' that can carry *any* needed sandwich type.
                    # So, if *any* needed type at 'p' is missing from trays at 'p', the place 'p' needs a delivery move.
                    delivery_needed_to_p = True
                    break # No need to check other types for this place, it needs a delivery move

            if delivery_needed_to_p:
                places_needing_delivery_move.add(p)

        h += len(places_needing_delivery_move)

        return h
