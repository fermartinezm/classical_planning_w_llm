from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
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

    Estimates the number of actions needed to serve all children.
    The heuristic sums up the estimated costs for:
    1. Serving each unserved child (1 action per child).
    2. Making sandwiches that are needed but don't exist (1 action per sandwich).
    3. Putting sandwiches that are at the kitchen onto trays (1 action per sandwich at kitchen).
    4. Moving trays to locations where unserved children are waiting if no tray is currently there (1 action per location).

    This heuristic is not admissible as it simplifies resource availability (bread, content, notexist)
    and tray usage (a tray at a location serves all children there, but moving it costs 1 regardless
    of how many children benefit). It aims to be efficiently computable and informative for greedy search.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and object lists.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Extract objects by type from static and initial state facts
        # Use sets to store unique objects
        self.all_children = set()
        self.all_trays = set()
        self.all_places = set()
        self.all_sandwiches = set() # Represents sandwich *objects* that can be made
        self.all_bread = set()
        self.all_content = set()


        # Store static mappings
        self.child_allergy = {} # child -> 'allergic_gluten' or 'not_allergic_gluten'
        self.child_place = {}   # child -> waiting_place

        # Combine static and initial state facts for object discovery
        all_facts = set(self.static_facts) | set(task.initial_state)

        for fact in all_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts

            predicate = parts[0]
            if predicate == 'allergic_gluten' or predicate == 'not_allergic_gluten':
                child = parts[1]
                self.all_children.add(child)
                self.child_allergy[child] = predicate # Store 'allergic_gluten' or 'not_allergic_gluten'
            elif predicate == 'waiting':
                child, place = parts[1], parts[2]
                self.all_children.add(child)
                self.all_places.add(place)
                self.child_place[child] = place
            elif predicate == 'at': # Tray location
                tray, place = parts[1], parts[2]
                self.all_trays.add(tray)
                self.all_places.add(place)
            elif predicate in ['at_kitchen_bread', 'no_gluten_bread']:
                 if len(parts) > 1: self.all_bread.add(parts[1])
            elif predicate in ['at_kitchen_content', 'no_gluten_content']:
                 if len(parts) > 1: self.all_content.add(parts[1])
            elif predicate in ['at_kitchen_sandwich', 'ontray', 'notexist', 'no_gluten_sandwich']:
                 if len(parts) > 1: self.all_sandwiches.add(parts[1])
            # 'served' predicate is a goal, doesn't define objects

        # Add 'kitchen' constant explicitly if it appears in any relevant facts
        # This is safer than relying solely on parsing variable positions
        for fact in all_facts:
             if 'kitchen' in fact:
                 self.all_places.add('kitchen')
                 break # Assume kitchen is a place if mentioned


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state.

        h = 0

        # 1. Count unserved children (cost for 'serve' action for each)
        unserved_children = {c for c in self.all_children if f'(served {c})' not in state}
        h += len(unserved_children)

        if not unserved_children:
            # Goal reached
            return 0

        # Determine total needed GF and Reg sandwiches for unserved children
        needed_gf = sum(1 for c in unserved_children if self.child_allergy.get(c) == 'allergic_gluten')
        needed_reg = sum(1 for c in unserved_children if self.child_allergy.get(c) == 'not_allergic_gluten')

        # Count available sandwiches (at kitchen or on tray)
        avail_gf_kitchen = 0
        avail_reg_kitchen = 0
        avail_gf_ontray = 0
        avail_reg_ontray = 0

        # Keep track of GF sandwiches currently in the state
        gf_sandwiches_in_state = {s for s in self.all_sandwiches if f'(no_gluten_sandwich {s})' in state}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'at_kitchen_sandwich':
                s = parts[1]
                is_gf = (s in gf_sandwiches_in_state)
                if is_gf:
                    avail_gf_kitchen += 1
                else:
                    avail_reg_kitchen += 1
            elif predicate == 'ontray':
                s = parts[1]
                is_gf = (s in gf_sandwiches_in_state)
                if is_gf:
                    avail_gf_ontray += 1
                else:
                    avail_reg_ontray += 1

        avail_gf_total = avail_gf_kitchen + avail_gf_ontray
        avail_reg_total = avail_reg_kitchen + avail_reg_ontray

        # 2. Count sandwiches that still need to be made
        # This is the deficit between needed and available sandwiches
        to_make_gf = max(0, needed_gf - avail_gf_total)
        to_make_reg = max(0, needed_reg - avail_reg_total)
        h += to_make_gf + to_make_reg # Each make action costs 1

        # 3. Count sandwiches at the kitchen that need to be put on trays
        # These are sandwiches that exist but haven't moved to a tray yet.
        # We count all of them as needing a 'put_on_tray' action.
        h += avail_gf_kitchen + avail_reg_kitchen # Each put_on_tray action costs 1

        # 4. Count places with unserved children that need a tray moved there
        places_with_unserved = {self.child_place.get(c) for c in unserved_children if self.child_place.get(c) is not None}
        places_needing_tray_move = 0
        for p in places_with_unserved:
            # Check if any tray is currently at place p
            tray_at_p = any(f'(at {t} {p})' in state for t in self.all_trays)
            if not tray_at_p:
                places_needing_tray_move += 1

        h += places_needing_tray_move # Each move_tray action costs 1

        return h
