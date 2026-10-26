from fnmatch import fnmatch
# Assuming heuristics.heuristic_base exists and defines a Heuristic base class
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
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
    This heuristic estimates the number of actions required to serve all waiting children.
    It sums up the estimated costs for:
    1. The final 'serve' action for each unserved child.
    2. Making sandwiches that are needed but not available (considering gluten requirements).
    3. Putting existing kitchen sandwiches onto trays.
    4. Moving trays to locations where unserved children are waiting and no tray with a sandwich is currently present.

    # Assumptions
    - Each child needs exactly one sandwich.
    - Allergic children require gluten-free sandwiches. Non-allergic children can take any sandwich.
    - Sandwiches must be on a tray to be served.
    - The tray must be at the child's location to serve.
    - Ingredients and 'notexist' sandwich objects are consumed upon making a sandwich.
    - The problem instance is solvable, implying sufficient total resources (ingredients, notexist sandwiches, trays) exist across the problem to satisfy the goal. The heuristic estimates steps based on *current* resource availability and needs.

    # Heuristic Initialization
    - Extracts all objects of each type (children, trays, sandwiches, places, bread, content) from the initial state and static facts.
    - Stores static facts and goal facts.
    - Maps each child to their waiting place (from static facts).
    - Identifies which children are allergic (from static facts).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all children who have not yet been served. If none, the heuristic is 0.
    2. Initialize the heuristic value `h` with the number of unserved children (representing the final 'serve' action for each).
    3. Calculate the number of gluten-free (NG) and regular (REG) sandwiches that are still needed on trays to satisfy the unserved children, accounting for suitable sandwiches already on trays.
    4. Calculate the number of NG and REG sandwiches currently in the kitchen that are not yet on trays.
    5. Calculate the number of available NG and REG ingredients and 'notexist' sandwich objects.
    6. Estimate the cost for 'put_on_tray' actions: Add 1 to `h` for each needed NG/REG sandwich that can be supplied by kitchen stock (up to the number needed).
    7. Estimate the cost for 'make_sandwich' actions: Add 1 to `h` for each remaining needed NG/REG sandwich that can be supplied by making new ones (up to the number needed and available ingredients/notexist objects). Prioritize making NG sandwiches if needed.
    8. Group unserved children by their waiting location.
    9. For each location where unserved children are waiting: Check if there is at least one tray at that location that has *any* sandwich on it. If not, add 1 to `h` (representing the need for a 'move_tray' action to bring a tray there).
    10. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and object lists.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Collect all objects by type from initial state and static facts
        all_facts = set(task.initial_state) | set(task.static)
        self.all_children = set()
        self.all_trays = set()
        self.all_sandwiches = set()
        self.all_bread = set()
        self.all_content = set()
        self.all_places = set() # Includes kitchen and tables

        # Add kitchen explicitly as it's a constant place
        self.all_places.add('kitchen')

        for fact in all_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts

            predicate = parts[0]
            if predicate in ['allergic_gluten', 'not_allergic_gluten', 'served', 'waiting']:
                if len(parts) > 1: self.all_children.add(parts[1])
                if predicate == 'waiting' and len(parts) > 2: self.all_places.add(parts[2])
            elif predicate in ['ontray']:
                 if len(parts) > 1: self.all_sandwiches.add(parts[1])
                 if len(parts) > 2: self.all_trays.add(parts[2])
            elif predicate in ['at_kitchen_sandwich', 'no_gluten_sandwich', 'notexist']:
                 if len(parts) > 1: self.all_sandwiches.add(parts[1])
            elif predicate in ['at_kitchen_bread', 'no_gluten_bread']:
                 if len(parts) > 1: self.all_bread.add(parts[1])
            elif predicate in ['at_kitchen_content', 'no_gluten_content']:
                 if len(parts) > 1: self.all_content.add(parts[1])
            elif predicate == 'at': # at tray place
                 if len(parts) > 1: self.all_trays.add(parts[1])
                 if len(parts) > 2: self.all_places.add(parts[2])


        # Map children to their waiting places (from static facts)
        self.child_waiting_place = {}
        for fact in self.static_facts:
            if match(fact, 'waiting', '*', '*'):
                c, p = get_parts(fact)[1:]
                self.child_waiting_place[c] = p

        # Identify allergic children (from static facts)
        self.allergic_children = {c for c in self.all_children if '(allergic_gluten ' + c + ')' in self.static_facts}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unserved children
        unserved_children = {c for c in self.all_children if '(served ' + c + ')' not in state}

        # 3. If none, the heuristic is 0.
        if not unserved_children:
            return 0

        # 4. Initialize heuristic
        h = 0

        # 5. Cost for serving: +1 for each unserved child
        h += len(unserved_children)

        # 6. Cost for getting sandwiches onto trays

        # Count suitable sandwiches already on trays
        ontray_ng = {s for s in self.all_sandwiches if '(no_gluten_sandwich ' + s + ')' in state and any('(ontray ' + s + ' ' + t + ')' in state for t in self.all_trays)}
        ontray_reg = {s for s in self.all_sandwiches if s not in ontray_ng and any('(ontray ' + s + ' ' + t + ')' in state for t in self.all_trays)}

        # Number of NG sandwiches needed *on trays*
        needed_ontray_ng = max(0, len({c for c in unserved_children if c in self.allergic_children}) - len(ontray_ng))
        # Number of *additional* REG sandwiches needed *on trays*
        needed_ontray_reg = max(0, len(unserved_children) - len(ontray_ng) - len(ontray_reg))

        # Count suitable sandwiches in kitchen *not* on trays
        kitchen_ng_not_on_tray = {s for s in self.all_sandwiches if '(no_gluten_sandwich ' + s + ')' in state and '(at_kitchen_sandwich ' + s + ')' in state}
        kitchen_reg_not_on_tray = {s for s in self.all_sandwiches if s not in kitchen_ng_not_on_tray and '(at_kitchen_sandwich ' + s + ')' in state}

        # Count available ingredients and 'notexist'
        N_bread_ng = sum(1 for f in state if match(f, 'at_kitchen_bread', '*') and match('(no_gluten_bread ' + get_parts(f)[1] + ')', 'no_gluten_bread', '*'))
        N_content_ng = sum(1 for f in state if match(f, 'at_kitchen_content', '*') and match('(no_gluten_content ' + get_parts(f)[1] + ')', 'no_gluten_content', '*'))
        N_bread_reg = sum(1 for f in state if match(f, 'at_kitchen_bread', '*') and not match('(no_gluten_bread ' + get_parts(f)[1] + ')', 'no_gluten_bread', '*'))
        N_content_reg = sum(1 for f in state if match(f, 'at_kitchen_content', '*') and not match('(no_gluten_content ' + get_parts(f)[1] + ')', 'no_gluten_content', '*'))
        N_notexist = sum(1 for f in state if match(f, 'notexist', '*'))

        # Cost for 'put_on_tray' actions:
        # Can satisfy NG need from kitchen
        put_ng_from_kitchen = min(needed_ontray_ng, len(kitchen_ng_not_on_tray))
        # Can satisfy REG need from kitchen
        put_reg_from_kitchen = min(needed_ontray_reg, len(kitchen_reg_not_on_tray))
        h += put_ng_from_kitchen + put_reg_from_kitchen

        # Remaining needed on trays
        rem_needed_ontray_ng = needed_ontray_ng - put_ng_from_kitchen
        rem_needed_ontray_reg = needed_ontray_reg - put_reg_from_kitchen

        # Cost for 'make_sandwich' actions:
        # Potential NG makes (prioritize if needed)
        potential_make_ng = min(N_bread_ng, N_content_ng, N_notexist)
        # Number of notexist objects consumed by NG makes
        notexist_used_for_ng = min(rem_needed_ontray_ng, potential_make_ng)
        # Remaining notexist objects for REG makes
        notexist_after_ng_makes = N_notexist - notexist_used_for_ng
        # Potential REG makes using remaining notexist
        potential_make_reg = min(N_bread_reg, N_content_reg, max(0, notexist_after_ng_makes))

        # Can satisfy remaining NG need by making
        make_ng = min(rem_needed_ontray_ng, potential_make_ng)
        # Can satisfy remaining REG need by making
        make_reg = min(rem_needed_ontray_reg, potential_make_reg)
        h += make_ng + make_reg

        # 7. Cost for moving trays

        # Group unserved children by their waiting place
        children_by_place = {}
        for c in unserved_children:
            p = self.child_waiting_place.get(c) # Get place from static facts
            if p: # Ensure place exists
                if p not in children_by_place:
                    children_by_place[p] = []
                children_by_place[p].append(c)

        # For each place with unserved children, check if a useful tray is present
        for place, children_at_place in children_by_place.items():
            if not children_at_place: continue # Should not happen based on how children_by_place is built

            useful_tray_at_p_exists = False
            for t in self.all_trays:
                # Check if tray is at the place
                if '(at ' + t + ' ' + place + ')' in state:
                    # Check if the tray has any sandwich on it
                    if any('(ontray ' + s + ' ' + t + ')' in state for s in self.all_sandwiches):
                        useful_tray_at_p_exists = True
                        break # Found a useful tray at this location

            if not useful_tray_at_p_exists:
                h += 1 # Need to move a tray to this location

        return h
