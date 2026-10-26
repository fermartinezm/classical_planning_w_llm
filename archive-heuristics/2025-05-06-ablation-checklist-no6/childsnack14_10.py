from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact string defensively
    if not fact or fact[0] != '(' or fact[-1] != ')':
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

    Estimates the number of actions needed to serve all children.
    The heuristic sums up the estimated costs for the main steps required
    to get sandwiches to unserved children:
    1. Serving each unserved child (1 action per child).
    2. Making necessary sandwiches based on demand and available supply
       (1 action per sandwich to make).
    3. Putting necessary sandwiches on trays (deficit count: number of
       sandwiches that still need to end up on a tray).
    4. Moving trays to locations where unserved children are waiting
       (deficit count: number of places needing a tray where none is present).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about children.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Map child name to their allergy status (True for allergic, False otherwise)
        self.child_allergy = {}
        # Map child name to their waiting place
        self.waiting_children = {}

        for fact in self.static_facts:
            parts = get_parts(fact)
            if not parts:
                continue
            predicate = parts[0]
            if predicate == 'allergic_gluten' and len(parts) == 2:
                self.child_allergy[parts[1]] = True
            elif predicate == 'not_allergic_gluten' and len(parts) == 2:
                self.child_allergy[parts[1]] = False
            elif predicate == 'waiting' and len(parts) == 3:
                self.waiting_children[parts[1]] = parts[2]

        # Identify all children mentioned in goals (these are the children to be served)
        self.children_to_serve = {get_parts(goal)[1] for goal in self.goals if match(goal, 'served', '*')}


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # 1. Count unserved children
        unserved_children = {c for c in self.children_to_serve if '(served ' + c + ')' not in state}
        num_unserved = len(unserved_children)

        # If no children are unserved, the goal is reached.
        if num_unserved == 0:
            return 0

        # Base heuristic: At least one 'serve' action per unserved child
        h = num_unserved

        # Identify demand for sandwich types
        unserved_allergic = {c for c in unserved_children if self.child_allergy.get(c, False)}
        unserved_non_allergic = unserved_children - unserved_allergic
        demand_gf = len(unserved_allergic)
        demand_reg = len(unserved_non_allergic)

        # 2. Count available sandwiches and their types in the current state
        # Available sandwiches are those that exist (not 'notexist').
        # In the state, existing sandwiches are typically those with predicates
        # like at_kitchen_sandwich, ontray, or no_gluten_sandwich.
        available_sandwiches = set()
        sandwich_is_gf = {} # Map sandwich name to True if GF

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate in ['at_kitchen_sandwich', 'ontray', 'no_gluten_sandwich'] and len(parts) >= 2:
                 s = parts[1]
                 available_sandwiches.add(s)
                 if predicate == 'no_gluten_sandwich':
                     sandwich_is_gf[s] = True

        available_gf_sandwiches = {s for s in available_sandwiches if sandwich_is_gf.get(s, False)}
        available_reg_sandwiches = available_sandwiches - available_gf_sandwiches

        supply_gf = len(available_gf_sandwiches)
        supply_reg = len(available_reg_sandwiches)

        # Calculate sandwiches to make
        # Prioritize using GF sandwiches for allergic children
        needed_gf_from_new = max(0, demand_gf - supply_gf)
        supply_gf_for_reg = max(0, supply_gf - demand_gf) # Remaining GF supply can serve non-allergic
        needed_reg_from_new = max(0, demand_reg - (supply_reg + supply_gf_for_reg))
        num_to_make = needed_gf_from_new + needed_reg_from_new

        h += num_to_make # Cost for 'make_sandwich' actions

        # 3. Count sandwiches currently on trays
        sandwiches_on_tray_current = {s for fact in state if fact.startswith('(ontray ')}
        num_ontray_current = len(sandwiches_on_tray_current)

        # Number of sandwiches that still need to be put on a tray
        # We need num_unserved sandwiches on trays eventually.
        # The deficit is the number of put_on_tray actions needed.
        num_need_put_on_tray = max(0, num_unserved - num_ontray_current)
        h += num_need_put_on_tray # Cost for 'put_on_tray' actions

        # 4. Count tray movements needed
        # Identify places where unserved children are waiting, excluding kitchen
        places_with_unserved_away_from_kitchen = {
            self.waiting_children[c] for c in unserved_children
            if c in self.waiting_children and self.waiting_children[c] != 'kitchen'
        }

        # Count trays currently at these places
        trays_at_unserved_places = {
            get_parts(fact)[1] for fact in state
            if match(fact, 'at', '*', '*') and get_parts(fact)[2] in places_with_unserved_away_from_kitchen
        }
        num_trays_at_unserved_places = len(trays_at_unserved_places)

        # Number of tray movements needed is the number of places needing a tray
        # minus the number of trays already there.
        num_tray_movements = max(0, len(places_with_unserved_away_from_kitchen) - num_trays_at_unserved_places)
        h += num_tray_movements # Cost for 'move_tray' actions

        return h
