from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty fact string or malformed facts gracefully
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
    Components:
    1. Number of unserved children (lower bound on 'serve' actions).
    2. Number of sandwiches that need to be made (lower bound on 'make' actions).
    3. Number of sandwiches that need to be put on trays (lower bound on 'put_on_tray' actions).
    4. Number of distinct locations (excluding kitchen) that need a tray delivery (lower bound on 'move_tray' actions).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Extract allergic status from static facts
        self.allergic_children = {get_parts(fact)[1] for fact in self.static_facts if match(fact, "allergic_gluten", "*")}
        self.not_allergic_children = {get_parts(fact)[1] for fact in self.static_facts if match(fact, "not_allergic_gluten", "*")}
        self.all_children = self.allergic_children | self.not_allergic_children

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state
        cost = 0

        # Identify unserved children
        unserved_children = {c for c in self.all_children if '(served ' + c + ')' not in state}
        num_unserved = len(unserved_children)

        # If all children are served, the goal is reached
        if num_unserved == 0:
            return 0

        # 1. Add cost for serving each unserved child
        cost += num_unserved

        # Categorize unserved children by allergy status
        unserved_allergic = {c for c in unserved_children if c in self.allergic_children}
        unserved_non_allergic = unserved_children - unserved_allergic
        num_allergic_unserved = len(unserved_allergic)
        num_non_allergic_unserved = len(unserved_non_allergic)

        # Get current sandwich status
        gf_sandwiches_in_state = {s for s_fact in state if match(s_fact, "no_gluten_sandwich", s) for s in [get_parts(s_fact)[1]] if get_parts(s_fact)}
        at_kitchen_sandwiches_in_state = {s for s_fact in state if match(s_fact, "at_kitchen_sandwich", s) for s in [get_parts(s_fact)[1]] if get_parts(s_fact)}
        ontray_sandwiches_in_state = {s for s_fact in state if match(s_fact, "ontray", s, "*") for s in [get_parts(s_fact)[1]] if get_parts(s_fact)}
        all_sandwiches_in_state = at_kitchen_sandwiches_in_state | ontray_sandwiches_in_state

        num_avail_gf = len(gf_sandwiches_in_state & all_sandwiches_in_state)
        num_avail_any = len(all_sandwiches_in_state) # Includes GF ones

        # 2. Count sandwiches that need to be made (for 'make' actions)
        # Assume infinite ingredients if at least one pair exists (as per PDDL modeling)
        # Count GF sandwiches that must be made
        num_make_gf = max(0, num_allergic_unserved - num_avail_gf)

        # Count Any sandwiches that must be made (after using available GF for A, then NA)
        # Total sandwiches needed = num_allergic_unserved + num_non_allergic_unserved
        # Total sandwiches available (any type) = num_avail_any
        # Total sandwiches that must be made = max(0, total_sandwiches_needed - num_avail_any)
        # Number of Any sandwiches to make = Total sandwiches to make - Number of GF sandwiches to make
        num_make_any = max(0, max(0, total_sandwiches_needed - num_avail_any) - num_make_gf)

        cost += num_make_gf + num_make_any

        # 3. Count sandwiches that need to be put on trays (for 'put_on_tray' actions)
        # These are the sandwiches that are needed but are not already on a tray.
        total_sandwiches_needed = num_allergic_unserved + num_non_allergic_unserved
        num_already_ontray = len(ontray_sandwiches_in_state)
        num_to_put_on_tray = max(0, total_sandwiches_needed - num_already_ontray)
        cost += num_to_put_on_tray

        # 4. Count distinct locations (excluding kitchen) that need a tray delivery (for 'move_tray' actions)
        # A location needs a tray delivery if there is an unserved child there
        # who is not already ready to be served (suitable sandwich on tray at their location).

        children_need_delivery = set()
        for child in unserved_children:
            child_place = next((get_parts(fact)[2] for fact in state if match(fact, "waiting", child, "*")), None)
            if child_place:
                is_allergic = child in self.allergic_children

                # Check if a suitable sandwich is on a tray at the child's location
                ready_at_place = False
                for s_fact in state:
                    if match(s_fact, "ontray", "*", "*"):
                        s, t = get_parts(s_fact)[1:3]
                        # Check if tray 't' is at 'child_place'
                        if '(at ' + t + ' ' + child_place + ')' in state:
                             is_gf_sandwich = '(no_gluten_sandwich ' + s + ')' in state
                             if (is_allergic and is_gf_sandwich) or (not is_allergic):
                                 ready_at_place = True
                                 break # Found a suitable sandwich on a tray at the location

                if not ready_at_place:
                    children_need_delivery.add(child)

        # Count distinct places (excluding kitchen) where children needing delivery are waiting
        places_needing_delivery = {get_parts(fact)[2] for child in children_need_delivery for fact in state if match(fact, "waiting", child, "*")}

        num_move_tray_destinations = len(places_needing_delivery - {'kitchen'})
        cost += num_move_tray_destinations

        return cost
