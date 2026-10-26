from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()

def match(fact, *args):
    """Check if a PDDL fact matches a pattern with wildcards."""
    parts = get_parts(fact)
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnack20Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    Estimates the number of actions required to serve all children by considering:
    - Whether each child needs a gluten-free sandwich.
    - Current locations of sandwiches and trays.
    - Steps to make, place, move, and serve sandwiches.

    # Assumptions
    - Each child requires one sandwich.
    - Trays can be moved between locations as needed.
    - Gluten-free sandwiches require specific bread and content, which are available if needed.

    # Heuristic Initialization
    - Extracts static information about allergic children and their waiting locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify unserved children and their locations.
    2. For each unserved child:
        a. Check if a suitable sandwich is already on a tray at their location.
        b. If not, check for a suitable sandwich in the kitchen.
        c. Calculate actions needed to make, place, move, and serve the sandwich.
    3. Sum actions for all unserved children, considering shared tray movements.
    """

    def __init__(self, task):
        self.allergic_children = set()
        self.waiting_locations = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'allergic_gluten':
                self.allergic_children.add(parts[1])
            elif parts[0] == 'waiting':
                self.waiting_locations[parts[1]] = parts[2]

    def __call__(self, node):
        state = node.state
        served = set()
        for fact in state:
            if match(fact, 'served', '*'):
                served.add(get_parts(fact)[1])
        unserved = [c for c in self.waiting_locations if c not in served]
        trays_in_kitchen = any(match(fact, 'at', '*', 'kitchen') for fact in state)
        total = 0

        for child in unserved:
            is_allergic = child in self.allergic_children
            location = self.waiting_locations[child]
            has_sandwich = False

            # Check for suitable sandwich on tray at location
            for fact in state:
                if match(fact, 'ontray', '*', '*'):
                    s = get_parts(fact)[1]
                    tray = get_parts(fact)[2]
                    if any(match(f, 'at', tray, location) for f in state):
                        if is_allergic:
                            if any(match(f, 'no_gluten_sandwich', s) for f in state):
                                has_sandwich = True
                                break
                        else:
                            has_sandwich = True
                            break
            if has_sandwich:
                total += 1
                continue

            # Check for suitable sandwich in kitchen
            suitable_in_kitchen = False
            if is_allergic:
                for fact in state:
                    if match(fact, 'at_kitchen_sandwich', '*'):
                        s = get_parts(fact)[1]
                        if any(match(f, 'no_gluten_sandwich', s) for f in state):
                            suitable_in_kitchen = True
                            break
            else:
                suitable_in_kitchen = any(match(fact, 'at_kitchen_sandwich', '*') for fact in state)

            if suitable_in_kitchen:
                if trays_in_kitchen:
                    total += 3  # put_on_tray, move, serve
                else:
                    total += 4  # move tray to kitchen, put, move, serve
            else:
                if trays_in_kitchen:
                    total += 4  # make, put, move, serve
                else:
                    total += 5  # make, move tray, put, move, serve

        return total
