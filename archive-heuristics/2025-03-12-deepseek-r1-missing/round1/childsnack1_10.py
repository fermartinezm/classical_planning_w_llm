from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class ChildsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Childsnack domain.

    # Summary
    This heuristic estimates the number of actions required to serve all unserved children by considering:
    - Whether a suitable sandwich is already on a tray at the child's location.
    - Whether a suitable sandwich is on a tray in the kitchen.
    - Whether a suitable sandwich is available in the kitchen but not on a tray.
    - The need to make new sandwiches and move trays to the required locations.

    # Assumptions
    - Each child requires exactly one sandwich.
    - Allergic children require gluten-free sandwiches, which must be on a tray at their location.
    - Non-allergic children can be served any sandwich.
    - Trays must be moved to the child's location if not already there.
    - Sandwiches not on trays must be placed on a tray in the kitchen before being moved.

    # Heuristic Initialization
    - Extract allergic status of each child from static facts.
    - Determine each child's waiting location from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unserved child:
        a. Check if a suitable sandwich is on a tray at their location (1 action).
        b. If not, check if a suitable sandwich is on a tray in the kitchen (2 actions if move needed).
        c. If not, check if a suitable sandwich is in the kitchen (3 actions: put, move, serve).
        d. If no suitable sandwich exists, account for making it (4 actions: make, put, move, serve).
    2. Sum the estimated actions for all unserved children.
    """

    def __init__(self, task):
        self.allergic_children = set()
        self.child_locations = {}
        self.static = task.static

        # Extract allergic_gluten and waiting predicates from static facts
        for fact in self.static:
            parts = fact.strip('()').split()
            if parts[0] == 'allergic_gluten':
                self.allergic_children.add(parts[1])
            elif parts[0] == 'waiting':
                child = parts[1]
                location = parts[2]
                self.child_locations[child] = location

    def __call__(self, node):
        state = node.state
        served = set()
        for fact in state:
            if fact.startswith('(served '):
                parts = fact[1:-1].split()
                served.add(parts[1])

        unserved = [child for child in self.child_locations if child not in served]
        total = 0

        for child in unserved:
            is_allergic = child in self.allergic_children
            location = self.child_locations[child]

            # Case 1: Suitable sandwich on tray at location
            case1 = False
            for fact in state:
                if fact.startswith('(ontray '):
                    parts = fact[1:-1].split()
                    s, t = parts[1], parts[2]
                    tray_at_loc = f'(at {t} {location})' in state
                    if not tray_at_loc:
                        continue
                    if is_allergic:
                        if f'(no_gluten_sandwich {s})' in state:
                            case1 = True
                            break
                    else:
                        case1 = True
                        break
            if case1:
                total += 1
                continue

            # Case 2: Suitable sandwich on tray in kitchen
            case2 = False
            for fact in state:
                if fact.startswith('(ontray '):
                    parts = fact[1:-1].split()
                    s, t = parts[1], parts[2]
                    tray_at_kitchen = f'(at {t} kitchen)' in state
                    if not tray_at_kitchen:
                        continue
                    if is_allergic:
                        if f'(no_gluten_sandwich {s})' in state:
                            case2 = True
                            break
                    else:
                        case2 = True
                        break
            if case2:
                if location == 'kitchen':
                    total += 1
                else:
                    total += 2
                continue

            # Case 3: Suitable sandwich in kitchen (not on tray)
            case3 = False
            for fact in state:
                if fact.startswith('(at_kitchen_sandwich '):
                    parts = fact[1:-1].split()
                    s = parts[1]
                    if is_allergic:
                        if f'(no_gluten_sandwich {s})' in state:
                            case3 = True
                            break
                    else:
                        case3 = True
                        break
            if case3:
                if location == 'kitchen':
                    total += 2
                else:
                    total += 3
                continue

            # Case 4: No suitable sandwich, need to make
            if location == 'kitchen':
                total += 3
            else:
                total += 4

        return total
