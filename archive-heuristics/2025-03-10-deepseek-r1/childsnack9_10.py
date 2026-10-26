from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class childsnack9Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    Estimates the number of actions needed to serve all children by considering sandwich preparation, tray placement, movement, and serving. Accounts for gluten-free requirements and optimizes steps.

    # Assumptions
    - Gluten-free sandwiches are required for allergic children.
    - Trays can be moved between locations in one action.
    - Required ingredients are available for solvable problems.

    # Heuristic Initialization
    Extracts static information about allergies and gluten-free ingredients.

    # Step-By-Step Thinking
    1. For each unserved child:
        a. Check if a suitable sandwich is on a tray at their location.
        b. If not, check trays elsewhere or sandwiches in the kitchen.
        c. Calculate costs for moving trays, making new sandwiches, and serving.
    2. Sum the minimal costs for all unserved children.
    """

    def __init__(self, task):
        self.allergic_children = set()
        self.gluten_free_breads = set()
        self.gluten_free_contents = set()

        for fact in task.static:
            parts = fact[1:-1].split()
            if parts[0] == 'allergic_gluten':
                self.allergic_children.add(parts[1])
            elif parts[0] == 'no_gluten_bread':
                self.gluten_free_breads.add(parts[1])
            elif parts[0] == 'no_gluten_content':
                self.gluten_free_contents.add(parts[1])

    def __call__(self, node):
        state = node.state
        total_cost = 0

        # Served children
        served = set()
        for fact in state:
            if fact.startswith('(served '):
                child = fact[1:-1].split()[1]
                served.add(child)

        # Unserved children and their locations
        unserved = []
        for fact in state:
            if fact.startswith('(waiting '):
                parts = fact[1:-1].split()
                child, loc = parts[1], parts[2]
                if child not in served:
                    unserved.append((child, loc))

        # Existing sandwiches (ontray or at_kitchen_sandwich)
        existing_sandwiches = set()
        for fact in state:
            if fact.startswith('(ontray ') or fact.startswith('(at_kitchen_sandwich '):
                s = fact[1:-1].split()[1]
                existing_sandwiches.add(s)

        # Tray at kitchen
        tray_at_kitchen = any(fact.startswith('(at ') and fact.endswith(' kitchen)') for fact in state)

        for child, loc in unserved:
            is_allergic = child in self.allergic_children
            min_cost = float('inf')

            # Check existing sandwiches
            for s in existing_sandwiches:
                if is_allergic and f'(no_gluten_sandwich {s})' not in state:
                    continue

                # Check if on tray
                on_tray = False
                tray = None
                for fact in state:
                    if fact.startswith(f'(ontray {s} '):
                        tray = fact[1:-1].split()[2]
                        on_tray = True
                        break

                if on_tray:
                    # Tray location
                    tray_loc = None
                    for fact in state:
                        if fact.startswith(f'(at {tray} '):
                            tray_loc = fact[1:-1].split()[2]
                            break
                    if tray_loc == loc:
                        cost = 1  # Serve
                    else:
                        cost = 2  # Move + serve
                else:
                    # In kitchen
                    if tray_at_kitchen:
                        cost = 3  # Put + move + serve
                    else:
                        cost = 4  # Move tray to kitchen + put + move + serve

                if cost < min_cost:
                    min_cost = cost

            # If no suitable sandwich, make new
            if min_cost == float('inf'):
                make_cost = 1
                if tray_at_kitchen:
                    put_cost = 1
                else:
                    put_cost = 2  # Move tray to kitchen + put
                move_cost = 1 if loc != 'kitchen' else 0
                total_new_cost = make_cost + put_cost + move_cost + 1  # serve
                min_cost = total_new_cost

            total_cost += min_cost

        return total_cost
