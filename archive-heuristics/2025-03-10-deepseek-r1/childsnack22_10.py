from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class childsnack22Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnack domain.

    # Summary
    This heuristic estimates the number of actions required to serve all unserved children by considering the steps needed to make sandwiches, place them on trays, move trays to the correct locations, and serve the children.

    # Assumptions
    - Each child requires their own sandwich.
    - Allergic children require gluten-free sandwiches, made from specific gluten-free bread and content portions.
    - Trays can be moved between locations in one action.
    - If a sandwich is on a tray not at the child's location, moving the tray takes one action.

    # Heuristic Initialization
    - Extracts static information about allergic children from the task's static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unserved child:
        a. Check if there's a suitable sandwich on a tray at their location.
        b. If not, check if a suitable sandwich exists on any tray (requires moving the tray).
        c. If no such sandwich exists, check if a sandwich can be made from ingredients in the kitchen.
        d. Account for moving trays to the kitchen if necessary.
    """

    def __init__(self, task):
        self.allergic_children = set()
        for fact in task.static:
            parts = fact.strip('()').split()
            if parts[0] == 'allergic_gluten':
                self.allergic_children.add(parts[1])

    def __call__(self, node):
        state = node.state
        served = set()
        waiting = {}
        trays_at = {}
        sandwich_on_tray = {}
        gluten_free_sandwiches = set()
        sandwiches_in_kitchen = set()

        for fact in state:
            parts = fact.strip('()').split()
            if parts[0] == 'served':
                served.add(parts[1])
            elif parts[0] == 'waiting':
                waiting[parts[1]] = parts[2]
            elif parts[0] == 'at' and len(parts) == 3 and parts[2] != 'kitchen':  # tray location
                trays_at[parts[1]] = parts[2]
            elif parts[0] == 'ontray':
                tray = parts[2]
                tray_loc = trays_at.get(tray, None)
                sandwich_on_tray[parts[1]] = (tray, tray_loc)
            elif parts[0] == 'no_gluten_sandwich':
                gluten_free_sandwiches.add(parts[1])
            elif parts[0] == 'at_kitchen_sandwich':
                sandwiches_in_kitchen.add(parts[1])

        unserved = [child for child in waiting if child not in served]
        cost = 0

        any_tray_at_kitchen = any(
            fact.strip('()').split() == ['at', tray, 'kitchen']
            for fact in state
            if fact.startswith('(at ')
        )

        for child in unserved:
            is_allergic = child in self.allergic_children
            child_loc = waiting[child]

            # Check if suitable sandwich is already on a tray at the child's location
            found_at_location = False
            if is_allergic:
                for s, (tray, tray_loc) in sandwich_on_tray.items():
                    if s in gluten_free_sandwiches and tray_loc == child_loc:
                        found_at_location = True
                        break
            else:
                for s, (tray, tray_loc) in sandwich_on_tray.items():
                    if tray_loc == child_loc:
                        found_at_location = True
                        break
            if found_at_location:
                continue

            # Check if suitable sandwich is on any tray (needs moving)
            found_on_tray = False
            if is_allergic:
                for s in gluten_free_sandwiches:
                    if s in sandwich_on_tray:
                        cost += 2  # move tray (1) and serve (1)
                        found_on_tray = True
                        break
            else:
                if sandwich_on_tray:
                    cost += 2
                    found_on_tray = True
            if found_on_tray:
                continue

            # Check if sandwich can be made from kitchen ingredients
            if is_allergic:
                gluten_free_in_kitchen = any(s in sandwiches_in_kitchen for s in gluten_free_sandwiches)
                if gluten_free_in_kitchen:
                    if any_tray_at_kitchen:
                        cost += 3  # put_on_tray (1), move (1), serve (1)
                    else:
                        cost += 4  # move tray to kitchen (1), put_on_tray (1), move (1), serve (1)
                else:
                    cost += 4  # make (1), put_on_tray (1), move (1), serve (1)
            else:
                any_in_kitchen = len(sandwiches_in_kitchen) > 0
                if any_in_kitchen:
                    if any_tray_at_kitchen:
                        cost += 3
                    else:
                        cost += 4
                else:
                    cost += 4

        return cost
