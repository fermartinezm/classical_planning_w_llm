import fnmatch
from heuristics.heuristic_base import Heuristic

class childsnack24Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    Estimates the number of actions needed to serve all children by considering the steps to make, place, move, and serve sandwiches. Accounts for gluten-free requirements and tray movements.

    # Assumptions
    - Each child requires one sandwich.
    - Gluten-free sandwiches are available if needed.
    - Trays can be moved directly to any location in one action.
    - Making a sandwich (gluten-free or regular) takes one action.

    # Heuristic Initialization
    - Extracts allergic and non-allergic children from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unserved child:
        a. Determine if they need a gluten-free sandwich.
        b. Check for existing suitable sandwiches on a tray at their location (cost 1).
        c. If not found, check for sandwiches on trays elsewhere (cost 2).
        d. If not found, check for sandwiches in the kitchen (cost 2 or 3).
        e. If no suitable sandwich exists, add cost to make, place, move, and serve (cost 3 or 4).
    2. Sum costs for all unserved children.
    """

    def __init__(self, task):
        self.allergic_children = set()
        self.non_allergic_children = set()
        for fact in task.static:
            parts = fact[1:-1].split()
            if parts[0] == 'allergic_gluten':
                self.allergic_children.add(parts[1])
            elif parts[0] == 'not_allergic_gluten':
                self.non_allergic_children.add(parts[1])
        self.all_children = self.allergic_children.union(self.non_allergic_children)

    def __call__(self, node):
        state = node.state
        served = set()
        waiting = {}
        ontray = {}
        at_tray = {}
        at_kitchen_sandwiches = set()
        no_gluten_sandwiches = set()
        notexist_sandwiches = set()

        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'served':
                served.add(parts[1])
            elif parts[0] == 'waiting':
                waiting[parts[1]] = parts[2]
            elif parts[0] == 'ontray':
                ontray[parts[1]] = parts[2]
            elif parts[0] == 'at' and parts[1].startswith('tray'):
                at_tray[parts[1]] = parts[2]
            elif parts[0] == 'at_kitchen_sandwich':
                at_kitchen_sandwiches.add(parts[1])
            elif parts[0] == 'no_gluten_sandwich':
                no_gluten_sandwiches.add(parts[1])
            elif parts[0] == 'notexist':
                notexist_sandwiches.add(parts[1])

        existing_sandwiches = set()
        for s in ontray:
            if s not in notexist_sandwiches:
                existing_sandwiches.add(s)
        existing_sandwiches.update(s for s in at_kitchen_sandwiches if s not in notexist_sandwiches)
        existing_sandwiches.update(s for s in no_gluten_sandwiches if s not in notexist_sandwiches)

        total_cost = 0

        for child in self.all_children:
            if child in served or child not in waiting:
                continue
            p = waiting[child]
            is_allergic = child in self.allergic_children

            suitable = []
            if is_allergic:
                suitable = [s for s in existing_sandwiches if s in no_gluten_sandwiches]
            else:
                suitable = list(existing_sandwiches)

            found = False
            for s in suitable:
                tray = ontray.get(s)
                if tray and at_tray.get(tray) == p:
                    total_cost += 1
                    found = True
                    break
            if found:
                continue

            for s in suitable:
                if s in ontray:
                    total_cost += 2
                    found = True
                    break
            if found:
                continue

            for s in suitable:
                if s in at_kitchen_sandwiches:
                    total_cost += 2 if p == 'kitchen' else 3
                    found = True
                    break
            if found:
                continue

            total_cost += 3 if p == 'kitchen' else 4

        return total_cost
