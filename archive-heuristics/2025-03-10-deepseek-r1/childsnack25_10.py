from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class childsnack25Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnack domain.

    # Summary
    Estimates the number of actions needed to serve all children by considering the steps required to make, place, move, and serve sandwiches for each unserved child.

    # Assumptions
    - Each child requires a unique sandwich.
    - Allergic children need a no-gluten sandwich made from available no-gluten ingredients.
    - Trays can be moved to children's locations, each move taking one action.
    - Sandwiches not on trays must be placed on a tray before moving.
    - Making a sandwich consumes one bread and one content portion.

    # Heuristic Initialization
    - Extracts static information about allergic children and no-gluten ingredients from the task's static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unserved child:
        a. Determine if allergic to gluten.
        b. For allergic children:
            i. Check existing no-gluten sandwiches on trays at their location (cost 1).
            ii. Check existing no-gluten sandwiches on other trays (cost 2: move + serve).
            iii. Check existing no-gluten sandwiches in the kitchen (cost 3: place on tray, move, serve).
            iv. If no existing sandwich, check availability of no-gluten ingredients and add cost 4 (make, place, move, serve).
        c. For non-allergic children:
            i. Check any existing sandwich on trays at their location (cost 1).
            ii. Check existing sandwiches on other trays (cost 2).
            iii. Check existing sandwiches in the kitchen (cost 3).
            iv. If no existing sandwich, check availability of any ingredients and add cost 4.
    2. Sum the minimal costs for all unserved children.
    """

    def __init__(self, task):
        self.allergic_children = set()
        self.no_gluten_breads = set()
        self.no_gluten_contents = set()

        for fact in task.static:
            parts = fact[1:-1].split()
            if parts[0] == 'allergic_gluten':
                self.allergic_children.add(parts[1])
            elif parts[0] == 'no_gluten_bread':
                self.no_gluten_breads.add(parts[1])
            elif parts[0] == 'no_gluten_content':
                self.no_gluten_contents.add(parts[1])

    def __call__(self, node):
        state = node.state
        served = set()
        waiting = {}
        existing_sandwiches = set()
        no_gluten_sandwiches = set()

        for fact in state:
            if fact.startswith('(served '):
                parts = fact[1:-1].split()
                served.add(parts[1])
            elif fact.startswith('(waiting '):
                parts = fact[1:-1].split()
                waiting[parts[1]] = parts[2]
            elif fact.startswith('(no_gluten_sandwich '):
                s = fact[1:-1].split()[1]
                no_gluten_sandwiches.add(s)
                existing_sandwiches.add(s)
            elif fact.startswith('(at_kitchen_sandwich '):
                s = fact[1:-1].split()[1]
                existing_sandwiches.add(s)
            elif fact.startswith('(ontray '):
                s = fact[1:-1].split()[1]
                existing_sandwiches.add(s)

        unserved_children = [child for child in waiting if child not in served]
        available_no_gluten_breads = sum(1 for b in self.no_gluten_breads if f'(at_kitchen_bread {b})' in state)
        available_no_gluten_contents = sum(1 for c in self.no_gluten_contents if f'(at_kitchen_content {c})' in state)
        available_breads = sum(1 for fact in state if fact.startswith('(at_kitchen_bread '))
        available_contents = sum(1 for fact in state if fact.startswith('(at_kitchen_content '))

        total_cost = 0

        for child in unserved_children:
            child_loc = waiting[child]
            is_allergic = child in self.allergic_children
            required_sandwiches = no_gluten_sandwiches if is_allergic else existing_sandwiches
            min_cost = float('inf')

            for s in required_sandwiches:
                cost = self._compute_sandwich_cost(s, child_loc, state)
                if cost < min_cost:
                    min_cost = cost

            if min_cost == float('inf'):
                if is_allergic:
                    if available_no_gluten_breads >= 1 and available_no_gluten_contents >= 1:
                        min_cost = 4
                    else:
                        min_cost = 4
                else:
                    if available_breads >= 1 and available_contents >= 1:
                        min_cost = 4
                    else:
                        min_cost = 4

            total_cost += min_cost

        return total_cost

    def _compute_sandwich_cost(self, s, child_loc, state):
        tray = None
        tray_loc = None
        for fact in state:
            if fact.startswith('(ontray '):
                parts = fact[1:-1].split()
                if parts[1] == s:
                    tray = parts[2]
                    break
        if tray:
            for fact in state:
                if fact.startswith('(at ') and fact[1:-1].split()[1] == tray:
                    tray_loc = fact[1:-1].split()[2]
                    break
            if tray_loc == child_loc:
                return 1
            else:
                return 2
        else:
            if f'(at_kitchen_sandwich {s})' in state:
                return 3
        return float('inf')
