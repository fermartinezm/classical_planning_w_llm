from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

def match(fact, *args):
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnack23Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    Estimates the number of actions needed to serve all children by considering steps to make, place, move, and serve sandwiches. Accounts for gluten-free requirements and tray locations.

    # Assumptions
    - Allergic children require gluten-free sandwiches.
    - Trays can be moved between locations in one action.
    - Kitchen has necessary ingredients for solvable problems.

    # Heuristic Initialization
    - Extracts allergic/non-allergic children and their locations from static facts.
    - Identifies goal children from the task's goals.

    # Step-By-Step Thinking
    1. For each unserved child:
        a. Allergic: Check for gluten-free sandwich on tray at their location. If missing, check kitchen and ingredients, adding steps to make, place, move, and serve.
        b. Non-allergic: Check for any sandwich on tray at their location. If missing, check kitchen and ingredients, adding steps to make, place, move, and serve.
    2. Sum steps for all unserved children.
    """

    def __init__(self, task):
        self.static = task.static
        self.goals = task.goals

        self.allergic = set()
        self.non_allergic = set()
        self.child_loc = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'allergic_gluten':
                self.allergic.add(parts[1])
            elif parts[0] == 'not_allergic_gluten':
                self.non_allergic.add(parts[1])
            elif parts[0] == 'waiting':
                self.child_loc[parts[1]] = parts[2]

        self.goal_children = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'served':
                self.goal_children.add(parts[1])

    def __call__(self, node):
        state = node.state
        cost = 0

        unserved = [c for c in self.goal_children if f'(served {c})' not in state]

        for child in unserved:
            loc = self.child_loc[child]
            if child in self.allergic:
                # Check for gluten-free sandwich on tray at location
                found = any(
                    match(f, 'ontray', s, '*') and f'(no_gluten_sandwich {s})' in state and f'(at {t} {loc})' in state
                    for f in state for s, t in [(get_parts(f)[1], get_parts(f)[2])] if match(f, 'ontray', '*', '*')
                )
                if found:
                    cost += 1
                    continue

                # Check kitchen for gluten-free sandwich
                sandwich_kitchen = [
                    s for f in state if match(f, 'at_kitchen_sandwich', s)
                    and f'(no_gluten_sandwich {s})' in state
                ]
                if sandwich_kitchen:
                    tray_kitchen = any(match(f, 'at', t, 'kitchen') for f in state for t in [get_parts(f)[1]])
                    cost += 3 if tray_kitchen else 4
                    continue

                # Check gluten-free ingredients
                breads = [
                    b for f in state if match(f, 'at_kitchen_bread', b)
                    and f'(no_gluten_bread {b})' in state
                ]
                contents = [
                    c for f in state if match(f, 'at_kitchen_content', c)
                    and f'(no_gluten_content {c})' in state
                ]
                if breads and contents:
                    tray_kitchen = any(match(f, 'at', t, 'kitchen') for f in state for t in [get_parts(f)[1]])
                    cost += 4 if tray_kitchen else 5
                else:
                    cost += 1000
            else:
                # Check for any sandwich on tray at location
                found = any(
                    match(f, 'ontray', s, t) and f'(at {t} {loc})' in state
                    for f in state for t in [get_parts(f)[2]]
                )
                if found:
                    cost += 1
                    continue

                # Check kitchen for any sandwich
                sandwich_kitchen = any(match(f, 'at_kitchen_sandwich', s) for f in state)
                if sandwich_kitchen:
                    tray_kitchen = any(match(f, 'at', t, 'kitchen') for f in state)
                    cost += 3 if tray_kitchen else 4
                    continue

                # Check regular ingredients
                breads = any(match(f, 'at_kitchen_bread', b) for f in state)
                contents = any(match(f, 'at_kitchen_content', c) for f in state)
                if breads and contents:
                    tray_kitchen = any(match(f, 'at', t, 'kitchen') for f in state)
                    cost += 4 if tray_kitchen else 5
                else:
                    cost += 1000

        return cost
