from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

def match(fact, pattern):
    parts = get_parts(fact)
    pattern_parts = pattern.split()
    if len(parts) != len(pattern_parts):
        return False
    for p, pat in zip(parts, pattern_parts):
        if not fnmatch(p, pat):
            return False
    return True

class Childsnack15Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Childsnack domain.

    # Summary
    Estimates the number of actions required to serve all children by considering the steps needed to prepare, transport, and serve each required sandwich. The heuristic accounts for gluten-free requirements and optimizes steps based on current state.

    # Assumptions
    - Each child requires a sandwich, gluten-free for allergic children.
    - Trays start at the kitchen and can be moved to children's locations.
    - Sandwiches can be made if ingredients are available in the kitchen.

    # Heuristic Initialization
    - Extracts static information: allergic children, their waiting locations, and initial objects (bread, content, trays, sandwiches).
    - Tracks all possible objects from initial and static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unserved child:
        a. Allergic: Check for existing gluten-free sandwich on a tray at their location. If found, add 1 (serve).
        b. If not found, check existing gluten-free sandwiches. If on a tray elsewhere, add move and serve. If in kitchen, add put, move, serve.
        c. If no existing sandwich, check ingredients and add steps to make, put, move, serve.
    2. Non-allergic: Similar steps but using any sandwich.
    3. Sum all steps, assuming minimal per-child actions without shared tray optimizations.
    """

    def __init__(self, task):
        self.static = task.static
        self.goals = task.goals

        self.allergic_children = set()
        self.waiting_locations = {}
        self.children = set()
        self.breads = set()
        self.contents = set()
        self.trays = set()
        self.all_sandwiches = set()

        for fact in task.static | task.initial_state:
            parts = get_parts(fact)
            if not parts:
                continue
            if parts[0] == 'allergic_gluten':
                self.allergic_children.add(parts[1])
                self.children.add(parts[1])
            elif parts[0] == 'not_allergic_gluten':
                self.children.add(parts[1])
            elif parts[0] == 'waiting':
                child, place = parts[1], parts[2]
                self.waiting_locations[child] = place
                self.children.add(child)
            elif parts[0] == 'at_kitchen_bread':
                self.breads.add(parts[1])
            elif parts[0] == 'at_kitchen_content':
                self.contents.add(parts[1])
            elif parts[0] == 'at' and parts[1].startswith('tray'):
                self.trays.add(parts[1])
            elif parts[0] == 'notexist' and parts[1].startswith('sandw'):
                self.all_sandwiches.add(parts[1])

    def __call__(self, node):
        state = node.state
        served = set()
        existing_sandwiches = set()
        gluten_free_sandwiches = set()
        tray_locations = {}
        sandwiches_on_tray = {}
        notexist_sandwiches = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            if parts[0] == 'served':
                served.add(parts[1])
            elif parts[0] == 'no_gluten_sandwich':
                gluten_free_sandwiches.add(parts[1])
            elif parts[0] == 'ontray':
                s, tray = parts[1], parts[2]
                sandwiches_on_tray[s] = tray
            elif parts[0] == 'at' and parts[1] in self.trays:
                tray_locations[parts[1]] = parts[2]
            elif parts[0] == 'notexist':
                notexist_sandwiches.add(parts[1])

        existing_sandwiches = self.all_sandwiches - notexist_sandwiches
        total_cost = 0

        for child in self.children:
            if child in served:
                continue
            is_allergic = child in self.allergic_children
            waiting_place = self.waiting_locations.get(child, 'table1')

            if is_allergic:
                found = False
                for s in gluten_free_sandwiches:
                    tray = sandwiches_on_tray.get(s, None)
                    if tray and tray_locations.get(tray, 'kitchen') == waiting_place:
                        found = True
                        break
                if found:
                    total_cost += 1
                    continue

                possible = existing_sandwiches & gluten_free_sandwiches
                if possible:
                    on_tray = [s for s in possible if s in sandwiches_on_tray]
                    if on_tray:
                        total_cost += 2
                    else:
                        total_cost += 3
                else:
                    has_bread = any(
                        f'(at_kitchen_bread {b})' in state and f'(no_gluten_bread {b})' in self.static
                        for b in self.breads
                    )
                    has_content = any(
                        f'(at_kitchen_content {c})' in state and f'(no_gluten_content {c})' in self.static
                        for c in self.contents
                    )
                    total_cost += 4 if has_bread and has_content else 4
            else:
                found = any(
                    tray_locations.get(sandwiches_on_tray.get(s, ''), '') == waiting_place
                    for s in existing_sandwiches
                )
                if found:
                    total_cost += 1
                    continue

                if existing_sandwiches:
                    on_tray = [s for s in existing_sandwiches if s in sandwiches_on_tray]
                    if on_tray:
                        total_cost += 2
                    else:
                        total_cost += 3
                else:
                    has_bread = any(f'(at_kitchen_bread {b})' in state for b in self.breads)
                    has_content = any(f'(at_kitchen_content {c})' in state for c in self.contents)
                    total_cost += 4 if has_bread and has_content else 4

        return total_cost
