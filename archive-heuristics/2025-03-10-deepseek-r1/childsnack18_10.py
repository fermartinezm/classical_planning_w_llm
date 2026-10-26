from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class Childsnack18Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    Estimates the number of actions needed to serve all children by considering steps to make, place, move, and serve sandwiches. Accounts for gluten-free requirements for allergic children.

    # Assumptions
    - Each child requires one sandwich.
    - Allergic children need gluten-free sandwiches made from available ingredients.
    - Trays start in the kitchen and can be moved directly to any place in one action.
    - The kitchen has sufficient ingredients for solvable problems.

    # Heuristic Initialization
    Extracts allergic children, their waiting places, gluten-free ingredients, and object types from static and initial state facts.

    # Step-By-Step Thinking
    1. For each unserved child:
        a. If allergic, check for existing gluten-free sandwich on a tray at their location.
        b. If not found, check for gluten-free sandwiches ready to be placed on trays.
        c. If none, calculate steps to make a new gluten-free sandwich and deliver it.
        d. For non-allergic children, repeat similar steps without gluten constraints.
    2. Sum actions for all children, considering optimal resource use and tray movements.
    """

    def __init__(self, task):
        self.static = task.static
        self.initial_state = task.initial_state

        self.children = set()
        self.breads = set()
        self.contents = set()
        self.sandwiches = set()
        self.trays = set()
        self.places = set()
        self.allergic_children = set()
        self.child_waiting_place = {}
        self.gluten_free_breads = set()
        self.gluten_free_contents = set()

        # Extract objects and static info
        for fact in self.static | self.initial_state:
            parts = fact[1:-1].split()
            predicate = parts[0]
            args = parts[1:]
            if predicate == 'allergic_gluten':
                self.allergic_children.add(args[0])
                self.children.add(args[0])
            elif predicate == 'not_allergic_gluten':
                self.children.add(args[0])
            elif predicate == 'waiting':
                child, place = args[0], args[1]
                self.children.add(child)
                self.child_waiting_place[child] = place
                self.places.add(place)
            elif predicate == 'no_gluten_bread':
                self.gluten_free_breads.add(args[0])
                self.breads.add(args[0])
            elif predicate == 'no_gluten_content':
                self.gluten_free_contents.add(args[0])
                self.contents.add(args[0])
            elif predicate == 'at_kitchen_bread':
                self.breads.add(args[0])
            elif predicate == 'at_kitchen_content':
                self.contents.add(args[0])
            elif predicate == 'at_kitchen_sandwich':
                self.sandwiches.add(args[0])
            elif predicate == 'ontray':
                self.sandwiches.add(args[0])
                self.trays.add(args[1])
            elif predicate == 'at':
                self.trays.add(args[0])
                self.places.add(args[1])
            elif predicate == 'notexist':
                self.sandwiches.add(args[0])

        self.places.add('kitchen')

    def __call__(self, node):
        state = node.state
        total_cost = 0

        served = set()
        for fact in state:
            if fact.startswith('(served '):
                served.add(fact.split()[1][:-1])

        unserved = self.children - served
        if not unserved:
            return 0

        # Precompute tray locations and sandwich info
        tray_locs = {t: 'kitchen' for t in self.trays}
        for fact in state:
            if fact.startswith('(at '):
                t, p = fact[1:-1].split()[1], fact[1:-1].split()[2]
                tray_locs[t] = p

        sandwich_on_tray = {}
        gluten_free = set()
        made_sandwiches = set()
        for s in self.sandwiches:
            on_tray = None
            for fact in state:
                if fact.startswith(f'(ontray {s} '):
                    on_tray = fact[1:-1].split()[2]
                    break
            sandwich_on_tray[s] = on_tray
            if f'(no_gluten_sandwich {s})' in state:
                gluten_free.add(s)
            if f'(at_kitchen_sandwich {s})' in state or on_tray is not None:
                made_sandwiches.add(s)

        for child in unserved:
            place = self.child_waiting_place[child]
            is_allergic = child in self.allergic_children

            if is_allergic:
                # Check existing gluten-free sandwich at place
                found = any(
                    s in gluten_free and sandwich_on_tray[s] is not None
                    and tray_locs[sandwich_on_tray[s]] == place
                    for s in self.sandwiches
                )
                if found:
                    total_cost += 1
                    continue

                # Check gluten-free sandwiches not on tray
                available = [s for s in gluten_free if s in made_sandwiches and sandwich_on_tray[s] is None]
                if available:
                    kitchen_trays = any(tray_locs[t] == 'kitchen' for t in self.trays)
                    cost = 3 if kitchen_trays else 4
                    total_cost += cost
                    continue

                # Make new gluten-free sandwich
                bread_available = any(f'(at_kitchen_bread {b})' in state for b in self.gluten_free_breads)
                content_available = any(f'(at_kitchen_content {c})' in state for c in self.gluten_free_contents)
                if bread_available and content_available:
                    kitchen_trays = any(tray_locs[t] == 'kitchen' for t in self.trays)
                    cost = 4 if kitchen_trays else 5
                    total_cost += cost
                else:
                    total_cost += 4  # Assume resources exist for solvable problems
            else:
                # Check any sandwich at place
                found = any(
                    sandwich_on_tray[s] is not None
                    and tray_locs[sandwich_on_tray[s]] == place
                    for s in made_sandwiches
                )
                if found:
                    total_cost += 1
                    continue

                # Check sandwiches not on tray
                available = [s for s in made_sandwiches if sandwich_on_tray[s] is None]
                if available:
                    kitchen_trays = any(tray_locs[t] == 'kitchen' for t in self.trays)
                    cost = 3 if kitchen_trays else 4
                    total_cost += cost
                    continue

                # Make new sandwich
                bread_available = any(f'(at_kitchen_bread {b})' in state for b in self.breads)
                content_available = any(f'(at_kitchen_content {c})' in state for c in self.contents)
                if bread_available and content_available:
                    kitchen_trays = any(tray_locs[t] == 'kitchen' for t in self.trays)
                    cost = 4 if kitchen_trays else 5
                    total_cost += cost
                else:
                    total_cost += 4  # Assume resources exist

        return total_cost
