from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class childsnack12Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all unserved children by considering the steps needed to make, place, move, and serve sandwiches for each child.

    # Assumptions
    - Allergic children require a no-gluten sandwich, which must be made with available no-gluten bread and content.
    - Non-allergic children can be served with any sandwich.
    - Trays can be moved between locations, requiring one action per move.
    - Sandwiches can be made if their required ingredients are available in the kitchen.
    - Optimistic assumptions are made about resource availability to ensure the heuristic is efficiently computable.

    # Heuristic Initialization
    - Extract static information about allergic children, no-gluten breads, and no-gluten contents.
    - Track which children are allergic and which breads/contents are no-gluten.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unserved child:
        a. Determine if the child is allergic or not.
        b. For allergic children:
            i. Check if a no-gluten sandwich is already on a tray at their location (1 action).
            ii. If not, check if a no-gluten sandwich exists on any tray (move tray + serve = 2 actions).
            iii. If not, check if a no-gluten sandwich is in the kitchen (put on tray, move, serve = 3-4 actions).
            iv. If not, check if ingredients are available to make a new no-gluten sandwich (make, put, move, serve = 4 actions).
        c. For non-allergic children:
            i. Check if any sandwich is on a tray at their location (1 action).
            ii. If not, check if any sandwich exists on any tray (move tray + serve = 2 actions).
            iii. If not, check if any sandwich is in the kitchen (put on tray, move, serve = 3-4 actions).
            iv. If not, check if any ingredients are available to make a new sandwich (make, put, move, serve = 4 actions).
    2. Sum the estimated actions for all unserved children.
    """

    def __init__(self, task):
        self.allergic_children = set()
        self.non_allergic_children = set()
        self.no_gluten_breads = set()
        self.no_gluten_contents = set()

        # Extract static information
        for fact in task.static:
            parts = fact[1:-1].split()
            if parts[0] == 'allergic_gluten':
                self.allergic_children.add(parts[1])
            elif parts[0] == 'not_allergic_gluten':
                self.non_allergic_children.add(parts[1])
            elif parts[0] == 'no_gluten_bread':
                self.no_gluten_breads.add(parts[1])
            elif parts[0] == 'no_gluten_content':
                self.no_gluten_contents.add(parts[1])

    def __call__(self, node):
        state = node.state
        total_cost = 0

        # Precompute existing sandwiches and their locations
        existing_sandwiches = set()
        ontray_sandwiches = {}
        trays_at_p = {}
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] in ['at_kitchen_sandwich', 'ontray', 'no_gluten_sandwich']:
                existing_sandwiches.add(parts[1])
            if parts[0] == 'ontray' and len(parts) == 3:
                s, t = parts[1], parts[2]
                ontray_sandwiches.setdefault(s, set()).add(t)
            if parts[0] == 'at' and len(parts) == 3:
                tray, p = parts[1], parts[2]
                trays_at_p.setdefault(p, set()).add(tray)

        # Precompute available ingredients
        available_no_gluten_breads = [b for b in self.no_gluten_breads if f'(at_kitchen_bread {b})' in state]
        available_no_gluten_contents = [c for c in self.no_gluten_contents if f'(at_kitchen_content {c})' in state]
        available_any_breads = [parts[1] for fact in state if fact.startswith('(at_kitchen_bread ')]
        available_any_contents = [parts[1] for fact in state if fact.startswith('(at_kitchen_content ')]

        # Collect unserved children
        unserved_children = []
        for child in self.allergic_children.union(self.non_allergic_children):
            if f'(served {child})' not in state:
                unserved_children.append(child)

        for child in unserved_children:
            # Get child's location
            child_p = None
            for fact in state:
                if fact.startswith(f'(waiting {child} '):
                    parts = fact[1:-1].split()
                    child_p = parts[2]
                    break
            if not child_p:
                continue  # Should not happen in valid states

            if child in self.allergic_children:
                # Check for existing no-gluten sandwich at child's location
                found = False
                for s in existing_sandwiches:
                    if f'(no_gluten_sandwich {s})' not in state:
                        continue
                    for t in ontray_sandwiches.get(s, []):
                        if t in trays_at_p.get(child_p, set()):
                            total_cost += 1
                            found = True
                            break
                    if found:
                        break
                if found:
                    continue

                # Check if any no-gluten sandwich on any tray
                has_ontray = any(
                    f'(no_gluten_sandwich {s})' in state and ontray_sandwiches.get(s, set())
                    for s in existing_sandwiches
                )
                if has_ontray:
                    total_cost += 2
                    continue

                # Check if no-gluten sandwich in kitchen
                has_kitchen_sandwich = any(
                    f'(at_kitchen_sandwich {s})' in state and f'(no_gluten_sandwich {s})' in state
                    for s in existing_sandwiches
                )
                if has_kitchen_sandwich:
                    # Check if any tray is at kitchen
                    if trays_at_p.get('kitchen', set()):
                        total_cost += 3  # put_on_tray, move, serve
                    else:
                        total_cost += 4  # move tray to kitchen, put_on_tray, move, serve
                    continue

                # Check if ingredients are available to make new no-gluten sandwich
                if available_no_gluten_breads and available_no_gluten_contents:
                    total_cost += 4  # make, put, move, serve
                else:
                    total_cost += 4  # optimistic assumption
            else:
                # Non-allergic child
                found = False
                for s in existing_sandwiches:
                    for t in ontray_sandwiches.get(s, []):
                        if t in trays_at_p.get(child_p, set()):
                            total_cost += 1
                            found = True
                            break
                    if found:
                        break
                if found:
                    continue

                # Check if any sandwich on any tray
                has_ontray = any(ontray_sandwiches.get(s, set()) for s in existing_sandwiches)
                if has_ontray:
                    total_cost += 2
                    continue

                # Check if any sandwich in kitchen
                has_kitchen_sandwich = any(f'(at_kitchen_sandwich {s})' in state for s in existing_sandwiches)
                if has_kitchen_sandwich:
                    if trays_at_p.get('kitchen', set()):
                        total_cost += 3
                    else:
                        total_cost += 4
                    continue

                # Check if any ingredients are available
                if available_any_breads and available_any_contents:
                    total_cost += 4
                else:
                    total_cost += 4  # optimistic assumption

        return total_cost
