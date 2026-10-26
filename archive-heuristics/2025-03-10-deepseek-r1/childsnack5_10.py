from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class childsnack5Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children by considering:
    - Whether each child is allergic to gluten and requires a gluten-free sandwich.
    - The current availability of suitable sandwiches on trays at the child's location.
    - The need to prepare new sandwiches and move trays if necessary.

    # Assumptions
    - Each child requires one sandwich.
    - Gluten-free children need a sandwich made with no_gluten_bread and no_gluten_content.
    - Trays can be moved to serve multiple children at the same location, but the heuristic counts each move per child for simplicity.

    # Heuristic Initialization
    - Extracts static information about which children are allergic to gluten.
    - Identifies which bread and content portions are gluten-free from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unserved child:
        a. Determine if they are allergic to gluten.
        b. Check if a suitable sandwich is already on a tray at their location.
        c. If not, check if a suitable sandwich is available in the kitchen and needs to be placed on a tray.
        d. If no suitable sandwich exists, check if gluten-free ingredients are available (for allergic children) or any ingredients (for others) to make a new sandwich.
        e. Sum the estimated actions for making, placing, moving, and serving the sandwich.
    """

    def __init__(self, task):
        self.allergic = {}  # Maps each child to True/False for gluten allergy
        self.static_no_gluten_breads = set()
        self.static_no_gluten_contents = set()
        self.children = set()

        # Extract static information from the task
        for fact in task.static:
            parts = fact[1:-1].split()
            if parts[0] == 'allergic_gluten':
                child = parts[1]
                self.children.add(child)
                self.allergic[child] = True
            elif parts[0] == 'not_allergic_gluten':
                child = parts[1]
                self.children.add(child)
                self.allergic[child] = False
            elif parts[0] == 'no_gluten_bread':
                self.static_no_gluten_breads.add(parts[1])
            elif parts[0] == 'no_gluten_content':
                self.static_no_gluten_contents.add(parts[1])

    def __call__(self, node):
        state = node.state
        total_cost = 0

        # Collect unserved children
        unserved_children = [c for c in self.children if f'(served {c})' not in state]

        for child in unserved_children:
            # Extract the child's waiting place
            waiting_facts = [f for f in state if f.startswith(f'(waiting {child} ')]
            if not waiting_facts:
                continue  # Should not happen in valid states
            waiting_parts = waiting_facts[0][1:-1].split()
            place = waiting_parts[2]

            if self.allergic[child]:
                # Check for existing gluten-free sandwich on tray at the child's place
                suitable_sandwich = False
                existing_sandwiches = []
                for fact in state:
                    if fact.startswith(('(no_gluten_sandwich ', '(ontray ', '(at_kitchen_sandwich ')):
                        parts = fact[1:-1].split()
                        s = parts[1]
                        if f'(notexist {s})' not in state:
                            existing_sandwiches.append(s)
                for s in existing_sandwiches:
                    if f'(no_gluten_sandwich {s})' not in state:
                        continue
                    # Check if on a tray at the correct place
                    tray_facts = [f for f in state if f.startswith(f'(ontray {s} ')]
                    if tray_facts:
                        t = tray_facts[0][1:-1].split()[2]
                        if f'(at {t} {place})' in state:
                            suitable_sandwich = True
                            break
                if suitable_sandwich:
                    total_cost += 1
                    continue

                # Check if gluten-free sandwich is in the kitchen
                in_kitchen = any(
                    f'(at_kitchen_sandwich {s})' in state and f'(no_gluten_sandwich {s})' in state
                    for s in existing_sandwiches
                )
                if in_kitchen:
                    total_cost += 3  # put_on_tray, move_tray, serve
                    continue

                # Check availability of gluten-free ingredients
                current_breads = [parts[1] for fact in state if fact.startswith('(at_kitchen_bread ') for parts in [fact[1:-1].split()]]
                current_contents = [parts[1] for fact in state if fact.startswith('(at_kitchen_content ') for parts in [fact[1:-1].split()]]
                gluten_free_breads = [b for b in current_breads if b in self.static_no_gluten_breads]
                gluten_free_contents = [c for c in current_contents if c in self.static_no_gluten_contents]

                if gluten_free_breads and gluten_free_contents:
                    total_cost += 4  # make_sandwich, put_on_tray, move_tray, serve
                else:
                    total_cost += 4  # Assume resources are available for solvable problems
            else:
                # Non-allergic child
                has_sandwich = False
                existing_sandwiches = []
                for fact in state:
                    if fact.startswith(('(ontray ', '(at_kitchen_sandwich ')):
                        parts = fact[1:-1].split()
                        s = parts[1]
                        if f'(notexist {s})' not in state:
                            existing_sandwiches.append(s)
                for s in existing_sandwiches:
                    tray_facts = [f for f in state if f.startswith(f'(ontray {s} ')]
                    if tray_facts:
                        t = tray_facts[0][1:-1].split()[2]
                        if f'(at {t} {place})' in state:
                            has_sandwich = True
                            break
                if has_sandwich:
                    total_cost += 1
                    continue

                # Check if any sandwich is in the kitchen
                in_kitchen = any(f'(at_kitchen_sandwich {s})' in state for s in existing_sandwiches)
                if in_kitchen:
                    total_cost += 3  # put_on_tray, move_tray, serve
                    continue

                # Check availability of any bread and content
                current_breads = [parts[1] for fact in state if fact.startswith('(at_kitchen_bread ') for parts in [fact[1:-1].split()]]
                current_contents = [parts[1] for fact in state if fact.startswith('(at_kitchen_content ') for parts in [fact[1:-1].split()]]
                if current_breads and current_contents:
                    total_cost += 4  # make_sandwich, put_on_tray, move_tray, serve
                else:
                    total_cost += 4  # Assume resources are available

        return total_cost
