from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class childsnack3Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnack domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children with appropriate sandwiches, considering gluten-free requirements and tray logistics.

    # Assumptions
    - Each unserved child requires exactly one sandwich.
    - Gluten-free sandwiches can only be made using gluten-free bread and content.
    - Trays can carry multiple sandwiches, but each tray movement is counted once per destination.
    - The kitchen is the initial location for all trays unless moved.

    # Heuristic Initialization
    - Extract static information about allergic children and gluten-free ingredients.
    - Precompute mappings for gluten-free bread and content portions.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Identify Unserved Children**: Determine which children are unserved and classify them into allergic (gluten-free required) and non-allergic.
    2. **Count Existing Sandwiches**: Track current gluten-free and regular sandwiches.
    3. **Calculate Required Sandwiches**: Determine how many additional gluten-free and regular sandwiches need to be made.
    4. **Track Tray Placements**: Count sandwiches not yet on trays and those needing placement.
    5. **Determine Tray Movements**: Check if trays are needed at each unserved child's location.
    6. **Sum Actions**: Combine counts for making sandwiches, placing them on trays, moving trays, and serving.
    """

    def __init__(self, task):
        self.task = task
        self.static = task.static
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.no_gluten_breads = set()
        self.no_gluten_contents = set()

        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] == 'allergic_gluten':
                self.allergic_children.add(parts[1])
            elif parts[0] == 'not_allergic_gluten':
                self.not_allergic_children.add(parts[1])
            elif parts[0] == 'no_gluten_bread':
                self.no_gluten_breads.add(parts[1])
            elif parts[0] == 'no_gluten_content':
                self.no_gluten_contents.add(parts[1])

    def __call__(self, node):
        state = node.state
        current_state = set(state)

        # Identify unserved children and their requirements
        unserved_children = []
        required_gluten_free = 0
        required_regular = 0
        for child in self.allergic_children.union(self.not_allergic_children):
            if f'(served {child})' not in current_state:
                unserved_children.append(child)
                if child in self.allergic_children:
                    required_gluten_free += 1
                else:
                    required_regular += 1

        # Count existing sandwiches
        existing_sandwiches = 0
        current_gluten_sandwiches = 0
        sandwiches_on_trays = 0

        for fact in current_state:
            if fact.startswith('(at_kitchen_sandwich ') or fact.startswith('(ontray '):
                existing_sandwiches += 1
            if fact.startswith('(no_gluten_sandwich '):
                current_gluten_sandwiches += 1
            if fact.startswith('(ontray '):
                sandwiches_on_trays += 1

        current_regular_sandwiches = existing_sandwiches - current_gluten_sandwiches

        # Calculate required make actions
        make_actions_gluten = max(0, required_gluten_free - current_gluten_sandwiches)
        make_actions_regular = max(0, required_regular - current_regular_sandwiches)
        make_actions = make_actions_gluten + make_actions_regular

        # Calculate put_on_tray actions
        existing_not_on_tray = existing_sandwiches - sandwiches_on_trays
        put_on_tray_actions = existing_not_on_tray + make_actions

        # Determine locations needing tray movements
        locations_with_unserved = set()
        for child in unserved_children:
            for fact in current_state:
                parts = fact[1:-1].split()
                if parts[0] == 'waiting' and parts[1] == child:
                    locations_with_unserved.add(parts[2])
                    break

        move_actions = 0
        for loc in locations_with_unserved:
            has_tray = any(fact.startswith('(at ') and fact.endswith(f' {loc})') for fact in current_state)
            if not has_tray:
                move_actions += 1

        # Serve actions
        serve_actions = len(unserved_children)

        heuristic_value = make_actions + put_on_tray_actions + move_actions + serve_actions
        return heuristic_value
