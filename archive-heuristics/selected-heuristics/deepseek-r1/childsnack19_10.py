from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class childsnack19Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children by considering:
    - The number of gluten-free and regular sandwiches needed.
    - The steps to make and place sandwiches on trays.
    - The movement of trays to required locations.
    - The serving actions for each child.

    # Assumptions
    - Allergic children require gluten-free sandwiches, others can have any type.
    - Each sandwich must be placed on a tray before serving.
    - Trays start at the kitchen and must be moved to children's locations if not already present.

    # Heuristic Initialization
    - Extracts static information about allergic and non-allergic children from the task's static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify unserved children and classify them into allergic and non-allergic.
    2. Count existing gluten-free and regular sandwiches (both on trays and in the kitchen).
    3. Calculate the number of additional gluten-free and regular sandwiches needed.
    4. Determine the number of trays required to be moved to children's locations.
    5. Sum the costs:
       - 2 actions per needed sandwich (make and place on tray).
       - 1 action per existing sandwich not on a tray (place on tray).
       - 1 action per tray movement to a required location.
       - 1 action per child to serve.
    """

    def __init__(self, task):
        self.static = task.static
        self.allergic_children = set()
        self.non_allergic_children = set()

        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] == 'allergic_gluten':
                self.allergic_children.add(parts[1])
            elif parts[0] == 'not_allergic_gluten':
                self.non_allergic_children.add(parts[1])

    def __call__(self, node):
        state = node.state
        served = set()
        unserved_allergic = 0
        unserved_non_allergic = 0

        # Identify served children
        for fact in state:
            if fact.startswith('(served '):
                parts = fact[1:-1].split()
                served.add(parts[1])

        # Count unserved allergic and non-allergic children
        unserved_allergic = sum(1 for c in self.allergic_children if c not in served)
        unserved_non_allergic = sum(1 for c in self.non_allergic_children if c not in served)
        total_unserved = unserved_allergic + unserved_non_allergic

        # Count existing sandwiches
        gluten_free = 0
        total_sandwiches = 0
        kitchen_sandwiches = 0

        for fact in state:
            if fact.startswith('(no_gluten_sandwich '):
                gluten_free += 1
            if fact.startswith('(ontray ') or fact.startswith('(at_kitchen_sandwich '):
                total_sandwiches += 1
            if fact.startswith('(at_kitchen_sandwich '):
                kitchen_sandwiches += 1

        regular = total_sandwiches - gluten_free

        # Calculate needed sandwiches
        needed_gluten = max(0, unserved_allergic - gluten_free)
        needed_regular = max(0, unserved_non_allergic - regular)
        make_cost = (needed_gluten + needed_regular) * 2

        # Existing sandwiches needing placement
        place_cost = kitchen_sandwiches

        # Determine tray movement needed
        locations = set()
        tray_locations = set()

        for fact in state:
            if fact.startswith('(waiting '):
                parts = fact[1:-1].split()
                if parts[1] not in served:
                    locations.add(parts[2])
            if fact.startswith('(at tray'):
                parts = fact[1:-1].split()
                tray_locations.add(parts[2])

        move_cost = len([loc for loc in locations if loc not in tray_locations])

        # Total heuristic
        return make_cost + place_cost + move_cost + total_unserved
