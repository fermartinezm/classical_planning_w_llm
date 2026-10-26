from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

class childsnack11Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    Estimates the number of actions needed to serve all children by considering:
    - Making required sandwiches (1 action per sandwich).
    - Placing sandwiches on trays (1 action per sandwich).
    - Moving trays to children's locations (1 action per tray per location).
    - Serving each child (1 action per child).

    # Assumptions
    - Each child requires a unique sandwich.
    - Allergic children need a no-gluten sandwich, which requires specific ingredients assumed to be available.
    - Trays start in the kitchen and must be moved to the child's location if not already there.

    # Heuristic Initialization
    - Extracts allergic children and their waiting locations from static facts.
    - Tracks which children are served via the current state.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unserved child:
        a. If allergic, check for a no-gluten sandwich on a tray in their location.
        b. If found, add 1 (serve).
        c. Else, check if such a sandwich is on any tray (add 2: move and serve).
        d. Else, check if the sandwich is made but not on a tray (add 3: put, move, serve).
        e. Else, add 4: make, put, move, serve.
    2. For non-allergic children, similar steps using any available sandwich.
    """

    def __init__(self, task):
        self.static = task.static
        self.allergic_children = set()
        self.child_location = {}
        self.children = set()

        # Extract allergic children and their locations from static facts
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'allergic_gluten':
                self.allergic_children.add(parts[1])
                self.children.add(parts[1])
            elif parts[0] == 'not_allergic_gluten':
                self.children.add(parts[1])
            elif parts[0] == 'waiting':
                child = parts[1]
                loc = parts[2]
                self.child_location[child] = loc

    def __call__(self, node):
        state = node.state
        total_cost = 0

        # Collect served children
        served = set()
        for fact in state:
            if fact.startswith('(served '):
                parts = get_parts(fact)
                served.add(parts[1])

        # Collect existing sandwiches and their details
        existing_sandwiches = set()
        no_gluten_sandwiches = set()
        sandwich_trays = {}
        tray_locations = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at_kitchen_sandwich':
                s = parts[1]
                existing_sandwiches.add(s)
            elif parts[0] == 'ontray':
                s, t = parts[1], parts[2]
                existing_sandwiches.add(s)
                sandwich_trays[s] = t
            elif parts[0] == 'no_gluten_sandwich':
                s = parts[1]
                no_gluten_sandwiches.add(s)
            elif parts[0] == 'at' and len(parts) == 3:
                t, loc = parts[1], parts[2]
                tray_locations[t] = loc

        # Process each unserved child
        for child in self.children:
            if child in served:
                continue
            is_allergic = child in self.allergic_children
            loc = self.child_location[child]

            if is_allergic:
                possible = no_gluten_sandwiches & existing_sandwiches
            else:
                possible = existing_sandwiches

            # Check if any possible sandwich is on a tray in the child's location
            found = False
            for s in possible:
                tray = sandwich_trays.get(s)
                if tray and tray_locations.get(tray) == loc:
                    found = True
                    break
            if found:
                total_cost += 1
                continue

            # Check if any possible sandwich is on any tray
            on_tray = any(s in sandwich_trays for s in possible)
            if on_tray:
                total_cost += 2
                continue

            # Check if any possible sandwich is in the kitchen
            in_kitchen = any(f'(at_kitchen_sandwich {s})' in state for s in possible)
            if in_kitchen:
                total_cost += 3
            else:
                # Need to make the sandwich
                total_cost += 4

        return total_cost
