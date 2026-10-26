from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class childsnack8Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnack domain.

    Summary:
    This heuristic estimates the number of actions required to serve all children by considering:
    - The need to make gluten-free sandwiches for allergic children.
    - The current locations of sandwiches (on trays or at the kitchen).
    - The movement of trays to children's waiting locations.
    - Serving actions for each child.

    Assumptions:
    - Each child requires a unique sandwich (once served, the sandwich is removed from the tray).
    - Moving a tray to a location allows all sandwiches on it to be served there (but each sandwich can only be used once).
    - The kitchen is the initial location for all trays unless moved.

    Heuristic Initialization:
    - Extracts static information about allergic children and their waiting locations.
    - Preprocesses goal conditions to determine which children need to be served.

    Step-By-Step Thinking for Computing Heuristic:
    1. For each child not yet served:
        a. Determine if they need a gluten-free sandwich.
        b. Check all existing sandwiches to find the minimum steps to serve them:
            i. If a suitable sandwich is on a tray at their location: 1 action (serve).
            ii. If a suitable sandwich is on a tray elsewhere: 2 actions (move tray and serve).
            iii. If a suitable sandwich is at the kitchen (not on a tray): 3 actions (put on tray, move tray, serve).
        c. If no suitable sandwich exists: 4 actions (make, put on tray, move tray, serve).
    2. Sum the minimum steps for all unserved children.
    """

    def __init__(self, task):
        self.static = task.static
        self.allergic_children = set()
        self.child_locations = {}

        # Extract allergic children and waiting locations from static facts
        for fact in self.static:
            parts = fact.strip('()').split()
            if parts[0] == 'allergic_gluten':
                self.allergic_children.add(parts[1])
            elif parts[0] == 'waiting':
                child = parts[1]
                place = parts[2]
                self.child_locations[child] = place

    def __call__(self, node):
        state = node.state
        total = 0

        # Collect existing sandwiches and their details
        existing_sandwiches = set()
        sandwich_info = {}

        for fact in state:
            parts = fact.strip('()').split()
            if not parts:
                continue
            if parts[0] in ['at_kitchen_sandwich', 'ontray', 'no_gluten_sandwich']:
                s = parts[1]
                existing_sandwiches.add(s)

        for s in existing_sandwiches:
            gluten_free = f'(no_gluten_sandwich {s})' in state
            on_tray = False
            tray = None
            tray_location = None

            # Check if the sandwich is on a tray
            for fact in state:
                if fact.startswith(f'(ontray {s} '):
                    parts = fact.strip('()').split()
                    tray = parts[2]
                    on_tray = True
                    break

            if on_tray:
                # Find the tray's current location
                for fact in state:
                    if fact.startswith(f'(at {tray} '):
                        parts = fact.strip('()').split()
                        tray_location = parts[2]
                        break
            else:
                # Check if the sandwich is at the kitchen
                if f'(at_kitchen_sandwich {s})' in state:
                    tray_location = 'kitchen'

            sandwich_info[s] = {
                'gluten_free': gluten_free,
                'tray_location': tray_location,
                'on_tray': on_tray
            }

        # Process each child
        for child, location in self.child_locations.items():
            if f'(served {child})' in state:
                continue

            is_allergic = child in self.allergic_children
            min_steps = 4  # Default to making a new sandwich

            for s, info in sandwich_info.items():
                if is_allergic and not info['gluten_free']:
                    continue  # Not suitable for allergic child

                if info['tray_location'] == location:
                    steps = 1
                elif info['on_tray']:
                    steps = 2  # Move tray and serve
                elif info['tray_location'] == 'kitchen':
                    steps = 3  # Put on tray, move, serve
                else:
                    continue  # Invalid state, treat as requiring new sandwich

                if steps < min_steps:
                    min_steps = steps

            total += min_steps

        return total
