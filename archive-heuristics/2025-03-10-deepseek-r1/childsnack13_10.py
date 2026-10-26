from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

def match(fact, pattern):
    parts = get_parts(fact)
    pattern_parts = pattern.split()
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern_parts))

class childsnack13Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnack domain.

    # Summary
    This heuristic estimates the number of actions required to serve all unserved children by considering:
    - Immediate serving of children with suitable sandwiches already on trays at their location.
    - Making new sandwiches for remaining children, putting them on trays, moving trays to the required locations, and serving.

    # Assumptions
    - Each child requires a unique sandwich.
    - Allergic children need gluten-free sandwiches, which require specific bread and content.
    - Trays start at the kitchen and must be moved to other locations if needed.
    - A tray can carry multiple sandwiches, but each sandwich requires a separate put action.

    # Heuristic Initialization
    - Extracts allergic and non-allergic children from static facts.
    - Identifies all possible sandwiches and trays from the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify unserved children and their locations.
    2. Check if any child can be served immediately with a suitable sandwich on a tray at their location.
    3. For remaining children:
       a. Count required gluten-free and regular sandwiches.
       b. Calculate actions needed to make, put on trays, move trays, and serve.
    4. Sum all actions to get the heuristic value.
    """

    def __init__(self, task):
        self.static = task.static
        self.initial_state = task.initial_state

        # Extract allergic and non-allergic children from static facts
        self.allergic_children = set()
        self.non_allergic_children = set()
        for fact in self.static:
            if match(fact, 'allergic_gluten *'):
                child = get_parts(fact)[1]
                self.allergic_children.add(child)
            elif match(fact, 'not_allergic_gluten *'):
                child = get_parts(fact)[1]
                self.non_allergic_children.add(child)
        self.all_children = self.allergic_children.union(self.non_allergic_children)

        # Extract all sandwiches from initial 'notexist' facts
        self.all_sandwiches = set()
        for fact in self.initial_state:
            if match(fact, 'notexist *'):
                sandwich = get_parts(fact)[1]
                self.all_sandwiches.add(sandwich)

        # Extract all trays from initial 'at' facts
        self.all_trays = set()
        for fact in self.initial_state:
            if match(fact, 'at * *'):
                tray = get_parts(fact)[1]
                self.all_trays.add(tray)

    def __call__(self, node):
        state = node.state
        served_immediately = 0
        allergic_remaining = 0
        non_allergic_remaining = 0
        places_remaining = set()

        # Identify unserved children and their locations
        unserved_children = []
        for child in self.all_children:
            served_fact = f'(served {child})'
            if served_fact not in state:
                # Check if child is waiting
                waiting = any(match(fact, f'waiting {child} *') for fact in state)
                if waiting:
                    unserved_children.append(child)

        # Check for immediate serving and collect remaining children
        for child in unserved_children:
            is_allergic = child in self.allergic_children
            # Get child's location
            child_place = None
            for fact in state:
                if match(fact, f'waiting {child} *'):
                    child_place = get_parts(fact)[2]
                    break
            if not child_place:
                continue

            # Check if suitable sandwich is on a tray at child_place
            found = False
            if is_allergic:
                # Look for gluten-free sandwich on tray at child_place
                for s in self.all_sandwiches:
                    if f'(no_gluten_sandwich {s})' in state:
                        for t in self.all_trays:
                            if f'(ontray {s} {t})' in state and f'(at {t} {child_place})' in state:
                                found = True
                                break
                        if found:
                            break
            else:
                # Look for any sandwich on tray at child_place
                for s in self.all_sandwiches:
                    if f'(notexist {s})' not in state:  # Sandwich exists
                        for t in self.all_trays:
                            if f'(ontray {s} {t})' in state and f'(at {t} {child_place})' in state:
                                found = True
                                break
                        if found:
                            break
            if found:
                served_immediately += 1
            else:
                if is_allergic:
                    allergic_remaining += 1
                else:
                    non_allergic_remaining += 1
                if child_place != 'kitchen':
                    places_remaining.add(child_place)

        # Calculate remaining actions
        existing_gluten_free = []
        existing_regular = []
        for s in self.all_sandwiches:
            if f'(notexist {s})' not in state:  # Sandwich exists
                on_tray = any(fact.startswith(f'(ontray {s} ') for fact in state)
                if not on_tray:
                    if f'(no_gluten_sandwich {s})' in state:
                        existing_gluten_free.append(s)
                    else:
                        existing_regular.append(s)

        available_gluten_free = len(existing_gluten_free)
        available_regular = len(existing_regular)

        needed_gluten_free = max(0, allergic_remaining - available_gluten_free)
        needed_regular = max(0, non_allergic_remaining - (available_regular + (available_gluten_free - max(0, allergic_remaining - needed_gluten_free))))

        make_actions = needed_gluten_free + needed_regular
        put_actions = allergic_remaining + non_allergic_remaining
        move_actions = len(places_remaining)
        serve_actions = allergic_remaining + non_allergic_remaining

        total = served_immediately + make_actions + put_actions + move_actions + serve_actions
        return total
