from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract components of a PDDL fact."""
    return fact[1:-1].split() if fact.startswith('(') else []

def match(fact, pattern):
    """Check if a fact matches a pattern using wildcards."""
    fact_parts = get_parts(fact)
    pattern_parts = pattern.split()
    if len(fact_parts) != len(pattern_parts):
        return False
    return all(fnmatch(fact_part, pattern_part) for fact_part, pattern_part in zip(fact_parts, pattern_parts))

class childsnack6Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnack domain.

    # Summary
    Estimates the number of actions needed to serve all children by considering:
    - Allergic children requiring gluten-free sandwiches.
    - Current locations of trays and sandwiches.
    - Availability of ingredients to make new sandwiches.

    # Assumptions
    - Each child requires a unique sandwich.
    - Moving a tray serves all children at its location.
    - Gluten-free sandwiches require specific ingredients available in the kitchen.

    # Heuristic Initialization
    - Extracts static information about children's allergies and locations.
    - Identifies gluten-free ingredients and initial positions of trays and sandwiches.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify unserved children and their locations.
    2. For each child:
        a. Allergic: Check for existing gluten-free sandwiches on trays or in the kitchen.
        b. Non-allergic: Check for any available sandwiches.
        c. Calculate actions needed to make, move, and serve sandwiches if not available.
    3. Sum actions for all unserved children, considering shared tray movements.
    """

    def __init__(self, task):
        self.static = task.static
        self.initial_state = task.initial_state

        # Extract children and their locations
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.child_locations = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'allergic_gluten':
                self.allergic_children.add(parts[1])
                self.child_locations[parts[1]] = parts[3] if len(parts) > 3 else parts[2]
            elif parts[0] == 'not_allergic_gluten':
                self.not_allergic_children.add(parts[1])
                self.child_locations[parts[1]] = parts[3] if len(parts) > 3 else parts[2]
            elif parts[0] == 'waiting':
                self.child_locations[parts[1]] = parts[2]

        # Extract gluten-free ingredients from static
        self.gluten_free_breads = set()
        self.gluten_free_contents = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'no_gluten_bread':
                self.gluten_free_breads.add(parts[1])
            elif parts[0] == 'no_gluten_content':
                self.gluten_free_contents.add(parts[1])

        # Extract all possible sandwiches, breads, contents, trays from initial state
        self.all_sandwiches = set()
        self.breads = set()
        self.contents = set()
        self.trays = set()
        for fact in self.initial_state:
            parts = get_parts(fact)
            if parts[0] == 'notexist':
                self.all_sandwiches.add(parts[1])
            elif parts[0] == 'at_kitchen_bread':
                self.breads.add(parts[1])
            elif parts[0] == 'at_kitchen_content':
                self.contents.add(parts[1])
            elif parts[0] == 'at' and len(parts) == 3:
                self.trays.add(parts[1])

    def __call__(self, node):
        state = node.state
        served = {parts[1] for fact in state if match(fact, 'served *') for parts in [get_parts(fact)]}
        unserved = (self.allergic_children | self.not_allergic_children) - served
        cost = 0

        # Precompute available ingredients
        available_gf_breads = [b for b in self.breads if f'(at_kitchen_bread {b})' in state and b in self.gluten_free_breads]
        available_gf_contents = [c for c in self.contents if f'(at_kitchen_content {c})' in state and c in self.gluten_free_contents]
        available_regular_breads = [b for b in self.breads if f'(at_kitchen_bread {b})' in state and b not in self.gluten_free_breads]
        available_regular_contents = [c for c in self.contents if f'(at_kitchen_content {c})' in state and c not in self.gluten_free_contents]

        # Track existing sandwiches and their states
        existing_sandwiches = []
        for s in self.all_sandwiches:
            if not any(match(fact, f'notexist {s}') for fact in state):
                is_gf = any(match(fact, f'no_gluten_sandwich {s}') for fact in state)
                on_tray = next((get_parts(fact)[2] for fact in state if match(fact, f'ontray {s} *')), None)
                tray_loc = next((get_parts(fact)[2] for fact in state if match(fact, f'at {on_tray} *')), None) if on_tray else None
                at_kitchen = any(match(fact, f'at_kitchen_sandwich {s}') for fact in state)
                existing_sandwiches.append((s, is_gf, on_tray, tray_loc, at_kitchen))

        # Process each unserved child
        for child in unserved:
            is_allergic = child in self.allergic_children
            location = self.child_locations.get(child, 'table1')  # default to table1 if missing

            # Find suitable sandwiches
            suitable = []
            for s in existing_sandwiches:
                s_id, s_gf, s_tray, s_tray_loc, s_kitchen = s
                if is_allergic and not s_gf:
                    continue
                suitable.append(s)

            # Check for on-tray at location
            served_flag = False
            for s in suitable:
                if s[3] == location:
                    cost += 1  # serve
                    existing_sandwiches.remove(s)
                    served_flag = True
                    break
            if served_flag:
                continue

            # Check for on-tray elsewhere
            for s in suitable:
                if s[2] is not None and s[3] != location:
                    cost += 2  # move and serve
                    existing_sandwiches.remove(s)
                    served_flag = True
                    break
            if served_flag:
                continue

            # Check for at kitchen
            for s in suitable:
                if s[4]:
                    cost += 3  # put, move, serve
                    existing_sandwiches.remove(s)
                    served_flag = True
                    break
            if served_flag:
                continue

            # Need to make new sandwich
            if is_allergic:
                if available_gf_breads and available_gf_contents:
                    available_gf_breads.pop(0)
                    available_gf_contents.pop(0)
                    cost += 4  # make, put, move, serve
                else:
                    cost += 1000  # penalty if missing ingredients
            else:
                if (available_regular_breads or available_gf_breads) and (available_regular_contents or available_gf_contents):
                    if available_regular_breads:
                        available_regular_breads.pop(0)
                    else:
                        available_gf_breads.pop(0)
                    if available_regular_contents:
                        available_regular_contents.pop(0)
                    else:
                        available_gf_contents.pop(0)
                    cost += 4
                else:
                    cost += 1000

        return cost
