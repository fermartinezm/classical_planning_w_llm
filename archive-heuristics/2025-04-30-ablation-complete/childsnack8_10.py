from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming this base class exists

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input, maybe log a warning or raise an error
        # For robustness, return empty list or handle specific cases
        # Assuming valid PDDL fact strings for this domain
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children.
    It counts the number of unserved children (representing the final 'serve' action needed for each)
    and adds the estimated minimum cost to get a suitable sandwich onto a tray at each child's location.
    The cost to get a sandwich to the location depends on its current state:
    - Already on a tray at the correct location: 0 additional cost (beyond the 'serve' action).
    - On a tray at a wrong location: 1 action (move_tray).
    - In the kitchen: 2 actions (put_on_tray + move_tray).
    - Needs to be made: 3 actions (make_sandwich + put_on_tray + move_tray).
    The heuristic prioritizes using sandwiches that are closer to being ready (on tray at wrong place > in kitchen > needs to be made).

    # Assumptions
    - Each unserved child requires exactly one suitable sandwich.
    - A tray is available at the kitchen when needed for the 'put_on_tray' action (relaxation).
    - Ingredients are sufficient if a 'notexist' sandwich object is available for the 'make' action (relaxation).
    - Tray movements between any two places cost 1 action (relaxation, ignores intermediate places).
    - Any sandwich on a tray at a wrong location can be moved to any needed location.
    - Any sandwich in the kitchen can be put on *a* tray and moved to any needed location.
    - Sandwich type (gluten-free/regular) is determined by the ingredients used to make it or is a static property once made. The heuristic checks the 'no_gluten_sandwich' predicate in the current state.
    - Child allergy and waiting place are static or in the initial state.

    # Heuristic Initialization
    - Extracts static information about child allergies (`allergic_gluten`, `not_allergic_gluten`),
      child waiting places (`waiting`), and gluten-free ingredients (`no_gluten_bread`, `no_gluten_content`).
    - Extracts initial state information about child waiting places if not in static facts, and existing `no_gluten_sandwich` facts.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Identify all children in the problem and their static properties (allergy, waiting place).
    2.  Identify which children are currently `served` in the given state.
    3.  Count the total number of unserved children (`N_unserved`). This contributes `N_unserved` to the heuristic (representing the final 'serve' action for each). If `N_unserved` is 0, the goal is reached, return 0.
    4.  Categorize unserved children by their waiting place and the type of sandwich they need (gluten-free if allergic, regular otherwise). Store counts in `unserved_children_by_place_type`.
    5.  Identify the current location of all trays (`at ?t ?p`).
    6.  Identify all existing sandwiches (`at_kitchen_sandwich ?s` or `ontray ?s ?t`). Determine their type (`no_gluten_sandwich ?s`) and current location (kitchen or the tray's location).
    7.  Count the number of `notexist` sandwich objects available.
    8.  For each combination of (place, sandwich_type) where unserved children are waiting, calculate the deficit: `needed_at_p_type = count` from step 4, `available_at_p_type = count` of suitable sandwiches currently `ontray` at that place from step 6. The deficit is `max(0, needed_at_p_type - available_at_p_type)`.
    9.  Sum the deficits for gluten-free sandwiches (`total_needed_delivery_gf`) and regular sandwiches (`total_needed_delivery_reg`) across all places. These represent the number of sandwiches of each type that need to be delivered to the correct locations on trays.
    10. Count available sandwiches not currently on trays at the correct location, categorized by type and current state: `ontray` at a wrong place, `at_kitchen_sandwich`, and `notexist`.
    11. Greedily fulfill the `total_needed_delivery_gf` requirement:
        - Use available gluten-free sandwiches `ontray` at wrong places first (cost 1 each).
        - Use available gluten-free sandwiches `at_kitchen_sandwich` next (cost 2 each: put_on_tray + move_tray).
        - Use available `notexist` sandwiches to make new gluten-free sandwiches if ingredients are implicitly available (cost 3 each: make + put_on_tray + move_tray). Add these costs to the heuristic.
    12. Greedily fulfill the `total_needed_delivery_reg` requirement similarly, using available regular sandwiches and remaining `notexist` sandwiches, adding costs (1, 2, or 3).
    13. The total heuristic value is the sum of `N_unserved` (from step 3) and the accumulated costs from steps 11 and 12.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static and initial state information.
        """
        self.goals = task.goals # Not strictly needed for this heuristic logic, but good practice
        self.static_facts = task.static
        self.initial_state_facts = task.initial_state

        # Extract static/initial info
        self.gluten_free_bread = set()
        self.gluten_free_content = set()
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.child_waiting_place = {} # child: place
        self.initial_gluten_free_sandwiches = set() # Sandwiches marked GF in initial state

        # Combine static and initial facts for parsing child info and initial GF sandwiches
        all_init_facts = self.static_facts | self.initial_state_facts

        for fact in all_init_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            if parts[0] == 'no_gluten_bread':
                self.gluten_free_bread.add(parts[1])
            elif parts[0] == 'no_gluten_content':
                self.gluten_free_content.add(parts[1])
            elif parts[0] == 'allergic_gluten':
                self.allergic_children.add(parts[1])
            elif parts[0] == 'not_allergic_gluten':
                self.not_allergic_children.add(parts[1])
            elif parts[0] == 'waiting':
                 self.child_waiting_place[parts[1]] = parts[2]
            elif parts[0] == 'no_gluten_sandwich':
                 self.initial_gluten_free_sandwiches.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings)

        current_state_facts = set(state)

        # 1. Count unserved children and categorize by location and allergy
        served_children = {parts[1] for fact in current_state_facts if match(fact, 'served', '*')}

        unserved_children_by_place_type = {} # {(place, type): count}
        all_children_in_problem = self.allergic_children | self.not_allergic_children

        for child in all_children_in_problem:
            if child not in served_children:
                place = self.child_waiting_place.get(child)
                if place is None:
                     # Child exists but isn't waiting? Problematic state? Assume valid problem.
                     continue
                is_allergic = child in self.allergic_children
                sandwich_type = 'gluten-free' if is_allergic else 'regular'
                key = (place, sandwich_type)
                unserved_children_by_place_type[key] = unserved_children_by_place_type.get(key, 0) + 1

        N_unserved = sum(unserved_children_by_place_type.values())

        if N_unserved == 0:
            return 0 # Goal reached

        h = N_unserved # Cost for serving each unserved child

        # 2. Identify current locations of trays
        tray_locations = {} # tray: place
        for fact in current_state_facts:
            if match(fact, 'at', '*', '*'):
                parts = get_parts(fact)
                if len(parts) == 3 and parts[1].startswith('tray'): # Ensure it's (at tray place)
                    tray, place = parts[1], parts[2]
                    tray_locations[tray] = place

        # 3. Identify existing sandwiches, their types, and locations
        sandwiches_kitchen_type = {'gluten-free': 0, 'regular': 0}
        sandwiches_ontray_wrong_place_type = {'gluten-free': 0, 'regular': 0}
        sandwiches_ontray_at_place_type_actual = {} # {(place, type): count} # Actual count in state

        existing_sandwiches = set()
        current_sandwich_types = {} # sandwich: type ('gluten-free' or 'regular')
        current_sandwich_locations = {} # sandwich: 'kitchen' or tray_object

        for fact in current_state_facts:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == 'at_kitchen_sandwich':
                s = parts[1]
                existing_sandwiches.add(s)
                current_sandwich_locations[s] = 'kitchen'
            elif parts[0] == 'ontray':
                if len(parts) == 3: # Ensure it's (ontray sandwich tray)
                    s, t = parts[1], parts[2]
                    existing_sandwiches.add(s)
                    current_sandwich_locations[s] = t # Store tray object
            elif parts[0] == 'no_gluten_sandwich':
                 if len(parts) == 2: # Ensure it's (no_gluten_sandwich sandwich)
                    s = parts[1]
                    current_sandwich_types[s] = 'gluten-free'

        # Infer regular type for existing sandwiches without no_gluten_sandwich predicate
        for s in existing_sandwiches:
            if s not in current_sandwich_types:
                current_sandwich_types[s] = 'regular'

        # Count available sandwiches by location and type
        for s in existing_sandwiches:
            s_type = current_sandwich_types.get(s)
            if s_type is None: continue # Should not happen with valid state

            loc = current_sandwich_locations.get(s)
            if loc == 'kitchen':
                sandwiches_kitchen_type[s_type] += 1
            elif loc in tray_locations: # It's on a tray with a known location
                tray_loc = tray_locations[loc]
                key = (tray_loc, s_type)
                sandwiches_ontray_at_place_type_actual[key] = sandwiches_ontray_at_place_type_actual.get(key, 0) + 1
            else:
                 # It's on a tray whose location is unknown or not 'at' any place.
                 # Treat as ontray at wrong place for heuristic purposes.
                 sandwiches_ontray_wrong_place_type[s_type] += 1

        # 4. Count available notexist sandwiches
        available_notexist = sum(1 for fact in current_state_facts if match(fact, 'notexist', '*'))

        # 5. Calculate sandwich-at-location deficit
        # This is the number of sandwiches of each type needed at each location
        # that are NOT currently on a tray at that location.
        deficit_gf = 0
        deficit_reg = 0

        for (place, s_type), needed_count in unserved_children_by_place_type.items():
            available_at_p_type = sandwiches_ontray_at_place_type_actual.get((place, s_type), 0)
            deficit = max(0, needed_count - available_at_p_type)
            if s_type == 'gluten-free':
                deficit_gf += deficit
            else: # regular
                deficit_reg += deficit

        # 6. Fulfill deficit using available sandwiches (cheapest delivery cost first)
        # Cost 1: ontray at wrong place -> move tray
        used_ontray_gf_wrong = min(deficit_gf, sandwiches_ontray_wrong_place_type['gluten-free'])
        h += used_ontray_gf_wrong * 1
        deficit_gf -= used_ontray_gf_wrong

        used_ontray_reg_wrong = min(deficit_reg, sandwiches_ontray_wrong_place_type['regular'])
        h += used_ontray_reg_wrong * 1
        deficit_reg -= used_ontray_reg_wrong

        # Cost 2: at kitchen -> put on tray + move tray
        used_kitchen_gf = min(deficit_gf, sandwiches_kitchen_type['gluten-free'])
        h += used_kitchen_gf * 2
        deficit_gf -= used_kitchen_gf

        used_kitchen_reg = min(deficit_reg, sandwiches_kitchen_type['regular'])
        h += used_kitchen_reg * 2
        deficit_reg -= used_kitchen_reg

        # Cost 3: needs to be made -> make + put on tray + move tray
        # Assume ingredients are available if notexist is available (relaxation).
        can_make_gf = min(deficit_gf, available_notexist)
        h += can_make_gf * 3
        deficit_gf -= can_make_gf
        available_notexist -= can_make_gf # Consume notexist

        can_make_reg = min(deficit_reg, available_notexist)
        h += can_make_reg * 3
        deficit_reg -= can_make_reg
        available_notexist -= can_make_reg # Consume notexist

        # Any remaining deficit implies unsolvability with available notexist sandwiches
        # or missing ingredients (which we relaxed). The heuristic just returns the current sum.

        return h
