from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
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
    This heuristic estimates the number of actions needed to serve all waiting
    children. It counts the required steps (make sandwich, put on tray, move
    tray, serve) for each unserved child, prioritizing the use of existing
    sandwiches and accounting for shared tray movements to a location.

    # Assumptions
    - Each unserved child requires one suitable sandwich (gluten-free if allergic).
    - Making a sandwich requires available ingredients and a 'notexist' sandwich object (implicitly assumed if needed for the heuristic count).
    - Putting a sandwich on a tray requires the sandwich in the kitchen and a tray in the kitchen.
    - Moving a tray requires the tray at a location.
    - Serving a sandwich requires the child waiting, the sandwich on a tray, and the tray at the child's location.
    - The cost of moving a tray to a specific waiting location is counted at most once, even if multiple children at that location need sandwiches delivered.

    # Heuristic Initialization
    The heuristic extracts static information about children from the task:
    - Which children are allergic to gluten.
    - The waiting place for each child who is initially waiting (i.e., appears in the goal).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:
    1. Identify all children who are initially waiting (from the task goals) but are not yet served in the current state. If all are served, the heuristic is 0.
    2. Parse the current state to find:
       - All available sandwiches (in the kitchen or on a tray) and their gluten status.
       - Which sandwich is on which tray (if applicable).
       - The current location of each tray.
    3. Initialize the total heuristic cost to 0.
    4. Initialize sets to track sandwiches that have been "assigned" to an unserved child in this calculation, and locations that have already had the cost for a tray move added.
    5. Iterate through each unserved child (sorting for deterministic output):
       a. Add 1 to the total cost for the final 'serve' action this child will require.
       b. Determine the child's waiting place and allergy status (from initialization).
       c. Find the "best" available suitable sandwich for this child that hasn't been used yet, prioritizing based on its current location:
          - Priority 3: A suitable sandwich already on a tray at the child's waiting place.
          - Priority 2: A suitable sandwich on a tray at a different location.
          - Priority 1: A suitable sandwich in the kitchen.
          - Priority 0: No suitable available sandwich exists (a new one must be made).
       d. Mark the chosen sandwich (if any) as used.
       e. Add additional costs based on the chosen sandwich's state (or lack thereof):
          - If Priority 3: No additional cost (serve action already counted, tray is ready).
          - If Priority 2: Add 1 for the 'move_tray' action, but only if the child's waiting place has not already had a 'move_tray' cost accounted for by a previous child.
          - If Priority 1: Add 1 for the 'put_on_tray' action. Add 1 for the 'move_tray' action, but only if the child's waiting place has not already had a 'move_tray' cost accounted for.
          - If Priority 0: Add 1 for the 'make_sandwich' action, 1 for the 'put_on_tray' action. Add 1 for the 'move_tray' action, but only if the child's waiting place has not already had a 'move_tray' cost accounted for.
    6. Return the total computed cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information:
        - Goal children (those initially waiting).
        - Waiting place for each goal child.
        - Allergy status for each child.
        """
        self.goals = task.goals
        static_facts = task.static

        self.child_waiting_place = {}
        self.child_is_allergic = {}
        all_children_in_static = set()

        # Find all children mentioned in static facts (waiting or allergy status)
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] in ['waiting', 'allergic_gluten', 'not_allergic_gluten']:
                if len(parts) > 1: # Ensure there's an object name
                    all_children_in_static.add(parts[1])

        # Store waiting place and allergy status for all children found in static facts
        for child in all_children_in_static:
            waiting_fact = next((f for f in static_facts if match(f, "waiting", child, "*")), None)
            if waiting_fact:
                self.child_waiting_place[child] = get_parts(waiting_fact)[2]

            is_allergic = any(match(f, "allergic_gluten", child) for f in static_facts)
            self.child_is_allergic[child] = is_allergic

        # Identify the set of children who are goals (i.e., initially waiting and need serving)
        self.goal_children = {get_parts(goal)[1] for goal in self.goals if get_parts(goal) and get_parts(goal)[0] == 'served'}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify unserved children who are goals
        unserved_children = {c for c in self.goal_children if f'(served {c})' not in state}

        if not unserved_children:
            return 0  # Goal reached

        total_cost = 0

        # 2. Parse state for available sandwiches and tray locations
        available_gf_sandwiches = set() # (sandwich name, location_type)
        available_ngf_sandwiches = set() # (sandwich name, location_type)
        sandwich_tray_map = {} # sandwich -> tray
        tray_location_map = {} # tray -> location
        sandwich_is_gf = {} # sandwich -> bool

        # First pass to find GF status of all sandwiches mentioned in state
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'no_gluten_sandwich':
                if len(parts) > 1:
                    sandwich_is_gf[parts[1]] = True

        # Second pass to find locations and add to available sets
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'at_kitchen_sandwich':
                if len(parts) > 1:
                    s = parts[1]
                    if sandwich_is_gf.get(s, False):
                        available_gf_sandwiches.add((s, 'kitchen'))
                    else:
                        available_ngf_sandwiches.add((s, 'kitchen'))
            elif parts and parts[0] == 'ontray':
                if len(parts) > 2:
                    s, t = parts[1], parts[2]
                    if sandwich_is_gf.get(s, False):
                        available_gf_sandwiches.add((s, 'tray'))
                    else:
                        available_ngf_sandwiches.add((s, 'tray'))
                    sandwich_tray_map[s] = t
            elif parts and parts[0] == 'at' and len(parts) == 3: # Assuming (at ?t ?p) where ?t is a tray
                 t, p = parts[1], parts[2]
                 tray_location_map[t] = p

        # 4. Initialize tracking sets
        used_sandwiches = set()
        locations_with_tray_move = set()

        # 5. Iterate through each unserved child (sorted for deterministic output)
        sorted_unserved_children = sorted(list(unserved_children))

        for child in sorted_unserved_children:
            # a. Add cost for serve action
            total_cost += 1

            waiting_place = self.child_waiting_place.get(child)
            is_allergic = self.child_is_allergic.get(child, False) # Default to False if status unknown

            # Should not happen in valid problems, but handle defensively
            if waiting_place is None:
                 continue

            # c. Find the best suitable available sandwich
            best_s_info = None # (s, location_type)
            best_priority = -1 # 3: on tray at location, 2: on tray wrong location, 1: in kitchen, 0: needs make

            # Combine all available sandwiches for easier iteration
            all_available_sandwiches_info = list(available_gf_sandwiches) + list(available_ngf_sandwiches)

            # Priority 3: On tray at the correct location
            suitable_on_tray_at_location = []
            for s, loc_type in all_available_sandwiches_info:
                 if s in used_sandwiches: continue
                 if loc_type == 'tray':
                     is_s_gf = sandwich_is_gf.get(s, False)
                     if (is_allergic and is_s_gf) or (not is_allergic): # Allergic needs GF, Non-allergic can use any
                         t = sandwich_tray_map.get(s)
                         if t and tray_location_map.get(t) == waiting_place:
                             suitable_on_tray_at_location.append((s, loc_type))

            if suitable_on_tray_at_location:
                best_s_info = suitable_on_tray_at_location[0] # Pick any
                best_priority = 3
            else:
                # Priority 2: On tray at wrong location
                suitable_on_tray_wrong_location = []
                for s, loc_type in all_available_sandwiches_info:
                    if s in used_sandwiches: continue
                    if loc_type == 'tray':
                        is_s_gf = sandwich_is_gf.get(s, False)
                        if (is_allergic and is_s_gf) or (not is_allergic):
                            t = sandwich_tray_map.get(s)
                            if t and tray_location_map.get(t) != waiting_place:
                                suitable_on_tray_wrong_location.append((s, loc_type))

                if suitable_on_tray_wrong_location:
                    best_s_info = suitable_on_tray_wrong_location[0] # Pick any
                    best_priority = 2
                else:
                    # Priority 1: In kitchen
                    suitable_in_kitchen = []
                    for s, loc_type in all_available_sandwiches_info:
                        if s in used_sandwiches: continue
                        if loc_type == 'kitchen':
                            is_s_gf = sandwich_is_gf.get(s, False)
                            if (is_allergic and is_s_gf) or (not is_allergic):
                                suitable_in_kitchen.append((s, loc_type))

                    if suitable_in_kitchen:
                        best_s_info = suitable_in_kitchen[0] # Pick any
                        best_priority = 1
                    # else: best_s_info remains None, best_priority remains -1 (Needs make)


            # e. Add additional costs based on priority
            if best_priority == 3:
                # Sandwich is on tray at the correct location. Only serve needed (already added +1).
                s, loc_type = best_s_info
                used_sandwiches.add(s)
                pass # Cost already added
            elif best_priority == 2:
                # Sandwich is on tray at wrong location. Needs move_tray, serve.
                s, loc_type = best_s_info
                used_sandwiches.add(s)
                # Needs move_tray to waiting_place
                if waiting_place != 'kitchen' and waiting_place not in locations_with_tray_move:
                    total_cost += 1 # move_tray
                    locations_with_tray_move.add(waiting_place)
            elif best_priority == 1:
                # Sandwich is in kitchen. Needs put_on_tray, move_tray, serve.
                s, loc_type = best_s_info
                used_sandwiches.add(s)
                total_cost += 1 # put_on_tray
                # Needs move_tray from kitchen to waiting_place
                if waiting_place != 'kitchen' and waiting_place not in locations_with_tray_move:
                    total_cost += 1 # move_tray
                    locations_with_tray_move.add(waiting_place)
            else: # best_priority == -1 (Needs make)
                # Needs make, put_on_tray, move_tray, serve.
                total_cost += 1 # make
                total_cost += 1 # put_on_tray
                # Needs move_tray from kitchen to waiting_place
                if waiting_place != 'kitchen' and waiting_place not in locations_with_tray_move:
                    total_cost += 1 # move_tray
                    locations_with_tray_move.add(waiting_place)

        # 6. Return total cost
        return total_cost
