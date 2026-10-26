from heuristics.heuristic_base import Heuristic
# No need for fnmatch if parsing manually

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all
    children specified in the goal. It does this by summing up the estimated
    minimum actions needed for each unserved child, based on the current
    state of available sandwiches relative to the child's location and needs.

    # Assumptions
    - Each child requires exactly one sandwich.
    - Trays are fungible and can be moved between any two places (kitchen or waiting locations)
      with a cost of 1 action (`move_tray`).
    - Sufficient bread and content portions are available in the kitchen to make
      any required sandwich type (gluten-free or regular) as long as a `notexist`
      sandwich object is available.
    - `waiting` locations and `allergic_gluten`/`not_allergic_gluten` status are static.

    # Heuristic Initialization
    - Extracts the set of children that need to be served from the goal.
    - Extracts the allergy status (gluten or none) for each child from the initial state.
    - Extracts the waiting location for each child from the initial state.
    - Extracts the static gluten-free status of sandwich objects from the initial state and static facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:

    1. Identify all children from the goal list who have not yet been served.
    2. For these unserved children, determine how many need a gluten-free sandwich
       and how many need a regular sandwich, based on their allergy status.
    3. Categorize the available sandwiches in the current state based on their type
       (gluten-free or regular) and their location relative to the unserved children
       who need that type:
       - State 0: The sandwich is on a tray, and that tray is currently at the
         waiting location of an unserved child who needs this type of sandwich.
         (Estimated cost to serve this child from this state: 1 action - `serve`)
       - State 1: The sandwich is on a tray, but that tray is *not* at the waiting
         location of any unserved child who needs this type of sandwich.
         (Estimated cost: 1 action - `move_tray` + 1 action - `serve` = 2)
       - State 2: The sandwich is `at_kitchen_sandwich`.
         (Estimated cost: 1 action - `put_on_tray` + 1 action - `move_tray` + 1 action - `serve` = 3)
    4. Count the number of `notexist` sandwich objects available. These represent
       sandwiches that can be made (State 3).
       (Estimated cost: 1 action - `make_sandwich` + 1 action - `put_on_tray` + 1 action - `move_tray` + 1 action - `serve` = 4)
    5. Greedily assign the available sandwiches/making capacity to the unserved
       children, prioritizing sandwiches in State 0, then State 1, then State 2,
       and finally using the capacity to make new sandwiches (State 3). This
       assignment is done separately for gluten-free and regular needs.
    6. Sum the estimated costs for all unserved children based on the state of the
       sandwich assigned to them (or the cost of making one).
    7. If the total number of unserved children exceeds the total number of
       existing sandwiches plus the number of `notexist` sandwich objects,
       the state is considered unsolvable with the given objects, and the
       heuristic returns infinity.
    8. If all children are served, the heuristic value is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts
        about children and sandwich types.
        """
        self.goals = task.goals
        # Use both initial state and static facts for static information
        self.static_facts = task.static
        self.initial_state = task.initial_state

        # Extract static child information from the initial state
        self.child_allergy = {}  # child_name -> 'gluten' or 'none'
        self.child_location = {} # child_name -> place

        for fact in self.initial_state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]
            if predicate == 'allergic_gluten':
                self.child_allergy[parts[1]] = 'gluten'
            elif predicate == 'not_allergic_gluten':
                self.child_allergy[parts[1]] = 'none'
            elif predicate == 'waiting':
                self.child_location[parts[1]] = parts[2]

        # Extract static sandwich type information from static facts and initial state
        self.sandwich_is_gf_init = {} # sandwich_name -> bool (True if GF)
        # Check both static facts and initial state facts for no_gluten_sandwich
        for fact in self.static_facts | self.initial_state:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] == 'no_gluten_sandwich':
                 self.sandwich_is_gf_init[parts[1]] = True

        # Extract goal children
        self.goal_children = set() # Set of child names that need to be served
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            if parts[0] == 'served':
                self.goal_children.add(parts[1])

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach the goal.
        """
        state = node.state

        # 1. Identify unserved children and their needs
        unserved_children = {c for c in self.goal_children if f'(served {c})' not in state}

        # 8. If all goal children are served, the heuristic is 0.
        if not unserved_children:
            return 0

        # Count unserved children by type
        unserved_gf_count = sum(1 for c in unserved_children if self.child_allergy.get(c) == 'gluten')
        unserved_reg_count = sum(1 for c in unserved_children if self.child_allergy.get(c) == 'none')

        # Build maps for current state facts related to sandwiches and trays
        sandwich_ontray_map = {} # sandwich -> tray
        tray_location_map = {} # tray -> place
        sandwich_at_kitchen = set()
        sandwiches_in_state = set() # All sandwich objects currently existing (not notexist)

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == 'ontray':
                sandwich, tray = parts[1], parts[2]
                sandwich_ontray_map[sandwich] = tray
                sandwiches_in_state.add(sandwich)
            elif predicate == 'at' and parts[1].startswith('tray'):
                tray, place = parts[1], parts[2]
                tray_location_map[tray] = place
            elif predicate == 'at_kitchen_sandwich':
                sandwich = parts[1]
                sandwich_at_kitchen.add(sandwich)
                sandwiches_in_state.add(sandwich)

        # 3. Categorize available sandwiches into states S0, S1, S2
        gf_s0_sandwiches = set()
        reg_s0_sandwiches = set()
        gf_s1_sandwiches = set()
        reg_s1_sandwiches = set()
        gf_s2_sandwiches = set()
        reg_s2_sandwiches = set()

        # Keep track of locations where unserved children are waiting, by type
        unserved_gf_locations = {self.child_location.get(c) for c in unserved_children if self.child_allergy.get(c) == 'gluten'}
        unserved_reg_locations = {self.child_location.get(c) for c in unserved_children if self.child_allergy.get(c) == 'none'}

        for s in sandwiches_in_state:
            is_gf = self.sandwich_is_gf_init.get(s, False) # Get static type

            if s in sandwich_ontray_map:
                t = sandwich_ontray_map[s]
                p = tray_location_map.get(t)
                if p: # If tray location is known
                    # Check for State 0: ontray at a relevant child's location
                    is_s0 = False
                    if is_gf and p in unserved_gf_locations:
                         is_s0 = True
                    elif not is_gf and p in unserved_reg_locations:
                         is_s0 = True

                    if is_s0:
                        if is_gf: gf_s0_sandwiches.add(s)
                        else: reg_s0_sandwiches.add(s)
                    else: # State 1: ontray elsewhere
                        if is_gf: gf_s1_sandwiches.add(s)
                        else: reg_s1_sandwiches.add(s)
                else: # State 1: ontray but tray location unknown (treat as elsewhere)
                    if is_gf: gf_s1_sandwiches.add(s)
                    else: reg_s1_sandwiches.add(s)

            elif s in sandwich_at_kitchen: # State 2: at_kitchen_sandwich
                if is_gf: gf_s2_sandwiches.add(s)
                else: reg_s2_sandwiches.add(s)

        # Note: The sets gf_s0, gf_s1, gf_s2 are naturally disjoint based on the
        # location predicates. Same for reg_s0, reg_s1, reg_s2.

        Avail_gf_s0 = len(gf_s0_sandwiches)
        Avail_reg_s0 = len(reg_s0_sandwiches)
        Avail_gf_s1 = len(gf_s1_sandwiches)
        Avail_reg_s1 = len(reg_s1_sandwiches)
        Avail_gf_s2 = len(gf_s2_sandwiches)
        Avail_reg_s2 = len(reg_s2_sandwiches)

        # 4. Count available notexist sandwich objects (potential for State 3)
        notexist_count = sum(1 for fact in state if get_parts(fact)[0] == 'notexist')

        # 5. Greedily assign sandwiches/making capacity to unserved children
        U_gf_rem = unserved_gf_count
        U_reg_rem = unserved_reg_count
        Notexist_rem = notexist_count
        total_cost = 0

        # Assign GF sandwiches from best state to worst
        served_gf_s0 = min(U_gf_rem, Avail_gf_s0)
        total_cost += served_gf_s0 * 1
        U_gf_rem -= served_gf_s0

        served_gf_s1 = min(U_gf_rem, Avail_gf_s1)
        total_cost += served_gf_s1 * 2
        U_gf_rem -= served_gf_s1

        served_gf_s2 = min(U_gf_rem, Avail_gf_s2)
        total_cost += served_gf_s2 * 3
        U_gf_rem -= served_gf_s2

        # Assign Regular sandwiches from best state to worst
        served_reg_s0 = min(U_reg_rem, Avail_reg_s0)
        total_cost += served_reg_s0 * 1
        U_reg_rem -= served_reg_s0

        served_reg_s1 = min(U_reg_rem, Avail_reg_s1)
        total_cost += served_reg_s1 * 2
        U_reg_rem -= served_reg_s1

        served_reg_s2 = min(U_reg_rem, Avail_reg_s2)
        total_cost += served_reg_s2 * 3
        U_reg_rem -= served_reg_s2

        # Assign sandwiches needing making (State 3)
        needed_to_make = U_gf_rem + U_reg_rem
        can_make = min(needed_to_make, Notexist_rem)
        total_cost += can_make * 4
        needed_to_make -= can_make

        # 7. Check if enough sandwiches can be made
        if needed_to_make > 0:
            # Cannot satisfy all unserved children even by making all possible sandwiches
            return float('inf')

        return total_cost
