from fnmatch import fnmatch
# Assuming heuristic_base is available in the execution environment
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper function to split a PDDL fact string into its predicate and arguments."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

def match(fact, *args):
    """Helper function to match a fact against a pattern using fnmatch."""
    parts = get_parts(fact)
    # Ensure we have enough parts to match against the pattern
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    Estimates the number of actions required to serve all children.
    The heuristic is calculated by summing the estimated number of actions
    of each type (make_sandwich, put_on_tray, move_tray, serve_sandwich)
    needed to transition from the current state to a state where all children
    are served. It considers the deficits in suitable sandwiches and the
    locations where deliveries are needed.

    Assumptions:
    - The problem instance is solvable (sufficient ingredients exist).
    - Trays have sufficient capacity to hold multiple sandwiches needed for a location.
    - The cost of each action is 1.
    - The heuristic is non-admissible, designed for greedy best-first search.

    Heuristic Initialization:
    In the constructor, static facts are processed to identify:
    - Which children are allergic to gluten.
    - The waiting place for each child.
    - Which bread portions are gluten-free.
    - Which content portions are gluten-free.
    This information is stored in dictionaries and sets for efficient lookup during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify unserved children: Count children for whom the '(served ?c)' predicate is false. This gives the number of 'serve' actions needed (N_unserved). Separate them into allergic (N_unserved_gf) and non-allergic (N_unserved_reg).
    2. Count available sandwiches: Determine the number of gluten-free (N_avail_gf) and regular (N_avail_reg) sandwiches that currently exist (either in the kitchen or on trays). Also count those specifically in the kitchen (N_kit_gf, N_kit_reg). A sandwich is considered regular if it's not explicitly marked as gluten-free.
    3. Estimate 'make_sandwich' actions: Calculate the deficit of suitable sandwiches needed (N_unserved_gf + N_unserved_reg) versus those available (N_avail_gf + N_avail_reg). The number of sandwiches that must be made (N_make_actions) is the maximum of 0 and this deficit.
    4. Estimate 'put_on_tray' actions: Sandwiches that are made end up in the kitchen. Sandwiches already in the kitchen also need to be put on trays. The number of 'put_on_tray' actions needed is the sum of sandwiches currently in the kitchen (N_kit_gf + N_kit_reg) and those that need to be made (N_make_actions).
    5. Estimate 'move_tray' actions: Identify places where unserved children are waiting. For each such place, count how many unserved children are waiting there (N_unserved_at_p). Then, count how many of these children *could* be served by the suitable sandwiches already present on trays *at that specific place* (N_ontray_children_served_at_p). If N_unserved_at_p is greater than N_ontray_children_served_at_p, it means more sandwiches need to be delivered to this location, requiring at least one 'move_tray' action to this place. N_tray_moves_needed is the count of distinct places that require such additional deliveries.
    6. Sum the estimated actions: The total heuristic value is the sum of N_unserved, N_make_actions, N_put_on_tray_actions, and N_tray_moves_needed.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        self.allergic_children = set()
        self.not_allergic_children = set()
        self.waiting_places = {} # child -> place
        self.gf_bread = set()
        self.gf_content = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'allergic_gluten':
                self.allergic_children.add(parts[1])
            elif parts[0] == 'not_allergic_gluten':
                self.not_allergic_children.add(parts[1])
            elif parts[0] == 'waiting':
                self.waiting_places[parts[1]] = parts[2]
            elif parts[0] == 'no_gluten_bread':
                self.gf_bread.add(parts[1])
            elif parts[0] == 'no_gluten_content':
                self.gf_content.add(parts[1])

        # All children are initially waiting according to the domain structure
        self.all_children = set(self.waiting_places.keys())

    def __call__(self, node):
        state = node.state

        # --- Step 1: Identify unserved children ---
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children = self.all_children - served_children

        if not unserved_children:
            return 0 # Goal state

        N_unserved = len(unserved_children)
        N_unserved_gf = len([c for c in unserved_children if c in self.allergic_children])
        N_unserved_reg = len([c for c in unserved_children if c in self.not_allergic_children])

        # --- Step 2: Count available sandwiches and ingredients ---
        at_kitchen_sandwich_set = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_sandwich", "*")}
        ontray_sandwiches_set = {get_parts(fact)[1] for fact in state if match(fact, "ontray", "*", "*")}
        no_gluten_sandwiches_state = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}

        # Determine gluten status of existing sandwiches
        sandwich_is_gf = {}
        all_existing_sandwiches = at_kitchen_sandwich_set | ontray_sandwiches_set
        for s in all_existing_sandwiches:
             sandwich_is_gf[s] = s in no_gluten_sandwiches_state

        # Count available sandwiches by type and location
        N_avail_gf = sum(1 for s in all_existing_sandwiches if sandwich_is_gf.get(s, False))
        N_avail_reg = sum(1 for s in all_existing_sandwiches if not sandwich_is_gf.get(s, False))

        N_kit_gf = sum(1 for s in at_kitchen_sandwich_set if sandwich_is_gf.get(s, False))
        N_kit_reg = sum(1 for s in at_kitchen_sandwich_set if not sandwich_is_gf.get(s, False))

        # --- Step 3: Estimate 'make_sandwich' actions ---
        # Deficit of suitable sandwiches needed vs total available
        D_sandwich_gf = max(0, N_unserved_gf - N_avail_gf)
        D_sandwich_reg = max(0, N_unserved_reg - N_avail_reg)
        N_make_actions = D_sandwich_gf + D_sandwich_reg

        # --- Step 4: Estimate 'put_on_tray' actions ---
        # Sandwiches that are made end up in the kitchen.
        # Sandwiches already in the kitchen also need put_on_tray.
        N_put_on_tray_actions = N_kit_gf + N_kit_reg + N_make_actions

        # --- Step 5: Estimate 'move_tray' actions ---
        # Identify places with unserved children
        places_with_unserved = {self.waiting_places[c] for c in unserved_children}

        # Map trays to their current location
        tray_locations = {}
        for fact in state:
             if match(fact, "at", "*", "*"):
                  parts = get_parts(fact)
                  # In this domain, only trays are 'at' places.
                  tray_locations[parts[1]] = parts[2]

        # Map trays to sandwiches on them
        sandwiches_on_tray_map = {} # tray -> set of sandwiches
        for fact in state:
            if match(fact, "ontray", "*", "*"):
                s, t = get_parts(fact)[1:]
                sandwiches_on_tray_map.setdefault(t, set()).add(s)

        N_tray_moves_needed = 0
        for place in places_with_unserved:
            unserved_children_at_p = [c for c in unserved_children if self.waiting_places[c] == place]
            N_unserved_at_p = len(unserved_children_at_p)

            # Count how many unserved children at 'p' can be served by the sandwiches
            # currently on trays *at p*.
            children_served_by_ontray_at_p = set()
            trays_at_p = {t for t, loc in tray_locations.items() if loc == place}

            for tray in trays_at_p:
                 sandwiches_on_this_tray = sandwiches_on_tray_map.get(tray, set())
                 for s in sandwiches_on_this_tray:
                     is_gf_sandwich = sandwich_is_gf.get(s, False)
                     for child in unserved_children_at_p:
                         child_is_allergic = child in self.allergic_children
                         # Check if sandwich s is suitable for child
                         if (is_gf_sandwich and child_is_allergic) or (not is_gf_sandwich and not child_is_allergic):
                             children_served_by_ontray_at_p.add(child) # Add the child if a suitable sandwich exists for them on a tray at 'p'

            N_ontray_children_served_at_p = len(children_served_by_ontray_at_p)

            # If the number of unserved children at 'p' is greater than the number
            # who can be served by sandwiches already on trays at 'p',
            # then at least one more tray move is needed to bring the deficit sandwiches.
            if N_unserved_at_p > N_ontray_children_served_at_p:
                 N_tray_moves_needed += 1

        # --- Step 6: Sum the estimated actions ---
        h = N_unserved + N_make_actions + N_put_on_tray_actions + N_tray_moves_needed

        return h
