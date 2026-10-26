from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all unserved children.
    It counts the total number of gluten-free and any-type sandwiches needed by unserved children.
    It then counts the available sandwiches in different "pools" based on their readiness (at child's location on tray, on tray elsewhere, in kitchen, makable).
    Finally, it greedily assigns available sandwiches from the cheapest pools to satisfy the needs, prioritizing gluten-free requirements, and sums the estimated costs.

    # Assumptions
    - Each unserved child requires exactly one sandwich of the appropriate type (gluten-free if allergic, any if not).
    - Trays can be moved between any two places with a single `move_tray` action.
    - Putting a sandwich on a tray requires a tray in the kitchen (heuristic simplifies this by assuming a tray is available when needed for a kitchen sandwich).
    - Making a sandwich requires available bread, content, and a `notexist` sandwich object in the kitchen.
    - Sufficient bread, content, and `notexist` sandwich objects are available in the initial state to make all potentially needed sandwiches (heuristic assumes this for the `makable` pool count).
    - The cost of actions is 1. The heuristic sums these unit costs based on the estimated steps for each needed sandwich.

    # Heuristic Initialization
    The heuristic extracts static information from the task:
    - The set of children that need to be served (from goal conditions).
    - The allergy status of each child (`allergic_gluten`).
    - the waiting location of each child (`waiting`).
    - The gluten-free status of bread and content types (`no_gluten_bread`, `no_gluten_content`).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all children who are in the goal set but are not yet marked as `served`. These are the unserved children.
    2. If there are no unserved children, the heuristic is 0.
    3. Count the total number of unserved allergic children (`needed_gf`) and unserved non-allergic children (`needed_any`).
    4. Extract relevant facts from the state: tray locations, sandwiches on trays, sandwiches in kitchen, `notexist` sandwich objects, kitchen bread/content, and sandwich gluten status.
    5. Count available sandwiches by type (GF/Any) and "pool" based on their current status:
       - Pool 1 (Cost 1: serve): Suitable sandwich on a tray at a location where an unserved child is waiting.
       - Pool 2 (Cost 2: move + serve): Suitable sandwich on a tray at a location where no unserved child is waiting.
       - Pool 3 (Cost 3: put + move + serve): Suitable sandwich in the kitchen.
       - Pool 4 (Cost 4: make + put + move + serve): Suitable sandwich that can be made from kitchen resources and `notexist` objects.
       (Note: A GF sandwich is suitable for both allergic and non-allergic children).
    6. Calculate the number of GF sandwiches that can be made (`makable_gf`) and the total number of sandwiches of any type that can be made (`makable_total`) based on kitchen resources and `notexist` objects.
    7. Initialize the heuristic cost to 0.
    8. Greedily satisfy the `needed_gf` sandwiches from the cheapest pools first (Pool 1 -> Pool 2 -> Pool 3 -> Pool 4), adding the corresponding cost multiplier (1, 2, 3, 4) for each sandwich used from that pool.
    9. Greedily satisfy the `needed_any` sandwiches from the remaining sandwiches in the cheapest pools (Pool 1 -> Pool 2 -> Pool 3 -> Pool 4), adding the corresponding cost multiplier (1, 2, 3, 4). When satisfying Any needs, remaining GF sandwiches in pools are used before non-GF ones in the same pool. For Pool 4, `makable_gf` is conceptually used first for GF needs, then the remaining `makable_total` is available for Any needs.
    10. The final heuristic value is the accumulated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract goal children
        self.goal_children = {get_parts(g)[1] for g in self.goals if get_parts(g)[0] == 'served'}

        # Extract static information
        self.is_allergic = {get_parts(s)[1]: True for s in static_facts if get_parts(s)[0] == 'allergic_gluten'}
        self.waiting_location = {get_parts(s)[1]: get_parts(s)[2] for s in static_facts if get_parts(s)[0] == 'waiting'}
        self.is_gf_bread = {get_parts(s)[1] for s in static_facts if get_parts(s)[0] == 'no_gluten_bread'}
        self.is_gf_content = {get_parts(s)[1] for s in static_facts if get_parts(s)[0] == 'no_gluten_content'}

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unserved children
        unserved_children = {c for c in self.goal_children if '(served ' + c + ')' not in state}

        # 2. If all children are served, the goal is reached.
        if not unserved_children:
            return 0

        # 3. Count needed sandwiches
        needed_gf = sum(1 for c in unserved_children if self.is_allergic.get(c, False))
        needed_any = sum(1 for c in unserved_children if not self.is_allergic.get(c, True))

        # 4. Extract state information
        tray_locations = {}
        ontray_sandwiches = {} # Map sandwich -> tray
        kitchen_sandwiches = set()
        notexist_sandwiches = set()
        kitchen_bread = set()
        kitchen_content = set()
        sandwich_is_gf = {} # Map sandwich -> bool

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1].startswith('tray'):
                tray_locations[parts[1]] = parts[2]
            elif parts[0] == 'ontray':
                ontray_sandwiches[parts[1]] = parts[2]
            elif parts[0] == 'at_kitchen_sandwich':
                kitchen_sandwiches.add(parts[1])
            elif parts[0] == 'notexist':
                notexist_sandwiches.add(parts[1])
            elif parts[0] == 'at_kitchen_bread':
                kitchen_bread.add(parts[1])
            elif parts[0] == 'at_kitchen_content':
                kitchen_content.add(parts[1])
            elif parts[0] == 'no_gluten_sandwich':
                sandwich_is_gf[parts[1]] = True

        # Helper to check if a sandwich is GF (defaults to False if not explicitly stated)
        is_gf_sandwich = lambda s: sandwich_is_gf.get(s, False)

        # 5. Count available sandwiches by type and pool
        avail_gf_p1_count = 0 # GF sandwich on tray at *any* location of an unserved child
        avail_any_p1_count = 0 # Any sandwich (non-GF) on tray at *any* location of an unserved child
        avail_gf_p2_count = 0 # GF sandwich on tray elsewhere
        avail_any_p2_count = 0 # Any sandwich (non-GF) on tray elsewhere
        avail_gf_p3_count = 0 # GF sandwich in kitchen
        avail_any_p3_count = 0 # Any sandwich (non-GF) in kitchen

        unserved_child_locations = set(self.waiting_location.get(c) for c in unserved_children if self.waiting_location.get(c))

        # Count Pool 1 (at location)
        for s, t in ontray_sandwiches.items():
            loc = tray_locations.get(t)
            if loc in unserved_child_locations:
                if is_gf_sandwich(s):
                    avail_gf_p1_count += 1
                else:
                    avail_any_p1_count += 1

        # Count Pool 2 (ontray elsewhere)
        for s, t in ontray_sandwiches.items():
            loc = tray_locations.get(t)
            if loc and loc not in unserved_child_locations:
                if is_gf_sandwich(s):
                    avail_gf_p2_count += 1
                else:
                    avail_any_p2_count += 1

        # Count Pool 3 (kitchen)
        for s in kitchen_sandwiches:
            if is_gf_sandwich(s):
                avail_gf_p3_count += 1
            else:
                avail_any_p3_count += 1

        # 6. Count makable sandwiches
        num_notexist = len(notexist_sandwiches)
        num_kitchen_bread_gf = len([b for b in kitchen_bread if b in self.is_gf_bread])
        num_kitchen_bread_any = len(kitchen_bread)
        num_kitchen_content_gf = len([c for c in kitchen_content if c in self.is_gf_content])
        num_kitchen_content_any = len(kitchen_content)

        makable_gf = min(num_notexist, num_kitchen_bread_gf, num_kitchen_content_gf)
        # Total makable sandwiches of any type
        makable_total = min(num_notexist, num_kitchen_bread_any, num_kitchen_content_any)


        # --- 7-9. Greedy assignment and cost calculation ---
        heuristic = 0
        needed_gf_rem = needed_gf
        needed_any_rem = needed_any

        # Pool 1 (Cost 1: serve)
        # Use GF from P1 for GF needs
        use_gf_p1 = min(needed_gf_rem, avail_gf_p1_count)
        heuristic += use_gf_p1 * 1
        needed_gf_rem -= use_gf_p1
        avail_gf_p1_count -= use_gf_p1 # Consume from pool

        # Use remaining Any from P1 for Any needs
        use_any_p1 = min(needed_any_rem, avail_any_p1_count) # Use non-GF Any at loc
        heuristic += use_any_p1 * 1
        needed_any_rem -= use_any_p1
        avail_any_p1_count -= use_any_p1 # Consume from pool

        # Pool 2 (Cost 2: move + serve)
        # Use GF from P2 for GF needs
        use_gf_p2 = min(needed_gf_rem, avail_gf_p2_count)
        heuristic += use_gf_p2 * 2
        needed_gf_rem -= use_gf_p2
        avail_gf_p2_count -= use_gf_p2 # Consume from pool

        # Use remaining Any from P2 for Any needs (including remaining GF)
        use_any_p2_gf = min(needed_any_rem, avail_gf_p2_count) # Use remaining GF from pool 2
        heuristic += use_any_p2_gf * 2
        needed_any_rem -= use_any_p2_gf
        avail_gf_p2_count -= use_any_p2_gf # Consume from pool

        use_any_p2_any = min(needed_any_rem, avail_any_p2_count) # Use non-GF Any elsewhere
        heuristic += use_any_p2_any * 2
        needed_any_rem -= use_any_p2_any
        avail_any_p2_count -= use_any_p2_any # Consume from pool


        # Pool 3 (Cost 3: put + move + serve)
        # Use GF from P3 for GF needs
        use_gf_p3 = min(needed_gf_rem, avail_gf_p3_count)
        heuristic += use_gf_p3 * 3
        needed_gf_rem -= use_gf_p3
        avail_gf_p3_count -= use_gf_p3 # Consume from pool

        # Use remaining Any from P3 for Any needs (including remaining GF)
        use_any_p3_gf = min(needed_any_rem, avail_gf_p3_count) # Use remaining GF from pool 3
        heuristic += use_any_p3_gf * 3
        needed_any_rem -= use_any_p3_gf
        avail_gf_p3_count -= use_any_p3_gf # Consume from pool

        use_any_p3_any = min(needed_any_rem, avail_any_p3_count) # Use non-GF Any kitchen
        heuristic += use_any_p3_any * 3
        needed_any_rem -= use_any_p3_any
        avail_any_p3_count -= use_any_p3_any # Consume from pool


        # Pool 4 (Cost 4: make + put + move + serve)
        # Use makable GF for GF needs
        use_gf_p4 = min(needed_gf_rem, makable_gf)
        heuristic += use_gf_p4 * 4
        needed_gf_rem -= use_gf_p4
        # makable_gf -= use_gf_p4 # Consume from makable pool (conceptually)

        # Use remaining makable (Any type) for Any needs
        # Total makable used for GF needs from Pool 4 is use_gf_p4.
        # Total makable sandwiches available is makable_total.
        # Remaining makable for Any needs is makable_total - use_gf_p4.
        makable_total_rem = makable_total - use_gf_p4
        use_any_p4 = min(needed_any_rem, makable_total_rem)
        heuristic += use_any_p4 * 4
        needed_any_rem -= use_any_p4

        # 10. Return the total heuristic value.
        # If needed_gf_rem > 0 or needed_any_rem > 0, it means we couldn't satisfy all needs
        # with the available/makable sandwiches according to this greedy assignment.
        # This heuristic doesn't return infinity, it just returns the cost for the sandwiches it could assign.
        # This is acceptable for a non-admissible heuristic.

        return heuristic
