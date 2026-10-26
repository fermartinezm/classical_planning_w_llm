from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the minimum number of actions required to serve
    all children. It counts the total demand for sandwiches (broken down by
    allergy type) and the available supply at different stages (on tray,
    in kitchen, makable). It satisfies the demand by drawing from the cheapest
    available global supply pools (cost 1 for serve if already at destination,
    cost 2 for move+serve if on tray elsewhere, cost 3 for put+move+serve if
    in kitchen, cost 4 for make+put+move+serve if needs making).

    # Assumptions
    - Each child needs exactly one sandwich.
    - Allergic children require gluten-free sandwiches.
    - Non-allergic children can eat any sandwich.
    - Multiple sandwiches can be placed on a single tray.
    - A tray can be moved between any two places.
    - Bread and content portions are consumed when making a sandwich.
    - A 'notexist' sandwich object is consumed when making a sandwich.
    - The costs of make, put, move, and serve actions are all 1.
    - Any sandwich on any tray can be moved to any location.
    - Any sandwich in the kitchen can be put on any tray in the kitchen.

    # Heuristic Initialization
    - Identify all children that need to be served from the goal conditions.
    - Extract static information: which children are allergic, where children are waiting,
      which bread/content portions are gluten-free.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children who are not yet served. If none, the heuristic is 0.
    2. Determine the type of sandwich needed (gluten-free or any) and the waiting location for each unserved child. Calculate the total global demand for gluten-free and regular sandwiches (`total_gf_demand`, `total_reg_demand`).
    3. Count the total available sandwiches by type (GF/Reg) and their initial stage globally:
       - On a tray anywhere (`initial_avail_gf_ontray`, `initial_avail_reg_ontray`).
       - In the kitchen (`avail_gf_kitchen_sandwich`, `avail_reg_kitchen_sandwich`).
       - Makable from components in the kitchen (`avail_gf_makable`, `avail_reg_makable`), limited by available 'notexist' sandwich objects (`avail_notexist`).
    4. Count the number of sandwiches already on trays at the *correct* destination locations that can satisfy the demand (`num_s3`). These sandwiches only require the 'serve' action (cost 1).
    5. Calculate the total heuristic cost by satisfying the total demand (`total_gf_demand`, `total_reg_demand`) from the available global supply pools, prioritizing the cheapest stages:
       - Use sandwiches from 'on tray anywhere' pool (cost 2: move + serve).
       - Use sandwiches from 'in kitchen' pool (cost 3: put + move + serve).
       - Use sandwiches from 'makable' pool (cost 4: make + put + move + serve).
    6. This global calculation assumes all sandwiches come from stages 0, 1, or 2 and have costs 4, 3, or 2 respectively. Correct the total cost by subtracting the cost difference (2 - 1 = 1) for each sandwich that was actually available at Stage 3 (on tray at the correct destination).
    7. If, after exhausting all available sandwiches/components/notexist objects, there is still demand, the state is likely unsolvable, and the heuristic returns infinity.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Identify all children that need to be served (from goal)
        self.goal_children = {get_parts(goal)[1] for goal in self.goals if get_parts(goal) and get_parts(goal)[0] == 'served'}

        # Extract static information
        self.allergic_children = {get_parts(fact)[1] for fact in static_facts if get_parts(fact) and get_parts(fact)[0] == 'allergic_gluten'}
        self.waiting_places = {get_parts(fact)[1]: get_parts(fact)[2] for fact in static_facts if get_parts(fact) and get_parts(fact)[0] == 'waiting'}
        self.no_gluten_breads = {get_parts(fact)[1] for fact in static_facts if get_parts(fact) and get_parts(fact)[0] == 'no_gluten_bread'}
        self.no_gluten_contents = {get_parts(fact)[1] for fact in static_facts if get_parts(fact) and get_parts(fact)[0] == 'no_gluten_content'}

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unserved children
        unserved_children = {c for c in self.goal_children if f'(served {c})' not in state}

        if not unserved_children:
            return 0  # Goal reached

        # 2. Determine demand per location and total global demand
        needed_gf_at = defaultdict(int)
        needed_reg_at = defaultdict(int)
        all_waiting_places = set()
        for child in unserved_children:
            # Ensure child is in waiting_places (should be based on static facts)
            if child in self.waiting_places:
                place = self.waiting_places[child]
                all_waiting_places.add(place)
                if child in self.allergic_children:
                    needed_gf_at[place] += 1
                else:
                    needed_reg_at[place] += 1
            # else: Child is unserved but not waiting? Assume solvable instances don't have this.

        total_gf_demand = sum(needed_gf_at.values())
        total_reg_demand = sum(needed_reg_at.values())

        # 3. Count available sandwiches/components globally by initial state
        avail_gf_ontray_at = defaultdict(int)
        avail_reg_ontray_at = defaultdict(int)
        avail_gf_kitchen_sandwich = 0
        avail_reg_kitchen_sandwich = 0
        avail_gf_bread_kitchen = 0
        avail_reg_bread_kitchen = 0
        avail_gf_content_kitchen = 0
        avail_reg_content_kitchen = 0
        avail_notexist = 0

        sandwich_is_gf = {}
        for fact in state:
             parts = get_parts(fact)
             if parts and parts[0] == 'no_gluten_sandwich':
                 sandwich_is_gf[parts[1]] = True

        tray_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'at' and parts[1].startswith('tray'):
                tray_locations[parts[1]] = parts[2]

        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'ontray':
                s, t = parts[1], parts[2]
                if t in tray_locations:
                    loc = tray_locations[t]
                    is_gf = sandwich_is_gf.get(s, False)
                    if is_gf:
                        avail_gf_ontray_at[loc] += 1
                    else:
                        avail_reg_ontray_at[loc] += 1
            elif parts and parts[0] == 'at_kitchen_sandwich':
                s = parts[1]
                is_gf = sandwich_is_gf.get(s, False)
                if is_gf:
                    avail_gf_kitchen_sandwich += 1
                else:
                    avail_reg_kitchen_sandwich += 1
            elif parts and parts[0] == 'at_kitchen_bread':
                b = parts[1]
                if b in self.no_gluten_breads:
                    avail_gf_bread_kitchen += 1
                else:
                    avail_reg_bread_kitchen += 1
            elif parts and parts[0] == 'at_kitchen_content':
                c = parts[1]
                if c in self.no_gluten_contents:
                    avail_gf_content_kitchen += 1
                else:
                    avail_reg_content_kitchen += 1
            elif parts and parts[0] == 'notexist':
                avail_notexist += 1

        avail_gf_makable = min(avail_gf_bread_kitchen, avail_gf_content_kitchen)
        avail_reg_makable = min(avail_reg_bread_kitchen, avail_reg_content_kitchen)

        initial_avail_gf_ontray = sum(avail_gf_ontray_at.values())
        initial_avail_reg_ontray = sum(avail_reg_ontray_at.values())
        initial_avail_gf_kitchen = avail_gf_kitchen_sandwich
        initial_avail_reg_kitchen = avail_reg_kitchen_sandwich
        initial_avail_gf_makable = avail_gf_makable
        initial_avail_reg_makable = avail_reg_makable
        initial_avail_notexist = avail_notexist

        # 4. Count sandwiches already at destination on tray (Stage 3)
        num_s3 = 0
        # Use copies of needed/avail counts for this calculation
        needed_gf_at_copy = needed_gf_at.copy()
        needed_reg_at_copy = needed_reg_at.copy()
        avail_gf_ontray_at_copy = avail_gf_ontray_at.copy()
        avail_reg_ontray_at_copy = avail_reg_ontray_at.copy()

        for p in all_waiting_places:
            needed_gf = needed_gf_at_copy[p]
            needed_reg = needed_reg_at_copy[p]
            avail_gf = avail_gf_ontray_at_copy[p]
            avail_reg = avail_reg_ontray_at_copy[p]

            # Serve allergic children first with GF sandwiches at p
            use_gf_for_gf = min(needed_gf, avail_gf)
            needed_gf -= use_gf_for_gf
            avail_gf -= use_gf_for_gf
            num_s3 += use_gf_for_gf

            # Serve non-allergic children with regular sandwiches at p
            use_reg_for_reg = min(needed_reg, avail_reg)
            needed_reg -= use_reg_for_reg
            avail_reg -= use_reg_for_reg
            num_s3 += use_reg_for_reg

            # Serve remaining non-allergic children with remaining GF sandwiches at p
            use_gf_for_reg = min(needed_reg, avail_gf)
            needed_reg -= use_gf_for_reg
            avail_gf -= use_gf_for_reg
            num_s3 += use_gf_for_reg

        # 5. Calculate global cost assuming all come from S0, S1, S2
        cost_from_s0_s1_s2 = 0
        gf_needed = total_gf_demand
        reg_needed = total_reg_demand

        # Use sandwiches on tray anywhere (cost 2)
        avail_gf_ontray = initial_avail_gf_ontray
        avail_reg_ontray = initial_avail_reg_ontray

        use_gf_ontray = min(gf_needed, avail_gf_ontray)
        gf_needed -= use_gf_ontray
        avail_gf_ontray -= use_gf_ontray
        cost_from_s0_s1_s2 += use_gf_ontray * 2

        use_reg_ontray = min(reg_needed, avail_reg_ontray)
        reg_needed -= use_reg_ontray
        avail_reg_ontray -= use_reg_ontray
        cost_from_s0_s1_s2 += use_reg_ontray * 2

        use_gf_ontray_for_reg = min(reg_needed, avail_gf_ontray)
        reg_needed -= use_gf_ontray_for_reg
        avail_gf_ontray -= use_gf_ontray_for_reg
        cost_from_s0_s1_s2 += use_gf_ontray_for_reg * 2

        # Use sandwiches in kitchen (cost 3)
        avail_gf_kitchen = initial_avail_gf_kitchen
        avail_reg_kitchen = initial_avail_reg_kitchen

        use_gf_kitchen = min(gf_needed, avail_gf_kitchen)
        gf_needed -= use_gf_kitchen
        avail_gf_kitchen -= use_gf_kitchen
        cost_from_s0_s1_s2 += use_gf_kitchen * 3

        use_reg_kitchen = min(reg_needed, avail_reg_kitchen)
        reg_needed -= use_reg_kitchen
        avail_reg_kitchen -= use_reg_kitchen
        cost_from_s0_s1_s2 += use_reg_kitchen * 3

        use_gf_kitchen_for_reg = min(reg_needed, avail_gf_kitchen)
        reg_needed -= use_gf_kitchen_for_reg
        avail_gf_kitchen -= use_gf_kitchen_for_reg
        cost_from_s0_s1_s2 += use_gf_kitchen_for_reg * 3

        # Use makable sandwiches (cost 4)
        avail_gf_s0 = initial_avail_gf_makable
        avail_reg_s0 = initial_avail_reg_makable
        avail_notexist_rem = initial_avail_notexist

        use_gf_makable = min(gf_needed, avail_gf_s0, avail_notexist_rem)
        gf_needed -= use_gf_makable
        avail_gf_s0 -= use_gf_makable
        avail_notexist_rem -= use_gf_makable
        cost_from_s0_s1_s2 += use_gf_makable * 4

        use_reg_makable = min(reg_needed, avail_reg_s0, avail_notexist_rem)
        reg_needed -= use_reg_makable
        avail_reg_s0 -= use_reg_makable
        avail_notexist_rem -= use_reg_makable
        cost_from_s0_s1_s2 += use_reg_makable * 4

        use_gf_makable_for_reg = min(reg_needed, avail_gf_s0, avail_notexist_rem)
        reg_needed -= use_gf_makable_for_reg
        avail_gf_s0 -= use_gf_makable_for_reg
        avail_notexist_rem -= use_gf_makable_for_reg
        cost_from_s0_s1_s2 += use_gf_makable_for_reg * 4

        # 7. If any demand remains, infinity
        if gf_needed > 0 or reg_needed > 0:
            return float('inf')

        # 6. Correct the cost for sandwiches already at Stage 3
        # Each sandwich at Stage 3 was counted with cost 2 in the global calculation
        # but should only cost 1 (the serve action).
        # So, subtract the cost difference (2 - 1 = 1) for each Stage 3 sandwich.
        total_cost = cost_from_s0_s1_s2 - num_s3 * 1

        return total_cost
