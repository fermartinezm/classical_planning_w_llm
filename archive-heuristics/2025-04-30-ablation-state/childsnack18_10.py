from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or empty facts
    fact = fact.strip()
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return [] # Or handle as an error if strict parsing is needed
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
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
    This heuristic estimates the number of actions required to serve all children
    who are currently waiting and not yet served. It breaks down the cost for
    each unserved child based on the current availability and location of
    appropriate sandwiches, prioritizing sandwiches that are closer to being
    delivered.

    # Assumptions
    - Each unserved child requires exactly one appropriate sandwich.
    - Gluten-allergic children require gluten-free sandwiches. Non-allergic
      children can receive regular sandwiches.
    - Ingredients for making sandwiches are always available in the kitchen
      when needed (ingredient counts are not strictly tracked by the heuristic).
    - Trays are available when needed (tray counts are not strictly tracked
      by the heuristic, except for their location).
    - A single tray move can potentially satisfy the needs of multiple children
      at the same location. This heuristic simplifies by counting delivery costs
      per *need* fulfilled, based on the sandwich's state, rather than counting
      distinct tray moves or put-on-tray actions globally.
    - The cost of actions is uniform (1).

    # Heuristic Initialization
    The heuristic pre-processes static information from the task:
    - Identifies the gluten status (allergic/not allergic) for each child.
    - Records the waiting location for each child.
    - Collects all relevant place names (kitchen and waiting locations).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:

    1.  **Identify Unserved Children:** Determine which children are in the goal
        set but do not have the `(served ?c)` predicate in the current state.
        If all children are served, the heuristic is 0. The number of unserved
        children is the base heuristic (cost of `serve` actions).

    2.  **Categorize Needs:** For each unserved child, identify their waiting
        location and the type of sandwich they need (regular or gluten-free)
        based on their allergy status. Sum up the counts of needed sandwiches
        per location and gluten type.

    3.  **Count Available Sandwiches:** Inventory the sandwiches in the current
        state based on their location (`at_kitchen_sandwich`, `ontray` at
        various places) and their gluten status (`no_gluten_sandwich`).

    4.  **Calculate Covered Needs:** For each location, determine how many of
        the needed sandwiches (by type) are *already* available on trays at
        that specific location. These needs are "covered" locally and do not
        require delivery actions for the sandwich itself (only the final serve).

    5.  **Calculate Unsatisfied Needs:** The total number of unserved children
        minus the total number of needs covered locally represents the number
        of sandwich needs that must be met by sandwiches from elsewhere or by
        making new ones. These needs require delivery actions.

    6.  **Allocate Unsatisfied Needs to Supply Pools:** These unsatisfied needs
        are fulfilled by sandwiches from different "supply pools", prioritized
        by their readiness for delivery:
        a.  **Pool A (Ontray Elsewhere):** Sandwiches already on trays anywhere,
            EXCEPT those already used to cover needs at the correct location.
            Each need met by this pool requires a `move_tray` action (cost 1
            for the delivery part).
        b.  **Pool B (At Kitchen Sandwich):** Sandwiches existing in the kitchen
            but not yet on a tray. Each need met by this pool requires a
            `put_on_tray` action and a `move_tray` action (cost 2 for the
            delivery part).
        c.  **Pool C (Needs Make):** Sandwiches that do not yet exist. Each need
            met by this pool requires a `make_sandwich` action, a `put_on_tray`
            action, and a `move_tray` action (cost 3 for the delivery part).

        The allocation is done greedily: satisfy as many needs as possible from
        Pool A, then from Pool B, and finally the remainder from Pool C.
        Allocation is done separately for regular and gluten-free needs.

    7.  **Sum Costs:** The total heuristic value is the sum of:
        -   1 for each unserved child (representing the final `serve` action).
        -   1 for each need met by Pool A (representing the `move_tray` cost).
        -   2 for each need met by Pool B (representing the `put_on_tray` + `move_tray` cost).
        -   3 for each need met by Pool C (representing the `make_sandwich` + `put_on_tray` + `move_tray` cost).
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        # The base class Heuristic might not have an __init__ that takes task,
        # but it's good practice to call super if it did.
        super().__init__(task)

        # Assuming task.goals is a frozenset of goal facts like `(served child1)`
        self.goal_served_children = {get_parts(goal)[1] for goal in task.goals if match(goal, "served", "*")}

        # Extract static facts
        self.child_gluten_status = {} # child_name -> 'gluten' or 'no_gluten' (type of sandwich needed)
        self.child_location = {}      # child_name -> place_name
        self.all_places = {'kitchen'} # Start with the constant kitchen

        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts
            predicate = parts[0]
            if predicate == 'allergic_gluten':
                self.child_gluten_status[parts[1]] = 'no_gluten' # Allergic needs NO_GLUTEN sandwich
            elif predicate == 'not_allergic_gluten':
                self.child_gluten_status[parts[1]] = 'gluten' # Not allergic needs GLUTEN (regular) sandwich
            elif predicate == 'waiting':
                self.child_location[parts[1]] = parts[2]
                self.all_places.add(parts[2])

        # Also collect places from initial tray locations
        for fact in task.initial_state:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] == 'at' and len(parts) == 3: # (at ?t ?p)
                 # Assuming only trays have 'at' predicates in initial state relevant to places
                 self.all_places.add(parts[2])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify Unserved Children
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children = {c for c in self.goal_served_children if c not in served_children}

        if not unserved_children:
            return 0 # Goal reached

        # Base heuristic: 1 action per unserved child (the serve action)
        h = len(unserved_children)

        # 2. Categorize Needs
        # Needs[(place, gluten_status)] = count
        needs = defaultdict(int)
        for child in unserved_children:
            # Get location and required sandwich type from pre-processed static info
            location = self.child_location.get(child)
            required_type = self.child_gluten_status.get(child)

            if location and required_type: # Ensure child has a waiting location and allergy status defined
                 needs[(location, required_type)] += 1
            # else: This child cannot be served based on static info. Ignore for heuristic.

        # 3. Count Available Sandwiches in State
        at_kitchen_sandwich_list = [get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_sandwich", "*")]
        ontray_list = [(get_parts(fact)[1], get_parts(fact)[2]) for fact in state if match(fact, "ontray", "*", "*")]
        at_tray_dict = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*")} # Tray -> Place
        no_gluten_sandwich_list = [get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")]

        available_at_kitchen_sandwich_reg = 0
        available_at_kitchen_sandwich_gf = 0
        for s in at_kitchen_sandwich_list:
            if s in no_gluten_sandwich_list:
                available_at_kitchen_sandwich_gf += 1
            else:
                available_at_kitchen_sandwich_reg += 1

        available_ontray_at_p_reg = {p: 0 for p in self.all_places}
        available_ontray_at_p_gf = {p: 0 for p in self.all_places}
        for s, t in ontray_list:
            if t in at_tray_dict: # Ensure tray location is known
                p = at_tray_dict[t]
                if p in self.all_places: # Ensure place is recognized
                    if s in no_gluten_sandwich_list:
                        available_ontray_at_p_gf[p] += 1
                    else:
                        available_ontray_at_p_reg[p] += 1

        # 4. Calculate Covered Needs (already on tray at location)
        covered_at_p_reg = {}
        covered_at_p_gf = {}
        for p in self.all_places:
            needed_reg = needs.get((p, 'gluten'), 0)
            needed_gf = needs.get((p, 'no_gluten'), 0)
            available_reg = available_ontray_at_p_reg.get(p, 0)
            available_gf = available_ontray_at_p_gf.get(p, 0)

            covered_at_p_reg[p] = min(needed_reg, available_reg)
            covered_at_p_gf[p] = min(needed_gf, available_gf)

        # 5. Calculate Unsatisfied Needs (need delivery or making)
        unsatisfied_needs_reg = sum(needs.get((p, 'gluten'), 0) - covered_at_p_reg[p] for p in self.all_places)
        unsatisfied_needs_gf = sum(needs.get((p, 'no_gluten'), 0) - covered_at_p_gf[p] for p in self.all_places)

        # 6. & 7. Allocate Unsatisfied Needs and Sum Costs for Delivery

        # Pool A (Ontray elsewhere): Sandwiches ontray anywhere, not already covering a need at location
        total_ontray_reg = sum(available_ontray_at_p_reg.values())
        total_ontray_gf = sum(available_ontray_at_p_gf.values())
        total_covered_at_location_reg = sum(covered_at_p_reg.values())
        total_covered_at_location_gf = sum(covered_at_p_gf.values())

        available_pool_A_reg = total_ontray_reg - total_covered_at_location_reg
        available_pool_A_gf = total_ontray_gf - total_covered_at_location_gf

        # Pool B (At kitchen sandwich)
        available_pool_B_reg = available_at_kitchen_sandwich_reg
        available_pool_B_gf = available_at_kitchen_sandwich_gf

        # Pool C (Needs make) - calculated during allocation

        # Allocate regular needs (cost 1 for Pool A, 2 for Pool B, 3 for Pool C)
        needs_reg = unsatisfied_needs_reg

        met_pool_A_reg = min(needs_reg, available_pool_A_reg)
        h += met_pool_A_reg * 1 # Cost for move_tray
        needs_reg -= met_pool_A_reg

        met_pool_B_reg = min(needs_reg, available_pool_B_reg)
        h += met_pool_B_reg * 2 # Cost for put_on_tray + move_tray
        needs_reg -= met_pool_B_reg

        met_pool_C_reg = needs_reg # Remaining needs must be made
        h += met_pool_C_reg * 3 # Cost for make_sandwich + put_on_tray + move_tray

        # Allocate gluten-free needs (cost 1 for Pool A, 2 for Pool B, 3 for Pool C)
        needs_gf = unsatisfied_needs_gf

        met_pool_A_gf = min(needs_gf, available_pool_A_gf)
        h += met_pool_A_gf * 1 # Cost for move_tray
        needs_gf -= met_pool_A_gf

        met_pool_B_gf = min(needs_gf, available_pool_B_gf)
        h += met_pool_B_gf * 2 # Cost for put_on_tray + move_tray
        needs_gf -= met_pool_B_gf

        met_pool_C_gf = needs_gf # Remaining needs must be made
        h += met_pool_C_gf * 3 # Cost for make_sandwich + put_on_tray + move_tray

        return h
