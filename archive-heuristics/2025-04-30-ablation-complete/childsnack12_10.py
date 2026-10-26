from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact string defensively
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
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
    This heuristic estimates the number of actions required to serve all waiting children.
    It sums the estimated costs for:
    1. Making necessary sandwiches (distinguishing gluten-free needs).
    2. Putting sandwiches onto trays (from the kitchen).
    3. Moving trays to locations where children are waiting.
    4. Serving the sandwiches to the children.

    # Assumptions
    - Each unserved child requires one sandwich.
    - Gluten-allergic children require gluten-free sandwiches.
    - Non-allergic children can accept any sandwich.
    - Sandwiches progress through stages: notexist -> at_kitchen_sandwich -> ontray -> served.
    - Trays are needed at locations where children are waiting (if not already present).
    - Enough bread, content, and 'notexist' sandwich objects are available to make needed sandwiches.
    - Enough trays exist to be moved to all required locations.
    - The cost of moving a tray *to* the kitchen (if needed for 'put_on_tray') is not explicitly counted, assuming a tray is usually available at the kitchen or the cost is negligible compared to other steps.

    # Heuristic Initialization
    - Extract static information about children: their waiting places and allergy status.
    - Identify the set of goal facts (which children need to be served).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all children who are in the goal state (need to be served) but are not yet served in the current state.
    2. Count the number of unserved children who are gluten-allergic (`needed_gf`) and the total number of unserved children (`needed_any`).
    3. Count the number of available sandwiches in the kitchen (`at_kitchen_sandwich`) and on trays (`ontray`), distinguishing between gluten-free and regular based on the `no_gluten_sandwich` predicate.
    4. Count the number of trays currently located at each place.
    5. Identify the set of places where unserved children are waiting.
    6. Calculate the estimated cost:
       a.  **Cost to make GF sandwiches:** Count how many GF sandwiches are needed (`needed_gf`) that are not currently available (either in kitchen or on tray). Each needed GF sandwich not available costs 1 'make_sandwich_no_gluten' action.
       b.  **Cost to make Regular sandwiches:** Count how many non-allergic children need sandwiches (`needed_any - needed_gf`) that cannot be satisfied by available regular sandwiches *or* by surplus available GF sandwiches (GF sandwiches not needed by allergic children). Each such needed sandwich costs 1 'make_sandwich' action.
       c.  **Cost to put on tray:** Count how many sandwiches are needed on trays (`needed_any`) that are not currently on trays. Each such sandwich needs to be moved from the kitchen to a tray, costing 1 'put_on_tray' action. This assumes the sandwiches are either already in the kitchen or will be made and appear in the kitchen.
       d.  **Cost to move trays:** For each place where unserved children are waiting (excluding the kitchen), if no tray is currently present at that place, count 1 'move_tray' action needed to bring a tray there.
       e.  **Cost to serve:** Each unserved child requires one 'serve_sandwich' action. Count the total number of unserved children.
    7. Sum up the costs from steps 6a, 6b, 6c, 6d, and 6e. This sum is the heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal conditions (which children need to be served).
        - Static facts (`waiting` places and `allergic_gluten` status for children).
        """
        self.goals = task.goals  # Goal conditions.
        # Combine static facts and initial state facts that are invariant for child properties
        # Waiting and allergy status are typically static or only in initial state
        static_and_initial_facts = task.static | task.initial_state

        # Map child to their waiting place
        self.waiting_places = {}
        for fact in static_and_initial_facts:
            if match(fact, "waiting", "*", "*"):
                _, child, place = get_parts(fact)
                self.waiting_places[child] = place

        # Map child to their allergy status
        self.allergy_status = {}
        for fact in static_and_initial_facts:
            if match(fact, "allergic_gluten", "*"):
                _, child = get_parts(fact)
                self.allergy_status[child] = True
            elif match(fact, "not_allergic_gluten", "*"):
                 _, child = get_parts(fact)
                 self.allergy_status[child] = False

        # Identify the set of children that need to be served in the goal
        self.children_to_serve_goal = {
            get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")
        }

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify unserved children
        unserved_children = [
            c for c in self.children_to_serve_goal if f"(served {c})" not in state
        ]

        # If all children are served, the heuristic is 0.
        if not unserved_children:
            return 0

        # 2. Count needed sandwiches by type
        needed_gf = sum(1 for c in unserved_children if self.allergy_status.get(c, False))
        needed_any = len(unserved_children) # Total sandwiches needed

        # 3. Count available sandwiches by state and type
        kitchen_sandwiches = []
        ontray_sandwiches = [] # Store just sandwich names here for counting
        sandwich_gluten_status = {} # Map sandwich name to True if GF

        for fact in state:
            if match(fact, "at_kitchen_sandwich", "*"):
                _, s = get_parts(fact)
                kitchen_sandwiches.append(s)
            elif match(fact, "ontray", "*", "*"):
                _, s, t = get_parts(fact)
                ontray_sandwiches.append(s) # Count sandwich regardless of tray
            elif match(fact, "no_gluten_sandwich", "*"):
                _, s = get_parts(fact)
                sandwich_gluten_status[s] = True

        available_gf_kitchen = sum(1 for s in kitchen_sandwiches if sandwich_gluten_status.get(s, False))
        available_reg_kitchen = len(kitchen_sandwiches) - available_gf_kitchen

        available_gf_ontray = sum(1 for s in ontray_sandwiches if sandwich_gluten_status.get(s, False))
        available_reg_ontray = len(ontray_sandwiches) - available_gf_ontray

        # 4. Count trays at places
        trays_at_place = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, t, p = get_parts(fact)
                trays_at_place[p] = trays_at_place.get(p, 0) + 1

        # 5. Identify places with unserved children
        places_with_unserved = {self.waiting_places.get(c) for c in unserved_children if self.waiting_places.get(c) is not None}
        # Ensure 'kitchen' is a valid place if trays can be there (domain constant)
        # We only care about places with unserved children or kitchen for trays.
        # The trays_at_place dict will naturally include kitchen if a tray is there.

        # 6. Calculate estimated costs
        cost = 0

        # a) Cost to make GF sandwiches
        available_gf_total = available_gf_kitchen + available_gf_ontray
        cost_make_gf = max(0, needed_gf - available_gf_total)
        cost += cost_make_gf

        # b) Cost to make Regular sandwiches (for non-allergic children)
        # Sandwiches needed for non-allergic children = needed_any - needed_gf
        # Available for non-allergic = available_reg_kitchen + available_reg_ontray + surplus_available_gf
        served_allergic_by_available_gf = min(needed_gf, available_gf_total)
        surplus_available_gf = available_gf_total - served_allergic_by_available_gf
        available_for_not_allergic = available_reg_kitchen + available_reg_ontray + surplus_available_gf
        needed_not_allergic = needed_any - needed_gf
        cost_make_reg = max(0, needed_not_allergic - available_for_not_allergic)
        cost += cost_make_reg

        # c) Cost to put on tray
        # Sandwiches that need to be on trays = needed_any
        # Sandwiches already on trays = len(ontray_sandwiches)
        cost_put_on_tray = max(0, needed_any - len(ontray_sandwiches))
        cost += cost_put_on_tray

        # d) Cost to move trays
        cost_move_tray = 0
        for p in places_with_unserved:
            # Only need to move a tray if the place is not the kitchen AND no tray is there
            if p != 'kitchen' and trays_at_place.get(p, 0) == 0:
                cost_move_tray += 1
        cost += cost_move_tray

        # e) Cost to serve
        cost_serve = needed_any # Each unserved child needs one serve action
        cost += cost_serve

        return cost
