from fnmatch import fnmatch
# Assuming heuristics.heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if running standalone for testing
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static

        def __call__(self, node):
            raise NotImplementedError


def get_parts(fact):
    """Helper function to split a PDDL fact string into parts."""
    # Remove surrounding parentheses and split by space
    return fact[1:-1].split()

def match(fact, *args):
    """Helper function to match fact parts with arguments using fnmatch."""
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    The heuristic estimates the cost to reach the goal (all children served)
    by summing up the estimated steps required for each unserved child.
    It models a pipeline where sandwiches are made, put on trays, moved
    to children's locations, and finally served. The heuristic counts how
    many unserved children can be satisfied by sandwiches currently in
    different stages of this pipeline (ready-to-serve, on tray at kitchen,
    at kitchen, makeable from ingredients) and assigns a cost multiplier
    (1, 2, 3, 4 respectively) based on the estimated actions needed to get
    a sandwich from that stage to the child. It prioritizes using no-gluten
    sandwiches for allergic children. The heuristic is non-admissible but
    aims to be informative for greedy best-first search.

    Assumptions:
    - Each child needs exactly one sandwich.
    - A sandwich on a tray at a child's location can serve that child in 1 action (`serve`).
    - A sandwich on a tray at the kitchen needs 1 `move_tray` + 1 `serve` = 2 actions.
    - A sandwich at the kitchen needs 1 `put_on_tray` + 1 `move_tray` + 1 `serve` = 3 actions.
    - A sandwich that needs to be made needs 1 `make` + 1 `put_on_tray` + 1 `move_tray` + 1 `serve` = 4 actions.
    - Trays and sandwich objects are implicitly assumed to be available as needed for the actions counted in the multipliers (e.g., a tray is available at the kitchen for `put_on_tray`). The heuristic counts potential actions based on sandwich/ingredient availability, not limited by tray availability beyond their current location.
    - The heuristic assumes optimal allocation of available sandwiches to children, prioritizing those closest to being served and prioritizing no-gluten sandwiches for allergic children.

    Heuristic Initialization:
    The constructor extracts static information from the task:
    - `allergic_children`: A set of children who are allergic to gluten.
    - `waiting_places`: A dictionary mapping each child to the place where they are waiting.
    - `no_gluten_bread_items`: A set of bread portion names that are no-gluten.
    - `no_gluten_content_items`: A set of content portion names that are no-gluten.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify all children who are not yet served by checking the goal facts against the current state's `served` facts.
    2. Divide the unserved children into two groups: allergic and non-allergic, using the pre-computed static information.
    3. Extract relevant information from the current state: which sandwiches are no-gluten, which sandwiches are on which trays, where each tray is located, which sandwiches are at the kitchen (not on trays), which bread/content portions are at the kitchen, and which sandwich objects `notexist`.
    4. Count the number of available sandwiches based on their type (no-gluten vs any) and their current status/location, categorizing them into pools:
       - Pool 1: Sandwiches on trays located at places where unserved children are waiting.
       - Pool 2: Sandwiches on trays located at the kitchen.
       - Pool 3: Sandwiches located at the kitchen but not on trays.
       - Pool 4: Sandwiches that can potentially be made from available ingredients and `notexist` sandwich objects.
    5. Calculate the number of ingredients available for making sandwiches, distinguishing between no-gluten and any ingredients. Count available `notexist` sandwich objects.
    6. Initialize the heuristic value to 0.
    7. Iterate through the pools from 1 to 4. For each pool, determine how many of the *remaining* unserved allergic and non-allergic children can be satisfied by the sandwiches in this pool. Prioritize satisfying allergic children with no-gluten sandwiches.
    8. For each child satisfied by a sandwich from a pool, add the corresponding cost multiplier (1 for Pool 1, 2 for Pool 2, 3 for Pool 3, 4 for Pool 4) to the total heuristic value.
    9. Update the counts of remaining unserved allergic and non-allergic children after considering each pool.
    10. For Pool 4 (makeable sandwiches), calculate the maximum number of no-gluten and any sandwiches that can be made using the available ingredients and `notexist` objects. Use these makeable counts to satisfy the remaining unserved children, prioritizing no-gluten for allergic children. Add the cost multiplier (4) for each child satisfied this way.
    11. The final accumulated value is the heuristic estimate.
    """
    def __init__(self, task):
        super().__init__(task)
        self.allergic_children = set()
        self.waiting_places = {}
        self.no_gluten_bread_items = set()
        self.no_gluten_content_items = set()

        # Extract static information from task.static
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'allergic_gluten':
                self.allergic_children.add(parts[1])
            elif parts[0] == 'waiting':
                self.waiting_places[parts[1]] = parts[2]
            elif parts[0] == 'no_gluten_bread':
                self.no_gluten_bread_items.add(parts[1])
            elif parts[0] == 'no_gluten_content':
                self.no_gluten_content_items.add(parts[1])

    def __call__(self, node):
        state = node.state

        # 1. Identify unserved children
        served_children = {get_parts(fact)[1] for fact in state if match(fact, 'served', '*')}
        all_children = set(self.waiting_places.keys())
        unserved_children = all_children - served_children

        # If all children are served, goal reached, heuristic is 0
        if not unserved_children:
            return 0

        unserved_allergic = {c for c in unserved_children if c in self.allergic_children}
        unserved_non_allergic = unserved_children - unserved_allergic

        N_allergic_unserved = len(unserved_allergic)
        N_non_allergic_unserved = len(unserved_non_allergic)

        # 2. Identify sandwich properties and locations
        sandwich_is_ng = {get_parts(fact)[1] for fact in state if match(fact, 'no_gluten_sandwich', '*')}
        sandwiches_ontray = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, 'ontray', '*', '*')}
        tray_locations = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, 'at', '*', '*')}
        at_kitchen_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, 'at_kitchen_sandwich', '*')}
        notexist_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, 'notexist', '*')}

        # 3. Count available ingredients
        at_kitchen_bread = {get_parts(fact)[1] for fact in state if match(fact, 'at_kitchen_bread', '*')}
        at_kitchen_content = {get_parts(fact)[1] for fact in state if match(fact, 'at_kitchen_content', '*')}

        NG_bread_avail = len(at_kitchen_bread.intersection(self.no_gluten_bread_items))
        NG_content_avail = len(at_kitchen_content.intersection(self.no_gluten_content_items))
        Any_bread_avail = len(at_kitchen_bread) # Total bread
        Any_content_avail = len(at_kitchen_content) # Total content
        Notexist_avail = len(notexist_sandwiches)

        # 4. Count sandwiches in different pools
        NG_ontray_at_place_count = 0
        Any_ontray_at_place_count = 0 # Non-NG sandwiches
        NG_ontray_kitchen_count = 0
        Any_ontray_kitchen_count = 0 # Non-NG sandwiches
        NG_at_kitchen_count = 0
        Any_at_kitchen_count = 0 # Non-NG sandwiches

        # Identify relevant places (where unserved children are waiting)
        relevant_places = {self.waiting_places[c] for c in unserved_children if c in self.waiting_places}

        # Count sandwiches on trays
        for s, t in sandwiches_ontray.items():
            is_ng = s in sandwich_is_ng
            tray_loc = tray_locations.get(t)
            if tray_loc:
                if tray_loc == 'kitchen':
                    if is_ng:
                        NG_ontray_kitchen_count += 1
                    else:
                        Any_ontray_kitchen_count += 1
                elif tray_loc in relevant_places:
                     # Only count sandwiches on trays that are at a place where an unserved child is waiting
                    if is_ng:
                        NG_ontray_at_place_count += 1
                    else:
                        Any_ontray_at_place_count += 1

        # Count sandwiches at kitchen (not on trays)
        for s in at_kitchen_sandwiches:
            is_ng = s in sandwich_is_ng
            if is_ng:
                NG_at_kitchen_count += 1
            else:
                Any_at_kitchen_count += 1

        # 5. Calculate heuristic based on pools
        h = 0

        # Pool 1: On tray at relevant place (Cost 1: Serve)
        # Prioritize NG for allergic children
        ServeNG_from_P1 = min(N_allergic_unserved, NG_ontray_at_place_count)
        N_allergic_unserved -= ServeNG_from_P1
        h += ServeNG_from_P1 * 1

        # Use remaining NG sandwiches from this pool for non-allergic children if needed
        RemainingNG_P1 = NG_ontray_at_place_count - ServeNG_from_P1
        ServeAny_from_P1 = min(N_non_allergic_unserved, Any_ontray_at_place_count + RemainingNG_P1)
        N_non_allergic_unserved -= ServeAny_from_P1
        h += ServeAny_from_P1 * 1

        # Pool 2: On tray at kitchen (Cost 2: Move + Serve)
        # Prioritize NG for allergic children
        ServeNG_from_P2 = min(N_allergic_unserved, NG_ontray_kitchen_count)
        N_allergic_unserved -= ServeNG_from_P2
        h += ServeNG_from_P2 * 2

        # Use remaining NG sandwiches from this pool for non-allergic children if needed
        RemainingNG_P2 = NG_ontray_kitchen_count - ServeNG_from_P2
        ServeAny_from_P2 = min(N_non_allergic_unserved, Any_ontray_kitchen_count + RemainingNG_P2)
        N_non_allergic_unserved -= ServeAny_from_P2
        h += ServeAny_from_P2 * 2

        # Pool 3: At kitchen (Cost 3: Put + Move + Serve)
        # Prioritize NG for allergic children
        ServeNG_from_P3 = min(N_allergic_unserved, NG_at_kitchen_count)
        N_allergic_unserved -= ServeNG_from_P3
        h += ServeNG_from_P3 * 3

        # Use remaining NG sandwiches from this pool for non-allergic children if needed
        RemainingNG_P3 = NG_at_kitchen_count - ServeNG_from_P3
        ServeAny_from_P3 = min(N_non_allergic_unserved, Any_at_kitchen_count + RemainingNG_P3)
        N_non_allergic_unserved -= ServeAny_from_P3
        h += ServeAny_from_P3 * 3

        # Pool 4: Makeable (Cost 4: Make + Put + Move + Serve)
        # Calculate how many NG sandwiches can be made
        CanMakeNG = min(NG_bread_avail, NG_content_avail, Notexist_avail)

        # Serve allergic children by making NG sandwiches
        MakeNG = min(N_allergic_unserved, CanMakeNG)
        N_allergic_unserved -= MakeNG
        h += MakeNG * 4

        # Update available resources after making NG sandwiches
        Notexist_avail -= MakeNG
        NG_bread_avail -= MakeNG
        NG_content_avail -= MakeNG
        Any_bread_avail -= MakeNG # NG bread is a subset of Any bread
        Any_content_avail -= MakeNG # NG content is a subset of Any content

        # Calculate how many Any sandwiches (non-NG or remaining NG) can be made
        # using remaining resources
        CanMakeAnyRemaining = min(Any_bread_avail, Any_content_avail, Notexist_avail)

        # Serve non-allergic children by making Any sandwiches
        MakeAny = min(N_non_allergic_unserved, CanMakeAnyRemaining)
        N_non_allergic_unserved -= MakeAny
        h += MakeAny * 4

        # Note: If N_allergic_unserved > 0 or N_non_allergic_unserved > 0 after this,
        # it means there are not enough sandwiches (existing or makeable) to serve
        # all children in the current state. The heuristic correctly reflects the
        # cost for the maximum number of children that *can* be served with available
        # resources in this simplified model.

        return h
