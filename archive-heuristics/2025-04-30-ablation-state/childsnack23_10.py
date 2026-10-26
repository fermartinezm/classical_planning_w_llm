from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions (from Logistics example)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at child1 table1)".
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
    This heuristic estimates the number of actions needed to serve all children.
    It sums up the estimated costs for four main stages that are typically required
    to serve a child: making a suitable sandwich, getting it onto a tray,
    moving the tray to the child's location, and finally serving the child.
    The heuristic counts the number of items (sandwiches, trays, children)
    that still need to pass through each stage to reach the goal state where all
    children are served.

    # Assumptions
    - Ingredients (bread, content) are always available in the kitchen if needed to make a sandwich.
    - Enough sandwich objects exist in the problem definition to make all required sandwiches.
    - Enough tray objects exist and are initially at the kitchen.
    - Tray capacity is sufficient to hold all sandwiches needed at a location.
    - The kitchen is a distinct place from other waiting places.
    - Any tray can be moved to any location.
    - Any suitable sandwich on a tray at a child's location can be served to that child.

    # Heuristic Initialization
    The heuristic extracts the following information from the task's initial state and static facts:
    - A set of all children defined in the problem.
    - A dictionary mapping each child to their allergy type ('ng' for gluten-allergic, 'reg' otherwise).
    - A dictionary mapping each child to their initial waiting place.
    - A set of all potential sandwich objects defined in the problem.
    - A set of all potential tray objects defined in the problem.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is calculated as the sum of four components:

    1.  **Cost to make sandwiches**: Estimate the number of new sandwiches that need to be made.
        - Count unserved children needing gluten-free sandwiches (`N_unserved_ng`).
        - Count unserved children needing regular sandwiches (`N_unserved_reg`).
        - Count available gluten-free sandwiches anywhere (`Avail_ng_total`: `at_kitchen_sandwich` or `ontray`).
        - Count available regular sandwiches anywhere (`Avail_reg_total`: `at_kitchen_sandwich` or `ontray`).
        - Number of new NG sandwiches needed = `max(0, N_unserved_ng - Avail_ng_total)`.
        - Number of new regular sandwiches needed (after using available regular and surplus NG) = `max(0, N_unserved_reg - (Avail_reg_total + max(0, Avail_ng_total - N_unserved_ng)))`.
        - Cost = (new NG needed) + (new regular needed). Each needs one `make_sandwich` action.

    2.  **Cost to put sandwiches on trays**: Estimate the number of sandwiches that need to reach the 'ontray' state.
        - Total sandwiches needed on trays = `N_unserved`.
        - Count sandwiches already `ontray` that are gluten-free (`Avail_ng_ontray`).
        - Count sandwiches already `ontray` that are regular (`Avail_reg_ontray`).
        - Sandwiches already on trays = `Avail_ng_ontray + Avail_reg_ontray`.
        - The number of sandwiches that still need to transition to the 'ontray' state.
        - Cost = `max(0, N_unserved - (Sandwiches already on trays))`.

    3.  **Cost to move trays**: Estimate the number of `move_tray` actions required.
        - Identify all places (excluding kitchen) where unserved children are waiting.
        - Identify all places where trays are currently located.
        - For each place where unserved children are waiting (and it's not the kitchen), if no tray is currently at that place, one `move_tray` action is needed to bring a tray there.
        - Cost = Number of distinct places `p != kitchen` with waiting unserved children that do not have a tray `at p`.

    4.  **Cost to serve**: Estimate the number of `serve` actions required.
        - Each unserved child needs one `serve` action.
        - Cost = Number of unserved children (`N_unserved`).

    The total heuristic value is the sum of these four costs.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting relevant static and initial state information."""
        self.goals = task.goals
        self.initial_state = task.initial_state
        self.static_facts = task.static

        # Extract children, their types, and initial waiting places
        self.all_children = set()
        self.child_type = {}  # child -> 'ng' or 'reg'
        self.child_place = {} # child -> place

        # Child info is typically in static facts or initial state
        # Iterate through both static and initial state to be robust
        for fact in self.static_facts | self.initial_state:
             if match(fact, "waiting", "*", "*"):
                 child, place = get_parts(fact)[1:3]
                 self.all_children.add(child)
                 self.child_place[child] = place
             elif match(fact, "allergic_gluten", "*"):
                 child = get_parts(fact)[1]
                 self.all_children.add(child)
                 self.child_type[child] = 'ng'
             elif match(fact, "not_allergic_gluten", "*"):
                 child = get_parts(fact)[1]
                 self.all_children.add(child)
                 self.child_type[child] = 'reg'

        # Extract all potential sandwich objects (those that initially do not exist)
        # Assuming all sandwich objects are listed with (notexist ?s) in the initial state
        self.all_sandwiches = {get_parts(fact)[1] for fact in self.initial_state if match(fact, "notexist", "*")}

        # Extract all potential tray objects (those initially at the kitchen)
        # Assuming all tray objects are listed with (at ?t kitchen) in the initial state
        self.all_trays = {get_parts(fact)[1] for fact in self.initial_state if match(fact, "at", "*", "kitchen")}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Count unserved children and categorize by allergy
        unserved_children = {c for c in self.all_children if f"(served {c})" not in state}
        N_unserved = len(unserved_children)

        if N_unserved == 0:
            return 0 # Goal state

        N_unserved_ng = len({c for c in unserved_children if self.child_type.get(c) == 'ng'})
        N_unserved_reg = len({c for c in unserved_children if self.child_type.get(c) == 'reg'})

        # 2. Count available sandwiches by type and location
        available_sandwiches_anywhere = {s for s in self.all_sandwiches if f"(notexist {s})" not in state}

        # Sandwiches currently at the kitchen
        Avail_ng_kitchen = len({s for s in available_sandwiches_anywhere if f"(at_kitchen_sandwich {s})" in state and f"(no_gluten_sandwich {s})" in state})
        Avail_reg_kitchen = len({s for s in available_sandwiches_anywhere if f"(at_kitchen_sandwich {s})" in state and f"(no_gluten_sandwich {s})" not in state})

        # Sandwiches currently on any tray
        Avail_ng_ontray = len({s for s in available_sandwiches_anywhere if any(match(fact, "ontray", s, "*") for fact in state) and f"(no_gluten_sandwich {s})" in state})
        Avail_reg_ontray = len({s for s in available_sandwiches_anywhere if any(match(fact, "ontray", s, "*") for fact in state) and f"(no_gluten_sandwich {s})" not in state})

        # Total available suitable sandwiches anywhere
        Avail_ng_total = Avail_ng_kitchen + Avail_ng_ontray
        Avail_reg_total = Avail_reg_kitchen + Avail_reg_ontray

        # 3. Estimate cost to make sandwiches
        # Need N_unserved_ng NG sandwiches. Available are Avail_ng_total.
        needed_ng_to_make = max(0, N_unserved_ng - Avail_ng_total)

        # Need N_unserved_reg Reg sandwiches. Available are Avail_reg_total + remaining Avail_ng_total.
        remaining_avail_ng = max(0, Avail_ng_total - N_unserved_ng) # NG sandwiches left after satisfying NG needs
        needed_reg_to_make = max(0, N_unserved_reg - (Avail_reg_total + remaining_avail_ng))

        cost_make_sandwiches = needed_ng_to_make + needed_reg_to_make

        # 4. Estimate cost to put sandwiches on trays
        # Total sandwiches needed on trays is N_unserved.
        # Sandwiches already on trays are Avail_ng_ontray + Avail_reg_ontray.
        # The number of sandwiches that still need to transition to the 'ontray' state.
        cost_put_on_tray = max(0, N_unserved - (Avail_ng_ontray + Avail_reg_ontray))

        # 5. Estimate cost to move trays
        places_with_waiting_unserved_children = {self.child_place[c] for c in unserved_children}
        
        # Find current locations of all trays
        tray_locations = {get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*") and get_parts(fact)[1] in self.all_trays}

        # Count places needing a tray move
        places_needing_tray_move = {p for p in places_with_waiting_unserved_children if p != 'kitchen' and p not in tray_locations}
        cost_move_trays = len(places_needing_tray_move)

        # 6. Estimate cost to serve children
        cost_serve = N_unserved

        # Total heuristic is the sum of estimated costs for each stage
        total_cost = cost_make_sandwiches + cost_put_on_tray + cost_move_trays + cost_serve

        return total_cost
