from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or malformed facts gracefully
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
        # Depending on expected input robustness, could log a warning or raise error
        return [] # Indicate invalid format
    return fact[1:-1].split()

# Helper function to check if a fact exists in the state set
def fact_in_state(state, predicate, *args):
    """Checks if a fact with the given predicate and arguments exists in the state."""
    # Construct the fact string
    fact_str = "(" + predicate + (" " + " ".join(args) if args else "") + ")"
    return fact_str in state

# Helper function to match facts with wildcards
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
    This heuristic estimates the number of actions required to serve all waiting
    children. It counts the necessary actions in different stages: making
    sandwiches, putting them on trays, moving trays to children's locations,
    and finally serving the sandwiches. It considers allergy constraints and
    the current location of sandwiches and trays.

    # Assumptions
    - All children need exactly one sandwich.
    - Allergy constraints must be met (allergic children need gluten-free).
    - Sandwiches must be made in the kitchen, put on a tray in the kitchen,
      and the tray moved to the child's location for serving.
    - Tray capacity is sufficient.
    - Sufficient bread, content, and sandwich objects exist to make needed sandwiches.
      (This assumption simplifies resource counting, making the heuristic non-admissible
       but faster to compute).
    - Each 'make', 'put_on_tray', 'move_tray', and 'serve' action costs 1.

    # Heuristic Initialization
    - Identifies all children from the goal state.
    - Determines the allergy status (allergic or not) for each child from static facts.
    - Determines the waiting location for each child from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic sums up estimated costs for several bottlenecks:

    1.  **Unserved Children Count:** The number of children who have not yet been served.
        Each unserved child requires at least one 'serve' action.

    2.  **Sandwiches to Make:** The number of suitable sandwiches that do not
        currently exist (neither in the kitchen nor on a tray) but are needed
        to serve the unserved children, respecting allergy constraints.
        Each such sandwich requires a 'make' action.

    3.  **Sandwiches to Put on Trays:** The number of sandwiches that need to
        transition from being in the kitchen (`at_kitchen_sandwich`) to being
        on a tray (`ontray`). This count is derived from the total number of
        sandwiches needed on trays (equal to the number of unserved children)
        minus those already on trays and those that will be newly made.
        Each such sandwich requires a 'put_on_tray' action.

    4.  **Tray Move to Kitchen:** If sandwiches need to be put on trays from
        the kitchen (i.e., there's a deficit of sandwiches on trays that aren't
        being newly made) and no tray is currently in the kitchen, a tray must
        be moved to the kitchen. This costs one 'move_tray' action.

    5.  **Tray Moves to Locations:** For each unique location where unserved
        children are waiting, if no tray is currently present at that location,
        a tray must be moved there. This costs one 'move_tray' action per such location.

    The total heuristic value is the sum of the counts from steps 1, 2, 3, 4, and 5.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting child information and static facts.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Identify all children from the goal facts (served ?c)
        self.all_children = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'served' and len(parts) == 2:
                self.all_children.add(parts[1])

        # Map child to allergy status and waiting location from static facts
        self.child_allergy = {} # child -> True if allergic, False otherwise
        self.child_location = {} # child -> place
        for fact in self.static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts
            if parts[0] == 'allergic_gluten' and len(parts) == 2:
                self.child_allergy[parts[1]] = True
            elif parts[0] == 'not_allergic_gluten' and len(parts) == 2:
                 self.child_allergy[parts[1]] = False
            elif parts[0] == 'waiting' and len(parts) == 3:
                 self.child_location[parts[1]] = parts[2]

        # Note: It's assumed that all children in the goal have corresponding
        # allergy and waiting facts in the static information.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Step 1: Count unserved children and categorize by allergy/location ---
        num_unserved = 0
        num_allergy_unserved = 0
        num_normal_unserved = 0
        waiting_places = set() # Places where unserved children are waiting

        for child in self.all_children:
            # Check if the child is served in the current state
            if fact_in_state(state, 'served', child):
                continue # This child is served

            num_unserved += 1
            # Get allergy status, default to non-allergic if not specified (though PDDL should specify)
            if self.child_allergy.get(child, False):
                num_allergy_unserved += 1
            else:
                num_normal_unserved += 1

            # Add the child's waiting location if known
            location = self.child_location.get(child)
            if location:
                 waiting_places.add(location)

        # If no children are unserved, the goal is reached.
        if num_unserved == 0:
            return 0

        # --- Step 2: Count available sandwiches ---
        sandwiches_on_trays_ng = 0
        sandwiches_on_trays_normal = 0
        sandwiches_kitchen_ng = 0
        sandwiches_kitchen_normal = 0

        # Find all sandwiches on trays and in kitchen and categorize by gluten status
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            if parts[0] == 'ontray' and len(parts) == 3:
                sandwich = parts[1]
                if fact_in_state(state, 'no_gluten_sandwich', sandwich):
                    sandwiches_on_trays_ng += 1
                else:
                    sandwiches_on_trays_normal += 1
            elif parts[0] == 'at_kitchen_sandwich' and len(parts) == 2:
                sandwich = parts[1]
                if fact_in_state(state, 'no_gluten_sandwich', sandwich):
                    sandwiches_kitchen_ng += 1
                else:
                    sandwiches_kitchen_normal += 1

        total_sandwiches_on_trays = sandwiches_on_trays_ng + sandwiches_on_trays_normal
        total_sandwiches_kitchen = sandwiches_kitchen_ng + sandwiches_kitchen_normal

        # --- Step 3: Calculate sandwiches that need to be made ---
        # We need num_allergy_unserved GF sandwiches.
        # We need num_normal_unserved normal sandwiches (can use surplus GF).
        available_ng = sandwiches_on_trays_ng + sandwiches_kitchen_ng
        available_normal = sandwiches_on_trays_normal + sandwiches_kitchen_normal

        needed_make_ng = max(0, num_allergy_unserved - available_ng)

        # Surplus GF sandwiches can serve normal children
        surplus_ng = max(0, available_ng - num_allergy_unserved)
        available_for_normal = available_normal + surplus_ng

        needed_make_normal = max(0, num_normal_unserved - available_for_normal)

        needed_make = needed_make_ng + needed_make_normal

        # Cost component 2: making sandwiches
        cost_make = needed_make

        # --- Step 4: Calculate sandwiches that need to be put on trays ---
        # Total sandwiches needed on trays is num_unserved.
        # Sandwiches already on trays: total_sandwiches_on_trays.
        # The deficit must be put on trays from kitchen stock or by making new ones.
        # The number of 'put_on_tray' actions needed is the total number of sandwiches
        # that must end up on trays minus those already there.
        needed_on_trays_total = max(0, num_unserved - total_sandwiches_on_trays)

        # Cost component 3: putting sandwiches on trays
        cost_put_on_tray = needed_on_trays_total

        # --- Step 5: Calculate tray moves to kitchen ---
        # Need to move a tray to kitchen if sandwiches need to be put on trays
        # from the kitchen (i.e., needed_on_trays_total > 0) and no tray is there.
        trays_in_kitchen = any(match(fact, 'at', '*', 'kitchen') for fact in state)

        cost_move_tray_to_kitchen = 0
        # If we need to put any sandwich on a tray (needed_on_trays_total > 0),
        # we need a tray in the kitchen to perform the 'put_on_tray' action.
        if needed_on_trays_total > 0 and not trays_in_kitchen:
             cost_move_tray_to_kitchen = 1

        # --- Step 6: Calculate tray moves to locations ---
        # Count places with waiting children that have no tray
        tray_locations = {get_parts(fact)[2] for fact in state if match(fact, 'at', '*', '*')}
        kitchen_location = 'kitchen' # Constant place

        num_waiting_places_without_trays = 0
        for place in waiting_places:
            # We need a tray at this waiting place if it's not the kitchen
            # and no tray is currently located there.
            # If the waiting place is the kitchen, the tray move to kitchen cost
            # already accounts for getting a tray there if needed.
            if place != kitchen_location and place not in tray_locations:
                 num_waiting_places_without_trays += 1

        # Cost component 5: moving trays to locations
        cost_move_trays_to_locations = num_waiting_places_without_trays

        # --- Step 7: Calculate serving actions ---
        # Each unserved child needs one serve action.
        cost_serve = num_unserved

        # --- Total Heuristic ---
        # Summing up the estimated costs for the necessary actions.
        total_heuristic = (
            cost_make +                  # Cost to make sandwiches
            cost_put_on_tray +           # Cost to put sandwiches on trays
            cost_move_tray_to_kitchen +  # Cost to get a tray to kitchen if needed for loading
            cost_move_trays_to_locations + # Cost to get trays to waiting locations
            cost_serve                   # Cost to serve each child
        )

        return total_heuristic
