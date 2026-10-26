# Required imports
from heuristics.heuristic_base import Heuristic
# No need for fnmatch if using get_parts

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
         return []
    return fact[1:-1].split()

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children.
    It sums the estimated costs for:
    1. Serving each unserved child.
    2. Making sandwiches that are needed but don't exist.
    3. Putting sandwiches onto trays (those currently at the kitchen or newly made).
    4. Moving trays to locations where unserved children are waiting if no tray is present there.

    # Assumptions
    - The primary goal is to serve all children listed in the problem goal.
    - Each child requires exactly one sandwich.
    - The type of sandwich (gluten-free or not) is relevant for serving, but this heuristic simplifies sandwich counting by focusing on the total number needed vs available, and the number needing the 'make' action. It does not strictly enforce GF matching beyond counting totals.
    - Sufficient bread and content are available to make needed sandwiches, provided 'notexist' sandwich objects are available.
    - There are enough 'notexist' sandwich objects to cover the number of sandwiches that need to be made.
    - Trays can be reused and moved to any location.
    - The cost of each relevant action (make, put-on-tray, move-tray, serve) is assumed to be 1.

    # Heuristic Initialization
    The heuristic pre-processes the task information to store:
    - The allergy status for each child (allergic_gluten or not_allergic_gluten).
    - The waiting place for each child.
    - The list of children that need to be served (from the goal state).
    - The total number of trays available in the problem instance (derived from the initial state).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is calculated as follows:

    1.  **Count Unserved Children:** Iterate through the list of children that need to be served (obtained during initialization). Count how many of them are not currently marked as `served` in the state. Let this be `N_unserved`. If `N_unserved` is 0, the goal is reached, and the heuristic is 0. Otherwise, this contributes `N_unserved` to the heuristic (representing the final `serve` action for each).

    2.  **Identify Waiting Places:** While counting unserved children, collect the set of distinct places where these unserved children are waiting. Let this set be `WaitingPlaces`.

    3.  **Count Existing Sandwiches:** Iterate through the state facts to count the total number of sandwiches that currently exist (either `at_kitchen_sandwich` or `ontray`). Let this be `E_total`. Also, count the number of sandwiches specifically `at_kitchen_sandwich`. Let this be `K`.

    4.  **Count Available Trays:** The total number of trays available in the problem is pre-calculated in `__init__`. Let this be `total_trays`.

    5.  **Estimate Needed Make Actions:** Calculate how many new sandwiches need to be made to satisfy the unserved children. This is the maximum of 0 and the difference between the number of unserved children and the total number of existing sandwiches (`max(0, N_unserved - E_total)`). Let this be `needed_make`. This contributes `needed_make` to the heuristic.

    6.  **Estimate Needed Put-On-Tray Actions:** Sandwiches that are `at_kitchen_sandwich` need to be put on a tray. This includes sandwiches that were initially at the kitchen and those that will be newly made. The total number of sandwiches that will need this action is the count of existing kitchen sandwiches (`K`) plus the number of sandwiches that need to be made (`needed_make`). This contributes `K + needed_make` to the heuristic.

    7.  **Estimate Needed Move-Tray Actions:** Trays need to be moved to locations where unserved children are waiting if no tray is currently at that location. The minimum number of tray movements required is the maximum of 0 and the difference between the number of distinct waiting places and the total number of trays (`max(0, |WaitingPlaces| - total_trays)`). This contributes `max(0, |WaitingPlaces| - total_trays)` to the heuristic.

    8.  **Sum Components:** The total heuristic value is the sum of the contributions from steps 1, 5, 6, and 7:
        `H = N_unserved + needed_make + (K + needed_make) + max(0, |WaitingPlaces| - total_trays)`
        `H = N_unserved + K + 2 * needed_make + max(0, |WaitingPlaces| - total_trays)`
        `H = N_unserved + K + 2 * max(0, N_unserved - E_total) + max(0, num_waiting_locations - self.total_trays)`

    This heuristic is non-admissible as it might overcount actions (e.g., a single tray move might serve multiple children's needs at that location) but aims to capture the main steps and bottlenecks towards the goal.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state # Need initial state to count total trays

        # Pre-process static facts and goals
        self.child_allergy = {} # Map child name to True (allergic) or False (not allergic)
        self.child_waiting_place = {} # Map child name to waiting place name
        self.children_to_serve = set() # Set of child names that need to be served
        self.total_trays = 0 # Total number of trays in the problem

        # Process static facts
        for fact in self.static:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            if parts[0] == "allergic_gluten" and len(parts) == 2:
                self.child_allergy[parts[1]] = True
            elif parts[0] == "not_allergic_gluten" and len(parts) == 2:
                self.child_allergy[parts[1]] = False
            elif parts[0] == "waiting" and len(parts) == 3:
                self.child_waiting_place[parts[1]] = parts[2]

        # Get children from goals
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip malformed goals
            if parts[0] == "served" and len(parts) == 2:
                self.children_to_serve.add(parts[1])

        # Count total trays from initial state
        # Assuming facts like (at trayX placeY) in initial state indicate trays
        for fact in self.initial_state:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == "at" and parts[1].startswith("tray"):
                self.total_trays += 1
        # If total_trays is 0, it might indicate an unsolvable problem or a different initial state structure.
        # The heuristic handles total_trays = 0 gracefully.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state # Current world state (frozenset of fact strings)
        state_facts = set(state) # Convert to set for efficient lookups

        # --- Step 1 & 2: Count unserved children and identify waiting places ---
        n_unserved = 0
        waiting_places = set() # Places where unserved children are waiting

        for child_name in self.children_to_serve:
            if f"(served {child_name})" not in state_facts:
                n_unserved += 1
                waiting_place = self.child_waiting_place.get(child_name)
                # It's possible a child is in the goal but not in static 'waiting' facts,
                # although problem examples suggest they are. Handle defensively.
                if waiting_place:
                    waiting_places.add(waiting_place)

        # If no unserved children, the goal is reached.
        if n_unserved == 0:
            return 0

        # --- Step 3: Count existing sandwiches and those at the kitchen ---
        k_sandwiches = 0 # Number of sandwiches at the kitchen
        existing_sandwiches = set() # Set of names of sandwiches that exist (at kitchen or ontray)

        for fact in state_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            if parts[0] == "at_kitchen_sandwich" and len(parts) == 2:
                k_sandwiches += 1
                existing_sandwiches.add(parts[1])
            elif parts[0] == "ontray" and len(parts) == 3:
                existing_sandwiches.add(parts[1])
            # We don't need to count notexist or GF status for this specific formula

        e_total = len(existing_sandwiches) # Total number of existing sandwiches

        # --- Step 5: Estimate Needed Make Actions ---
        # Number of sandwiches that need to be created from 'notexist' slots
        needed_make = max(0, n_unserved - e_total)

        # --- Step 6: Estimate Needed Put-On-Tray Actions ---
        # Sandwiches at kitchen + sandwiches that will be made need putting on tray
        cost_put_on_tray = k_sandwiches + needed_make

        # --- Step 7: Estimate Needed Move-Tray Actions ---
        # Number of distinct waiting locations that need a tray, assuming total_trays are available
        num_waiting_locations = len(waiting_places)
        cost_move_tray = max(0, num_waiting_locations - self.total_trays)

        # --- Step 8: Sum Components ---
        # H = N_unserved (serve) + needed_make (make) + cost_put_on_tray (put) + cost_move_tray (move)
        # Substituting cost_put_on_tray = K + needed_make:
        # H = N_unserved + needed_make + (K + needed_make) + cost_move_tray
        # H = N_unserved + K + 2 * needed_make + cost_move_tray
        # Substituting needed_make = max(0, N_unserved - E_total):
        # H = N_unserved + K + 2 * max(0, N_unserved - E_total) + max(0, num_waiting_locations - self.total_trays)

        heuristic_value = n_unserved + k_sandwiches + 2 * needed_make + cost_move_tray

        return heuristic_value
