from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Import math for potential use, though not strictly needed for this heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args for a strict match,
    # while allowing fnmatch wildcards to work on individual parts.
    # A simpler check: just zip and compare as fnmatch handles length differences gracefully
    return all(fnmatch(part, arg) for part, arg in zip(parts, args)) and len(parts) == len(args)


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting
    children by summing up the estimated costs for different stages of the
    sandwich delivery pipeline: making sandwiches, putting them on trays,
    moving trays to children's locations, and serving the children.

    # Assumptions
    - Each unserved child requires one suitable sandwich.
    - Trays are a shared resource; one tray move can potentially serve multiple
      children at the same location.
    - Ingredients (bread, content) and sandwich objects are sufficient if they
      exist in counts greater than zero for making purposes (simplified).
    - The kitchen is the starting point for trays and sandwich making.
    - Children do not wait in the kitchen (simplification based on typical problems).

    # Heuristic Initialization
    - Extracts the set of children who need to be served from the goal state.
    - Extracts static information about child allergies and their waiting places.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic is calculated as the sum of four main cost components,
    representing necessary actions in the sandwich delivery process:

    1.  **Cost to Serve:** Each unserved child requires one 'serve' action
        as the final step. This cost is the total number of children who are
        in the goal state but not yet marked as served in the current state.

    2.  **Cost to Move Trays:** For each unique location where unserved children
        are waiting (excluding the kitchen), if there is no tray currently
        present at that location, a tray must be moved there from elsewhere.
        This cost is the number of such locations needing a tray. This models
        the tray movement cost as a per-location cost, assuming one tray is
        sufficient per location.

    3.  **Cost to Make Sandwiches:** Sandwiches must be made from ingredients
        if the total number of suitable sandwiches required by unserved children
        exceeds the number of suitable sandwiches already made (either in the
        kitchen or already on trays). This cost is the number of sandwiches
        that need to be created using the 'make_sandwich' or
        'make_sandwich_no_gluten' actions.

    4.  **Cost to Put on Tray:** Sandwiches must be on trays to be transported
        and served. This cost applies to sandwiches that are currently in the
        kitchen (`at_kitchen_sandwich`) and need the 'put_on_tray' action,
        plus any sandwiches that will be newly made (as they will also end up
        `at_kitchen_sandwich` and require a 'put_on_tray' action). This cost
        is the number of sandwiches currently in the kitchen plus the number
        of sandwiches that need to be made.

    The total heuristic value is the sum of these four cost components.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Identify all children who are goals (need to be served)
        self.goal_children = {
            c for goal in self.goals if match(goal, "served", "*")
            for c in get_parts(goal)[1:]
        }

        # Map children to their allergy status
        self.child_allergies = {}
        for fact in static_facts:
            if match(fact, "allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.child_allergies[child] = 'allergic'
            elif match(fact, "not_allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.child_allergies[child] = 'not_allergic'

        # Map children to their waiting place
        self.child_waiting_places = {}
        for fact in static_facts:
             if match(fact, "waiting", "*", "*"):
                 child, place = get_parts(fact)[1:]
                 self.child_waiting_places[child] = place

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # 1. Identify unserved children and their needs
        unserved_children = {c for c in self.goal_children if f"(served {c})" not in state}

        if not unserved_children:
            return 0 # Goal state reached

        num_allergic_unserved = sum(1 for c in unserved_children if self.child_allergies.get(c) == 'allergic')
        num_not_allergic_unserved = sum(1 for c in unserved_children if self.child_allergies.get(c) == 'not_allergic')
        num_unserved_total = num_allergic_unserved + num_not_allergic_unserved

        # Cost Component 1: Cost to Serve
        # Each unserved child needs one 'serve' action.
        cost_serve = num_unserved_total
        total_cost += cost_serve

        # 2. Identify places needing tray moves
        # Places where unserved children are waiting
        unserved_waiting_at = {self.child_waiting_places[c] for c in unserved_children}
        # Places where trays are currently located
        trays_at = {p for fact in state if match(fact, "at", "*", "*") for t, p in [get_parts(fact)[1:]]}

        # Cost Component 2: Cost to Move Trays
        # For each place with unserved children that doesn't have a tray, a tray must be moved there.
        # Assume children don't wait *in* the kitchen for tray moves to their location.
        places_needing_tray_move = {p for p in unserved_waiting_at if p != 'kitchen'} - trays_at
        cost_move = len(places_needing_tray_move)
        total_cost += cost_move

        # 3. Count available sandwiches at different stages
        # Sandwiches currently in the kitchen
        gf_in_kitchen = sum(1 for fact in state if match(fact, "at_kitchen_sandwich", "*") and f"(no_gluten_sandwich {get_parts(fact)[1]})" in state)
        regular_in_kitchen = sum(1 for fact in state if match(fact, "at_kitchen_sandwich", "*") and f"(no_gluten_sandwich {get_parts(fact)[1]})" not in state)

        # Sandwiches currently on trays (anywhere)
        gf_on_trays = sum(1 for fact in state if match(fact, "ontray", "*", "*") and f"(no_gluten_sandwich {get_parts(fact)[1]})" in state)
        regular_on_trays = sum(1 for fact in state if match(fact, "ontray", "*", "*") and f"(no_gluten_sandwich {get_parts(fact)[1]})" not in state)

        # Total sandwiches already made (in kitchen or on trays)
        avail_gf_made = gf_in_kitchen + gf_on_trays
        avail_reg_made = regular_in_kitchen + regular_on_trays

        # Total sandwiches of each type needed
        N_gf_needed = num_allergic_unserved
        N_reg_needed = num_not_allergic_unserved

        # Cost Component 3: Cost to Make Sandwiches
        # Sandwiches must be made if the total needed exceeds those already made.
        # We count the deficit regardless of ingredient availability for simplicity
        # in this non-admissible heuristic.
        must_make_gf = max(0, N_gf_needed - avail_gf_made)
        must_make_regular = max(0, N_reg_needed - avail_reg_made)

        cost_make = must_make_gf + must_make_regular
        total_cost += cost_make

        # Cost Component 4: Cost to Put on Tray
        # Sandwiches that are currently at_kitchen_sandwich or will be made
        # need to be put on a tray.
        cost_put = gf_in_kitchen + regular_in_kitchen + cost_make
        total_cost += cost_put

        return total_cost
