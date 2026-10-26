from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions for PDDL fact parsing
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
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
    This heuristic estimates the number of actions required to serve all waiting
    children. It sums up the estimated costs for four main types of actions
    needed across all unserved children: serving, moving trays to locations
    where children are waiting, making necessary sandwiches, and putting
    sandwiches onto trays.

    # Assumptions
    - Each unserved child requires one suitable sandwich and one serving action.
    - Each location (excluding the kitchen) where unserved children are waiting
      requires at least one tray to be moved there if no tray is currently present.
    - Making a sandwich requires available bread and content (availability is
      not strictly checked beyond counting needed vs available made sandwiches).
    - Putting a sandwich on a tray requires a tray available in the kitchen
      (availability is not strictly checked beyond counting the action).
    - Action costs are uniform (cost 1).

    # Heuristic Initialization
    The heuristic extracts the initial set of waiting children, their locations,
    and their allergy status from the static facts of the task. This information
    is stored to determine which children still need to be served in any given state.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:

    1.  **Identify Unserved Children:** Determine which children were initially
        waiting but are not yet marked as `served` in the current state. Count
        the total number of such children and categorize them by allergy status
        (allergic needing GF, non-allergic needing any). Identify the distinct
        locations where these unserved children are waiting.

    2.  **Cost for Serving:** Add 1 to the total cost for each unserved child.
        This represents the final `serve_sandwich` or `serve_sandwich_no_gluten`
        action needed for each child.

    3.  **Cost for Tray Movement:** For each distinct location (excluding the
        kitchen) where unserved children are waiting, check if any tray is
        currently present at that location in the current state. Add 1 to the
        total cost for each such location that does *not* have a tray. This
        represents the `move_tray` action needed to bring a tray to that location.

    4.  **Count Available Sandwiches:** Count the number of gluten-free and
        regular sandwiches that have already been made (i.e., are currently
        `at_kitchen_sandwich` or `ontray`) in the current state.

    5.  **Cost for Make Sandwich:** Calculate how many additional gluten-free
        sandwiches are needed (demand from allergic children minus available GF
        sandwiches). Calculate how many additional regular sandwiches are needed
        (demand from non-allergic children minus available regular sandwiches,
        using any surplus GF sandwiches to meet regular demand). Add 1 to the
        total cost for each sandwich that still needs to be made.

    6.  **Cost for Put on Tray:** Count the number of sandwiches currently
        `at_kitchen_sandwich` in the current state. Add this count, plus the
        number of sandwiches calculated in step 5 that still need to be made,
        to the total cost. This represents the `put_on_tray` action needed for
        sandwiches in the kitchen or those yet to be made.

    7.  **Total Heuristic:** The sum of costs from steps 2, 3, 5, and 6 is the
        heuristic value for the state. If no children are unserved, the heuristic is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting initial waiting children,
        their locations, and allergy status from static facts.
        """
        super().__init__(task)
        # self.goals = task.goals # Goals are (served child), we track initial waiting instead
        static_facts = task.static

        self.initial_waiting = {}
        self.allergic_children = set()

        # Extract initial waiting children and their places/allergy status
        for fact in static_facts:
            if match(fact, "waiting", "*", "*"):
                _, child, place = get_parts(fact)
                self.initial_waiting[child] = {'place': place, 'allergic': False}
            elif match(fact, "allergic_gluten", "*"):
                _, child = get_parts(fact)
                self.allergic_children.add(child)

        # Update allergy status for waiting children
        for child in self.initial_waiting:
            if child in self.allergic_children:
                self.initial_waiting[child]['allergic'] = True

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify Unserved Children
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        unserved_children_details = {}
        places_with_unserved = set()
        needed_gf = 0
        needed_reg = 0

        for child, details in self.initial_waiting.items():
            if child not in served_children:
                unserved_children_details[child] = details
                # Children only wait at non-kitchen places according to domain/examples
                if details['place'] != 'kitchen':
                     places_with_unserved.add(details['place'])
                if details['allergic']:
                    needed_gf += 1
                else:
                    needed_reg += 1

        # If no children are unserved, we are in a goal state (or equivalent for this heuristic)
        if not unserved_children_details:
            return 0

        total_cost = 0

        # 2. Cost for Serving
        total_cost += len(unserved_children_details)

        # 3. Cost for Tray Movement
        trays_at_place = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                 parts = get_parts(fact)
                 # Check if the fact is like (at trayX placeY)
                 if len(parts) == 3 and parts[1].startswith('tray'):
                    _, tray, place = parts
                    trays_at_place[place] = trays_at_place.get(place, 0) + 1

        places_needing_tray_move = 0
        for place in places_with_unserved:
            # We need at least one tray at this place if children are waiting there
            if place not in trays_at_place or trays_at_place[place] == 0:
                 places_needing_tray_move += 1
        total_cost += places_needing_tray_move

        # 4. Count Available Sandwiches (made)
        available_gf_made = 0
        available_reg_made = 0
        made_sandwiches_in_state = set()

        for fact in state:
            if match(fact, "ontray", "*", "*"):
                _, sandwich, tray = get_parts(fact)
                made_sandwiches_in_state.add(sandwich)
            elif match(fact, "at_kitchen_sandwich", "*"):
                 _, sandwich = get_parts(fact)
                 made_sandwiches_in_state.add(sandwich)

        # Check which made sandwiches are gluten-free
        gf_sandwiches_in_state = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}

        for sandwich in made_sandwiches_in_state:
             if sandwich in gf_sandwiches_in_state:
                 available_gf_made += 1
             else:
                 available_reg_made += 1

        # 5. Cost for Make Sandwich
        needed_to_make_gf = max(0, needed_gf - available_gf_made)
        # Use available GF sandwiches first for GF demand, then surplus for Reg demand
        gf_surplus = max(0, available_gf_made - needed_gf)
        needed_to_make_reg = max(0, needed_reg - available_reg_made - gf_surplus)

        total_cost += needed_to_make_gf + needed_to_make_reg

        # 6. Cost for Put on Tray
        sandwiches_at_kitchen = sum(1 for fact in state if match(fact, "at_kitchen_sandwich", "*"))
        # Sandwiches needing put_on_tray are those currently in kitchen + those that need making
        total_cost += sandwiches_at_kitchen + needed_to_make_gf + needed_to_make_reg

        return total_cost
