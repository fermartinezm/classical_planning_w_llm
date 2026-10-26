from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle cases that are not standard PDDL facts represented as strings
        # For this domain, facts are expected to be strings like '(predicate arg1 arg2)'
        # If it's not a string or not in the expected format, return an empty list
        # or raise an error, depending on desired robustness.
        # Given the problem description, we expect valid fact strings.
        # Let's return an empty list for malformed input to avoid crashes.
        return []
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
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
    This heuristic estimates the number of actions required to serve all children.
    It breaks down the process into stages: making sandwiches, putting them on trays,
    moving trays to children's locations, and finally serving the children.
    It sums the estimated actions needed for each stage, counting deficits based
    on the current state, ignoring some resource constraints (like limited ingredients
    or sandwich objects) for computational efficiency.

    # Assumptions
    - Enough bread and content portions of both types are available to make needed sandwiches.
    - Enough unique sandwich objects (`notexist`) are available to represent new sandwiches.
    - Enough trays are available in total.
    - Trays can be moved between any two places in one action.
    - Trays have unlimited capacity for sandwiches.
    - Sandwiches can be put on a tray if the sandwich is at the kitchen and a tray is at the kitchen.

    # Heuristic Initialization
    - Extracts static information about children: their allergy status (allergic_gluten or not_allergic_gluten)
      and their waiting place (`waiting ?c ?p`). This information is stored for quick lookup.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic calculates the sum of estimated actions needed across four main stages
    for all unserved children:

    1.  **Serve Cost**: Count the number of children who are currently `waiting` but not yet `served`.
        Each such child requires one `serve_sandwich` or `serve_sandwich_no_gluten` action.
        Cost = Number of unserved children.

    2.  **Tray Move Cost**: Identify the distinct places where unserved children are waiting.
        For each such place, check if there is at least one tray currently located `at` that place.
        If no tray is present at a place where children are waiting, a tray needs to be moved there.
        Cost = Number of distinct waiting places that do not currently have a tray.

    3.  **Put on Tray Cost**: Count the number of sandwiches (distinguishing between gluten-free and regular)
        that are required to be `ontray` to serve the unserved children. This is equal to the number
        of unserved allergic children (for GF sandwiches) plus the number of unserved non-allergic
        children (for regular sandwiches). Subtract the number of GF/regular sandwiches that are
        already `ontray` in the current state. The deficit represents sandwiches that still need
        to undergo the `put_on_tray` action.
        Cost = `max(0, needed_gf_ontray - current_gf_ontray) + max(0, needed_reg_ontray - current_reg_ontray)`.

    4.  **Make Sandwich Cost**: Count the number of sandwiches (distinguishing between gluten-free and regular)
        that are required to be `at_kitchen_sandwich` to satisfy the deficit calculated in step 3.
        Subtract the number of GF/regular sandwiches that are already `at_kitchen_sandwich` in the
        current state. The deficit represents sandwiches that still need to undergo a `make_sandwich`
        or `make_sandwich_no_gluten` action.
        Cost = `max(0, needed_gf_kitchen - current_gf_kitchen) + max(0, needed_reg_kitchen - current_reg_kitchen)`,
        where `needed_gf_kitchen = max(0, needed_gf_ontray - current_gf_ontray)` and
        `needed_reg_kitchen = max(0, needed_reg_ontray - current_reg_ontray)`.

    The total heuristic value is the sum of the costs from steps 1, 2, 3, and 4.
    If all children are served, the heuristic value is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about children.
        """
        self.goals = task.goals # Not strictly needed for this heuristic, but good practice
        static_facts = task.static

        self.allergic_children = set()
        self.not_allergic_children = set()
        self.waiting_places_map = {} # Maps child name to their waiting place
        self.all_children = set() # Set of all children in the problem

        # Extract static information from the task
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "allergic_gluten" and len(parts) == 2:
                child = parts[1]
                self.allergic_children.add(child)
                self.all_children.add(child)
            elif predicate == "not_allergic_gluten" and len(parts) == 2:
                child = parts[1]
                self.not_allergic_children.add(child)
                self.all_children.add(child)
            elif predicate == "waiting" and len(parts) == 3:
                child, place = parts[1], parts[2]
                self.waiting_places_map[child] = place
                # Note: Children in waiting_places_map should also be in all_children,
                # but adding here ensures we cover all children mentioned in static facts.
                self.all_children.add(child)


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach a goal state.
        """
        state = node.state

        # 1. Identify unserved children
        served_children_in_state = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children = {child for child in self.all_children if child not in served_children_in_state}

        # If all children are served, the goal is reached.
        if not unserved_children:
            return 0

        # Count unserved children by allergy status
        unserved_allergic = {c for c in unserved_children if c in self.allergic_children}
        unserved_regular = {c for c in unserved_children if c in self.not_allergic_children}

        N_unserved = len(unserved_children)
        N_gf_needed_total = len(unserved_allergic)
        N_reg_needed_total = len(unserved_regular)

        # --- Calculate heuristic components ---

        # Component 1: Serve Cost
        # Each unserved child needs one serve action.
        cost_serve = N_unserved

        # Component 2: Tray Move Cost
        # Identify places where unserved children are waiting
        places_with_unserved_children = {self.waiting_places_map[c] for c in unserved_children if c in self.waiting_places_map}

        # Identify places where trays are currently located
        places_with_trays = {get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*") and get_parts(fact)[1].startswith("tray")}

        # Count places needing a tray (where children wait but no tray is present)
        places_needing_tray = places_with_unserved_children - places_with_trays
        cost_move_tray = len(places_needing_tray)

        # Component 3: Put on Tray Cost
        # Count sandwiches currently on trays, by type
        num_gf_ontray_sandwiches = sum(1 for fact in state if match(fact, "ontray", "*", "*") and match(fact, "no_gluten_sandwich", get_parts(fact)[1]))
        num_reg_ontray_sandwiches = sum(1 for fact in state if match(fact, "ontray", "*", "*") and not match(fact, "no_gluten_sandwich", get_parts(fact)[1]))

        # Calculate deficit of sandwiches needing to be put on tray
        gf_needing_put_on_tray = max(0, N_gf_needed_total - num_gf_ontray_sandwiches)
        reg_needing_put_on_tray = max(0, N_reg_needed_total - num_reg_ontray_sandwiches)
        cost_put_on_tray = gf_needing_put_on_tray + reg_needing_put_on_tray

        # Component 4: Make Sandwich Cost
        # Count sandwiches currently at the kitchen, by type
        num_gf_kitchen_sandwiches = sum(1 for fact in state if match(fact, "at_kitchen_sandwich", "*") and match(fact, "no_gluten_sandwich", get_parts(fact)[1]))
        num_reg_kitchen_sandwiches = sum(1 for fact in state if match(fact, "at_kitchen_sandwich", "*") and not match(fact, "no_gluten_sandwich", get_parts(fact)[1]))

        # Calculate deficit of sandwiches needing to be made (to fulfill the 'put on tray' deficit)
        gf_needing_make = max(0, gf_needing_put_on_tray - num_gf_kitchen_sandwiches)
        reg_needing_make = max(0, reg_needing_put_on_tray - num_reg_kitchen_sandwiches)
        cost_make = gf_needing_make + reg_needing_make

        # Total heuristic is the sum of costs for each stage
        total_heuristic = cost_serve + cost_move_tray + cost_put_on_tray + cost_make

        return total_heuristic

