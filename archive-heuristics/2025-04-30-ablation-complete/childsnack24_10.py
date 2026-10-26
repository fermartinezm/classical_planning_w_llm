from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected fact format, returning empty list
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting children.
    It breaks down the problem into sub-goals for each unserved child and sums up
    estimated costs for making sandwiches, putting them on trays, moving trays
    to the children's locations, and finally serving the children.

    # Assumptions
    - Ingredients (bread, content) and sandwich objects (`notexist`) are assumed
      to be sufficient to make any required sandwich type, unless explicitly
      accounted for by counting available ingredients (which this version simplifies
      by only counting existing sandwiches and those that *must* be made based on deficit).
    - Trays are assumed to be available in the kitchen or elsewhere to be moved
      to waiting locations.
    - The heuristic sums costs independently for different stages (make, put, move, serve)
      and different sandwich types, ignoring potential resource conflicts (e.g.,
      multiple sandwiches needing the same tray or multiple trays needing the same path).
    - Each action (make, put, move, serve) is assumed to have a cost of 1.

    # Heuristic Initialization
    - Extracts static information about children: their allergy status (allergic_gluten
      or not_allergic_gluten) and their waiting place. This information is stored
      in dictionaries mapping child names to their properties.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic is calculated as the sum of estimated costs for four main types of actions:
    Serve, Make, Put_on_tray, and Move_tray.

    1.  **Identify Unserved Children and Their Needs:**
        - Iterate through all children identified in the static facts (via child_waiting_place).
        - For each child, check if they are marked as `served` in the current state.
        - If a child is not served, determine their required sandwich type (gluten or no-gluten)
          based on their allergy status (stored during initialization).
        - Count the total number of unserved children needing a gluten sandwich (`N_g`)
          and those needing a no-gluten sandwich (`N_ng`).
        - Identify the set of unique places where these unserved children are waiting.

    2.  **Estimate Serve Actions (H_serve):**
        - Each unserved child requires one `serve_sandwich` action.
        - `H_serve = N_g + N_ng`.

    3.  **Estimate Tray Movement Actions (H_move):**
        - Each unique place where unserved children are waiting needs at least one tray
          to be present for serving.
        - Count the number of unique waiting places (`Num_waiting_places`).
        - Count the number of trays currently located at any of these waiting places
          (`Num_trays_at_waiting_places`).
        - The estimated number of tray movements needed to bring trays to places
          that currently lack one is `H_move = max(0, Num_waiting_places - Num_trays_at_waiting_places)`.
          (This assumes any tray can be moved to any needed location).

    4.  **Estimate Sandwich Preparation Actions (H_make and H_put):**
        - We need a total of `N_g` gluten sandwiches and `N_ng` no-gluten sandwiches
          to be available *on trays* to serve all unserved children.
        - Count the number of suitable sandwiches already on trays (`A_g_ontray`, `A_ng_ontray`).
        - Count the number of suitable sandwiches currently in the kitchen (`A_g_kitchen`, `A_ng_kitchen`).
        - Estimate the number of sandwiches that *must* be made (`make_sandwich`):
          These are the needed sandwiches that are neither already on trays nor in the kitchen.
          `Must_make_g = max(0, N_g - (A_g_ontray + A_g_kitchen))`
          `Must_make_ng = max(0, N_ng - (A_ng_ontray + A_ng_kitchen))`
          `H_make = Must_make_g + Must_make_ng`. (Assumes ingredients and sandwich objects are available).
        - Estimate the number of sandwiches that *must* be put on trays (`put_on_tray`):
          These are the needed sandwiches that are not already on trays. They must either
          come from the kitchen stock or be newly made.
          `Must_put_g = max(0, N_g - A_g_ontray)`
          `Must_put_ng = max(0, N_ng - A_ng_ontray)`
          `H_put = Must_put_g + Must_put_ng`. (Assumes trays are available in the kitchen when needed).

    5.  **Calculate Total Heuristic:**
        - The total heuristic value is the sum of the estimated costs for each action type:
        - `H = H_serve + H_make + H_put + H_move`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about children.
        """
        self.goals = task.goals # Store goals for completeness, though not directly used in this heuristic formula
        static_facts = task.static

        # Map child name to their allergy status
        self.child_allergy = {}
        # Map child name to their waiting place
        self.child_waiting_place = {}

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            predicate = parts[0]
            if predicate == "allergic_gluten":
                child = parts[1]
                self.child_allergy[child] = "allergic_gluten"
            elif predicate == "not_allergic_gluten":
                child = parts[1]
                self.child_allergy[child] = "not_allergic_gluten"
            elif predicate == "waiting":
                child = parts[1]
                place = parts[2]
                self.child_waiting_place[child] = place

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach the goal.
        """
        state = node.state

        # --- Step 1: Identify Unserved Children and Their Needs ---
        served_children = set()
        kitchen_sandwiches = set() # Sandwiches at_kitchen_sandwich
        ontray_sandwiches = set()  # Sandwiches ontray
        no_gluten_sandwiches = set() # Sandwiches that are no_gluten_sandwich
        tray_locations = {} # Map tray -> place

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            predicate = parts[0]
            if predicate == "served":
                served_children.add(parts[1])
            elif predicate == "at_kitchen_sandwich":
                kitchen_sandwiches.add(parts[1])
            elif predicate == "ontray":
                ontray_sandwiches.add(parts[1])
            elif predicate == "no_gluten_sandwich":
                no_gluten_sandwiches.add(parts[1])
            elif predicate == "at" and len(parts) == 3: # (at ?t ?p)
                 tray, place = parts[1], parts[2]
                 tray_locations[tray] = place

        # Count unserved children and their needs (gluten/no-gluten)
        N_g = 0 # Number of unserved children needing gluten sandwich
        N_ng = 0 # Number of unserved children needing no-gluten sandwich
        waiting_places = set() # Unique places where unserved children are waiting

        # Iterate through children identified in static facts
        for child in self.child_waiting_place:
            if child not in served_children:
                waiting_places.add(self.child_waiting_place[child])
                # Get allergy from static info stored in __init__
                allergy_status = self.child_allergy.get(child)
                if allergy_status == "allergic_gluten":
                    N_ng += 1
                elif allergy_status == "not_allergic_gluten":
                    N_g += 1
                # Note: Children present in state but not in static (waiting/allergy) are ignored

        # If all children known from static facts are served, the heuristic is 0
        if N_g == 0 and N_ng == 0:
             # Check if the goal is fully satisfied based on task.goals
             # A simpler check is if all children from static are served.
             # If task.goals contains served predicates for children not in static,
             # this heuristic might underestimate, but based on typical PDDL,
             # all relevant children are defined in init/static.
             all_static_children_served = True
             for child in self.child_waiting_place:
                 if child not in served_children:
                     all_static_children_served = False
                     break # Should not happen if N_g and N_ng are 0, but as a safeguard
             if all_static_children_served:
                 return 0


        # --- Step 2: Estimate Serve Actions (H_serve) ---
        H_serve = N_g + N_ng

        # --- Step 3: Estimate Tray Movement Actions (H_move) ---
        Num_waiting_places = len(waiting_places)
        Num_trays_at_waiting_places = 0
        for place in tray_locations.values():
             if place in waiting_places:
                 Num_trays_at_waiting_places += 1

        H_move = max(0, Num_waiting_places - Num_trays_at_waiting_places)

        # --- Step 4: Estimate Sandwich Preparation Actions (H_make and H_put) ---

        # Count available sandwiches by type and location (kitchen vs. ontray)
        A_g_kitchen = sum(1 for s in kitchen_sandwiches if s not in no_gluten_sandwiches)
        A_ng_kitchen = sum(1 for s in kitchen_sandwiches if s in no_gluten_sandwiches)
        A_g_ontray = sum(1 for s in ontray_sandwiches if s not in no_gluten_sandwiches)
        A_ng_ontray = sum(1 for s in ontray_sandwiches if s in no_gluten_sandwiches)

        # Estimate sandwiches that must be made
        # These are needed sandwiches not already on trays or in the kitchen
        Must_make_g = max(0, N_g - (A_g_ontray + A_g_kitchen))
        Must_make_ng = max(0, N_ng - (A_ng_ontray + A_ng_kitchen))
        H_make = Must_make_g + Must_make_ng

        # Estimate sandwiches that must be put on trays
        # These are needed sandwiches not already on trays
        Must_put_g = max(0, N_g - A_g_ontray)
        Must_put_ng = max(0, N_ng - A_ng_ontray)
        H_put = Must_put_g + Must_put_ng

        # --- Step 5: Calculate Total Heuristic ---
        total_heuristic = H_serve + H_make + H_put + H_move

        return total_heuristic
