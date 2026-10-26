import os
import sys
from fnmatch import fnmatch

# Ensure the heuristics directory is in the Python path if needed
# This might be necessary depending on how the planner loads heuristics.
# Example: sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import base class for heuristics
# The exact import path depends on the planner's project structure.
# This assumes 'heuristic_base.py' is accessible, e.g., in a 'heuristics' package.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Fallback if the standard path doesn't work (e.g., running script directly)
    # Assumes heuristic_base.py is in the same directory or Python path.
    print("Could not import Heuristic from heuristics.heuristic_base, trying direct import.")
    try:
        from heuristic_base import Heuristic
    except ImportError:
        raise ImportError("Could not find Heuristic base class. Ensure it's in the Python path.")


# Helper function to parse PDDL facts like "(predicate obj1 obj2)" into a list ["predicate", "obj1", "obj2"]
def get_parts(fact):
    """Extract the components of a PDDL fact string by removing parentheses and splitting."""
    return fact[1:-1].split()

# Helper function to match facts against patterns using fnmatch for wildcard support
def match(fact, *args):
    """
    Check if a PDDL fact string matches a given pattern.

    Args:
        fact (str): The PDDL fact string (e.g., "(at tray1 kitchen)").
        *args: A sequence of strings representing the pattern elements. '*' can be used as a wildcard.

    Returns:
        bool: True if the fact matches the pattern, False otherwise.
    """
    parts = get_parts(fact)
    # Check if the number of parts in the fact matches the number of elements in the pattern
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding pattern element using fnmatch
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Childsnack domain.

    # Summary
    This heuristic estimates the total number of actions required to serve all
    children specified in the goal state. It calculates the minimum estimated
    actions needed for each unserved child independently and sums these costs.
    The cost for a single child considers the necessary steps: making a sandwich (if needed),
    putting it on a tray (if needed), moving the tray (if needed), and serving.
    It checks the current state to determine if some steps are already completed
    (e.g., a suitable sandwich already exists on a tray at the right location).

    # Assumptions
    - The heuristic calculates the cost for each child independently. It assumes
      that necessary resources (like ingredients for making sandwiches, or a tray
      being available at the kitchen when needed for 'put_on_tray') are available
      when required for that child's estimated plan steps.
    - It does not model resource contention. For example, if only one gluten-free
      sandwich exists but two children need one, the heuristic might underestimate
      the true cost because it calculates costs independently.
    - It does not model positive interactions. For example, if one tray movement
      to a table could potentially allow serving multiple children waiting there,
      the heuristic might overestimate the true cost by counting the move for each child.
      This overestimation is acceptable for a non-admissible heuristic used in
      Greedy Best-First Search.
    - It assumes that objects involved in `(at ?obj ?location)` predicates are
      trays, as other objects like ingredients and sandwiches use specific
      `at_kitchen_*` predicates in this domain.

    # Heuristic Initialization
    - The constructor (`__init__`) parses the static facts provided by the task object.
    - It stores the allergy status (`allergic_gluten` or `not_allergic_gluten`)
      for each child in a dictionary `self.child_allergy`.
    - It stores the waiting location (`waiting ?c ?p`) for each child in a
      dictionary `self.child_waiting_location`.
    - It identifies the set of all children that need to be served according to the
      goal predicates `(served ?c)` and stores them in `self.goal_children`.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Identify Unserved Children:** Compare the set of goal children (`self.goal_children`)
        with the children currently served (`(served ?c)`) in the state. If all goal
        children are served, the heuristic value is 0 (goal reached).
    2.  **Parse Current State:** Extract relevant information from the current state's facts:
        -   Find all sandwiches currently on trays using `(ontray ?s ?t)` facts.
        -   Find all sandwiches currently at the kitchen using `(at_kitchen_sandwich ?s)` facts.
        -   Determine which sandwiches are gluten-free using `(no_gluten_sandwich ?s)` facts.
        -   Find the current location `?p` of each tray `?t` using `(at ?t ?p)` facts. Store these
            in suitable data structures (dictionaries mapping sandwich name to properties/location).
    3.  **Calculate Cost per Child:** Iterate through each unserved child `c`:
        a.  Retrieve the child's allergy status (needs gluten-free?) and waiting place `p`
            from the precomputed dictionaries.
        b.  **Check Case 1 (Minimal Cost):** Is there a suitable sandwich `s` (matching allergy needs)
            already on *any* tray `t` that is currently located at the child's waiting place `p`?
            If yes, the estimated cost for this child is 1 (only the `serve` action is needed).
        c.  **Check Case 2:** If Case 1 is false, is there a suitable sandwich `s` on *any*
            tray `t`, but that tray `t` is currently at a different location `p'` (which could be
            the kitchen or another place)?
            If yes, the estimated cost is 2 (one `move_tray` action + one `serve` action).
        d.  **Check Case 3:** If Cases 1 and 2 are false, is there a suitable sandwich `s`
            currently at the kitchen `(at_kitchen_sandwich s)`?
            If yes, the estimated cost depends on the child's waiting location `p`:
            - If `p` is 'kitchen': Cost is 2 (`put_on_tray` + `serve`).
            - If `p` is not 'kitchen': Cost is 3 (`put_on_tray` + `move_tray` + `serve`).
        e.  **Check Case 4 (Maximum Cost):** If none of the above conditions are met, it implies
            a suitable sandwich must be made from scratch.
            The estimated cost depends on the child's waiting location `p`:
            - If `p` is 'kitchen': Cost is 3 (`make_sandwich*` + `put_on_tray` + `serve`).
            - If `p` is not 'kitchen': Cost is 4 (`make_sandwich*` + `put_on_tray` + `move_tray` + `serve`).
    4.  **Sum Costs:** The total heuristic value is the sum of the minimum estimated
        costs calculated independently for each unserved child in step 3.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing static information from the task.

        Args:
            task: The planning task object, containing task.static (static facts),
                  task.goals (goal facts), etc.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store allergy status for each child (True if allergic_gluten)
        self.child_allergy = {}
        # Store waiting location for each child
        self.child_waiting_location = {}

        # Parse static facts to populate the dictionaries
        for fact in static_facts:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "allergic_gluten":
                self.child_allergy[parts[1]] = True
            elif predicate == "not_allergic_gluten":
                self.child_allergy[parts[1]] = False
            elif predicate == "waiting":
                child = parts[1]
                # Ensure child has an entry in allergy map, default to False if not specified
                # (though a well-formed PDDL should specify for all children)
                if child not in self.child_allergy:
                     self.child_allergy[child] = False
                self.child_waiting_location[child] = parts[2] # Store place

        # Identify the set of children that need to be served to reach the goal
        self.goal_children = {get_parts(g)[1] for g in self.goals if match(g, "served", "*")}


    def __call__(self, node):
        """
        Computes the heuristic value (estimated cost to goal) for the given state node.

        Args:
            node: The node in the search space. The state is accessed via `node.state`.

        Returns:
            int: An integer estimate of the number of actions required to reach the goal.
                 Returns 0 if the current state satisfies all goal conditions.
        """
        state = node.state
        total_cost = 0

        # 1. Identify unserved children
        served_children_in_state = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children = self.goal_children - served_children_in_state

        # If all goal children are served, the cost is 0
        if not unserved_children:
            return 0

        # 2. Parse current state for relevant information
        ontray_tuples = set()           # Stores (sandwich, tray) tuples
        kitchen_sandwiches_set = set()  # Stores {sandwich} names at kitchen
        no_gluten_sandwiches = set()    # Stores {sandwich} names that are gluten-free
        at_facts = {}                   # Stores obj -> location mapping from (at obj loc) facts

        for fact in state:
            if match(fact, "ontray", "*", "*"):
                ontray_tuples.add(tuple(get_parts(fact)[1:]))
            elif match(fact, "at_kitchen_sandwich", "*"):
                kitchen_sandwiches_set.add(get_parts(fact)[1])
            elif match(fact, "no_gluten_sandwich", "*"):
                no_gluten_sandwiches.add(get_parts(fact)[1])
            elif match(fact, "at", "*", "*"):
                # Assumes (at obj loc) primarily refers to tray locations in this domain
                obj, loc = get_parts(fact)[1:]
                at_facts[obj] = loc
            # Other facts are ignored for this heuristic calculation

        # Determine current locations of trays based on 'at' facts
        tray_locations = {} # tray -> location
        for obj, loc in at_facts.items():
            # Simple assumption: if an object has an 'at' predicate, it's a tray.
            # This works because ingredients/sandwiches use specific 'at_kitchen_*' predicates.
            tray_locations[obj] = loc

        # Create structured info about sandwiches: where they are and if they are gluten-free
        # Map: sandwich_name -> (tray_name, is_gluten_free)
        ontray_sandwiches = {s: (t, s in no_gluten_sandwiches) for s, t in ontray_tuples}
        # Map: sandwich_name -> is_gluten_free
        kitchen_sandwiches = {s: (s in no_gluten_sandwiches) for s in kitchen_sandwiches_set}

        # 3. Calculate cost for each unserved child
        for child in unserved_children:
            child_cost = 0
            # Retrieve child's requirements from precomputed static info
            needs_gf = self.child_allergy.get(child)
            target_place = self.child_waiting_location.get(child)

            # Basic check for valid problem setup (should always pass for valid instances)
            if needs_gf is None or target_place is None:
                 print(f"Warning: Heuristic calculation skipped for child {child} due to missing static info.")
                 # Assign a high cost or skip, depending on desired behavior for ill-formed problems
                 # For now, just skip, potentially underestimating if problem is malformed.
                 continue

            # --- Determine minimum cost based on available sandwiches ---

            # Case 1: Suitable sandwich on a tray at the target location?
            found_suitable_ontray_at_place = False
            for s, (t, is_gf) in ontray_sandwiches.items():
                tray_loc = tray_locations.get(t)
                if tray_loc == target_place:
                    # Check if sandwich type matches child's need
                    if (needs_gf and is_gf) or (not needs_gf): # Matches if (needs GF and is GF) OR (doesn't need GF)
                        found_suitable_ontray_at_place = True
                        break # Found a suitable one
            if found_suitable_ontray_at_place:
                child_cost = 1 # Only needs 'serve' action
                total_cost += child_cost
                continue # Proceed to the next child

            # Case 2: Suitable sandwich on a tray, but tray is elsewhere?
            found_suitable_ontray_elsewhere = False
            for s, (t, is_gf) in ontray_sandwiches.items():
                tray_loc = tray_locations.get(t)
                # Check if tray location is known and is *not* the target place
                if tray_loc is not None and tray_loc != target_place:
                    if (needs_gf and is_gf) or (not needs_gf):
                        found_suitable_ontray_elsewhere = True
                        break
            if found_suitable_ontray_elsewhere:
                child_cost = 2 # Needs 'move_tray' + 'serve'
                total_cost += child_cost
                continue

            # Case 3: Suitable sandwich exists, but it's still at the kitchen?
            found_suitable_at_kitchen = False
            for s, is_gf in kitchen_sandwiches.items():
                 if (needs_gf and is_gf) or (not needs_gf):
                    found_suitable_at_kitchen = True
                    break
            if found_suitable_at_kitchen:
                cost_put = 1 # Need 'put_on_tray'
                cost_move = 1 if target_place != 'kitchen' else 0 # Need 'move_tray' only if target isn't kitchen
                cost_serve = 1 # Need 'serve'
                child_cost = cost_put + cost_move + cost_serve # Total cost: 2 or 3 actions
                total_cost += child_cost
                continue

            # Case 4: No suitable sandwich exists anywhere; must make one.
            cost_make = 1 # Need 'make_sandwich*'
            cost_put = 1  # Need 'put_on_tray'
            cost_move = 1 if target_place != 'kitchen' else 0 # Need 'move_tray' if target isn't kitchen
            cost_serve = 1 # Need 'serve'
            child_cost = cost_make + cost_put + cost_move + cost_serve # Total cost: 3 or 4 actions
            total_cost += child_cost

        # 4. Return the sum of costs for all unserved children
        return total_cost

