import sys
import os
from fnmatch import fnmatch

# Ensure the path includes the directory containing heuristic_base
# This might be needed if the script is run directly or imported from a different location
# Example: Add the parent directory of the 'heuristics' folder to the path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Attempt to import the base class
try:
    # Assumes the script is run from a context where 'heuristics.heuristic_base' is accessible
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Fallback if the standard import path is not set up correctly
    # This assumes heuristic_base.py is in the same directory or PYTHONPATH is set
    print("Could not import Heuristic from heuristics.heuristic_base. Trying local import.", file=sys.stderr)
    try:
        from heuristic_base import Heuristic
    except ImportError:
        # If it still fails, provide a clear error message
        raise ImportError("Could not find Heuristic base class. Ensure 'heuristic_base.py' is accessible.")


# Helper function to parse PDDL facts represented as strings
def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string like "(predicate arg1 arg2)".
    Returns a list of strings: [predicate, arg1, arg2].
    Returns an empty list if the fact format is invalid.
    """
    if isinstance(fact, str) and len(fact) > 2 and fact.startswith("(") and fact.endswith(")"):
        return fact[1:-1].split()
    return []


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain 'childsnacks'.

    # Summary
    Estimates the cost to serve all children in the ChildSnacks domain by summing
    the estimated number of actions required for the following sub-tasks:
    1. Making sandwiches (`make_sandwich*`) if not enough suitable ones exist.
    2. Putting newly made or existing kitchen sandwiches onto trays (`put_on_tray`).
    3. Moving trays to the children's locations (`move_tray`).
    4. Serving the sandwiches (`serve_sandwich*`).
    The heuristic aims to be informative for Greedy Best-First Search and does
    not need to be admissible (i.e., it might overestimate the true cost).

    # Assumptions
    - Assumes an infinite supply of ingredients (bread, content) if needed for
      making sandwiches. The heuristic calculates the cost assuming sandwiches
      *can* be made, ignoring potential dead ends due to lack of specific
      ingredients (e.g., gluten-free). This is acceptable for a non-admissible heuristic.
    - Assumes an infinite supply of abstract sandwich objects (`sandw1`, `sandw2`, ...)
      can be created, ignoring the `(notexist ?s)` precondition check during
      heuristic calculation. This simplifies the estimation.
    - Tray movement cost (`h_move`) is estimated by counting the number of distinct
      child locations that currently lack a tray with a suitable sandwich for
      any child waiting there. Each such location is assumed to require at least
      one `move_tray` action directed towards it. This simplifies complex routing
      or shared trips but provides a basic estimate of delivery effort.
    - The heuristic prioritizes using existing sandwiches over making new ones.
    - It prioritizes using sandwiches already on trays (especially those at the
      target location) over using sandwiches from the kitchen.

    # Heuristic Initialization
    - Parses the task's static facts provided during initialization to store:
        - Each child's allergy status (`child_allergy`: child -> bool). True if allergic to gluten.
        - Each child's waiting location (`child_location`: child -> place).
    - Identifies the set of children that need to be served based on the goal
      description (`goal_children`).
    - Performs basic validation to ensure all goal children have associated
      static information (allergy, location).

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Identify Unserved Children:** Determine the set of `unserved_children`
        by comparing `goal_children` with the current state's `(served ?c)` facts.
        If this set is empty, the goal is reached, and the heuristic value is 0.
    2.  **Inventory Current State:** Scan the current state (set of facts) to
        build data structures representing the current situation:
        *   `sandwiches_at_kitchen`: `{ name: is_gluten_free }` (from `(at_kitchen_sandwich ?s)`)
        *   `sandwiches_on_tray`: `{ name: (is_gluten_free, tray_name, tray_location) }`
            (derived from `(ontray ?s ?t)`, `(at ?t ?loc)`, `(no_gluten_sandwich ?s)`)
        *   `tray_locations`: `{ tray_name: location }` (from `(at ?t ?loc)`)
        *   `sandwich_gf_status`: `{ name: is_gluten_free }` (from `(no_gluten_sandwich ?s)`)
    3.  **Calculate Make/Put_on_Tray Cost (`h_make_put`):**
        *   Initialize `h_make_put = 0`.
        *   Create mutable copies of the available sandwich resources (`avail_kitchen`, `avail_ontray`).
        *   Group `avail_ontray` sandwiches by location (`ontray_by_loc`) for efficient lookup.
        *   Iterate through each `child` in `unserved_children` (sorted for determinism):
            *   Determine if the child needs a gluten-free sandwich (`is_allergic`). Get child's location `child_loc`.
            *   **Priority 1:** Check if a suitable sandwich exists in `ontray_by_loc` at `child_loc`.
                If yes, "consume" it (remove from mutable resources) and assign cost 0 for this child's make/put step.
            *   **Priority 2:** If not found, check if a suitable sandwich exists in `ontray_by_loc` at any *other* location.
                If yes, consume it and assign cost 0 for this child's make/put step.
            *   **Priority 3:** If not found, check if a suitable sandwich exists in `avail_kitchen`.
                If yes, consume it, assign cost 1 (for the required `put_on_tray` action), and add 1 to `h_make_put`.
            *   **Priority 4:** If no suitable sandwich found anywhere, assume one must be made.
                Assign cost 2 (1 for `make_sandwich*`, 1 for `put_on_tray`), and add 2 to `h_make_put`.
    4.  **Calculate Serve Cost (`h_serve`):**
        *   `h_serve = len(unserved_children)`. Each unserved child requires exactly one `serve_sandwich*` action.
    5.  **Calculate Move Cost (`h_move`):**
        *   Initialize `h_move = 0`.
        *   Group `unserved_children` by their waiting location (`children_by_loc`).
        *   Use the *original* state's `sandwiches_on_tray` information (gathered in step 2, not the consumed version from step 3). Group these by location (`current_sandwiches_on_tray_at_loc`).
        *   For each `loc` that has unserved children waiting (i.e., `loc` in `children_by_loc`):
            *   Set a flag `trip_needed = True`.
            *   Check if `loc` exists in `current_sandwiches_on_tray_at_loc`.
            *   If yes, iterate through sandwiches `s` on trays currently at `loc`. If any sandwich `s` is suitable for *any* child waiting at `loc`, set `trip_needed = False` and break the inner checks for this location (as potentially no new trip *to* this location is needed immediately).
            *   If, after checking all possibilities, `trip_needed` remains `True`, increment `h_move` by 1. This signifies that a tray likely needs to be moved *to* this location at some point.
    6.  **Total Heuristic Value:** Return the sum `h = h_make_put + h_move + h_serve`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static information from the task.
        Args:
            task: The planning task object containing goals, initial state, operators, and static facts.
        Raises:
            ValueError: If static information (allergy/location) is missing for any child listed in the goals.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # --- Pre-process static facts ---
        self.child_allergy = {} # child -> is_allergic (bool)
        self.child_location = {} # child -> place

        for fact in self.static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid or empty facts

            pred = parts[0]
            # Check arity for safety
            if pred == "allergic_gluten" and len(parts) == 2:
                self.child_allergy[parts[1]] = True
            elif pred == "not_allergic_gluten" and len(parts) == 2:
                self.child_allergy[parts[1]] = False
            elif pred == "waiting" and len(parts) == 3:
                child_name = parts[1]
                self.child_location[child_name] = parts[2]

        # Identify goal children from the task's goal specification
        self.goal_children = set()
        for goal in self.goals:
             parts = get_parts(goal)
             # Check if it's a 'served' goal predicate
             if parts and parts[0] == "served" and len(parts) == 2:
                 self.goal_children.add(parts[1])

        # --- Validation ---
        # Ensure all children mentioned in goals have corresponding static info
        for child in self.goal_children:
            if child not in self.child_allergy:
                # Default assumption if static facts are incomplete (e.g., assume not allergic)
                # This might happen if a child is only mentioned in the goal.
                # A warning could be useful, but for heuristic calculation, assuming a default might be necessary.
                print(f"Warning: Allergy info missing for goal child {child}. Assuming not allergic.", file=sys.stderr)
                self.child_allergy[child] = False
            if child not in self.child_location:
                # This is critical. If we don't know where the child is, we cannot estimate delivery cost.
                raise ValueError(f"Heuristic Initialization Error: Waiting location missing for goal child {child}")


    def __call__(self, node):
        """
        Calculates the heuristic value for a given state node.
        Args:
            node: The state node in the search space, containing the current state.
        Returns:
            An integer estimate of the remaining actions to reach the goal.
        """
        state = node.state

        # --- 1. Identify Unserved Children ---
        unserved_children = set()
        for child in self.goal_children:
            # Efficiently check if the '(served child)' fact string exists in the state set
            if f"(served {child})" not in state:
                unserved_children.add(child)

        # If all goal children are served, the state is a goal state (or surpasses it)
        if not unserved_children:
            return 0

        # --- 2. Inventory Current State ---
        sandwiches_at_kitchen = {} # name -> is_gluten_free (bool)
        sandwiches_on_tray_interim = {} # name -> [is_gluten_free, tray_name] # Temp storage until tray location known
        tray_locations = {} # tray -> location
        sandwich_gf_status = {} # name -> is_gluten_free (from no_gluten_sandwich facts)

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            pred = parts[0]
            # Check arity for safety
            if pred == "no_gluten_sandwich" and len(parts) == 2:
                 sandwich_gf_status[parts[1]] = True
            elif pred == "at" and len(parts) == 3: # Could be tray or other object location
                obj, loc = parts[1], parts[2]
                # We store all 'at' facts; tray identity confirmed via 'ontray' facts
                tray_locations[obj] = loc
            elif pred == "at_kitchen_sandwich" and len(parts) == 2:
                s_name = parts[1]
                # Use known GF status or default to False
                is_gf = sandwich_gf_status.get(s_name, False)
                sandwiches_at_kitchen[s_name] = is_gf
            elif pred == "ontray" and len(parts) == 3:
                s_name, t_name = parts[1], parts[2]
                is_gf = sandwich_gf_status.get(s_name, False)
                sandwiches_on_tray_interim[s_name] = [is_gf, t_name]

        # Finalize sandwiches_on_tray with actual tray locations
        sandwiches_on_tray = {} # name -> (is_gluten_free, tray, tray_location)
        for s_name, (is_gf, t_name) in sandwiches_on_tray_interim.items():
             if t_name in tray_locations:
                 sandwiches_on_tray[s_name] = (is_gf, t_name, tray_locations[t_name])
             # else: If tray location is unknown, we cannot use this sandwich effectively
             # This indicates an inconsistent state or an object assumed to be a tray wasn't.
             # We simply ignore such sandwiches for the heuristic calculation.

        # --- 3. Calculate Make/Put_on_Tray Cost (h_make_put) ---
        h_make_put = 0
        # Create mutable copies of resources for simulating consumption
        avail_kitchen = dict(sandwiches_at_kitchen)
        avail_ontray = dict(sandwiches_on_tray)

        # Group available ontray sandwiches by location for efficient lookup
        ontray_by_loc = {} # loc -> { s_name: (is_gf, tray) }
        for s_name, (is_gf, tray, loc) in avail_ontray.items():
            if loc not in ontray_by_loc:
                ontray_by_loc[loc] = {}
            ontray_by_loc[loc][s_name] = (is_gf, tray)

        # Process children deterministically
        sorted_unserved_children = sorted(list(unserved_children))

        for child in sorted_unserved_children:
            is_allergic = self.child_allergy[child]
            child_loc = self.child_location[child]
            found_sandwich = False
            consumed_sandwich_name = None # Track which sandwich was consumed from a tray

            # Priority 1: Use sandwich on a tray already at the child's location
            if child_loc in ontray_by_loc:
                # Iterate over a copy for safe deletion
                for s_name, (is_gf, tray) in list(ontray_by_loc[child_loc].items()):
                    if (is_allergic and is_gf) or (not is_allergic): # Check suitability
                        consumed_sandwich_name = s_name
                        found_sandwich = True
                        del ontray_by_loc[child_loc][s_name] # Consume from location group
                        if not ontray_by_loc[child_loc]: del ontray_by_loc[child_loc] # Clean up if empty
                        break # Found one, stop checking this location
                if found_sandwich:
                    pass # Cost = 0 for make/put

            # Priority 2: Use sandwich on a tray at a different location
            if not found_sandwich:
                # Iterate over copy of location keys
                other_locs = [loc for loc in list(ontray_by_loc.keys()) if loc != child_loc]
                for loc in other_locs:
                     if loc not in ontray_by_loc: continue # Skip if location was emptied
                     # Iterate over copy of items
                     for s_name, (is_gf, tray) in list(ontray_by_loc[loc].items()):
                         if (is_allergic and is_gf) or (not is_allergic):
                             consumed_sandwich_name = s_name
                             found_sandwich = True
                             del ontray_by_loc[loc][s_name] # Consume from location group
                             if not ontray_by_loc[loc]: del ontray_by_loc[loc] # Clean up
                             break # Found one, stop checking this location
                     if found_sandwich: break # Stop checking other locations

            # If a sandwich was consumed from any tray, remove it from the main avail_ontray dict
            if consumed_sandwich_name is not None and consumed_sandwich_name in avail_ontray:
                 del avail_ontray[consumed_sandwich_name]

            # Priority 3: Use sandwich from the kitchen
            if not found_sandwich:
                consumed_sandwich_name_kitchen = None
                # Iterate over copy of kitchen items
                for s_name, is_gf in list(avail_kitchen.items()):
                    if (is_allergic and is_gf) or (not is_allergic):
                        consumed_sandwich_name_kitchen = s_name
                        found_sandwich = True
                        break
                if found_sandwich:
                    del avail_kitchen[consumed_sandwich_name_kitchen] # Consume from kitchen
                    h_make_put += 1 # Cost = 1 (for put_on_tray)

            # Priority 4: Make a new sandwich
            if not found_sandwich:
                # Assume we can make one (infinite ingredients/sandwich objects)
                h_make_put += 2 # Cost = 1 (make) + 1 (put_on_tray)

        # --- 4. Calculate Serve Cost (h_serve) ---
        h_serve = len(unserved_children)

        # --- 5. Calculate Move Cost (h_move) ---
        # This calculation uses the *original* state inventory (sandwiches_on_tray),
        # not the consumed versions from step 3.
        h_move = 0
        children_by_loc = {} # loc -> {child1, child2, ...}
        for child in unserved_children:
             loc = self.child_location[child]
             if loc not in children_by_loc:
                 children_by_loc[loc] = set()
             children_by_loc[loc].add(child)

        # Group the original state's on-tray sandwiches by location
        current_sandwiches_on_tray_at_loc = {} # loc -> { s_name: (is_gf, tray) }
        for s_name, (is_gf, tray, loc) in sandwiches_on_tray.items():
             if loc not in current_sandwiches_on_tray_at_loc:
                 current_sandwiches_on_tray_at_loc[loc] = {}
             current_sandwiches_on_tray_at_loc[loc][s_name] = (is_gf, tray)

        # Check each location with waiting children
        for loc, children_at_this_loc in children_by_loc.items():
            trip_needed = True # Assume a trip is needed unless proven otherwise
            if loc in current_sandwiches_on_tray_at_loc:
                # Check if any sandwich on a tray at this location is suitable for any child here
                for s_name, (is_gf, tray) in current_sandwiches_on_tray_at_loc[loc].items():
                    # Is this sandwich suitable for *any* child waiting here?
                    for child in children_at_this_loc:
                         is_allergic = self.child_allergy[child]
                         if (is_allergic and is_gf) or (not is_allergic):
                             # Yes, a suitable sandwich exists on a tray currently at this location.
                             # This might satisfy the need, potentially avoiding a new trip *to* here.
                             trip_needed = False
                             break # Found a suitable sandwich for one child
                    if not trip_needed:
                        break # No need to check other sandwiches at this location

            # If no suitable sandwich was found on any tray currently at this location
            if trip_needed:
                h_move += 1 # Increment cost, estimating one move action to bring a tray here

        # --- 6. Total Heuristic Value ---
        h_total = h_make_put + h_move + h_serve
        return h_total

