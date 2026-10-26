import itertools
from fnmatch import fnmatch
# Import the base class - adjust the path if necessary based on the planner's structure
# Assuming the heuristic base class is accessible via this path
from heuristics.heuristic_base import Heuristic

# Helper functions - defined globally for use within the heuristic class
def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Example: "(at tray1 kitchen)" -> ["at", "tray1", "kitchen"]
    Returns an empty list if the fact format is unexpected (e.g., not a string,
    doesn't start/end with parentheses).
    """
    if isinstance(fact, str) and len(fact) > 2 and fact.startswith('(') and fact.endswith(')'):
        # Strip parentheses and split by space
        return fact[1:-1].split()
    return []

def match(fact_str, *pattern):
    """
    Checks if a PDDL fact string matches a given pattern (predicate + arguments).
    Uses fnmatch for wildcard support ('*') in the pattern arguments.
    Example: match("(at tray1 kitchen)", "at", "*", "kitchen") -> True
             match("(served child1)", "served", "*") -> True
    """
    parts = get_parts(fact_str)
    # The number of components in the fact must match the pattern length
    if len(parts) != len(pattern):
        return False
    # Check each part against the corresponding pattern argument using fnmatch
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))


class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the PDDL Childsnack domain.

    # Summary
    Estimates the cost to reach the goal state by summing the estimated minimum actions
    required to serve each currently unserved child. The estimation for each child
    considers the cheapest path based on the current state: using a ready sandwich
    on a tray at the child's location (cost 1), using a sandwich on a tray elsewhere
    (cost 2), using a sandwich from the kitchen (cost 3), or making a new sandwich
    (cost 4).

    # Assumptions
    - The goal is always to have all children listed in the goal served (`(served ?c)`).
    - Static predicates like `waiting`, `allergic_gluten`, `no_gluten_bread`, etc., do not change
      during planning and are available from task initialization (`task.static`).
    - Enough trays exist, and it's assumed at least one can be at the kitchen (or moved there implicitly
      within the cost estimate) when needed for putting a sandwich on a tray. This is relevant
      for estimating costs 3 (using kitchen sandwich) and 4 (making sandwich).
    - Enough ingredients (bread, content) exist to make sandwiches if no suitable ones are readily available;
      the heuristic assumes underlying solvability of the task instance. It does not check for ingredient
      exhaustion when estimating cost 4.
    - The heuristic does not account for resource contention (e.g., one sandwich potentially
      serving multiple children if suitable for both) or shared actions (e.g., one tray move serving multiple
      children at the same destination). It sums the estimated cost for each child independently, which
      might overestimate the true optimal plan cost but serves as a guiding heuristic.

    # Heuristic Initialization (`__init__`)
    - Stores the goal predicates (`task.goals`).
    - Parses static facts provided during task initialization (`task.static`) to build efficient lookups:
        - `child_locations`: dict mapping child name -> waiting location name.
        - `allergic_children`: set containing names of children with `(allergic_gluten ?c)`.
        - `gluten_free_bread`: set containing names of bread portions with `(no_gluten_bread ?b)`.
        - `gluten_free_content`: set containing names of content portions with `(no_gluten_content ?c)`.
        - `goal_children`: set containing names of all children mentioned in the goal facts `(served ?c)`.

    # Step-By-Step Thinking for Computing Heuristic (`__call__`)
    1.  Identify all children `c` present in `goal_children` but not yet served (i.e., `(served c)` is not in the current `node.state`). If no such children exist, the goal is reached, return 0.
    2.  Initialize `total_cost = 0`.
    3.  Parse the current state (`node.state`) to determine the status and location of relevant dynamic objects:
        - `tray_locations`: dict mapping tray name -> current place name.
        - `kitchen_sandwiches`: dict mapping kitchen sandwich name -> is_gluten_free (boolean).
        - `sandwiches_on_tray`: dict mapping sandwich name -> (tray_name, is_gluten_free). (Assumes a sandwich is only on one tray).
        - `kitchen_bread`: dict mapping kitchen bread name -> is_gluten_free.
        - `kitchen_content`: dict mapping kitchen content name -> is_gluten_free.
        - Determine `is_gluten_free` status for sandwiches by checking for `(no_gluten_sandwich s)` facts present in the current state. For ingredients, use the pre-processed static info.
    4.  Determine if at least one tray is currently at the kitchen (`tray_at_kitchen` flag). This is needed to estimate costs involving the `put_on_tray` action.
    5.  For each unserved child `c`:
        a. Retrieve the child's waiting location `p` and allergy status (`is_allergic`) using the pre-processed static info. Handle potential missing info gracefully.
        b. Estimate the minimum cost (`cost_for_child`) to serve this child by checking the following conditions sequentially. The cost is set by the first condition that is met:
            i.   **Cost 1 (Serve):** Is there a suitable sandwich `s` (gluten-free if `is_allergic`) on *any* tray `t` where `(at t p)` is true in the state? If yes, `cost_for_child = 1`. Action sequence: `serve_sandwich*`.
            ii.  **Cost 2 (Move Tray, Serve):** If not Cost 1, is there a suitable sandwich `s` on *any* tray `t` where `(at t p')` is true and `p' != p`? If yes, `cost_for_child = 2`. Action sequence: `move_tray`, `serve_sandwich*`.
            iii. **Cost 3 (Put on Tray, Move Tray, Serve):** If not Cost 1 or 2, is there a suitable sandwich `s` in the kitchen (`(at_kitchen_sandwich s)`) *and* is the `tray_at_kitchen` flag true? If yes, `cost_for_child = 3`. Action sequence: `put_on_tray`, `move_tray`, `serve_sandwich*`.
            iv.  **Cost 4 (Make, Put, Move, Serve):** If none of the above conditions are met, the default estimated cost is 4. This assumes it's possible to perform the sequence: `make_sandwich*`, `put_on_tray`, `move_tray`, `serve_sandwich*`.
        c. Add the determined `cost_for_child` (1, 2, 3, or 4) to `total_cost`.
    6.  Return `total_cost`.
    """

    def __init__(self, task):
        super().__init__(task) # Initialize the base Heuristic class
        self.goals = task.goals
        static_facts = task.static

        # Pre-process static information for efficient lookup during heuristic evaluation
        self.child_locations = {}
        self.allergic_children = set()
        self.gluten_free_bread = set()
        self.gluten_free_content = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip potential empty strings or malformed facts

            predicate = parts[0]
            # Use explicit length checks for robustness against malformed PDDL facts
            if predicate == "waiting" and len(parts) == 3:
                # Map child name to their waiting location name
                self.child_locations[parts[1]] = parts[2]
            elif predicate == "allergic_gluten" and len(parts) == 2:
                # Add child name to the set of allergic children
                self.allergic_children.add(parts[1])
            elif predicate == "no_gluten_bread" and len(parts) == 2:
                # Add bread portion name to the set of gluten-free breads
                self.gluten_free_bread.add(parts[1])
            elif predicate == "no_gluten_content" and len(parts) == 2:
                # Add content portion name to the set of gluten-free contents
                self.gluten_free_content.add(parts[1])

        # Identify all children that need to be served according to the goal definition
        self.goal_children = set()
        for goal_fact in self.goals:
             # Use the match helper for consistency and potential pattern matching
             if match(goal_fact, "served", "*"):
                 parts = get_parts(goal_fact)
                 # Ensure the fact has the expected structure (served child)
                 if len(parts) == 2:
                     self.goal_children.add(parts[1])


    def __call__(self, node):
        """
        Calculates the heuristic value for a given state node.
        """
        state = node.state

        # --- Identify unserved children in the current state ---
        served_children = set()
        for fact in state:
            # Use match for robust parsing of (served c) facts
            if match(fact, "served", "*"):
                parts = get_parts(fact)
                if len(parts) == 2:
                    served_children.add(parts[1])

        # Find children that are in the goal but not yet served
        unserved_children = self.goal_children - served_children

        # If all goal children are served, the heuristic value is 0 (goal reached)
        if not unserved_children:
            return 0

        # --- Parse current state to gather dynamic information ---
        kitchen_sandwiches = {} # Maps sandwich name -> is_gluten_free (bool)
        sandwiches_on_tray = {} # Maps sandwich name -> (tray_name, is_gluten_free)
        tray_locations = {} # Maps tray name -> place name
        kitchen_bread = {} # Maps bread name -> is_gluten_free
        kitchen_content = {} # Maps content name -> is_gluten_free
        # Keep track of sandwiches explicitly marked as gluten-free in this state
        gluten_free_sandwiches_in_state = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]

            # Use if/elif structure for clarity and potential minor efficiency gain
            if predicate == "at" and len(parts) == 3: # Tray location (at tray place)
                tray_locations[parts[1]] = parts[2]
            elif predicate == "at_kitchen_sandwich" and len(parts) == 2:
                # Assume not gluten-free initially; will be updated if (no_gluten_sandwich s) exists
                kitchen_sandwiches[parts[1]] = False
            elif predicate == "ontray" and len(parts) == 3:
                # Assume not gluten-free initially; will be updated later
                sandwiches_on_tray[parts[1]] = (parts[2], False) # Store (tray_name, is_gf)
            elif predicate == "at_kitchen_bread" and len(parts) == 2:
                bread = parts[1]
                # Determine GF status using pre-processed static info
                kitchen_bread[bread] = (bread in self.gluten_free_bread)
            elif predicate == "at_kitchen_content" and len(parts) == 2:
                content = parts[1]
                # Determine GF status using pre-processed static info
                kitchen_content[content] = (content in self.gluten_free_content)
            elif predicate == "no_gluten_sandwich" and len(parts) == 2:
                 # Record sandwiches marked as gluten-free in this specific state
                 gluten_free_sandwiches_in_state.add(parts[1])

        # Update the gluten-free status for sandwiches based on (no_gluten_sandwich s) facts
        # It's safe to iterate directly over keys/items if not modifying dict size
        for s in kitchen_sandwiches:
            if s in gluten_free_sandwiches_in_state:
                kitchen_sandwiches[s] = True
        for s, (t, _) in sandwiches_on_tray.items():
             if s in gluten_free_sandwiches_in_state:
                 # Update the tuple value associated with the sandwich key
                 sandwiches_on_tray[s] = (t, True)

        # --- Calculate heuristic value by summing costs for each unserved child ---
        total_cost = 0
        # Check if *any* tray is currently at the kitchen location
        tray_at_kitchen = any(loc == 'kitchen' for loc in tray_locations.values())

        for child in unserved_children:
            # Retrieve child's location and allergy status from pre-processed data
            child_loc = self.child_locations.get(child)
            # Handle robustness: If child's waiting location isn't defined (e.g., bad PDDL), skip.
            if child_loc is None:
                 # This might indicate an issue with the PDDL instance or static parsing.
                 # Consider logging a warning here in a real implementation.
                 continue

            is_allergic = child in self.allergic_children
            # Start with the highest possible cost (make->put->move->serve) and try to find cheaper options.
            cost_for_child = 4

            # --- Check Level 1: Serve (Cost 1) ---
            # Can we serve directly using a suitable sandwich on a tray already at the child's location?
            found_level1 = False
            for s, (t, s_is_gf) in sandwiches_on_tray.items():
                # Check if the tray 't' containing sandwich 's' is at the child's location
                if tray_locations.get(t) == child_loc:
                    # Check if the sandwich is suitable for the child's allergy status
                    if is_allergic:
                        if s_is_gf: # Allergic child requires a gluten-free sandwich
                            found_level1 = True; break # Found suitable sandwich, stop checking Level 1
                    else: # Non-allergic child can have any sandwich
                        found_level1 = True; break # Found suitable sandwich, stop checking Level 1
            if found_level1:
                cost_for_child = 1
                total_cost += cost_for_child
                continue # Move to the next unserved child

            # --- Check Level 2: Move, Serve (Cost 2) ---
            # Can we serve by moving a tray that has a suitable sandwich but is currently elsewhere?
            found_level2 = False
            for s, (t, s_is_gf) in sandwiches_on_tray.items():
                 tray_loc = tray_locations.get(t)
                 # Check if the tray 't' exists and is *not* at the child's location
                 if tray_loc is not None and tray_loc != child_loc:
                    # Check if the sandwich 's' is suitable
                    if is_allergic:
                        if s_is_gf:
                            found_level2 = True; break
                    else: # Non-allergic
                        found_level2 = True; break
            if found_level2:
                cost_for_child = 2
                total_cost += cost_for_child
                continue # Move to the next unserved child

            # --- Check Level 3: Put, Move, Serve (Cost 3) ---
            # Can we serve using a suitable sandwich currently in the kitchen?
            # This requires a tray to be available at the kitchen.
            found_level3 = False
            if tray_at_kitchen: # Only possible if a tray is ready at the kitchen
                for s, s_is_gf in kitchen_sandwiches.items():
                    # Check if the kitchen sandwich 's' is suitable
                    if is_allergic:
                        if s_is_gf:
                            found_level3 = True; break
                    else: # Non-allergic
                        found_level3 = True; break
            if found_level3:
                cost_for_child = 3
                total_cost += cost_for_child
                continue # Move to the next unserved child

            # --- Level 4: Make, Put, Move, Serve (Cost 4) ---
            # If none of the cheaper options (Levels 1, 2, 3) were applicable,
            # we estimate the cost as 4, assuming the sequence make->put->move->serve is needed.
            # The heuristic assumes this is possible if the state is part of a valid plan.
            # No need for further checks here; the cost remains the default value of 4.
            total_cost += cost_for_child # Add the cost (which is 4 if we reach here)

        # Return the total estimated cost for serving all remaining children
        return total_cost
