import fnmatch
from heuristics.heuristic_base import Heuristic
# No other external imports are typically needed for basic heuristic calculation
# The 'task' object and 'node' object (with 'node.state') are provided by the planner environment.

# Helper function to parse PDDL facts represented as strings
def get_parts(fact):
    """
    Extracts the predicate and arguments from a PDDL fact string.
    Example: '(at tray1 kitchen)' -> ['at', 'tray1', 'kitchen']
    Handles potential extra whitespace and assumes standard PDDL fact format.
    Returns an empty list if parsing fails or the fact is malformed.
    """
    try:
        # Remove leading/trailing whitespace and the surrounding parentheses
        content = fact.strip()[1:-1]
        # Split by whitespace; filters out empty strings resulting from multiple spaces
        parts = [part for part in content.split() if part]
        return parts
    except IndexError:
        # Return empty list if the fact string is too short or malformed (e.g., "()")
        return []

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain childsnacks.

    # Summary
    This heuristic estimates the minimum number of actions required to reach a goal state
    from the current state in the childsnacks domain. The goal is typically to have
    served a specific set of children. The heuristic calculates an estimate by summing
    the minimum estimated costs required to serve each *unserved* goal child independently.
    It considers the cheapest way to get a suitable sandwich (regular or gluten-free,
    based on the child's allergy) to the child's location via a tray.

    # Assumptions
    - The heuristic calculates the cost for each child independently and sums these costs.
      This means it does not perfectly model resource contention (e.g., limited number
      of sandwiches, ingredients, or trays) or positive interactions (e.g., a single
      tray movement serving multiple children at the same location). This is a common
      simplification for efficiently computable heuristics.
    - It assumes that if a tray needs to be moved (e.g., to the kitchen to pick up a
      sandwich, or from the kitchen to the child's location), it requires at most one
      `move_tray` action per necessary movement segment.
    - It assumes the existence of a unique place named 'kitchen' as defined in the domain.
    - It assumes the problem instance defines enough `sandwich` objects to satisfy the
      needs, if making new sandwiches is required. It does not track the `(notexist ?s)`
      predicate explicitly but relies on finding ingredients if no suitable sandwich exists.
    - If ingredients required to make a necessary sandwich are unavailable at the kitchen,
      the heuristic correctly identifies this path as impossible (infinite cost) and, if
      no other path exists for that child, returns infinity for the state, indicating
      potential unreachability.

    # Heuristic Initialization
    - The constructor (`__init__`) pre-processes information from the task definition.
    - It identifies the set of children that need to be served based on the `task.goals`.
    - It parses `task.static` facts to build efficient lookups for:
        - Each child's allergy status (`child -> is_allergic`: boolean).
        - Each child's waiting location (`child -> place`).
        - Which specific bread portions are gluten-free (`set of GF bread names`).
        - Which specific content portions are gluten-free (`set of GF content names`).
    - It stores the name of the kitchen place ('kitchen').

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Identify Unserved Children:** Determine the set of children specified in the goal
        that are not yet marked as `(served ?c)` in the current `node.state`.
    2.  **Goal Check:** If this set is empty, the goal is reached. Return 0.
    3.  **Initialize Cost:** Set `total_heuristic_cost = 0`.
    4.  **Parse Current State:** Extract dynamic information from `node.state`:
        - Current locations of all trays `(at ?t ?p)`.
        - Sandwiches currently located at the kitchen `(at_kitchen_sandwich ?s)`.
        - Sandwiches currently on a tray `(ontray ?s ?t)`.
        - Which existing sandwiches are gluten-free `(no_gluten_sandwich ?s)`.
        - Which bread portions are currently at the kitchen `(at_kitchen_bread ?b)`.
        - Which content portions are currently at the kitchen `(at_kitchen_content ?c')`.
    5.  **Check Tray at Kitchen:** Determine if at least one tray is currently at the kitchen.
    6.  **Iterate Through Unserved Children:** For each unserved goal child `c`:
        a.  Retrieve the child's allergy requirement (`child_needs_gf`) and target location
            (`target_place`) from the pre-processed static info.
        b.  Initialize `min_child_cost = float('inf')` for this child. This variable will
            store the minimum cost found across the possible ways to serve this child.
        c.  **Evaluate Option 1: Serve from Suitable Sandwich on Tray at Target Location.**
            - Search if any tray `t` at `target_place` holds a suitable sandwich `s`
              (matching `child_needs_gf`).
            - If yes, the cost is 1 (one `serve_sandwich` action). Update `min_child_cost = min(min_child_cost, 1)`.
        d.  **Evaluate Option 2: Serve from Suitable Sandwich on Tray Elsewhere.**
            - Search if any tray `t` at a location *other* than `target_place` holds a
              suitable sandwich `s`.
            - If yes, the cost is 2 (one `move_tray` action + one `serve_sandwich` action).
              Update `min_child_cost = min(min_child_cost, 2)`.
        e.  **Evaluate Option 3: Serve using Suitable Existing Kitchen Sandwich.**
            - Search if a suitable sandwich `s` exists at the kitchen.
            - If yes, calculate the cost based on `target_place` and whether a tray is
              currently at the kitchen:
                - If `target_place == kitchen`: Cost is 2 (put+serve) if tray at kitchen, else 3 (move_tray+put+serve).
                - If `target_place != kitchen`: Cost is 3 (put+move+serve) if tray at kitchen, else 4 (move_tray+put+move+serve).
            - Update `min_child_cost = min(min_child_cost, cost3)`.
        f.  **Evaluate Option 4: Serve by Making a New Sandwich.**
            - Check if suitable ingredients (bread `b`, content `c'`, matching
              `child_needs_gf`) are available at the kitchen.
            - If yes, calculate the cost based on `target_place` and tray availability at kitchen:
                - If `target_place == kitchen`: Cost is 3 (make+put+serve) if tray at kitchen, else 4 (make+move_tray+put+serve).
                - If `target_place != kitchen`: Cost is 4 (make+put+move+serve) if tray at kitchen, else 5 (make+move_tray+put+move+serve).
            - Update `min_child_cost = min(min_child_cost, cost4)`. If ingredients are not available, this option has infinite cost.
        g.  **Accumulate Cost or Detect Unsolvability:**
            - After checking all options, if `min_child_cost` remains `float('inf')`, it implies
              the child cannot be served from the current state (e.g., needed ingredients are
              missing and no suitable sandwich exists). The overall goal is unreachable. Return `float('inf')`.
            - Otherwise, add the found `min_child_cost` for this child to `total_heuristic_cost`.
    7.  **Return Total Cost:** After iterating through all unserved children, return the final
        `total_heuristic_cost`.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
        # Assume 'kitchen' is the standard name for the kitchen place constant
        self.kitchen = 'kitchen'

        # --- Pre-process static information ---
        self.goal_children = set()
        for goal in self.goals:
            parts = get_parts(goal)
            # Ensure the goal is '(served <child>)'
            if len(parts) == 2 and parts[0] == 'served':
                self.goal_children.add(parts[1])

        self.child_allergy = {} # child -> is_allergic (bool)
        self.child_location = {} # child -> place
        # Store GF status from static facts. Assume objects not mentioned are not GF.
        self.static_is_gluten_free_bread = set()
        self.static_is_gluten_free_content = set()

        for fact in self.static:
            parts = get_parts(fact)
            if not parts: continue # Skip if parsing failed
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'allergic_gluten' and len(args) == 1:
                self.child_allergy[args[0]] = True
            elif predicate == 'not_allergic_gluten' and len(args) == 1:
                self.child_allergy[args[0]] = False
            elif predicate == 'waiting' and len(args) == 2:
                self.child_location[args[0]] = args[1]
            elif predicate == 'no_gluten_bread' and len(args) == 1:
                self.static_is_gluten_free_bread.add(args[0])
            elif predicate == 'no_gluten_content' and len(args) == 1:
                self.static_is_gluten_free_content.add(args[0])


    def __call__(self, node):
        state = node.state

        # --- Parse current state ---
        served_children = set()
        kitchen_bread = set()
        kitchen_content = set()
        kitchen_sandwiches = set() # Stores sandwich names
        sandwiches_on_trays = {} # sandwich -> tray name
        tray_locations = {} # tray name -> place name
        is_gluten_free_sandwich = set() # Stores names of GF sandwiches

        for fact in state:
            parts = get_parts(fact)
            # Basic validation of parsed parts
            if not parts: continue
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'served' and len(args) == 1:
                served_children.add(args[0])
            elif predicate == 'at_kitchen_bread' and len(args) == 1:
                kitchen_bread.add(args[0])
            elif predicate == 'at_kitchen_content' and len(args) == 1:
                kitchen_content.add(args[0])
            elif predicate == 'at_kitchen_sandwich' and len(args) == 1:
                kitchen_sandwiches.add(args[0])
            elif predicate == 'ontray' and len(args) == 2:
                sandwiches_on_trays[args[0]] = args[1] # s -> t
            elif predicate == 'at' and len(args) == 2:
                 # Assuming the first arg is the object (tray) and second is location
                 # This relies on the domain structure where only trays use 'at'
                 tray_locations[args[0]] = args[1] # t -> p
            elif predicate == 'no_gluten_sandwich' and len(args) == 1:
                is_gluten_free_sandwich.add(args[0])

        # --- Calculate heuristic value ---
        unserved_goal_children = self.goal_children - served_children

        if not unserved_goal_children:
            return 0 # Goal state reached

        total_heuristic_cost = 0

        # Check if any tray is currently at the kitchen
        tray_at_kitchen = any(loc == self.kitchen for loc in tray_locations.values())

        # Get available ingredients at kitchen and their GF status
        # These sets represent ingredients currently usable for making sandwiches
        available_gf_bread_kitchen = {b for b in kitchen_bread if b in self.static_is_gluten_free_bread}
        available_reg_bread_kitchen = kitchen_bread - available_gf_bread_kitchen
        available_gf_content_kitchen = {c for c in kitchen_content if c in self.static_is_gluten_free_content}
        available_reg_content_kitchen = kitchen_content - available_gf_content_kitchen

        # --- Calculate cost for each unserved child independently ---
        for child in unserved_goal_children:
            child_needs_gf = self.child_allergy.get(child)
            target_place = self.child_location.get(child)

            # Safety check: if static info for a goal child is missing, something is wrong
            if child_needs_gf is None or target_place is None:
                # Indicate an issue with the problem definition or initialization
                return float('inf')

            min_child_cost = float('inf')

            # --- Evaluate Option 1: Serve from Tray at Target ---
            cost1 = float('inf')
            for s, t in sandwiches_on_trays.items():
                s_is_gf = s in is_gluten_free_sandwich
                # Check if sandwich 's' meets the child's dietary needs
                is_suitable = (child_needs_gf and s_is_gf) or (not child_needs_gf)
                if is_suitable:
                    tray_loc = tray_locations.get(t)
                    # Check if the tray 't' holding the suitable sandwich 's' is at the target place
                    if tray_loc == target_place:
                        cost1 = 1 # Cost is 1 action: serve_sandwich
                        break # Found the absolute cheapest way for this child
            min_child_cost = min(min_child_cost, cost1)
            # Optimization: If the minimum cost is already 1, we can't do better.
            if min_child_cost == 1:
                total_heuristic_cost += 1
                continue # Move to the next child

            # --- Evaluate Option 2: Serve from Tray Elsewhere ---
            cost2 = float('inf')
            for s, t in sandwiches_on_trays.items():
                s_is_gf = s in is_gluten_free_sandwich
                is_suitable = (child_needs_gf and s_is_gf) or (not child_needs_gf)
                if is_suitable:
                    tray_loc = tray_locations.get(t)
                    # Check if the tray 't' is at a known location, but NOT the target place
                    if tray_loc is not None and tray_loc != target_place:
                        cost2 = 2 # Cost is 2 actions: move_tray + serve_sandwich
                        break # Found the best cost for this type of option
            min_child_cost = min(min_child_cost, cost2)
            # Optimization: If the minimum cost is now 2, proceed to next child
            if min_child_cost == 2:
                total_heuristic_cost += 2
                continue # Move to the next child

            # --- Evaluate Option 3: Serve from Kitchen Sandwich ---
            cost3 = float('inf')
            found_suitable_kitchen_sandwich = False
            for s in kitchen_sandwiches:
                s_is_gf = s in is_gluten_free_sandwich
                is_suitable = (child_needs_gf and s_is_gf) or (not child_needs_gf)
                if is_suitable:
                    found_suitable_kitchen_sandwich = True
                    break # Found a suitable sandwich at the kitchen

            if found_suitable_kitchen_sandwich:
                if target_place == self.kitchen:
                    # Need: put_on_tray + serve. Requires tray at kitchen for put_on_tray.
                    cost3 = 2 if tray_at_kitchen else 3 # 2 actions or 3 (if tray needs moving first)
                else:
                    # Need: put_on_tray + move_tray + serve. Requires tray at kitchen for put_on_tray.
                    cost3 = 3 if tray_at_kitchen else 4 # 3 actions or 4 (if tray needs moving first)
            min_child_cost = min(min_child_cost, cost3)
            # Cannot continue yet, Option 4 might still be cheaper or equal if cost3 is high

            # --- Evaluate Option 4: Serve by Making Sandwich ---
            cost4 = float('inf')
            # Check if necessary ingredients are available at the kitchen
            ingredients_available = False
            if child_needs_gf:
                # Need at least one GF bread AND one GF content at kitchen
                if available_gf_bread_kitchen and available_gf_content_kitchen:
                    ingredients_available = True
            else:
                # Need at least one bread (any type) AND one content (any type) at kitchen
                if (available_gf_bread_kitchen or available_reg_bread_kitchen) and \
                   (available_gf_content_kitchen or available_reg_content_kitchen):
                    ingredients_available = True

            if ingredients_available:
                if target_place == self.kitchen:
                    # Need: make + put + serve. Requires tray at kitchen for put.
                    cost4 = 3 if tray_at_kitchen else 4 # 3 actions or 4 (if tray needs moving first)
                else:
                    # Need: make + put + move + serve. Requires tray at kitchen for put.
                    cost4 = 4 if tray_at_kitchen else 5 # 4 actions or 5 (if tray needs moving first)
            # If ingredients are not available, cost4 remains infinity

            min_child_cost = min(min_child_cost, cost4)

            # --- Final check and accumulation for this child ---
            if min_child_cost == float('inf'):
                # If no option was possible (cost1-4 all infinity), goal is unreachable from this state
                # This typically happens if ingredients are needed but unavailable.
                return float('inf')
            else:
                # Add the minimum cost found for serving this child to the total
                total_heuristic_cost += min_child_cost

        # Return the total estimated cost for all unserved children
        return total_heuristic_cost

