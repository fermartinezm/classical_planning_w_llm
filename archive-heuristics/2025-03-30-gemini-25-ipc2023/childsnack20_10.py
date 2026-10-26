import sys
from fnmatch import fnmatch
# Assume the base class Heuristic is available in the Python path,
# typically expected in a directory structure like:
# project_root/
#   heuristics/
#     heuristic_base.py
#     childsnack_heuristic.py (this file)
#   planner.py (or similar)
# If the structure differs, the import path might need adjustment.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Provide a dummy class if the import fails, e.g., for standalone testing
    # This helps with basic syntax checks but won't work for actual planning.
    print("Warning: Heuristic base class not found. Using a dummy base class.", file=sys.stderr)
    class Heuristic:
        def __init__(self, task): pass
        def __call__(self, node): return 0


# Helper functions for parsing PDDL fact strings
def get_parts(fact: str) -> list[str]:
    """
    Extracts the predicate and arguments from a PDDL fact string.
    Example: "(at tray1 kitchen)" -> ["at", "tray1", "kitchen"]
    Removes parentheses and splits by whitespace.
    """
    return fact.strip()[1:-1].split()

def match(fact: str, *pattern: str) -> bool:
    """
    Checks if a PDDL fact string matches a given pattern.
    The pattern should have the same number of elements as the fact (predicate + arguments).
    Pattern elements can be '*' to act as a wildcard for that position.
    Uses fnmatch for wildcard matching within each element comparison.
    Example: match("(at tray1 kitchen)", "at", "*", "kitchen") -> True
    Example: match("(at tray1 kitchen)", "at", "tray?", "*") -> True
    Example: match("(served child1)", "served", "*") -> True
    """
    parts = get_parts(fact)
    if len(parts) != len(pattern):
        return False
    # Compare each part of the fact with the corresponding pattern element
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain childsnacks.

    # Summary
    This heuristic estimates the number of actions required to serve all children
    as specified in the goal state. It calculates the cost for each unserved
    child individually and sums these costs. The cost for a child involves
    finding or making a suitable sandwich, putting it on a tray, moving the
    tray to the child's location, and serving the sandwich. The heuristic tries
    to reuse existing sandwiches and considers tray locations to estimate costs
    more accurately. It prioritizes using sandwiches already on trays, then
    sandwiches in the kitchen, and finally resorts to making new sandwiches.

    # Assumptions
    - The primary goal is always a conjunction of `(served ?c)` predicates for children.
    - There are enough raw ingredients (bread, content) to make sandwiches for
      all children who need one made. Ingredient availability is not checked.
    - A tray can be moved between any two locations in a single `move_tray` action.
    - The cost calculation assumes a greedy assignment of existing sandwiches
      to children. It processes children one by one and "consumes" the chosen
      resource for that child, preventing its reuse in the calculation for
      subsequent children within the same state evaluation. This makes the
      heuristic potentially non-admissible but aims for better guidance in a
      greedy search.
    - The constant place representing the kitchen is named 'kitchen'.
    - At least one tray exists in the problem instance if sandwiches need to be made or moved from the kitchen.

    # Heuristic Initialization
    - Stores the goal predicates `(served ?c)`.
    - Parses static facts provided in the task object to create mappings for:
        - `child_is_allergic`: dictionary mapping child name (str) to boolean (True if allergic to gluten).
        - `child_waiting_location`: dictionary mapping child name (str) to place name (str).
    - Identifies the set of children that need to be served according to the goal predicates.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Identify State:** Get the current state's facts (a frozenset of strings). Parse them to find:
        - `served_children`: A set of names of children already served.
        - `trays_at_location`: A dictionary mapping tray name (str) to its current location name (str).
        - `sandwiches_on_tray`: A dictionary mapping sandwich name (str) to the tray name (str) it is on.
        - `sandwiches_at_kitchen`: A set of names of sandwiches currently at the kitchen.
        - `gf_sandwiches`: A set of names of sandwiches that are gluten-free `(no_gluten_sandwich ?s)`.
    2.  **Identify Needs:** Determine the set of `unserved_goal_children` by taking the difference between the goal children and the `served_children`. If this set is empty, the goal is reached, return 0.
    3.  **Resource Availability:** Create mutable copies of the available sandwiches (`current_sandwiches_on_tray`, `current_sandwiches_at_kitchen`) to track consumption during the heuristic calculation. Check if any tray is currently at the kitchen. Calculate `cost_to_get_tray_to_kitchen` (0 if a tray is already there, 1 otherwise, assuming a tray can always be moved if needed).
    4.  **Iterate Through Unserved Children:** For each `child` in `unserved_goal_children`:
        a.  Get the child's waiting location `child_loc` and whether they need a gluten-free sandwich (`needs_gf`).
        b.  Initialize `min_cost_for_child` to infinity and `chosen_option_details` to None.
        c.  **Option 1: Serve from Tray at Location:** Check if a suitable sandwich `s` (matching `needs_gf`) exists on any tray `t` currently at `child_loc`.
            - If yes, Cost = 1 (serve). Update `min_cost_for_child` to 1 and record `chosen_option_details = (1, s)`. Since this is the minimum possible cost, break the search for this child's options.
        d.  **Option 2: Serve from Tray Elsewhere:** If Option 1 didn't apply or cost > 2, check if a suitable sandwich `s` exists on any tray `t` at a different location.
            - If yes, Cost = 1 (move) + 1 (serve) = 2. If 2 < `min_cost_for_child`, update `min_cost_for_child` to 2 and record `chosen_option_details = (2, s)`. Break the inner loop (found a candidate for this option).
        e.  **Option 3: Serve from Kitchen:** If current `min_cost_for_child` is greater than the potential cost here, check if a suitable sandwich `s` exists at the kitchen.
            - Cost = 1 (put_on_tray) + 1 (move, if `child_loc` != kitchen) + 1 (serve) + `cost_to_get_tray_to_kitchen`.
            - If this cost < `min_cost_for_child`, update `min_cost_for_child` and record `chosen_option_details = (3, s)`. Break the inner loop.
        f.  **Option 4: Make New Sandwich:** If current `min_cost_for_child` is greater than the potential cost here.
            - Cost = 1 (make) + 1 (put_on_tray) + 1 (move, if `child_loc` != kitchen) + 1 (serve) + `cost_to_get_tray_to_kitchen`.
            - If this cost < `min_cost_for_child`, update `min_cost_for_child` and record `chosen_option_details = (4, None)`.
        g.  **Accumulate Cost:** Add the final `min_cost_for_child` for this child to the total `h_value`. Handle the unlikely case where no option was found (assign a large cost).
        h.  **Update Resources:** If `chosen_option_details` indicates an existing sandwich was used (type 1, 2, or 3), remove that sandwich `s` from the corresponding mutable resource set/dictionary (`current_sandwiches_on_tray` or `current_sandwiches_at_kitchen`) to prevent it from being used again for another child in this state evaluation.
    5.  **Return Total:** The final `h_value` is the sum of the minimum costs calculated for each unserved child.
    """

    def __init__(self, task):
        # Initialize base class if it has an __init__ method
        super().__init__(task)
        self.goals = task.goals
        self.static_facts = task.static
        self.child_is_allergic = {}
        self.child_waiting_location = {}
        # Define the kitchen constant name used in the domain
        self.kitchen_place = 'kitchen'

        # Parse static facts once during initialization
        for fact in self.static_facts:
            parts = get_parts(fact)
            predicate = parts[0]
            # Check length to avoid errors on malformed facts
            if len(parts) > 1:
                if predicate == 'allergic_gluten':
                    self.child_is_allergic[parts[1]] = True
                elif predicate == 'not_allergic_gluten':
                    self.child_is_allergic[parts[1]] = False
            if len(parts) > 2:
                 if predicate == 'waiting':
                    # Child name is parts[1], place name is parts[2]
                    self.child_waiting_location[parts[1]] = parts[2]

        # Identify children mentioned in goals that need serving
        self.goal_children = set()
        for goal in self.goals:
             # Use match helper function for robustness
             if match(goal, "served", "*"):
                 self.goal_children.add(get_parts(goal)[1])


    def __call__(self, node):
        state = node.state
        h_value = 0

        # --- 1. Identify State Information ---
        served_children = set()
        trays_at_location = {} # Map: tray_name -> location_name
        sandwiches_on_tray = {} # Map: sandwich_name -> tray_name
        sandwiches_at_kitchen = set() # Set: {sandwich_name}
        gf_sandwiches = set() # Set: {sandwich_name}

        for fact in state:
            # Use match helper function for safety and clarity
            if match(fact, "served", "*"):
                served_children.add(get_parts(fact)[1])
            elif match(fact, "at", "*", "*"): # Matches (at trayX placeY)
                parts = get_parts(fact)
                # Basic check: assume first arg is tray, second is place
                # A more robust check could use task object types if available
                trays_at_location[parts[1]] = parts[2]
            elif match(fact, "ontray", "*", "*"): # Matches (ontray sandwichX trayY)
                parts = get_parts(fact)
                sandwiches_on_tray[parts[1]] = parts[2]
            elif match(fact, "at_kitchen_sandwich", "*"): # Matches (at_kitchen_sandwich sandwichX)
                sandwiches_at_kitchen.add(get_parts(fact)[1])
            elif match(fact, "no_gluten_sandwich", "*"): # Matches (no_gluten_sandwich sandwichX)
                gf_sandwiches.add(get_parts(fact)[1])

        # --- 2. Identify Unserved Children ---
        unserved_goal_children = self.goal_children - served_children

        # If all goal children are served, heuristic value is 0
        if not unserved_goal_children:
            return 0

        # --- 3. Prepare Resource Tracking for Calculation ---
        # Create mutable copies to simulate resource consumption within this heuristic call
        current_sandwiches_on_tray = sandwiches_on_tray.copy()
        current_sandwiches_at_kitchen = sandwiches_at_kitchen.copy()

        # Check if any tray is currently at the kitchen
        tray_at_kitchen_exists = any(loc == self.kitchen_place for loc in trays_at_location.values())
        # Cost to bring a tray to kitchen if needed for put_on_tray action.
        # Assumes at least one tray exists and can be moved if needed.
        cost_to_get_tray_to_kitchen = 0 if tray_at_kitchen_exists else 1

        # --- 4. Iterate Through Unserved Children and Calculate Cost ---
        # Convert set to list for ordered iteration (order doesn't strictly matter here)
        children_to_process = list(unserved_goal_children)

        for child in children_to_process:
            child_loc = self.child_waiting_location.get(child)
            # If a goal child isn't in the waiting list (problem inconsistency), skip them.
            if child_loc is None:
                 print(f"Warning: Goal child {child} has no waiting location in static facts.", file=sys.stderr)
                 continue

            # Determine if the child needs a gluten-free sandwich
            needs_gf = self.child_is_allergic.get(child, False) # Default to False if info missing

            min_cost_for_child = float('inf')
            # Store tuple: (option_type, consumed_sandwich_name_or_None)
            chosen_option_details = None

            # --- Evaluate Option 1: Serve from Tray at Child's Location ---
            best_s_opt1 = None
            cost_opt1 = 1 # Cost is just the 'serve' action
            for s, t in current_sandwiches_on_tray.items():
                tray_loc = trays_at_location.get(t) # Get the tray's current location
                if tray_loc == child_loc:
                    is_gf = s in gf_sandwiches
                    # Check if sandwich type matches need
                    if (needs_gf and is_gf) or (not needs_gf):
                        min_cost_for_child = cost_opt1
                        best_s_opt1 = s
                        break # Found the best possible option (cost 1)

            if best_s_opt1 is not None:
                 chosen_option_details = (1, best_s_opt1)
                 # If cost is 1, no need to check other options for this child

            # --- Evaluate Option 2: Serve from Tray Elsewhere ---
            # Only check if the current best cost is more than 2
            if min_cost_for_child > 2:
                best_s_opt2 = None
                cost_opt2 = 2 # move(1) + serve(1)
                for s, t in current_sandwiches_on_tray.items():
                    tray_loc = trays_at_location.get(t)
                    if tray_loc != child_loc: # Ensure tray is not at the child's location
                        is_gf = s in gf_sandwiches
                        if (needs_gf and is_gf) or (not needs_gf):
                            min_cost_for_child = cost_opt2
                            best_s_opt2 = s
                            break # Found a candidate for option 2

                if best_s_opt2 is not None:
                    chosen_option_details = (2, best_s_opt2)

            # --- Evaluate Option 3: Serve from Kitchen ---
            cost_opt3 = 1 # put_on_tray
            if child_loc != self.kitchen_place:
                cost_opt3 += 1 # move tray
            cost_opt3 += 1 # serve
            cost_opt3 += cost_to_get_tray_to_kitchen # Add cost if no tray at kitchen initially

            # Only check if potentially better than the current minimum cost found
            if cost_opt3 < min_cost_for_child:
                best_s_opt3 = None
                for s in current_sandwiches_at_kitchen:
                    is_gf = s in gf_sandwiches
                    if (needs_gf and is_gf) or (not needs_gf):
                        min_cost_for_child = cost_opt3
                        best_s_opt3 = s
                        break # Found a suitable sandwich at the kitchen

                if best_s_opt3 is not None:
                    chosen_option_details = (3, best_s_opt3)

            # --- Evaluate Option 4: Make New Sandwich ---
            # Assumes ingredients are available
            cost_opt4 = 1 # make action
            cost_opt4 += 1 # put_on_tray action
            if child_loc != self.kitchen_place:
                cost_opt4 += 1 # move tray action
            cost_opt4 += 1 # serve action
            cost_opt4 += cost_to_get_tray_to_kitchen # Add cost if no tray at kitchen initially

            # Only check if potentially better than the current minimum cost found
            if cost_opt4 < min_cost_for_child:
                min_cost_for_child = cost_opt4
                chosen_option_details = (4, None) # No specific existing sandwich consumed

            # --- Accumulate Cost and Update Resources ---
            if min_cost_for_child == float('inf'):
                 # This implies no option was found, which shouldn't happen if Option 4
                 # (make new) is always considered possible. Could indicate an unsolvable
                 # state if ingredients were tracked, or an issue here.
                 # Assign a large cost for Greedy Best First Search robustness.
                 print(f"Warning: Could not find any path to serve child {child}. Assigning high cost.", file=sys.stderr)
                 h_value += 1000 # Arbitrarily large cost
                 continue # Skip resource update for this child

            # Add the minimum cost found for this child to the total heuristic value
            h_value += min_cost_for_child

            # Simulate resource consumption for the chosen option
            if chosen_option_details is not None:
                option_type, consumed_s = chosen_option_details
                # If an existing sandwich was used (Options 1, 2, or 3)
                if consumed_s is not None:
                    if option_type == 1 or option_type == 2:
                        # Remove sandwich from the mutable 'on tray' list
                        if consumed_s in current_sandwiches_on_tray:
                            del current_sandwiches_on_tray[consumed_s]
                    elif option_type == 3:
                        # Remove sandwich from the mutable 'at kitchen' list
                        # Use discard() as it doesn't raise an error if key is already gone
                        current_sandwiches_at_kitchen.discard(consumed_s)
                # Option 4 (make new) doesn't consume an existing tracked sandwich resource.

        # Return the total estimated cost for all unserved children
        return h_value
