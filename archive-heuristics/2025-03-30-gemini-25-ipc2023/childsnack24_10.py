import math
from fnmatch import fnmatch
# Import base class assumed to be available via heuristics.heuristic_base
# If the environment differs, this import path might need adjustment.
from heuristics.heuristic_base import Heuristic

# Helper function (as provided in examples)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Robust check for empty strings or unexpected formats
    if not fact or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        # Return empty list for invalid format to avoid errors in subsequent processing
        return []
    return fact[1:-1].split()


# Helper function (as provided in examples)
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check arity only if args are provided
    if args and len(parts) != len(args):
        return False
    # Handle case where parts might be empty if fact was invalid
    if not parts and args:
        return False
    # Handle case where no args are provided (match any fact) - not typical usage here
    if not args:
        # Match if fact is non-empty, return False otherwise? Or True?
        # Let's assume specific patterns are always given, so this case isn't hit.
        # If it were, returning True might be expected (matches anything).
        # For safety, let's return False if no args given, requires explicit pattern.
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    # Summary
    Estimates the cost to reach the goal state in the ChildSnacks domain by summing
    the estimated costs for serving each currently unserved child. The estimate for
    each child considers the cheapest way to get them a suitable sandwich (using
    existing ones on trays, existing ones at the kitchen, or making a new one),
    ignoring resource contention. This heuristic is designed for Greedy Best-First
    Search and is likely not admissible.

    # Assumptions
    - The primary goal is to achieve all `(served ?c)` facts specified in the task goal.
    - Each unserved child requires a sequence of actions: potentially make sandwich (1),
      put on tray (1), move tray (1 if needed), serve (1).
    - The heuristic checks if existing sandwiches (at kitchen or on tray) can
      be used to shorten this sequence for a child.
    - Allergy requirements (gluten-free) specified by `(allergic_gluten ?c)` and
      `(no_gluten_...)` predicates are strictly followed.
    - Resource contention (e.g., limited trays preventing parallel serving, limited
      ingredients, or a single tray serving multiple children in one trip) is ignored
      for simplicity and computational efficiency. The estimate for each child assumes
      resources (trays, ingredients) are available for their specific path.
    - Tray movement cost is simplified: 0 if the tray is already at the target
      location (child's location for serving, or kitchen for putting sandwich),
      1 otherwise (representing one `move_tray` action).
    - Assumes 'kitchen' is the unique, constant place for making sandwiches and
      loading them onto trays initially.

    # Heuristic Initialization
    - Stores the set of children that need to be served based on `(served ?c)`
      predicates in the goal definition (`self.goal_children`).
    - Parses static facts (`task.static`) provided during initialization to build
      efficient lookups:
        - `self.child_location`: Maps each child object to their waiting place object.
        - `self.child_allergy`: Maps each child object to a boolean (True if allergic
          to gluten, False otherwise).
        - `self.gluten_free_bread`: A set containing all bread portion objects marked
          as `(no_gluten_bread ...)`.
        - `self.gluten_free_content`: A set containing all content portion objects marked
          as `(no_gluten_content ...)`.
    - Performs basic validation to warn if goal children lack necessary static info.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Current State:** Iterate through the facts in the current state (`node.state`)
        to identify:
        - `served_children`: Set of children already served.
        - `sandwiches_at_kitchen`: Set of sandwiches currently at the kitchen.
        - `sandwiches_on_tray`: Dictionary mapping sandwich -> tray for sandwiches on trays.
        - `is_gluten_free_sandwich`: Set of sandwiches marked as gluten-free.
        - `tray_locations`: Dictionary mapping tray -> current location.
        - `available_bread`: Set of bread portions available at the kitchen.
        - `available_content`: Set of content portions available at the kitchen.
    2.  **Identify Unserved Children:** Calculate the set difference:
        `unserved_children = self.goal_children - served_children`.
    3.  **Handle Goal State:** If `unserved_children` is empty, the goal is reached. Return 0.
    4.  **Initialize Total Cost:** `total_heuristic_value = 0`.
    5.  **Iterate Through Unserved Children:** For each `child` in `unserved_children`:
        a. **Retrieve Child Info:** Get the child's waiting location (`child_loc`) and
           allergy status (`is_allergic`) from the pre-processed static info stored
           during initialization. If this info is missing (due to invalid PDDL or setup),
           return `float('inf')` as the state is likely invalid or unsolvable.
        b. **Calculate Minimum Cost for Child:** Initialize `min_cost_for_child = float('inf')`.
           Evaluate three possibilities to serve this child:
            i.  **Use Sandwich on Tray:** Find all sandwiches `s` currently on any tray `t`
                that are suitable for the child (gluten-free if `is_allergic`, any otherwise).
                For each suitable `s` on tray `t` at location `p_tray`, calculate the cost:
                `cost = (1 if p_tray != child_loc else 0) [move_tray] + 1 [serve]`.
                Update `min_cost_for_child = min(min_cost_for_child, cost)`.
            ii. **Use Sandwich at Kitchen:** Find all sandwiches `s` currently at the kitchen
                that are suitable for the child. If any exist, calculate the estimated cost:
                `cost = 1 [put_on_tray] + (1 if child_loc != 'kitchen' else 0) [move_tray] + 1 [serve]`.
                This assumes a tray is available at the kitchen.
                Update `min_cost_for_child = min(min_cost_for_child, cost)`.
            iii. **Make New Sandwich:** Check if the required ingredients are available at the
                kitchen (specific gluten-free bread/content if `is_allergic`, or any
                bread/content pair otherwise). If yes, calculate the estimated cost:
                `cost = 1 [make] + 1 [put_on_tray] + (1 if child_loc != 'kitchen' else 0) [move_tray] + 1 [serve]`.
                This assumes a tray is available at the kitchen.
                Update `min_cost_for_child = min(min_cost_for_child, cost)`.
        c.  **Check Solvability for Child:** If `min_cost_for_child` is still `float('inf')`
            after checking all options, it means this child cannot be served even under
            optimistic assumptions. The state is considered a dead end. Return `float('inf')`.
        d.  **Accumulate Cost:** Add the calculated `min_cost_for_child` to
            `total_heuristic_value`.
    6.  **Return Total Cost:** Return the final `total_heuristic_value`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing static information and goal states.
        """
        super().__init__(task) # Ensure base class init is called if necessary
        self.goals = task.goals
        self.static_facts = task.static

        # Pre-process static facts for efficient lookup
        self.child_location = {}
        self.child_allergy = {} # True if allergic_gluten
        self.gluten_free_bread = set()
        self.gluten_free_content = set()
        self.goal_children = set()

        for fact in self.static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip if fact is malformed

            predicate = parts[0]
            try:
                # Extract relevant static information
                if predicate == "waiting" and len(parts) == 3:
                    self.child_location[parts[1]] = parts[2]
                elif predicate == "allergic_gluten" and len(parts) == 2:
                    self.child_allergy[parts[1]] = True
                elif predicate == "not_allergic_gluten" and len(parts) == 2:
                     self.child_allergy[parts[1]] = False
                elif predicate == "no_gluten_bread" and len(parts) == 2:
                    self.gluten_free_bread.add(parts[1])
                elif predicate == "no_gluten_content" and len(parts) == 2:
                    self.gluten_free_content.add(parts[1])
            except IndexError:
                # Log or ignore potential errors from malformed facts
                # print(f"Warning: Malformed static fact encountered: {fact}")
                pass

        # Identify children mentioned in the goal
        for goal_fact in self.goals:
             parts = get_parts(goal_fact)
             if not parts: continue # Skip if goal is malformed
             try:
                 if parts[0] == "served" and len(parts) == 2:
                     self.goal_children.add(parts[1])
             except IndexError:
                 # Log or ignore potential errors from malformed goals
                 # print(f"Warning: Malformed goal fact encountered: {goal_fact}")
                 pass

        # Optional validation: Check if all goal children have necessary info
        for child in self.goal_children:
            if child not in self.child_location:
                print(f"Warning: Goal child {child} missing 'waiting' info in static facts.")
            if child not in self.child_allergy:
                print(f"Warning: Goal child {child} missing allergy info in static facts.")


    def __call__(self, node):
        """
        Computes the heuristic value for the given state node.
        Estimates the remaining actions needed to serve all goal children.
        """
        state = node.state
        heuristic_value = 0

        # --- State Parsing ---
        # Efficiently gather current state information relevant to the heuristic
        served_children = set()
        sandwiches_at_kitchen = set()
        sandwiches_on_tray = {} # Map sandwich -> tray
        is_gluten_free_sandwich = set()
        tray_locations = {} # Map tray -> current location
        available_bread = set()
        available_content = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]

            try:
                # Map facts to internal data structures
                if predicate == "served" and len(parts) == 2:
                    served_children.add(parts[1])
                elif predicate == "at_kitchen_sandwich" and len(parts) == 2:
                    sandwiches_at_kitchen.add(parts[1])
                elif predicate == "ontray" and len(parts) == 3:
                    sandwiches_on_tray[parts[1]] = parts[2] # sandwich -> tray
                elif predicate == "no_gluten_sandwich" and len(parts) == 2:
                    is_gluten_free_sandwich.add(parts[1])
                elif predicate == "at" and len(parts) == 3: # Assume (at tray place)
                     tray_locations[parts[1]] = parts[2] # tray -> location
                elif predicate == "at_kitchen_bread" and len(parts) == 2:
                    available_bread.add(parts[1])
                elif predicate == "at_kitchen_content" and len(parts) == 2:
                    available_content.add(parts[1])
            except IndexError:
                # Silently ignore malformed facts in state
                pass

        # --- Heuristic Calculation ---
        unserved_children = self.goal_children - served_children

        # If all goal children are served, heuristic is 0
        if not unserved_children:
            return 0

        # Categorize available ingredients for easier checking
        available_gf_bread = {b for b in available_bread if b in self.gluten_free_bread}
        available_reg_bread = available_bread - available_gf_bread
        available_gf_content = {c for c in available_content if c in self.gluten_free_content}
        available_reg_content = available_content - available_gf_content

        # Calculate cost for each unserved child
        for child in unserved_children:
            child_loc = self.child_location.get(child)
            is_allergic = self.child_allergy.get(child)

            # If static info is missing, state is likely invalid or unsolvable path
            if child_loc is None or is_allergic is None:
                return float('inf')

            min_cost_for_child = float('inf')

            # --- Possibility 1: Use an existing sandwich on a tray ---
            cost1 = float('inf')
            suitable_sandwiches_on_tray = []
            if is_allergic:
                suitable_sandwiches_on_tray = [s for s, t in sandwiches_on_tray.items() if s in is_gluten_free_sandwich]
            else:
                suitable_sandwiches_on_tray = list(sandwiches_on_tray.keys())

            for s in suitable_sandwiches_on_tray:
                tray = sandwiches_on_tray[s]
                tray_loc = tray_locations.get(tray)
                if tray_loc is not None:
                    # Cost = move tray (if needed) + serve
                    move_cost = 0 if tray_loc == child_loc else 1
                    serve_cost = 1
                    cost1 = min(cost1, move_cost + serve_cost)
            min_cost_for_child = min(min_cost_for_child, cost1)

            # --- Possibility 2: Use an existing sandwich from the kitchen ---
            cost2 = float('inf')
            suitable_sandwiches_at_kitchen = []
            if is_allergic:
                suitable_sandwiches_at_kitchen = [s for s in sandwiches_at_kitchen if s in is_gluten_free_sandwich]
            else:
                suitable_sandwiches_at_kitchen = list(sandwiches_at_kitchen)

            if suitable_sandwiches_at_kitchen:
                # Cost = put on tray + move tray (if needed) + serve
                put_cost = 1
                move_cost = 0 if child_loc == 'kitchen' else 1 # Assumes tray starts at kitchen
                serve_cost = 1
                cost2 = put_cost + move_cost + serve_cost
            min_cost_for_child = min(min_cost_for_child, cost2)

            # --- Possibility 3: Make a new sandwich ---
            cost3 = float('inf')
            can_make = False
            if is_allergic:
                # Need specific GF ingredients
                if available_gf_bread and available_gf_content:
                    can_make = True
            else:
                # Need any pair of bread/content
                if (available_gf_bread or available_reg_bread) and \
                   (available_gf_content or available_reg_content):
                    can_make = True

            if can_make:
                 # Cost = make + put on tray + move tray (if needed) + serve
                 make_cost = 1
                 put_cost = 1
                 move_cost = 0 if child_loc == 'kitchen' else 1 # Assumes tray starts at kitchen
                 serve_cost = 1
                 cost3 = make_cost + put_cost + move_cost + serve_cost
            min_cost_for_child = min(min_cost_for_child, cost3)

            # --- Final check for the child ---
            if min_cost_for_child == float('inf'):
                 # If no option is possible, this state is a dead end for this child's goal.
                 # Return infinity for the entire state heuristic.
                 return float('inf')

            # Add the minimum cost for this child to the total heuristic value
            heuristic_value += min_cost_for_child

        # Return the total estimated cost for all unserved children
        return heuristic_value
