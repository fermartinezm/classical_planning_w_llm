import sys
from fnmatch import fnmatch
# Assuming the planner infrastructure provides the Heuristic base class
# If running standalone, you might need to adjust imports or provide a dummy base class
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Dummy base class for standalone testing
    class Heuristic:
        def __init__(self, task): pass
        def __call__(self, node): raise NotImplementedError

# --- Utility Functions ---

def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Example: "(at tray1 kitchen)" -> ["at", "tray1", "kitchen"]

    Args:
        fact (str): A PDDL fact string (e.g., "(predicate arg1 arg2)").

    Returns:
        list: A list containing the predicate name and its arguments.
              Returns an empty list if the fact format is invalid.
    """
    if fact.startswith("(") and fact.endswith(")"):
        # Remove parentheses and split by space
        return fact[1:-1].split()
    return [] # Return empty list for invalid format

def match(fact, *pattern):
    """
    Checks if a fact string matches a pattern.
    Supports '*' wildcard in the pattern elements.

    Args:
        fact (str): The PDDL fact string to check.
        *pattern: A sequence of strings representing the pattern to match against
                  the fact's predicate and arguments.

    Returns:
        bool: True if the fact matches the pattern, False otherwise.
    """
    parts = get_parts(fact)
    # Check if the number of parts in the fact matches the pattern length
    if len(parts) != len(pattern):
        return False
    # Compare each part with the corresponding pattern element using fnmatch
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))

# --- Heuristic Class ---

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain 'childsnacks'.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state,
    where the goal is typically to have served a set of children. It calculates
    the cost by summing the estimated minimum actions needed for each unserved child
    individually. It considers the location of the child, their potential gluten allergy,
    and the availability and location of sandwiches, ingredients, and trays in the
    current state. It aims for informativeness and computational efficiency, suitable
    for guiding a greedy best-first search.

    # Assumptions
    - The goal consists solely of `(served ?child)` predicates.
    - Every child mentioned in the goal has a defined `(waiting ?child ?place)` predicate and
      either `(allergic_gluten ?child)` or `(not_allergic_gluten ?child)` in the static facts.
    - The constant `kitchen` represents the place where ingredients are and sandwiches are made.
    - Objects involved in the `(at ?obj ?loc)` predicate where `?loc` is a known place (derived
      from `waiting` facts or the `kitchen` constant) are assumed to be trays. This relies
      on the domain structure where only trays move between these places.
    - The heuristic does not perform detailed resource tracking (e.g., consuming items mentally
      when calculating cost for one child before calculating for the next) or mutex reasoning.
      It checks for the existence of required items/states for each child independently based
      on the current state. This simplification keeps the heuristic fast but means it might
      underestimate the cost in resource-constrained scenarios where multiple children compete
      for the same item.

    # Heuristic Initialization
    - Stores the goal predicates provided by the task.
    - Parses static facts (`task.static`) to build efficient lookup structures:
        - `waiting_location[child] -> place`: Maps each child to their waiting location.
        - `is_allergic[child] -> bool`: Maps each child to their gluten allergy status.
        - `no_gluten_bread`: A set of bread portions that are gluten-free.
        - `no_gluten_content`: A set of content portions that are gluten-free.
        - `places`: A set of all known place names (including the kitchen).
    - Identifies the set of children that need to be served according to the goal predicates.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Goal Check**: If the current state (`node.state`) already satisfies all goal predicates (`self.goals`), the heuristic value is 0.
    2.  **Identify Unserved Children**: Determine the set of children required in the goal (`self.goal_children`) that do not have a `(served child)` fact in the current state.
    3.  **State Analysis**: Parse the current state (`node.state`) to gather dynamic information:
        - Location of each tray (`tray_locations`).
        - Sandwiches currently in the kitchen (`kitchen_sandwiches`), noting their gluten status.
        - Sandwiches currently on trays (`ontray_sandwiches`), noting their gluten status and the tray they are on.
        - Available bread portions in the kitchen (`kitchen_bread_avail`), noting their gluten status based on static facts.
        - Available content portions in the kitchen (`kitchen_content_avail`), noting their gluten status based on static facts.
        - Available sandwich identifiers (unused sandwich objects marked by `notexist`).
    4.  **Initialize Cost**: Set the total heuristic cost `H = 0`.
    5.  **Iterate Over Unserved Children**: For each unserved child `c`:
        a.  Retrieve the child's required waiting place `p` and gluten requirement (`req_no_gluten`) from the pre-processed static information.
        b.  Estimate the minimum cost `cost_c` to serve this child by evaluating the following options sequentially and taking the minimum cost found:
            i.  **Serve from tray at `p` (Cost=1)**: Check if a suitable sandwich (matching `req_no_gluten`) exists on any tray currently located at `p`.
            ii. **Serve from tray elsewhere (Cost=2)**: Check if a suitable sandwich exists on any tray located at a place different from `p`. (Cost = 1 move + 1 serve).
            iii.**Serve using kitchen sandwich (Cost=3 or 4)**: Check if a suitable sandwich exists in the kitchen.
                - If a tray is also at the kitchen: Cost = 3 (1 put + 1 move + 1 serve).
                - If no tray is at the kitchen, but trays exist elsewhere: Cost = 4 (1 move tray to kitchen + 1 put + 1 move + 1 serve).
            iv. **Make new sandwich (Cost=4 or 5)**: Check if suitable ingredients (bread, content matching `req_no_gluten`) and an unused sandwich identifier (`notexist`) are available.
                - If a tray is at the kitchen: Cost = 4 (1 make + 1 put + 1 move + 1 serve).
                - If no tray is at the kitchen, but trays exist elsewhere: Cost = 5 (1 make + 1 move tray to kitchen + 1 put + 1 move + 1 serve).
        c.  **Accumulate Cost**: If a possible sequence of actions (cost < infinity) was found for the child, add the minimum estimated cost `cost_c` to the total heuristic cost `H`.
        d.  **Handle Unreachable**: If no sequence of actions could be found for the child (e.g., required ingredients or trays are missing), add a fixed penalty value (e.g., 10) to `H`. This signifies that achieving this sub-goal is currently difficult or impossible from this state, discouraging the search from exploring this path deeply if alternatives exist.
    6.  **Return Total Cost**: Return the final accumulated value `H`.
    """

    def __init__(self, task):
        """Initializes the heuristic by processing static information from the task."""
        super().__init__(task) # Ensure base class initialization if needed
        self.goals = task.goals
        self.static_facts = task.static
        # Assuming 'kitchen' is the standard name for the kitchen place constant
        self.kitchen = "kitchen"

        # Pre-process static facts for efficient lookups during heuristic evaluation
        self.waiting_location = {} # child -> place
        self.is_allergic = {}      # child -> bool
        self.no_gluten_bread = set()
        self.no_gluten_content = set()
        self.children = set()
        self.places = set() # Populated from waiting facts and includes kitchen

        for fact in self.static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "waiting" and len(parts) == 3:
                # waiting ?c - child ?p - place
                child, place = parts[1], parts[2]
                self.waiting_location[child] = place
                self.children.add(child)
                self.places.add(place)
            elif predicate == "allergic_gluten" and len(parts) == 2:
                # allergic_gluten ?c - child
                child = parts[1]
                self.is_allergic[child] = True
                self.children.add(child)
            elif predicate == "not_allergic_gluten" and len(parts) == 2:
                # not_allergic_gluten ?c - child
                child = parts[1]
                self.is_allergic[child] = False
                self.children.add(child)
            elif predicate == "no_gluten_bread" and len(parts) == 2:
                # no_gluten_bread ?b - bread-portion
                self.no_gluten_bread.add(parts[1])
            elif predicate == "no_gluten_content" and len(parts) == 2:
                # no_gluten_content ?c - content-portion
                self.no_gluten_content.add(parts[1])
            # Other static facts like types (e.g., (child c1)) are ignored here

        # Ensure the kitchen is included in the set of known places
        self.places.add(self.kitchen)

        # Store the set of children that need to be served according to the goal
        self.goal_children = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}


    def __call__(self, node):
        """
        Estimates the cost to reach the goal state from the given node's state.
        Uses the pre-processed static information and analyzes the current dynamic state.
        """
        state = node.state

        # --- Goal Check ---
        # If all goal predicates are present in the current state, the cost is 0.
        if self.goals <= state:
            return 0

        # --- State Analysis ---
        # Identify children already served in the current state
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        # Determine which goal children are still unserved
        unserved_children = self.goal_children - served_children

        # Fallback check: If no children are left to serve, but the goal isn't met
        if not unserved_children and not self.goals <= state:
             # This might happen if goals include non-served predicates,
             # or if a child was served but somehow became unserved (unlikely in this domain).
             # Return a small non-zero cost to indicate goal not fully met.
             return 1

        # Analyze dynamic facts in the current state
        kitchen_sandwiches = {} # s -> is_no_gluten (bool)
        ontray_sandwiches = {}  # s -> (tray, is_no_gluten)
        tray_locations = {}     # t -> place
        kitchen_bread_avail = {} # b -> is_no_gluten (bool)
        kitchen_content_avail = {} # c -> is_no_gluten (bool)
        notexist_sandwiches = set() # Set of available sandwich names

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]

            if predicate == "at_kitchen_sandwich" and len(parts) == 2:
                s = parts[1]
                # Check if the sandwich is marked as no_gluten in the state
                is_ng = f"(no_gluten_sandwich {s})" in state
                kitchen_sandwiches[s] = is_ng
            elif predicate == "ontray" and len(parts) == 3:
                s, t = parts[1], parts[2]
                is_ng = f"(no_gluten_sandwich {s})" in state
                ontray_sandwiches[s] = (t, is_ng)
            elif predicate == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # Assumption: If an object is 'at' a known place, it's a tray.
                if loc in self.places:
                    tray_locations[obj] = loc
            elif predicate == "at_kitchen_bread" and len(parts) == 2:
                b = parts[1]
                # Check static facts for gluten status
                kitchen_bread_avail[b] = (b in self.no_gluten_bread)
            elif predicate == "at_kitchen_content" and len(parts) == 2:
                c = parts[1]
                # Check static facts for gluten status
                kitchen_content_avail[c] = (c in self.no_gluten_content)
            elif predicate == "notexist" and len(parts) == 2:
                notexist_sandwiches.add(parts[1])

        # Identify trays currently at the kitchen vs elsewhere
        trays_at_kitchen = {t for t, loc in tray_locations.items() if loc == self.kitchen}
        trays_elsewhere = set(tray_locations.keys()) - trays_at_kitchen
        any_trays_exist = bool(trays_at_kitchen or trays_elsewhere)

        # --- Heuristic Calculation ---
        total_heuristic_cost = 0

        # Iterate through each unserved child and estimate the cost to serve them
        for child in unserved_children:
            # Retrieve child-specific requirements from pre-processed static info
            req_no_gluten = self.is_allergic.get(child)
            req_place = self.waiting_location.get(child)

            # Handle potential errors if static info is missing (should not happen in valid PDDL tasks)
            if req_no_gluten is None or req_place is None:
                # This indicates an issue with the PDDL problem definition or static fact parsing
                total_heuristic_cost += 100 # Assign a large penalty
                continue

            # Helper function to check if a resource (sandwich, ingredient) is suitable
            # based on the child's gluten requirement.
            def is_suitable(resource_is_no_gluten):
                # If child requires no_gluten, resource must be no_gluten.
                # If child does not require no_gluten, any resource is suitable.
                return resource_is_no_gluten or not req_no_gluten

            cost_for_child = float('inf') # Initialize cost for this child to infinity

            # --- Evaluate Options (find the minimum cost) ---

            # Option 1: Serve from tray already at the child's location (Cost: 1)
            # Check only if cost_for_child is currently > 1
            if cost_for_child > 1:
                for s, (t, is_ng) in ontray_sandwiches.items():
                    # Check if tray 't' is at the required place and sandwich 's' is suitable
                    if tray_locations.get(t) == req_place and is_suitable(is_ng):
                        cost_for_child = 1 # Found the cheapest possible option
                        break # Exit loop for Option 1

            # Option 2: Serve from tray at a different location (Cost: 2)
            # Check only if cost_for_child is currently > 2
            if cost_for_child > 2:
                for s, (t, is_ng) in ontray_sandwiches.items():
                     tray_loc = tray_locations.get(t)
                     # Check if tray 't' is at a known, different location and sandwich 's' is suitable
                     if tray_loc is not None and tray_loc != req_place and is_suitable(is_ng):
                         cost_for_child = min(cost_for_child, 2) # 1 move_tray + 1 serve
                         # Don't break here, continue checking all trays elsewhere
                         # as we only want the minimum cost (which is 2 for this category)

            # Option 3: Serve using a sandwich from the kitchen (Cost: 3 or 4)
            # Check only if cost_for_child is currently > 3 (or > 4 if no trays at kitchen)
            if cost_for_child > 3:
                for s, is_ng in kitchen_sandwiches.items():
                     if is_suitable(is_ng):
                         # Check tray availability
                         if trays_at_kitchen:
                             cost_for_child = min(cost_for_child, 3) # 1 put + 1 move + 1 serve
                         elif trays_elsewhere:
                             # Assumes a tray from elsewhere can be moved to kitchen first
                             cost_for_child = min(cost_for_child, 4) # 1 move_to_kitchen + 1 put + 1 move + 1 serve
                         # If no trays exist at all, cost remains inf for this path

            # Option 4: Make a new sandwich (Cost: 4 or 5)
            # Check only if cost_for_child is currently > 4 (or > 5 if no trays at kitchen)
            if cost_for_child > 4:
                # Check if there's an available sandwich ID
                if notexist_sandwiches:
                    # Check if suitable ingredients are available in the kitchen
                    has_suitable_bread = any(is_suitable(is_ng_b) for b, is_ng_b in kitchen_bread_avail.items())
                    has_suitable_content = any(is_suitable(is_ng_c) for c, is_ng_c in kitchen_content_avail.items())

                    if has_suitable_bread and has_suitable_content:
                        # Check tray availability
                        if trays_at_kitchen:
                            cost_for_child = min(cost_for_child, 4) # 1 make + 1 put + 1 move + 1 serve
                        elif trays_elsewhere:
                            cost_for_child = min(cost_for_child, 5) # 1 make + 1 move_to_kitchen + 1 put + 1 move + 1 serve
                        # If no trays exist, cost remains inf for this path

            # --- Accumulate Cost for this Child ---
            # If cost_for_child is still infinity after checking all options,
            # it means this child cannot be served from the current state with available resources.
            if cost_for_child == float('inf'):
                 # Add a penalty to the heuristic value.
                 total_heuristic_cost += 10 # Arbitrary penalty value
            else:
                 # Add the minimum cost found for serving this child
                 total_heuristic_cost += cost_for_child

        return total_heuristic_cost
