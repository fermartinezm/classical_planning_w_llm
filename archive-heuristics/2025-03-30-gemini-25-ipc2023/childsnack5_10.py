import math
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(at tray1 kitchen)" -> ["at", "tray1", "kitchen"]
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern using fnmatch for wildcard support.
    - `fact`: The complete fact string, e.g., "(at tray1 kitchen)".
    - `args`: A list or tuple representing the pattern, e.g., ["at", "tray*", "kitchen"].
    - Returns `True` if the fact's parts match the pattern, `False` otherwise.
    """
    parts = get_parts(fact)
    # Ensure the number of parts in the fact matches the pattern length
    if len(parts) != len(args):
        return False
    # Check each part against the corresponding pattern argument using fnmatch
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class ChildsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain 'childsnacks'.

    # Summary
    Estimates the cost to reach the goal state in the Childsnack domain by summing
    the estimated minimum actions required for serving each currently unserved child
    independently. The goal is assumed to be serving a specific set of children.
    This heuristic is designed for Greedy Best-First Search and does not need to be admissible.

    # Assumptions
    - Assumes sufficient bread and content portions are always available in the kitchen
      when a 'make_sandwich' action is required. It does not model ingredient inventory levels.
    - Assumes sandwiches, once made or placed on a tray, can be used to satisfy
      the need of any suitable child for the purpose of cost estimation. It does not
      model resource contention (e.g., one sandwich cannot serve two children) or
      exclusive reservation of sandwiches/trays during the estimation.
    - Estimates tray movement costs simply: 1 action ('move_tray') is needed if the
      tray holding the required sandwich is not at the child's waiting location.
      Moving a tray to the kitchen (if needed for 'put_on_tray') is implicitly
      handled within the cost estimates of later steps, assuming it takes at most 1 action.
    - Assumes the existence of at least one tray in the problem instance.
    - Assumes object names follow conventions (e.g., trays start with 'tray') for internal parsing,
      although this is only used for identifying tray locations.

    # Heuristic Initialization
    - Parses the static facts provided in the task (`task.static`) to build a dictionary (`child_info`)
      mapping each child object to their allergy status (`allergic`: True/False) and
      their waiting location (`place`: string, e.g., 'table1').
    - Identifies the set of children that need to be served to satisfy the goal conditions
      (`task.goals`) by extracting children `c` from `(served c)` goal predicates.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value (`h`) is computed by iterating through all children required
    to be served according to the goal state (`self.goal_children`). For each child
    not yet served in the current state (`node.state`):
    1. Retrieve the child's allergy status (`needs_gf`) and waiting place (`wait_place`)
       from the precomputed `child_info`.
    2. Determine the minimum estimated actions (`cost_c`) needed to serve this child
       by checking the current state for the most advanced situation possible, evaluated
       sequentially. The first scenario that matches determines the cost for this child:
        a. **Scenario 1: Suitable Sandwich Ready on Tray at Location.**
           - Check if there exists a sandwich `s` in the state such that:
             - `s` is suitable: `(no_gluten_sandwich s)` is true if `needs_gf` is true.
             - `s` is on a tray `t`: `(ontray s t)` is true.
             - Tray `t` is at the child's waiting place `p`: `(at t p)` is true.
           - If yes, `cost_c = 1` (only the 'serve' action is needed).
        b. **Scenario 2: Suitable Sandwich Ready on Tray Elsewhere.**
           - If Scenario 1 is not met, check if there exists a suitable sandwich `s` on a tray `t` (`(ontray s t)`), but the tray is at a different location `loc` (`(at t loc)`, `loc != p`).
           - If yes, `cost_c = 2` (one 'move_tray' action + one 'serve' action).
        c. **Scenario 3: Suitable Sandwich Available at Kitchen.**
           - If Scenarios 1 and 2 are not met, check if there exists a suitable sandwich `s` at the kitchen (`(at_kitchen_sandwich s)`).
           - If yes, the estimated sequence needed is: 'put_on_tray', 'move_tray' (if `p` is not kitchen), 'serve'.
           - `cost_c = 1 (put) + (0 if p == kitchen else 1) (move) + 1 (serve)`. This results in `cost_c = 2` (if waiting at kitchen) or `3` (if waiting elsewhere).
        d. **Scenario 4: Need to Make a New Sandwich.**
           - If none of the above scenarios are met, assume a new sandwich must be made.
           - The estimated sequence needed is: 'make_sandwich', 'put_on_tray', 'move_tray' (if `p` is not kitchen), 'serve'.
           - `cost_c = 1 (make) + 1 (put) + (0 if p == kitchen else 1) (move) + 1 (serve)`. This results in `cost_c = 3` (if waiting at kitchen) or `4` (if waiting elsewhere).
    3. Add the calculated `cost_c` for the current child to the total heuristic value `h`.
    4. After checking all goal children, the final `h` value is returned.
    5. **Goal State Handling:** If the set of unserved goal children is empty, the state is a goal state, and the heuristic returns 0.
    6. **Non-Goal State with h=0:** If the goal is not met but the calculated `h` is 0 (an unlikely edge case, possibly due to missing static info for a goal child), the heuristic returns 1 to ensure the search makes progress.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing static facts and goal conditions from the task.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.child_info = {} # Stores {child: {'allergic': bool, 'place': place}}
        self.goal_children = set() # Stores children that need to be served

        # Parse static facts to get child properties (allergy, waiting place)
        for fact in self.static_facts:
            parts = get_parts(fact)
            predicate = parts[0]
            # Ensure facts have the expected structure before accessing parts
            if predicate in ["allergic_gluten", "not_allergic_gluten"] and len(parts) == 2:
                child = parts[1]
                if child not in self.child_info:
                    self.child_info[child] = {'allergic': False, 'place': None} # Initialize if first time seeing child
                self.child_info[child]['allergic'] = (predicate == "allergic_gluten")
            elif predicate == "waiting" and len(parts) == 3:
                child, place = parts[1], parts[2]
                if child not in self.child_info:
                    self.child_info[child] = {'allergic': False, 'place': None} # Initialize if first time seeing child
                self.child_info[child]['place'] = place

        # Parse goals to find which children need serving
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == "served" and len(parts) == 2:
                 self.goal_children.add(parts[1])

        # Optional: Validate that all goal children have complete info loaded from static facts
        for child in self.goal_children:
            if child not in self.child_info or self.child_info[child]['place'] is None:
                print(f"Warning: Child {child} required in goal but missing static info (allergy/waiting place). Heuristic might be inaccurate.")


    def _is_sandwich_gluten_free(self, sandwich, state):
        """Checks if the '(no_gluten_sandwich sandwich)' predicate is true in the state."""
        # Construct the fact string to check for existence in the state set
        return f"(no_gluten_sandwich {sandwich})" in state

    def __call__(self, node):
        """
        Calculates the heuristic estimate (estimated number of actions to goal)
        for the given state node.
        """
        state = node.state
        h_value = 0

        # --- Parse current state efficiently ---
        # Sets and dictionaries to store relevant information from the current state
        served_children_in_state = set()
        trays_locations = {} # {tray_name: location_name}
        sandwiches_info = {} # {sandwich_name: {'loc_type': 'kitchen'/'tray', 'tray': t/None, 'loc': place/None, 'is_gf': bool}}
        sandwiches_on_tray_map = {} # Temporary map {sandwich_name: tray_name}

        # First pass: Identify tray locations, sandwiches on trays, and served children
        for fact in state:
            parts = get_parts(fact)
            # Skip empty or malformed facts
            if not parts:
                continue
            predicate = parts[0]

            # Store tray locations
            if predicate == "at" and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 # Basic check assuming trays are named starting with 'tray'.
                 # This might need adjustment if naming conventions differ.
                 if obj.startswith("tray"):
                     trays_locations[obj] = loc
            # Store which sandwich is on which tray
            elif predicate == "ontray" and len(parts) == 3:
                s, t = parts[1], parts[2]
                sandwiches_on_tray_map[s] = t
            # Store which children are already served
            elif predicate == "served" and len(parts) == 2:
                served_children_in_state.add(parts[1])

        # Second pass: Populate sandwiches_info using tray locations and check GF status
        # Iterate through the state again to find sandwiches at kitchen and determine GF status
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            predicate = parts[0]

            # Check for sandwiches at the kitchen
            if predicate == "at_kitchen_sandwich" and len(parts) == 2:
                s = parts[1]
                # Add info only if not already processed as being on a tray (consistency check)
                if s not in sandwiches_on_tray_map:
                    is_gf = self._is_sandwich_gluten_free(s, state)
                    sandwiches_info[s] = {'loc_type': 'kitchen', 'tray': None, 'loc': 'kitchen', 'is_gf': is_gf}

        # Now, add info for sandwiches identified as being on a tray
        for s, t in sandwiches_on_tray_map.items():
             is_gf = self._is_sandwich_gluten_free(s, state)
             tray_loc = trays_locations.get(t) # Get the location of the tray it's on
             if tray_loc is not None:
                  # Store detailed info about the sandwich on the tray
                  sandwiches_info[s] = {'loc_type': 'tray', 'tray': t, 'loc': tray_loc, 'is_gf': is_gf}
             else:
                  # This indicates an inconsistent state (e.g., tray exists but has no 'at' predicate)
                  # Log a warning or skip this sandwich for heuristic calculation.
                  # print(f"Warning: Tray {t} for sandwich {s} has no location in state.")
                  pass


        # --- Calculate cost per unserved child ---
        # Determine which children required by the goal are not yet served
        unserved_goal_children = self.goal_children - served_children_in_state

        # Calculate the estimated cost for each unserved child
        for child in unserved_goal_children:
            # Ensure child info exists before proceeding; skip if static info was missing
            if child not in self.child_info or self.child_info[child]['place'] is None:
                h_value += 4 # Add a default high cost if info is missing
                continue

            child_needs_gf = self.child_info[child]['allergic']
            child_wait_place = self.child_info[child]['place']
            cost_c = math.inf # Initialize cost for this child to infinity

            # Check scenarios sequentially to find the minimum cost path estimate

            # Scenario 1: Suitable Sandwich Ready on Tray at Location.
            found1 = any(
                info['loc_type'] == 'tray' and                     # Is it on a tray?
                info['loc'] == child_wait_place and                # Is the tray at the right place?
                (info['is_gf'] if child_needs_gf else True)        # Is the sandwich suitable (GF if needed)?
                for info in sandwiches_info.values()
            )
            if found1:
                cost_c = 1 # 1 action: serve
            else:
                # Scenario 2: Suitable Sandwich Ready on Tray Elsewhere.
                found2 = any(
                    info['loc_type'] == 'tray' and                     # Is it on a tray?
                    info['loc'] != child_wait_place and                # Is the tray elsewhere?
                    (info['is_gf'] if child_needs_gf else True)        # Is the sandwich suitable?
                    for info in sandwiches_info.values()
                )
                if found2:
                    cost_c = 2 # 2 actions: move_tray + serve
                else:
                    # Scenario 3: Suitable Sandwich Available at Kitchen.
                    found3 = any(
                        info['loc_type'] == 'kitchen' and                  # Is it at the kitchen?
                        (info['is_gf'] if child_needs_gf else True)        # Is the sandwich suitable?
                        for info in sandwiches_info.values()
                    )
                    if found3:
                        # Estimate: put_on_tray + move_tray (if needed) + serve
                        move_cost = 0 if child_wait_place == 'kitchen' else 1
                        cost_c = 1 + move_cost + 1 # Total 2 or 3 actions
                    else:
                        # Scenario 4: Need to Make a New Sandwich (Fallback).
                        # Assumes ingredients are available.
                        # Estimate: make_sandwich + put_on_tray + move_tray (if needed) + serve
                        move_cost = 0 if child_wait_place == 'kitchen' else 1
                        cost_c = 1 + 1 + move_cost + 1 # Total 3 or 4 actions

            # Add the determined cost for this child to the total heuristic value
            # If cost_c remained infinity (shouldn't happen with fallback), add a default cost.
            h_value += cost_c if cost_c != math.inf else 4


        # --- Final adjustments ---
        # If there are no unserved children, the goal is met.
        if not unserved_goal_children:
             return 0
        # Ensure the heuristic value is at least 1 if the goal state has not been reached.
        # This prevents the search from terminating prematurely if h_value somehow calculated to 0.
        elif h_value == 0:
             return 1
        else:
             # Return the calculated sum of costs
             return h_value
