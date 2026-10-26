import sys
import os
from fnmatch import fnmatch

# Ensure the heuristic base class can be imported.
# Assumes the script is placed in a directory where 'heuristics.heuristic_base' is accessible.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Fallback if the heuristic_base is in the same directory or PYTHONPATH is set
    try:
        from heuristic_base import Heuristic
    except ImportError:
        raise ImportError("Could not import Heuristic base class. Please ensure it's accessible via PYTHONPATH or placed correctly.")


# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handles facts like "(predicate)" or "(predicate obj1 obj2)"
    content = fact[1:-1].strip()
    if not content:
        return []
    return content.split()

def match(fact_parts, *pattern_args):
    """
    Check if a list of fact parts matches a given pattern.

    Args:
        fact_parts: A list of strings representing the parts of the fact (predicate + arguments).
        *pattern_args: The pattern parts to match against (strings, allowing '*' wildcard).

    Returns:
        True if the parts match the pattern, False otherwise.
    """
    if len(fact_parts) != len(pattern_args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(fact_parts, pattern_args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Childsnack domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children
    specified in the goal `(served ?c)`. It calculates the minimum estimated steps needed for
    each unserved child individually and sums these estimates. The estimate for
    a single child considers the steps required to find or make a suitable sandwich,
    put it on a tray (if necessary), move the tray to the child's location (if necessary),
    and serve the sandwich. It prioritizes using existing sandwiches based on their location.

    # Assumptions
    - The heuristic assumes that resources like ingredients and trays are sufficiently
      available or can be made available with minimal effort (e.g., moving a tray).
      It does not perform detailed checks on ingredient counts or tray contention.
    - It does not account for potential resource conflicts (e.g., needing the same
      tray or ingredient for multiple children simultaneously).
    - It assumes the shortest path to serve each child independently and sums these costs.
      This might overestimate if actions can be shared (e.g., moving one tray serves
      multiple children at the same location), but is suitable for greedy search guidance.
    - The cost of moving a tray between any two locations is assumed to be 1 action.
    - The cost of moving a tray to the kitchen (if needed to pick up a sandwich)
      is implicitly included in the steps estimated when a sandwich is taken from the kitchen.

    # Heuristic Initialization
    - Extracts the set of children that need to be served from the goal conditions `(served ?c)`.
    - Stores static information about children's allergies (`allergic_gluten`, `not_allergic_gluten`)
      and their waiting locations (`waiting ?c ?p`) using dictionaries for quick lookup during evaluation.
    - Verifies that all goal children have the necessary static information defined.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic cost `h` to 0.
    2. Pre-process the current `state` (a frozenset of fact strings) to efficiently access:
        - Set of children already served `(served ?c)`.
        - Set of sandwiches currently at the kitchen `(at_kitchen_sandwich ?s)`.
        - Dictionary mapping sandwiches on trays to the tray they are on `(ontray ?s ?t)`.
        - Dictionary mapping trays to their current location `(at ?t ?p)`.
        - Set of sandwiches that are marked as gluten-free `(no_gluten_sandwich ?s)`.
    3. Identify all children `c` that are part of the goal `(served c)` but are not yet in the `served_children` set.
    4. For each such unserved child `c`:
        a. Retrieve the child's allergy status (`is_allergic`) and waiting location (`child_loc`) from the pre-calculated static info.
        b. Estimate the minimum cost (`min_cost_for_child`) to serve this child, starting with a default maximum cost of 4 (representing make + put + move + serve).
        c. Check the following conditions in order of preference (cheapest first), updating `min_cost_for_child` and stopping the check for this child as soon as the cheapest path is found:
            i. **Case 1: Suitable sandwich on a tray at the child's location `p`:** Cost = 1 (serve).
               - Iterate through sandwiches `s` currently on trays `t` (`sandwiches_on_tray`).
               - Check if the tray `t` is at the child's location (`tray_locations.get(t) == child_loc`).
               - Check if the sandwich `s` is suitable (gluten-free status matches allergy requirement).
               - If found, set `min_cost_for_child = 1` and proceed to the next child.
            ii. **Case 2: Suitable sandwich on a tray `t'` at any other location `p'`:** Cost = 2 (move + serve).
               - If `min_cost_for_child` is still > 2, iterate through sandwiches `s` on trays `t'`.
               - Check if the tray `t'` has a known location (`tray_loc = tray_locations.get(t')`) which is *not* the child's location.
               - Check if the sandwich `s` is suitable.
               - If found, update `min_cost_for_child = 2` and proceed to the next child.
            iii. **Case 3: Suitable sandwich `s` at the kitchen:** Cost = 3 (put + move + serve).
               - If `min_cost_for_child` is still > 3, iterate through sandwiches `s` at the kitchen (`sandwiches_at_kitchen`).
               - Check if the sandwich `s` is suitable.
               - If found, update `min_cost_for_child = 3` and proceed to the next child.
            iv. **Case 4: No suitable sandwich exists readily available:** Cost = 4 (make + put + move + serve).
               - If none of the above cases were met, the cost remains the default value of 4.
        d. Add the final `min_cost_for_child` for this child to the total heuristic cost `h`.
    5. Return the total cost `h`. If `h` is 0, it implies all goal children are served (goal state).
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static information from the task.
        Args:
            task: The planning task object containing static facts, goals, etc.
        """
        super().__init__(task) # Call parent constructor if it exists and is needed
        self.goals = task.goals
        static_facts = task.static

        # --- Extract Goal Children ---
        self.goal_children = set()
        for goal in self.goals:
            parts = get_parts(goal)
            # Ensure the fact is '(served <child>)'
            if len(parts) == 2 and parts[0] == "served":
                self.goal_children.add(parts[1])

        # --- Extract Static Information ---
        self.child_allergies = {} # child -> bool (True if allergic)
        self.child_locations = {} # child -> place
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or invalid facts

            predicate = parts[0]
            # Use direct predicate comparison and check arity for robustness
            if predicate == "allergic_gluten" and len(parts) == 2:
                self.child_allergies[parts[1]] = True
            elif predicate == "not_allergic_gluten" and len(parts) == 2:
                 self.child_allergies[parts[1]] = False
            elif predicate == "waiting" and len(parts) == 3:
                 self.child_locations[parts[1]] = parts[2]

        # --- Validate Static Info for Goal Children ---
        for child in self.goal_children:
            if child not in self.child_allergies:
                raise ValueError(f"Allergy status (allergic_gluten/not_allergic_gluten) for goal child '{child}' not found in static facts.")
            if child not in self.child_locations:
                raise ValueError(f"Waiting location (waiting) for goal child '{child}' not found in static facts.")


    def __call__(self, node):
        """
        Calculates the heuristic value for a given state node.
        Args:
            node: The node in the search space containing the state.
        Returns:
            An integer estimate of the cost to reach the goal.
        """
        state = node.state
        h_value = 0

        # --- Pre-process state for faster lookups ---
        served_children = set()
        sandwiches_at_kitchen = set() # set of sandwiches at kitchen
        sandwiches_on_tray = {} # sandwich -> tray
        tray_locations = {} # tray -> location
        sandwiches_gluten_free = set() # set of sandwiches that are gluten free

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            # Use direct predicate comparison and check arity
            if predicate == "served" and len(parts) == 2:
                served_children.add(parts[1])
            elif predicate == "at_kitchen_sandwich" and len(parts) == 2:
                sandwiches_at_kitchen.add(parts[1])
            elif predicate == "ontray" and len(parts) == 3:
                sandwiches_on_tray[parts[1]] = parts[2] # sandwich -> tray
            elif predicate == "at" and len(parts) == 3:
                obj = parts[1]
                loc = parts[2]
                # Simple check based on common naming convention for trays
                # Assumes trays are named starting with 'tray'
                if obj.startswith("tray"):
                     tray_locations[obj] = loc
            elif predicate == "no_gluten_sandwich" and len(parts) == 2:
                 sandwiches_gluten_free.add(parts[1])
        # --- End Pre-processing ---


        # Calculate cost for each unserved goal child
        for child in self.goal_children:
            if child in served_children:
                continue # Already served

            # Retrieve child-specific info (guaranteed to exist by __init__)
            child_loc = self.child_locations[child]
            is_allergic = self.child_allergies[child]

            # Determine the minimum cost to serve this child
            min_cost_for_child = 4 # Default cost: make(1) + put(1) + move(1) + serve(1)

            # Check Case 1: Suitable sandwich on tray at child's location
            for sandwich, tray in sandwiches_on_tray.items():
                is_gf = sandwich in sandwiches_gluten_free
                if tray_locations.get(tray) == child_loc: # Check tray is at correct location
                    # Check if sandwich is suitable for the child's allergy
                    if (is_allergic and is_gf) or (not is_allergic):
                        min_cost_for_child = 1 # Cost: serve(1)
                        break # Found the best case (cost 1)
            if min_cost_for_child == 1:
                h_value += min_cost_for_child
                continue # Move to the next child

            # Check Case 2: Suitable sandwich on tray at different location
            if min_cost_for_child > 2: # Only check if cost can be improved to 2
                for sandwich, tray in sandwiches_on_tray.items():
                    is_gf = sandwich in sandwiches_gluten_free
                    tray_loc = tray_locations.get(tray)
                    # Check tray is at a known location, but not the child's location
                    if tray_loc is not None and tray_loc != child_loc:
                        # Check if sandwich is suitable
                        if (is_allergic and is_gf) or (not is_allergic):
                            min_cost_for_child = 2 # Cost: move(1) + serve(1)
                            break # Found cost 2
            if min_cost_for_child == 2:
                 h_value += min_cost_for_child
                 continue # Move to the next child

            # Check Case 3: Suitable sandwich at kitchen
            if min_cost_for_child > 3: # Only check if cost can be improved to 3
                for sandwich in sandwiches_at_kitchen:
                    is_gf = sandwich in sandwiches_gluten_free
                    # Check if sandwich is suitable
                    if (is_allergic and is_gf) or (not is_allergic):
                        min_cost_for_child = 3 # Cost: put(1) + move(1) + serve(1)
                        break # Found cost 3
            # No need for 'if min_cost_for_child == 3: continue' as this is the last check before adding

            # Add the determined minimum cost for this child (1, 2, 3, or 4)
            h_value += min_cost_for_child

        # The heuristic value is 0 only if all goal children are served.
        return h_value
