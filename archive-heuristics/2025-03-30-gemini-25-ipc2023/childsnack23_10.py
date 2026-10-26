import os
import sys
from fnmatch import fnmatch

# Ensure the heuristics directory is in the Python path if needed.
# If the script calling this heuristic is in the parent directory of 'heuristics',
# this might not be necessary. Adjust if your structure differs.
# Example:
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)

try:
    # Assumes heuristic_base is in a 'heuristics' directory accessible via Python path
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Provide a more informative error message if the import fails.
    raise ImportError(
        "Could not import Heuristic base class from heuristics.heuristic_base. "
        "Ensure the 'heuristics' directory is in the Python path or structured correctly "
        "relative to the execution script."
    )


def get_parts(fact):
    """
    Extract the components of a PDDL fact string.

    Args:
        fact (str): A PDDL fact string, e.g., "(predicate obj1 obj2)".

    Returns:
        list: A list of strings, e.g., ["predicate", "obj1", "obj2"],
              or an empty list if the fact is malformed.
    """
    # Basic check for parentheses and non-empty content
    if not fact or len(fact) < 3 or fact[0] != '(' or fact[-1] != ')':
        # Return empty list for malformed facts
        return []
    # Split the content within the parentheses
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern using fnmatch for wildcards.

    Args:
        fact (str): The complete fact string, e.g., "(at tray1 kitchen)".
        *args: A sequence of strings representing the pattern components,
               e.g., "at", "tray*", "kitchen". Wildcards (*) are supported.

    Returns:
        bool: True if the fact's parts match the pattern, False otherwise.
    """
    parts = get_parts(fact)
    # The number of parts in the fact must match the number of pattern arguments
    if len(parts) != len(args):
        return False
    # Check each part against the corresponding pattern argument using fnmatch
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain 'childsnacks'.

    # Summary
    This heuristic estimates the minimum number of actions required to reach a
    goal state from the current state in the childsnacks domain. The goal is
    typically to have served a specific set of children. The heuristic counts
    the necessary 'make_sandwich', 'put_on_tray', 'move_tray', and 'serve'
    actions required to satisfy the unserved children goals. It is designed
    for greedy best-first search and is not necessarily admissible but aims
    to be informative and efficiently computable.

    # Assumptions
    - The primary goal is always a conjunction of `(served ?c)` predicates for one or more children `?c`.
    - Sufficient bread and content portions are assumed to be available in the kitchen whenever a 'make_sandwich' action is estimated (this is a common heuristic simplification).
    - Trays are assumed to have sufficient capacity to carry the necessary sandwiches for a delivery trip, as the domain actions don't model capacity limits explicitly for `move_tray`.
    - The heuristic prioritizes using existing sandwiches (on trays or in the kitchen) before estimating the cost of making new ones.

    # Heuristic Initialization
    - The constructor (`__init__`) processes the task's static facts (those true in all states) to build efficient lookups:
        - `child_allergy`: A dictionary mapping each child object name (str) to `True` if they are allergic to gluten, `False` otherwise.
        - `child_waiting_location`: A dictionary mapping each child object name (str) to the place object name (str) where they are waiting.
    - It also extracts the set of `goal_children` (strings) from the task's goal specification, containing the names of children who must be served.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Identify Unserved Children:** Determine the set of children specified in the goal who are not yet marked as `(served ?c)` in the current `state`. If this set is empty, the goal state is reached, and the heuristic value is 0.
    2.  **Count Needed Sandwiches by Type:** Calculate the number of unserved children who are allergic (`N_allergic_unserved`) and those who are not (`N_nonallergic_unserved`). These counts represent the minimum number of gluten-free (GF) and regular sandwiches needed, respectively.
    3.  **Assess Available Sandwiches:** Scan the current `state` to count existing sandwiches:
        - `S_gf_kitchen`, `S_reg_kitchen`: Number of GF/regular sandwiches at the kitchen.
        - `S_gf_ontray`, `S_reg_ontray`: Number of GF/regular sandwiches currently on any tray.
    4.  **Calculate `cost_serve`:** This cost is equal to the total number of unserved children (`N_unserved`), as each requires one `serve_sandwich` or `serve_sandwich_no_gluten` action.
    5.  **Calculate `cost_make`:** Estimate the number of 'make' actions needed:
        a. Calculate the shortfall of GF sandwiches: `make_gf = max(0, N_allergic_unserved - (S_gf_kitchen + S_gf_ontray))`.
        b. Calculate the shortfall of regular sandwiches needed for non-allergic children, considering that leftover GF sandwiches can be used. Available sandwiches for them = `max(0, (S_gf_kitchen + S_gf_ontray) - N_allergic_unserved) + S_reg_kitchen + S_reg_ontray`. Then, `make_reg = max(0, N_nonallergic_unserved - available_sandwiches_for_them)`.
        c. `cost_make = make_gf + make_reg`.
    6.  **Calculate `cost_put_on_tray`:** Estimate the number of 'put_on_tray' actions:
        a. Total sandwiches needed is `N_unserved`.
        b. Sandwiches already on trays is `S_gf_ontray + S_reg_ontray`.
        c. The number of sandwiches that still need to be moved from the kitchen (either existing there or newly made) onto a tray is `max(0, N_unserved - sandwiches_already_on_trays)`.
        d. `cost_put_on_tray` equals this number.
    7.  **Calculate `cost_move_tray`:** Estimate the minimum number of 'move_tray' actions:
        a. Determine if sandwiches need to be picked up from the kitchen (`needs_pickup`), which is true if `cost_put_on_tray > 0`.
        b. Check if any tray is currently located at the kitchen (`is_tray_at_kitchen`).
        c. If pickup is needed and no tray is at the kitchen, increment `cost_move_tray` by 1 (a tray must move *to* the kitchen).
        d. Identify the set of distinct locations `p` (excluding the kitchen) where unserved children are waiting (`unserved_locations`).
        e. Find the set of locations where trays are currently present (`trays_present_at`).
        f. Initialize a set `destinations_to_visit`. For each location `p` in `unserved_locations` (where `p` is not the kitchen): if `p` needs a sandwich delivered from the kitchen (approximated by `needs_pickup` being true) OR if no tray is currently at `p`, then add `p` to `destinations_to_visit`.
        g. Increment `cost_move_tray` by the number of unique locations in `destinations_to_visit`. This estimates the moves *to* the necessary waiting locations.
    8.  **Total Heuristic Value:** The final heuristic estimate is the sum of the calculated costs: `cost_serve + cost_make + cost_put_on_tray + cost_move_tray`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goals and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.child_allergy = {} # child_name (str) -> is_allergic (bool)
        self.child_waiting_location = {} # child_name (str) -> place_name (str)
        self.goal_children = set() # set of child_names (str)

        # Parse static facts to populate lookup dictionaries
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip if fact is malformed

            predicate = parts[0]
            if predicate == "allergic_gluten" and len(parts) == 2:
                self.child_allergy[parts[1]] = True
            elif predicate == "not_allergic_gluten" and len(parts) == 2:
                self.child_allergy[parts[1]] = False
            elif predicate == "waiting" and len(parts) == 3:
                # parts[1] is child, parts[2] is place
                self.child_waiting_location[parts[1]] = parts[2]
            # Add parsing for other static predicates if they become relevant

        # Parse goal facts to identify children to be served
        for goal_fact in self.goals:
             # Use match function for robustness
             if match(goal_fact, "served", "*"):
                 child = get_parts(goal_fact)[1]
                 self.goal_children.add(child)
                 # Basic validation during initialization (optional but helpful)
                 if child not in self.child_allergy:
                     print(f"Warning: Child {child} in goal but no allergy info found in static facts.")
                 if child not in self.child_waiting_location:
                     print(f"Warning: Child {child} in goal but no waiting location found in static facts.")


    def __call__(self, node):
        """
        Calculate the heuristic estimate for the given state node.

        Args:
            node: A node object containing the state (node.state).
                  The state is expected to be a frozenset of PDDL fact strings.

        Returns:
            int: The estimated cost (number of actions) to reach the goal.
        """
        state = node.state

        # 1. Identify Unserved Children
        unserved_children = {
            child for child in self.goal_children if f"(served {child})" not in state
        }

        N_unserved = len(unserved_children)
        # If no children remain unserved, the goal is reached.
        if N_unserved == 0:
            return 0

        # 2. Count Needed Sandwiches by Type based on unserved children's allergies
        N_allergic_unserved = sum(1 for c in unserved_children if self.child_allergy.get(c, False)) # Default to False if info missing
        N_nonallergic_unserved = N_unserved - N_allergic_unserved

        # 3. Assess Available Sandwiches and Tray Locations from Current State
        S_gf_kitchen = 0
        S_reg_kitchen = 0
        S_gf_ontray = 0
        S_reg_ontray = 0
        sandwiches_at_kitchen = set() # Store names of sandwiches at kitchen
        sandwiches_ontray_set = set() # Store names of sandwiches on trays
        gluten_free_sandwiches = set() # Store names of GF sandwiches
        tray_locations = {} # tray_name (str) -> place_name (str)
        current_sandwiches_on_trays_map = {} # sandwich_name (str) -> tray_name (str)

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            # Identify locations and properties of sandwiches and trays
            if predicate == "at_kitchen_sandwich" and len(parts) == 2:
                sandwiches_at_kitchen.add(parts[1])
            elif predicate == "ontray" and len(parts) == 3:
                sandwich_name, tray_name = parts[1], parts[2]
                sandwiches_ontray_set.add(sandwich_name)
                current_sandwiches_on_trays_map[sandwich_name] = tray_name
            elif predicate == "no_gluten_sandwich" and len(parts) == 2:
                gluten_free_sandwiches.add(parts[1])
            elif predicate == "at" and len(parts) == 3:
                 # Heuristically identify trays based on object name patterns
                 # Assumes trays aren't named like children, sandwiches, ingredients, or places
                 obj, loc = parts[1], parts[2]
                 known_places = set(self.child_waiting_location.values()) | {'kitchen'}
                 if not (obj.startswith('child') or obj.startswith('sandw') or \
                         obj.startswith('bread') or obj.startswith('content') or \
                         obj in known_places):
                     tray_locations[obj] = loc

        # Consolidate sandwich counts based on location and GF status
        for s in sandwiches_at_kitchen:
            if s in gluten_free_sandwiches: S_gf_kitchen += 1
            else: S_reg_kitchen += 1

        for s in sandwiches_ontray_set:
            # Check if the sandwich is currently mapped to a tray (consistency check)
            if s in current_sandwiches_on_trays_map:
                if s in gluten_free_sandwiches: S_gf_ontray += 1
                else: S_reg_ontray += 1

        # 4. Calculate cost_serve: One serve action per unserved child.
        cost_serve = N_unserved

        # 5. Calculate cost_make: Actions needed to create missing sandwiches.
        needed_gf = N_allergic_unserved
        available_gf = S_gf_kitchen + S_gf_ontray
        make_gf = max(0, needed_gf - available_gf) # GF sandwiches to make

        needed_reg = N_nonallergic_unserved
        # Calculate sandwiches available for non-allergic (leftover GF + existing regular)
        remaining_gf_available = max(0, available_gf - needed_gf)
        available_for_reg = remaining_gf_available + S_reg_kitchen + S_reg_ontray
        make_reg = max(0, needed_reg - available_for_reg) # Regular sandwiches to make

        cost_make = make_gf + make_reg

        # 6. Calculate cost_put_on_tray: Actions to move sandwiches from kitchen to tray.
        sandwiches_on_tray_total = S_gf_ontray + S_reg_ontray
        sandwiches_needed_total = N_unserved
        # Sandwiches to put = total needed - already on tray (min 0)
        sandwiches_to_put = max(0, sandwiches_needed_total - sandwiches_on_tray_total)
        cost_put_on_tray = sandwiches_to_put

        # 7. Calculate cost_move_tray: Estimate necessary tray movements.
        cost_move_tray = 0
        # Pickup needed if sandwiches must be put onto trays from kitchen
        needs_pickup = (cost_put_on_tray > 0)
        trays_present_at = set(tray_locations.values()) # Set of locations currently having a tray
        is_tray_at_kitchen = 'kitchen' in trays_present_at

        # Cost for moving a tray TO the kitchen (if pickup needed and no tray is there)
        if needs_pickup and not is_tray_at_kitchen:
            cost_move_tray += 1

        # Cost for moving trays TO waiting locations
        # Get unique locations (excluding kitchen) where unserved children wait
        unserved_locations = {
            loc for child, loc in self.child_waiting_location.items()
            if child in unserved_children and loc != 'kitchen'
        }

        destinations_to_visit = set() # Track unique destinations needing a visit
        for p in unserved_locations:
            # Location p needs a visit if:
            # a) It requires sandwiches delivered from the kitchen (approximated by needs_pickup)
            # b) OR no tray is currently at p to serve the child(ren) there.
            if needs_pickup or (p not in trays_present_at):
                destinations_to_visit.add(p)

        # Add cost for moving to each unique destination identified
        cost_move_tray += len(destinations_to_visit)

        # 8. Total Heuristic Value: Sum of estimated costs for each action type.
        total_cost = cost_serve + cost_make + cost_put_on_tray + cost_move_tray
        return total_cost
