import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Not strictly needed now, but potentially useful

# Helper function to extract parts of a fact string like "(pred obj1 obj2)" -> ["pred", "obj1", "obj2"]
def get_parts(fact):
    """Extracts the components of a PDDL fact string, removing parentheses."""
    # Remove leading '(' and trailing ')' and split by space
    return fact[1:-1].split()

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the ChildSnack PDDL domain.

    # Summary
    Estimates the number of actions required to reach a goal state where specific children are served.
    The heuristic calculates the estimated cost for each unserved child required by the goal
    and sums these costs. The cost for serving a single child is estimated by finding the
    cheapest sequence of actions (make sandwich, put on tray, move tray, serve) based on the
    current state of sandwiches and trays. It prioritizes using existing resources efficiently.
    This heuristic is designed for Greedy Best-First Search and is not necessarily admissible.

    # Assumptions
    - Assumes that if a sandwich needs to be made, sufficient ingredients (bread, content)
      and an available sandwich object (in the 'notexist' state) are present. This simplifies the
      calculation by not needing to check ingredient availability.
    - Assumes at least one tray object exists in the problem instance.
    - Ignores potential conflicts or resource contention (e.g., multiple children needing the
      same tray, the last gluten-free sandwich, or specific ingredients). Costs are calculated
      independently for each child.
    - Simplifies tray movement cost: assumes moving a tray between any two distinct locations
      (including the kitchen) costs exactly 1 'move_tray' action.
    - Assumes that if a tray is needed at the kitchen (e.g., for 'put_on_tray') and none is
      present, moving an existing tray there costs 1 action.

    # Heuristic Initialization
    - The constructor (`__init__`) parses the task's static facts (`task.static`) and goals (`task.goals`).
    - It stores essential static information derived from these facts:
        - `child_is_allergic`: A dictionary mapping child names (str) to a boolean indicating
          if they are allergic to gluten.
        - `child_waiting_at`: A dictionary mapping child names (str) to the place name (str)
          where they are waiting.
    - It identifies the set of `target_children` (str) that need to be served according to the goal facts.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Initialization**: Set the total heuristic estimate `h = 0`.
    2.  **Goal Check**: If the current state already satisfies all goals, return 0 immediately.
    3.  **State Parsing**: Parse the current state (`node.state`, a frozenset of fact strings)
        to extract dynamic information into efficient data structures:
        - `served_children`: Set of names of children already served.
        - `tray_locations`: Dictionary mapping tray names to their current location names.
        - `sandwiches_at_kitchen`: Set of names of sandwiches currently at the kitchen.
        - `sandwiches_on_tray`: Dictionary mapping names of sandwiches on trays to the name of the tray they are on.
        - `gluten_free_sandwiches`: Set of names of sandwiches that are gluten-free.
    4.  **Iterate Through Goals**: For each `child` in `self.target_children`:
        a.  **Check if Served**: If `child` is already in `served_children`, this goal component is satisfied, so add 0 to `h` for this child and continue to the next child.
        b.  **Estimate Cost if Unserved**: If the child is not served:
            i.  Retrieve the child's waiting `place` and `is_allergic` status from the precomputed static info dictionaries. Handle potential `KeyError` if static info is missing for a goal child (indicates a problem definition issue, return infinity).
            ii. **Calculate Minimum Cost (`cost_c`)**:
                - Initialize `cost_c = 4`. This is the baseline cost assuming a new sandwich must be made: make(1) + put_on_tray(1) + move_tray(1) + serve(1).
                - **Option 1: Sandwich on Tray at Child's Location**: Iterate through `sandwiches_on_tray`. If a suitable sandwich `s` (gluten-free if needed) is on tray `t`, and `tray_locations[t]` is the child's `place`, then the cost is 1 (serve). Set `cost_c = 1` and break all inner loops for this child (as 1 is the minimum possible cost).
                - **Option 2: Sandwich on Tray Elsewhere**: If Option 1 didn't apply, continue iterating through `sandwiches_on_tray`. If a suitable sandwich `s` is on tray `t` but `tray_locations[t]` is *not* the child's `place`, the cost is 2 (move_tray + serve). Update `cost_c = min(cost_c, 2)`.
                - **Option 3: Sandwich at Kitchen**: If the current `cost_c` is still greater than 2 (meaning options 1 and 2 didn't yield a cost of 1 or 2), check `sandwiches_at_kitchen`. For each suitable sandwich `s` found:
                    - Determine the cost to put it on a tray (`cost_put`): Check if any tray is currently at the 'kitchen' location in `tray_locations`. If yes, `cost_put = 1` (put_on_tray action). If no, `cost_put = 2` (1 move_tray action to bring a tray to the kitchen + 1 put_on_tray action).
                    - The total cost for this path is `cost_put + 1 (move_tray from kitchen to place) + 1 (serve)`. Update `cost_c = min(cost_c, cost_put + 2)`.
                - **Option 4: Make New Sandwich**: This cost is implicitly represented by the initial `cost_c = 4`. If none of the above options provide a lower cost, `cost_c` remains 4.
            iii. **Add Child's Cost**: Add the final determined `cost_c` for this child to the total heuristic value `h`.
    5.  **Return Total**: Return the accumulated heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing static information and goals from the task.
        """
        self.goals = task.goals
        self.static = task.static

        # Pre-process static information from task.static
        self.child_is_allergic = {}
        self.child_waiting_at = {}

        for fact in self.static:
            parts = get_parts(fact)
            # Ensure parts is not empty before accessing parts[0]
            if not parts:
                continue
            predicate = parts[0]
            try:
                if predicate == "allergic_gluten" and len(parts) == 2:
                    self.child_is_allergic[parts[1]] = True
                elif predicate == "not_allergic_gluten" and len(parts) == 2:
                    self.child_is_allergic[parts[1]] = False
                elif predicate == "waiting" and len(parts) == 3:
                    # parts[1] is child, parts[2] is place
                    self.child_waiting_at[parts[1]] = parts[2]
            except IndexError:
                # Log or handle potential errors in static fact format
                # print(f"Warning: Skipping potentially malformed static fact: {fact}")
                pass # Ignore malformed facts silently for now

        # Identify target children from task.goals
        self.target_children = set()
        for goal in self.goals:
             parts = get_parts(goal)
             if not parts:
                 continue
             try:
                 # Assuming goals are only of the form (served child)
                 if parts[0] == "served" and len(parts) == 2:
                     self.target_children.add(parts[1])
             except IndexError:
                 # Log or handle potential errors in goal fact format
                 # print(f"Warning: Skipping potentially malformed goal fact: {goal}")
                 pass # Ignore malformed goals silently

        # Optional: Add checks here to ensure all target children have required static info
        # This helps catch potential issues in PDDL definitions early.
        for child in self.target_children:
            if child not in self.child_is_allergic:
                print(f"Warning: Child {child} in goal but allergy status unknown from static facts.")
            if child not in self.child_waiting_at:
                 print(f"Warning: Child {child} in goal but waiting location unknown from static facts.")


    def __call__(self, node):
        """
        Calculates the heuristic value for a given state node.
        """
        state = node.state

        # Check if goal is already reached - if so, heuristic is 0
        # This uses the subset check: are all goal facts present in the current state?
        if self.goals <= state:
            return 0

        # Parse current state information efficiently
        served_children = set()
        sandwiches_at_kitchen = set()
        sandwiches_on_tray = {} # sandwich_name -> tray_name
        gluten_free_sandwiches = set()
        tray_locations = {} # tray_name -> place_name

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            predicate = parts[0]
            try:
                if predicate == "served" and len(parts) == 2:
                    served_children.add(parts[1])
                elif predicate == "at_kitchen_sandwich" and len(parts) == 2:
                    sandwiches_at_kitchen.add(parts[1])
                elif predicate == "ontray" and len(parts) == 3:
                    # parts[1] is sandwich, parts[2] is tray
                    sandwiches_on_tray[parts[1]] = parts[2]
                elif predicate == "no_gluten_sandwich" and len(parts) == 2:
                    gluten_free_sandwiches.add(parts[1])
                elif predicate == "at" and len(parts) == 3:
                     # parts[1] is tray, parts[2] is place
                     # We assume only trays use the 'at' predicate like this based on domain structure
                     tray_locations[parts[1]] = parts[2]
            except IndexError:
                 # Ignore potential errors in state fact format silently
                 pass

        h = 0
        for child in self.target_children:
            # Skip if this child's goal is already met in the current state
            if child in served_children:
                continue

            # Get child's requirements (place, allergy) from stored static info
            try:
                place = self.child_waiting_at[child]
                is_allergic = self.child_is_allergic[child]
            except KeyError:
                # This indicates an inconsistency between goals and static facts.
                # Return a high value or infinity to strongly discourage exploring this path.
                # print(f"Error: Missing static info for goal child {child}. Returning infinity.")
                return float('inf')

            # --- Estimate minimum cost to serve this child ---
            cost_c = 4 # Base cost: make(1) + put(1) + move(1) + serve(1)
            found_best_option = False # Flag to stop early if cost 1 is found

            # Check sandwiches on trays first (Options 1 & 2)
            for s, t in sandwiches_on_tray.items():
                # Check if sandwich 's' is suitable for the child's allergy needs
                is_suitable = (not is_allergic) or (s in gluten_free_sandwiches)
                if is_suitable:
                    # Check if the tray 't' holding the sandwich has a known location
                    if t in tray_locations:
                        tray_loc = tray_locations[t]
                        if tray_loc == place:
                            # Option 1: Sandwich on tray at the correct place -> cost = 1 (serve)
                            cost_c = 1
                            found_best_option = True
                            break # Found the absolute minimum cost, no need to check further for this child
                        else:
                            # Option 2: Sandwich on tray elsewhere -> cost = 2 (move + serve)
                            cost_c = min(cost_c, 2)
                    # else: Inconsistency: sandwich on tray but tray location unknown. Ignore this sandwich?

            # If cost 1 was found, proceed to the next child
            if found_best_option:
                h += cost_c
                continue

            # Check sandwiches at kitchen (Option 3) - only if cost is still > 2
            if cost_c > 2:
                # Check if any tray is currently at the kitchen location
                is_tray_at_kitchen = False
                for t_loc in tray_locations.values():
                    if t_loc == 'kitchen':
                        is_tray_at_kitchen = True
                        break
                # Cost to put a sandwich from kitchen onto a tray at kitchen
                # 1 action (put_on_tray) if tray is there, 2 actions (move_tray + put_on_tray) if not
                cost_put = 1 if is_tray_at_kitchen else 2

                for s in sandwiches_at_kitchen:
                    # Check if sandwich 's' is suitable
                    is_suitable = (not is_allergic) or (s in gluten_free_sandwiches)
                    if is_suitable:
                        # Calculate total cost via this kitchen sandwich path
                        # cost = cost_to_put_on_tray + move_tray_to_place + serve
                        current_kitchen_path_cost = cost_put + 1 + 1
                        cost_c = min(cost_c, current_kitchen_path_cost)
                        # Optimization: if cost becomes 3, we can't do better from kitchen,
                        # but maybe another kitchen sandwich yields 3 if cost_put was 1?
                        # Keep checking all kitchen sandwiches.

            # Add the minimum cost found (or the base cost of 4) for this child to the total
            h += cost_c

        # Return the total estimated heuristic cost
        # The logic should ensure h >= 0, but max(0, h) is a safe return.
        return max(0, h)

