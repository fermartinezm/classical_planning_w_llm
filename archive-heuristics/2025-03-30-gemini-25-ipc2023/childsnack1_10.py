import math
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts represented as strings
def get_parts(fact):
    """
    Removes parentheses and splits a PDDL fact string into its components.
    Example: "(at tray1 kitchen)" -> ["at", "tray1", "kitchen"]
    """
    return fact[1:-1].split()

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL ChildSnack domain.

    # Summary
    This heuristic estimates the minimum number of actions required to reach a
    goal state from the current state. The goal is typically to have all
    specified children served `(served ?c)`. The heuristic calculates the
    estimated cost for each unserved child individually and sums these costs.
    The cost for serving a single child is estimated based on the most efficient
    way to provide them with a suitable sandwich (considering allergies), checking
    these options in increasing order of estimated actions:
    1. Use a suitable sandwich already on a tray at the child's location.
    2. Use a suitable sandwich on a tray at a different location (requires moving the tray).
    3. Use a suitable sandwich currently in the kitchen (requires putting on a tray and moving).
    4. Make a new suitable sandwich (requires ingredients, making, putting on tray, moving).

    # Assumptions
    - Each PDDL action has a cost of 1.
    - Trays can be moved between any two places (including 'kitchen') with a single 'move_tray' action.
    - The heuristic assumes resources (ingredients, available sandwiches, trays) are sufficient or can be made available with minimal actions (like moving a tray). It does not perform detailed resource tracking or allocation, focusing on estimating the necessary steps for each child independently.
    - Gluten-free sandwiches (`no_gluten_sandwich`) can serve any child (allergic or not). Regular sandwiches can only serve non-allergic children.
    - The problem instances are solvable, meaning a path to the goal exists.

    # Heuristic Initialization
    - The constructor (`__init__`) processes the task's static information and goal conditions.
    - It identifies the set of children that need to be served (`goal_children`).
    - It parses static facts to build efficient lookups for:
        - Child allergies: `child_is_allergic` (dictionary mapping child to boolean).
        - Child waiting locations: `child_waiting_location` (dictionary mapping child to place).
        - Gluten-free status of initial bread and content portions: `is_gluten_free_bread`, `is_gluten_free_content` (sets).

    # Step-By-Step Thinking for Computing Heuristic
    The `__call__` method computes the heuristic value for a given state (`node.state`):
    1. **State Parsing:** Iterate through the facts in the current state to build data structures representing the current situation:
        - `served_children`: Set of children already served.
        - `sandwiches_on_tray`: Dictionary mapping trays to sets of sandwiches currently on them.
        - `sandwiches_at_kitchen`: Set of sandwiches currently at the kitchen.
        - `sandwich_is_gluten_free`: Dictionary mapping sandwiches to their gluten-free status (True/False).
        - `tray_locations`: Dictionary mapping trays to their current place.
        - `available_bread_at_kitchen`: Dictionary mapping available bread portions at the kitchen to their gluten-free status.
        - `available_content_at_kitchen`: Dictionary mapping available content portions at the kitchen to their gluten-free status.
        - `available_sandwich_slots`: Count of `(notexist ?s)` facts, representing how many new sandwiches can be made.
    2. **Identify Unserved Children:** Determine the set of `unserved_goal_children` by subtracting `served_children` from the `goal_children` identified during initialization.
    3. **Goal Check:** If `unserved_goal_children` is empty, the goal is reached, return 0.
    4. **Calculate Cost Per Child:** For each `child` in `unserved_goal_children`:
        a. Retrieve the child's `waiting_place` and `is_allergic` status.
        b. Initialize `min_cost_for_child` to infinity.
        c. **Check Level 1 (Serve from Tray at Correct Location):** Iterate through trays `t` located at `waiting_place`. Check if any sandwich `s` on `t` is suitable (gluten-free if `is_allergic`). If yes, `min_cost_for_child = min(min_cost_for_child, 1)` (for the 'serve' action).
        d. **Check Level 2 (Serve from Tray at Different Location):** If `min_cost_for_child > 1`, iterate through trays `t` *not* at `waiting_place`. Check if any sandwich `s` on `t` is suitable. If yes, `min_cost_for_child = min(min_cost_for_child, 2)` (for 'move_tray' + 'serve').
        e. **Check Level 3 (Serve from Kitchen Sandwich):** If `min_cost_for_child > 2`, iterate through sandwiches `s` in `sandwiches_at_kitchen`. Check if `s` is suitable. If yes, estimate cost: 1 ('put_on_tray') + (1 if `waiting_place` != 'kitchen' else 0) ('move_tray') + 1 ('serve'). Update `min_cost_for_child = min(min_cost_for_child, cost)` (cost is 2 or 3). This assumes a tray is available at/can be moved to the kitchen.
        f. **Check Level 4 (Make New Sandwich):** If `min_cost_for_child > 3`, check if a suitable sandwich can be made (requires appropriate ingredients `at_kitchen` and an available `notexist` slot). If yes, estimate cost: 1 ('make_sandwich*') + 1 ('put_on_tray') + (1 if `waiting_place` != 'kitchen' else 0) ('move_tray') + 1 ('serve'). Update `min_cost_for_child = min(min_cost_for_child, cost)` (cost is 3 or 4).
        g. **Add Child's Cost:** Add the final `min_cost_for_child` to `total_cost`. If it remained infinity (heuristic deems it impossible to serve the child from this state), add a large penalty value (e.g., 100) instead to signify difficulty without using actual infinity.
    5. **Final Adjustment:** If `total_cost` is 0 but the state is not actually a goal state (e.g., due to heuristic inaccuracy or intermediate states), return 1 to ensure non-zero cost for non-goal states. Otherwise, return `total_cost`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static task information.
        """
        self.goals = task.goals
        static_facts = task.static

        # Identify children that need to be served according to the goal
        self.goal_children = set()
        for goal in self.goals:
            parts = get_parts(goal)
            # Ensure the goal is well-formed before accessing parts[1]
            if parts and parts[0] == 'served' and len(parts) > 1:
                self.goal_children.add(parts[1])

        # Store static information about children and ingredients
        self.child_is_allergic = {}
        self.child_waiting_location = {}
        self.is_gluten_free_bread = set()
        self.is_gluten_free_content = set()

        for fact in static_facts:
            parts = get_parts(fact)
            predicate = parts[0]
            # Ensure facts are well-formed before accessing parts[1], parts[2]
            if not parts: continue

            if predicate == 'allergic_gluten' and len(parts) > 1:
                self.child_is_allergic[parts[1]] = True
            elif predicate == 'not_allergic_gluten' and len(parts) > 1:
                self.child_is_allergic[parts[1]] = False
            elif predicate == 'waiting' and len(parts) > 2:
                self.child_waiting_location[parts[1]] = parts[2]
            elif predicate == 'no_gluten_bread' and len(parts) > 1:
                self.is_gluten_free_bread.add(parts[1])
            elif predicate == 'no_gluten_content' and len(parts) > 1:
                self.is_gluten_free_content.add(parts[1])
            # Note: 'kitchen' is a constant place, handled implicitly.

    def __call__(self, node):
        """
        Calculates the heuristic value for the given state.
        """
        state = node.state

        # --- 1. Pre-process state to get current status ---
        served_children = set()
        sandwiches_on_tray = {} # tray -> set of sandwiches
        sandwiches_at_kitchen = set()
        sandwich_is_gluten_free = {} # sandwich -> bool (True if GF, False otherwise)
        tray_locations = {} # tray -> place
        available_bread_at_kitchen = {} # bread -> is_gluten_free (bool)
        available_content_at_kitchen = {} # content -> is_gluten_free (bool)
        available_sandwich_slots = 0
        all_sandwiches_in_state = set() # Keep track of all existing sandwiches

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if not parts: continue

            if predicate == 'served' and len(parts) > 1:
                served_children.add(parts[1])
            elif predicate == 'ontray' and len(parts) > 2:
                sandwich, tray = parts[1], parts[2]
                if tray not in sandwiches_on_tray:
                    sandwiches_on_tray[tray] = set()
                sandwiches_on_tray[tray].add(sandwich)
                all_sandwiches_in_state.add(sandwich)
            elif predicate == 'at_kitchen_sandwich' and len(parts) > 1:
                sandwich = parts[1]
                sandwiches_at_kitchen.add(sandwich)
                all_sandwiches_in_state.add(sandwich)
            elif predicate == 'no_gluten_sandwich' and len(parts) > 1:
                sandwich_is_gluten_free[parts[1]] = True
                all_sandwiches_in_state.add(parts[1]) # Ensure it's tracked
            elif predicate == 'at' and len(parts) > 2: # Tray location
                # Assuming first arg is tray, second is place based on domain actions
                obj, place = parts[1], parts[2]
                # Simple check if obj looks like a tray (heuristic assumption)
                # A better approach would be to know tray objects from task definition
                if 'tray' in obj:
                     tray_locations[obj] = place
            elif predicate == 'at_kitchen_bread' and len(parts) > 1:
                bread = parts[1]
                available_bread_at_kitchen[bread] = (bread in self.is_gluten_free_bread)
            elif predicate == 'at_kitchen_content' and len(parts) > 1:
                content = parts[1]
                available_content_at_kitchen[content] = (content in self.is_gluten_free_content)
            elif predicate == 'notexist' and len(parts) > 1:
                available_sandwich_slots += 1

        # Assume sandwiches not explicitly marked as 'no_gluten_sandwich' are not GF
        for s in all_sandwiches_in_state:
             if s not in sandwich_is_gluten_free:
                 sandwich_is_gluten_free[s] = False

        # --- 2. Identify unserved children ---
        unserved_goal_children = self.goal_children - served_children

        # --- 3. Goal Check ---
        if not unserved_goal_children:
            return 0 # Goal reached

        # --- Check resource availability for making sandwiches ---
        has_gf_bread = any(is_gf for bread, is_gf in available_bread_at_kitchen.items() if is_gf)
        has_any_bread = bool(available_bread_at_kitchen)
        has_gf_content = any(is_gf for content, is_gf in available_content_at_kitchen.items() if is_gf)
        has_any_content = bool(available_content_at_kitchen)

        can_make_gf_sandwich = has_gf_bread and has_gf_content and available_sandwich_slots > 0
        # Can make regular sandwich if any bread and any content exist
        can_make_regular_sandwich = has_any_bread and has_any_content and available_sandwich_slots > 0

        # --- 4. Calculate cost for each unserved child ---
        total_cost = 0
        for child in unserved_goal_children:
            # Retrieve child's static info, skip if missing (shouldn't happen in valid tasks)
            is_allergic = self.child_is_allergic.get(child)
            waiting_place = self.child_waiting_location.get(child)
            if waiting_place is None or is_allergic is None:
                # Assign high cost if static info is missing for a goal child
                total_cost += 100
                continue

            min_cost_for_child = math.inf

            # --- Check Level 1: Suitable sandwich on tray at waiting_place ---
            found_level1 = False
            for tray, location in tray_locations.items():
                if location == waiting_place and tray in sandwiches_on_tray:
                    for sandwich in sandwiches_on_tray[tray]:
                        is_gf = sandwich_is_gluten_free.get(sandwich, False)
                        if (is_allergic and is_gf) or (not is_allergic):
                            min_cost_for_child = min(min_cost_for_child, 1) # serve
                            found_level1 = True
                            break # Found cheapest option for this tray
                    if found_level1: break # Found cheapest overall for level 1

            if found_level1:
                total_cost += min_cost_for_child
                continue # Move to next child

            # --- Check Level 2: Suitable sandwich on tray at other location ---
            found_level2 = False
            for tray, location in tray_locations.items():
                if location != waiting_place and tray in sandwiches_on_tray:
                    for sandwich in sandwiches_on_tray[tray]:
                        is_gf = sandwich_is_gluten_free.get(sandwich, False)
                        if (is_allergic and is_gf) or (not is_allergic):
                            min_cost_for_child = min(min_cost_for_child, 2) # move + serve
                            found_level2 = True
                            break
                    if found_level2: break

            if found_level2:
                total_cost += min_cost_for_child
                continue

            # --- Check Level 3: Suitable sandwich at kitchen ---
            found_level3 = False
            for sandwich in sandwiches_at_kitchen:
                 is_gf = sandwich_is_gluten_free.get(sandwich, False)
                 if (is_allergic and is_gf) or (not is_allergic):
                     cost = 1 # put_on_tray
                     if waiting_place != 'kitchen':
                         cost += 1 # move_tray
                     cost += 1 # serve
                     min_cost_for_child = min(min_cost_for_child, cost) # cost is 2 or 3
                     found_level3 = True
                     # Don't break, find the minimum cost if multiple kitchen sandwiches exist
                     # (though cost is same unless waiting_place is kitchen)

            if found_level3:
                 total_cost += min_cost_for_child
                 continue

            # --- Check Level 4: Make new sandwich ---
            can_make_suitable = False
            if is_allergic and can_make_gf_sandwich:
                can_make_suitable = True
            elif not is_allergic and can_make_regular_sandwich:
                 can_make_suitable = True

            if can_make_suitable:
                cost = 1 # make_sandwich*
                cost += 1 # put_on_tray
                if waiting_place != 'kitchen':
                    cost += 1 # move_tray
                cost += 1 # serve
                min_cost_for_child = min(min_cost_for_child, cost) # cost is 3 or 4

            # --- Add Child's Cost ---
            if min_cost_for_child == math.inf:
                 # Child seems unservable from this state based on heuristic's view
                 # Add a large penalty instead of infinity
                 total_cost += 100
            else:
                 total_cost += min_cost_for_child

        # --- 5. Final Adjustment ---
        # Ensure heuristic is non-zero for non-goal states
        if total_cost == 0 and not self.goals <= state:
             return 1

        return total_cost

