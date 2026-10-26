from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic): # Assuming Heuristic is defined elsewhere
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the minimum number of actions required to serve
    each unserved child, summing these minimum costs. It considers the current
    state of suitable sandwiches (ready to serve, on tray elsewhere, at kitchen,
    needs making) and the availability of trays at the kitchen.

    # Assumptions
    - Each child needs exactly one sandwich.
    - Allergic children require gluten-free sandwiches.
    - Non-allergic children can accept any sandwich.
    - Ingredients (bread, content) are consumed when making a sandwich.
    - Sandwich objects (`notexist`) are consumed when making a sandwich.
    - Trays are resources that can be moved and used to transport sandwiches.
    - The 'kitchen' is a constant place where ingredients and newly made
      sandwiches start, and where trays might need to be moved to pick up
      sandwiches.
    - The heuristic assigns a base cost for reaching different stages of
      sandwich preparation and delivery for each child.

    # Heuristic Initialization
    The heuristic extracts static information from the task:
    - The waiting location for each child.
    - Which children are allergic or not allergic.
    - Which bread and content portions are gluten-free.
    This information is stored in dictionaries and sets for quick lookup.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is the sum of the minimum costs calculated for each
    unserved child. For an unserved child C waiting at place P:

    1.  **Check if Served:** If the fact `(served C)` is in the current state,
        the cost for this child is 0.

    2.  **Estimate Minimum Cost if Not Served:** If the child is not served,
        we estimate the minimum actions needed to get a suitable sandwich to them
        and serve it. We look for the "most ready" suitable sandwich:
        -   **Level 1 (Ready at Location):** Is there a suitable sandwich S on a
            tray T, and is tray T currently at the child's waiting place P?
            (Check `(ontray S T)` and `(at T P)` for a suitable S).
            If yes, the minimum cost for this child is 1 (the `serve` action).
        -   **Level 2 (On Tray, Needs Moving):** If not Level 1, is there a
            suitable sandwich S on a tray T, but tray T is *not* at place P?
            (Check `(ontray S T)` for a suitable S, and verify `(at T P)` is false).
            If yes, the minimum cost is 2 (a `move_tray` action and the `serve` action).
        -   **Level 3 (At Kitchen, Needs Tray & Moving):** If not Level 1 or 2,
            is there a suitable sandwich S currently at the kitchen?
            (Check `(at_kitchen_sandwich S)` for a suitable S).
            If yes:
                -   Check if *any* tray T is currently at the kitchen `(at T kitchen)`.
                -   If a tray is at the kitchen, the minimum cost is 3 (a `put_on_tray`
                    action, a `move_tray` action to P, and the `serve` action).
                -   If no tray is at the kitchen, a tray must first be moved there.
                    The minimum cost is 4 (a `move_tray` to kitchen, `put_on_tray`,
                    `move_tray` to P, `serve`).
        -   **Level 4 (Needs Making, Tray & Moving):** If not Level 1, 2, or 3,
            can a suitable sandwich be made? This requires:
            -   An unused sandwich object S (`(notexist S)`).
            -   Suitable ingredients (bread B, content C) at the kitchen
                (`(at_kitchen_bread B)`, `(at_kitchen_content C)`). Gluten status
                must match if the child is allergic.
            If these resources exist:
                -   Check if *any* tray T is currently at the kitchen `(at T kitchen)`.
                -   If a tray is at the kitchen, the minimum cost is 4 (a `make_sandwich`
                    action, `put_on_tray`, `move_tray` to P, `serve`).
                -   If no tray is at the kitchen, a tray must first be moved there.
                    The minimum cost is 5 (`make_sandwich`, `move_tray` to kitchen,
                    `put_on_tray`, `move_tray` to P, `serve`).
        -   **Level 5 (No Resources):** If none of the above conditions are met,
            it's likely impossible to serve this child with the current state and
            available initial resources. A large cost (e.g., 1000) is assigned.

    3.  The total heuristic is the sum of the minimum costs calculated for each
        unserved child.

    This heuristic is not admissible but provides a reasonable estimate for
    greedy best-first search by prioritizing states where children are closer
    to being served.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals # Store goals to check if state is goal state (h=0)

        # Extract static information
        self.child_location = {}
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.no_gluten_breads = set()
        self.no_gluten_contents = set()

        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "waiting":
                child, place = parts[1], parts[2]
                self.child_location[child] = place
            elif predicate == "allergic_gluten":
                self.allergic_children.add(parts[1])
            elif predicate == "not_allergic_gluten":
                self.not_allergic_children.add(parts[1])
            elif predicate == "no_gluten_bread":
                self.no_gluten_breads.add(parts[1])
            elif predicate == "no_gluten_content":
                self.no_gluten_contents.add(parts[1])

        # Identify all children from static waiting facts
        self.all_children = set(self.child_location.keys())


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Heuristic is 0 if and only if the goal is reached
        if self.goals <= state:
            return 0

        total_heuristic = 0
        large_value = 1000 # Cost for children that seem unservable

        # Get dynamic facts from the current state
        served_children_in_state = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        ontray_sandwiches = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "ontray", "*", "*")} # {sandwich: tray}
        tray_locations = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*")} # {tray: place}
        kitchen_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_sandwich", "*")}
        no_gluten_sandwiches_in_state = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}
        kitchen_breads_in_state = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_bread", "*")}
        kitchen_contents_in_state = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_content", "*")}
        notexist_sandwiches_in_state = {get_parts(fact)[1] for fact in state if match(fact, "notexist", "*")}

        # Check if *any* tray is at the kitchen
        is_tray_at_kitchen = any(place == "kitchen" for place in tray_locations.values())

        # Check if suitable ingredients exist at the kitchen
        can_make_regular_sandwich = len(kitchen_breads_in_state) > 0 and len(kitchen_contents_in_state) > 0
        can_make_gluten_free_sandwich = (
            len(self.no_gluten_breads.intersection(kitchen_breads_in_state)) > 0 and
            len(self.no_gluten_contents.intersection(kitchen_contents_in_state)) > 0
        )
        can_make_any_sandwich = can_make_regular_sandwich or can_make_gluten_free_sandwich

        # Check if an unused sandwich object is available
        can_use_new_sandwich_object = len(notexist_sandwiches_in_state) > 0


        for child in self.all_children:
            if child in served_children_in_state:
                continue # Child is already served, cost is 0 for this child

            child_place = self.child_location[child]
            is_allergic = child in self.allergic_children

            min_cost_for_child = large_value # Default to unservable

            # --- Level 1: Ready to serve? ---
            # Is there a suitable sandwich S on a tray T at child_place P?
            found_level_1 = False
            for s, t in ontray_sandwiches.items():
                if tray_locations.get(t) == child_place:
                    is_suitable = (not is_allergic) or (s in no_gluten_sandwiches_in_state)
                    if is_suitable:
                        min_cost_for_child = 1
                        found_level_1 = True
                        break # Found the best case for this child

            if found_level_1:
                total_heuristic += min_cost_for_child
                continue # Move to the next child

            # --- Level 2: On tray elsewhere? ---
            # Is there a suitable sandwich S on a tray T not at child_place P?
            found_level_2 = False
            for s, t in ontray_sandwiches.items():
                 # Check if tray exists and is not at the child's place
                 if tray_locations.get(t) != child_place:
                    is_suitable = (not is_allergic) or (s in no_gluten_sandwiches_in_state)
                    if is_suitable:
                        min_cost_for_child = 2
                        found_level_2 = True
                        break # Found the next best case

            if found_level_2:
                total_heuristic += min_cost_for_child
                continue # Move to the next child

            # --- Level 3: At kitchen? ---
            # Is there a suitable sandwich S at the kitchen?
            found_level_3 = False
            for s in kitchen_sandwiches:
                is_suitable = (not is_allergic) or (s in no_gluten_sandwiches_in_state)
                if is_suitable:
                    # Cost depends on tray availability at kitchen
                    min_cost_for_child = 3 if is_tray_at_kitchen else 4
                    found_level_3 = True
                    break # Found the next best case

            if found_level_3:
                total_heuristic += min_cost_for_child
                continue # Move to the next child

            # --- Level 4: Can be made? ---
            # Are there suitable ingredients and a notexist sandwich object?
            can_make_suitable = False
            if can_use_new_sandwich_object: # Need an unused sandwich object first
                if is_allergic:
                    if can_make_gluten_free_sandwich:
                        can_make_suitable = True
                else: # Not allergic, any sandwich is suitable
                    if can_make_any_sandwich:
                         can_make_suitable = True

            if can_make_suitable:
                 # Cost depends on tray availability at kitchen
                 min_cost_for_child = 4 if is_tray_at_kitchen else 5

            # If we reached here and min_cost_for_child is still large_value,
            # it means no suitable sandwich exists or can be made with current resources.
            total_heuristic += min_cost_for_child # Add the calculated min_cost (could be large_value)


        return total_heuristic
