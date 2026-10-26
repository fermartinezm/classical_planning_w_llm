from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
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

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the total number of actions required to serve all
    children who are currently waiting and not yet served. It calculates the
    minimum steps needed for each unserved child independently, considering
    the availability and location of suitable sandwiches and the possibility
    of making new ones.

    # Assumptions
    - Each unserved child requires one suitable sandwich.
    - Suitability depends on the child's allergy status (gluten-free or not).
    - Actions have a cost of 1.
    - Trays are available when needed in the kitchen or can be moved there with cost 1.
    - Ingredients are sufficient in total across the problem, such that if a
      sandwich is needed and cannot be found or made from current kitchen
      ingredients, it can still be made (this prevents infinite heuristic values
      in solvable states where ingredients might be temporarily unavailable
      in the kitchen). This is a simplification for a non-admissible heuristic.

    # Heuristic Initialization
    - Extracts static information:
        - All children and their waiting locations.
        - Which children are allergic to gluten.
        - Which bread, content, and sandwich items are gluten-free.
        - All tray objects (inferred from initial state facts).
        - Whether any GF bread/content exists in the problem (from static facts).

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is the sum of the estimated minimum steps for each
    child who is not yet served. For an unserved child `c` waiting at place `p`:

    1.  **Check if already served:** If `(served c)` is true in the state, the cost for this child is 0.
    2.  **Determine suitability requirement:** Check if the child `c` is allergic to gluten. If yes, a gluten-free sandwich is required. Otherwise, any sandwich is suitable.
    3.  **Find minimum steps to get a suitable sandwich to the child's location `p` and serve it:**
        -   Initialize `min_steps_for_child` to infinity.
        -   **Option A: Suitable sandwich already on a tray at `p`?**
            -   Look for facts `(ontray s t)` and `(at t p)` where `s` is suitable for `c`.
            -   If found, `min_steps_for_child = min(min_steps_for_child, 1)` (cost of `serve` action).
        -   **Option B: Suitable sandwich on a tray elsewhere?**
            -   Look for facts `(ontray s t)` and `(at t p')` where `p' is not None`, `p' != p` and `s` is suitable for `c`.
            -   If found, `min_steps_for_child = min(min_steps_for_child, 2)` (cost of `move_tray` + `serve`).
        -   **Option C: Suitable sandwich in the kitchen?**
            -   Look for facts `(at_kitchen_sandwich s)` where `s` is suitable for `c`.
            -   If found, estimate cost: 1 (`put_on_tray`) + (1 if `p` is not `kitchen` for `move_tray`) + 1 (`serve`). Assume a tray is available at the kitchen.
            -   `min_steps_for_child = min(min_steps_for_child, 3 if p != 'kitchen' else 2)`.
        -   **Option D: Can a suitable sandwich be made?**
            -   Check if required ingredients (GF or any bread/content) are `at_kitchen_bread` and `at_kitchen_content` in the current state, considering GF status from static facts.
            -   Determine if a suitable sandwich *can* be made using available kitchen ingredients OR if it's possible based on the fallback assumption (ingredients exist globally).
            -   If a suitable sandwich can be made: Estimate cost: 1 (`make_sandwich`) + 1 (`put_on_tray`) + (1 if `p` is not `kitchen` for `move_tray`) + 1 (`serve`). Assume a tray is available at the kitchen.
            -   `min_steps_for_child = min(min_steps_for_child, 4 if p != 'kitchen' else 3)`.
        -   **Handle unreachable child:** If `min_steps_for_child` is still infinity after checking all options (should ideally not happen in solvable problems with the fallback), return a large value to indicate a bad state.

    4.  **Sum costs:** Add `min_steps_for_child` to the total heuristic value.

    The final heuristic value is the accumulated total cost for all unserved children.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state
        self.kitchen_place = 'kitchen' # Constant

        self.children = set()
        self.waiting_locations = {} # child -> place
        self.allergic_children = set()
        self.gluten_free_items = set() # bread, content, sandwich
        self.all_trays = set()

        # Extract static information
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'waiting':
                child, place = parts[1], parts[2]
                self.children.add(child)
                self.waiting_locations[child] = place
            elif parts[0] == 'allergic_gluten':
                self.allergic_children.add(parts[1])
                self.children.add(parts[1])
            elif parts[0] == 'not_allergic_gluten':
                 self.children.add(parts[1])
            elif parts[0] in ['no_gluten_bread', 'no_gluten_content', 'no_gluten_sandwich']:
                self.gluten_free_items.add(parts[1])

        # Collect all objects that appear as trays in relevant initial state facts
        for fact in self.initial_state:
            parts = get_parts(fact)
            # Trays appear as the first argument of 'at' when located somewhere
            if parts[0] == 'at' and len(parts) == 3: # (at ?t ?p)
                 self.all_trays.add(parts[1])
            # Trays appear as the second argument of 'ontray'
            elif parts[0] == 'ontray' and len(parts) == 3: # (ontray ?s ?t)
                 self.all_trays.add(parts[2])

        # Check if any GF bread/content exists globally (in static facts)
        self.any_gf_bread_exists_globally = any(match(fact, "no_gluten_bread", "*") for fact in self.static)
        self.any_gf_content_exists_globally = any(match(fact, "no_gluten_content", "*") for fact in self.static)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_heuristic = 0

        # Identify served children in the current state
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        # Identify current locations of trays
        tray_locations = {} # tray -> place
        for fact in state:
            if match(fact, "at", "*", "*"):
                 parts = get_parts(fact)
                 # Check if this object is one of the known trays
                 if parts[1] in self.all_trays:
                     tray_locations[parts[1]] = parts[2]

        # Identify sandwiches currently on trays and their tray
        sandwiches_on_trays = {} # sandwich -> tray
        for fact in state:
            if match(fact, "ontray", "*", "*"):
                sandwich, tray = get_parts(fact)[1], get_parts(fact)[2]
                sandwiches_on_trays[sandwich] = tray

        # Identify sandwiches currently in the kitchen
        kitchen_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_sandwich", "*")}

        # Check available ingredients in kitchen in the current state
        has_any_bread_kitchen = any(match(fact, "at_kitchen_bread", "*") for fact in state)
        has_any_content_kitchen = any(match(fact, "at_kitchen_content", "*") for fact in state)
        has_any_gf_bread_kitchen = any(match(fact, "at_kitchen_bread", b) and (b in self.gluten_free_items) for fact in state)
        has_any_gf_content_kitchen = any(match(fact, "at_kitchen_content", c) and (c in self.gluten_free_items) for fact in state)


        # Calculate cost for each unserved child
        for child in self.children:
            if child not in served_children:
                waiting_place = self.waiting_locations.get(child)
                # If a child is not in waiting_locations, they are not part of the goal
                # or the problem is malformed. Assuming children in self.children
                # are the ones that need serving and are waiting somewhere.
                if waiting_place is None:
                    continue # Should not happen in valid problems

                needs_gf = child in self.allergic_children
                min_steps_for_child = float('inf')

                # Option A: Suitable sandwich on tray at child's location
                for s, t in sandwiches_on_trays.items():
                    if tray_locations.get(t) == waiting_place:
                        is_suitable = (s in self.gluten_free_items) if needs_gf else True
                        if is_suitable:
                            min_steps_for_child = min(min_steps_for_child, 1) # serve

                # Option B: Suitable sandwich on tray elsewhere
                if min_steps_for_child == float('inf'): # Only consider if Option A wasn't met
                    for s, t in sandwiches_on_trays.items():
                         if tray_locations.get(t) is not None and tray_locations.get(t) != waiting_place:
                            is_suitable = (s in self.gluten_free_items) if needs_gf else True
                            if is_suitable:
                                min_steps_for_child = min(min_steps_for_child, 2) # move + serve

                # Option C: Suitable sandwich in the kitchen
                if min_steps_for_child == float('inf'): # Only consider if Options A, B weren't met
                    for s in kitchen_sandwiches:
                        is_suitable = (s in self.gluten_free_items) if needs_gf else True
                        if is_suitable:
                            # Assume tray available at kitchen or can be moved there (cost 1)
                            cost = 1 # put
                            if waiting_place != self.kitchen_place:
                                cost += 1 # move
                            cost += 1 # serve
                            min_steps_for_child = min(min_steps_for_child, cost)

                # Option D: Make suitable sandwich
                if min_steps_for_child == float('inf'): # Only consider if Options A, B, C weren't met
                    can_make_suitable = False
                    if needs_gf:
                        # Can make GF if GF bread AND GF content are in the kitchen
                        if has_any_gf_bread_kitchen and has_any_gf_content_kitchen:
                             can_make_suitable = True
                        # Fallback: If ingredients not in kitchen, assume they can be obtained if needed
                        # (possible if GF ingredients exist globally)
                        elif self.any_gf_bread_exists_globally and self.any_gf_content_exists_globally:
                             can_make_suitable = True # Assume ingredients can be moved/generated
                    else: # not allergic
                        # Can make non-GF if any bread AND any content are in the kitchen
                        if has_any_bread_kitchen and has_any_content_kitchen:
                            can_make_suitable = True
                        # Fallback: Assume ingredients can be obtained if needed (always possible if objects exist)
                        else:
                             can_make_suitable = True # Assume ingredients can be moved/generated

                    if can_make_suitable:
                        cost = 1 # make
                        cost += 1 # put
                        if waiting_place != self.kitchen_place:
                            cost += 1 # move
                        cost += 1 # serve
                        min_steps_for_child = min(min_steps_for_child, cost)

                # If still infinity, this child is unreachable with current heuristic logic.
                # This should ideally not happen in solvable problems with the fallback.
                # Return a large value to indicate a bad state.
                if min_steps_for_child == float('inf'):
                     min_steps_for_child = 1000 # Large penalty

                total_heuristic += min_steps_for_child

        return total_heuristic
