from fnmatch import fnmatch
# Assuming heuristics.heuristic_base is available in the environment
# If not, you might need to define a dummy base class or remove inheritance
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy base class if the actual one is not available
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError("Subclasses must implement __call__")
        def __str__(self):
            return self.__class__.__name__
        def __repr__(self):
            return self.__str__()


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and not empty
    if not isinstance(fact, str) or not fact:
        return []
    # Remove outer parentheses and split by spaces
    # Handle cases like "(at tray1 kitchen)" -> ['at', 'tray1', 'kitchen']
    # Handle cases like "(served child1)" -> ['served', 'child1']
    parts = fact[1:-1].split()
    return parts

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj1 loc1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the total minimum number of actions required to serve all unserved children.
    For each unserved child, it estimates the cost based on the current location and availability
    of a suitable sandwich (on a tray at their location, on a tray elsewhere, in the kitchen,
    or needing to be made from ingredients). The total heuristic is the sum of these minimum
    costs for all unserved children.

    # Assumptions
    - Ingredients in the kitchen are sufficient *in total* across the problem to make needed sandwiches
      (the heuristic checks current kitchen stock but doesn't track consumption across children).
    - A tray is always available at the kitchen to put a sandwich on if needed.
    - Any tray can be moved to any place in one action.
    - Non-allergic children can eat any sandwich (regular or gluten-free).
    - Allergic children require a gluten-free sandwich.

    # Heuristic Initialization
    The heuristic extracts static information from the task:
    - The set of all children.
    - The waiting place for each child.
    - Which children are allergic to gluten.
    - Which bread and content portions are gluten-free (these facts are static).

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is calculated as follows:
    1. Initialize the total heuristic cost `h` to 0.
    2. Count the currently available bread and content portions in the kitchen from the current state, distinguishing
       between gluten-free and regular based on static facts identified during initialization.
    3. Identify all sandwiches that currently exist (either in the kitchen or on a tray)
       and determine which ones are gluten-free based on facts in the current state.
    4. Iterate through each child identified during initialization:
       - If the child is already marked as `served` in the current state, their contribution to the heuristic is 0.
       - If the child is not served:
         - Determine the child's waiting place `?p` and allergy status.
         - Calculate the minimum estimated cost `cost_c` to serve this specific child:
           - Initialize `cost_c` to a large value (e.g., 1000, representing a high cost or penalty).
           - **Check 1: Suitable sandwich on a tray at the child's location `?p`?**
             - Look for a sandwich `?s` on a tray `?t` (`(ontray ?s ?t)` in state) where the tray `?t` is at the child's location `?p` (`(at ?t ?p)` in state).
             - If the child is allergic, check if `?s` is gluten-free (`(no_gluten_sandwich ?s)` in state). If not allergic, any sandwich is suitable.
             - If a suitable sandwich is found on a tray at `?p`, set `cost_c = 1` (representing the `serve` action).
           - **Check 2: Suitable sandwich on a tray at a different location `?p'`?**
             - If `cost_c` is still high (i.e., > 1), look for a suitable sandwich `?s` on a tray `?t` (`(ontray ?s ?t)` in state) where the tray `?t` is at a location `?p'` different from `?p` (`(at ?t ?p')` in state, `?p' != ?p`).
             - Check suitability as in Check 1.
             - If found, set `cost_c = 2` (representing `move_tray` from `?p'` to `?p` + `serve`).
           - **Check 3: Suitable sandwich in the kitchen?**
             - If `cost_c` is still high (i.e., > 2), look for a suitable sandwich `?s` in the kitchen (`(at_kitchen_sandwich ?s)` in state).
             - Check suitability as in Check 1.
             - If found, set `cost_c = 1` (`put_on_tray`) + (1 if `?p` != "kitchen" else 0) (`move_tray`) + 1 (`serve`).
           - **Check 4: Can a suitable sandwich be made from ingredients in the kitchen?**
             - If `cost_c` is still high (i.e., > 3 if `?p` is not kitchen, or > 2 if `?p` is kitchen), check if the necessary ingredients are available in the kitchen based on the counts from step 2.
             - If the child is allergic, check if there is at least one GF bread and one GF content portion available.
             - If the child is not allergic, check if there is at least one bread (GF or regular) and one content portion (GF or regular) available.
             - If ingredients are available, set `cost_c = 1` (`make_sandwich`) + 1 (`put_on_tray`) + (1 if `?p` != "kitchen" else 0) (`move_tray`) + 1 (`serve`).
           - **Handle Unsolvable Case:** If `cost_c` remains at the initial large value (e.g., 1000) after all checks, it implies no suitable sandwich exists or can be made with current kitchen ingredients. This state is likely a dead end for this child. The initial large value of `cost_c` serves as the penalty for this child.
         - Add the calculated `cost_c` to the total heuristic `h`.
    5. Return the total heuristic cost `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static # Static facts are provided

        self.children = set()
        self.waiting_places = {}  # child -> place
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.gluten_free_breads_static = set()  # bread objects that are GF
        self.gluten_free_contents_static = set() # content objects that are GF

        # Extract static information from the static facts
        for fact in self.static:
             parts = get_parts(fact)
             if not parts: continue # Skip empty or invalid facts

             predicate = parts[0]
             if predicate == "waiting" and len(parts) == 3:
                 child, place = parts[1:]
                 self.children.add(child)
                 self.waiting_places[child] = place
             elif predicate == "allergic_gluten" and len(parts) == 2:
                 child = parts[1]
                 self.allergic_children.add(child)
             elif predicate == "not_allergic_gluten" and len(parts) == 2:
                 child = parts[1]
                 self.not_allergic_children.add(child)
             elif predicate == "no_gluten_bread" and len(parts) == 2:
                 bread = parts[1]
                 self.gluten_free_breads_static.add(bread)
             elif predicate == "no_gluten_content" and len(parts) == 2:
                 content = parts[1]
                 self.gluten_free_contents_static.add(content)

        # It's good practice to ensure all children mentioned in waiting are accounted for
        # in allergy lists, though PDDL instances should ideally be well-formed.
        # We rely on self.allergic_children and self.not_allergic_children for allergy status.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0
        penalty = 1000 # Large penalty for seemingly unsolvable subgoals

        # Count available ingredients in kitchen from current state
        available_bread_kitchen = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_bread", "*")}
        available_content_kitchen = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_content", "*")}

        num_gf_bread_kitchen = sum(1 for b in available_bread_kitchen if b in self.gluten_free_breads_static)
        num_reg_bread_kitchen = len(available_bread_kitchen) - num_gf_bread_kitchen
        num_gf_content_kitchen = sum(1 for c in available_content_kitchen if c in self.gluten_free_contents_static)
        num_reg_content_kitchen = len(available_content_kitchen) - num_gf_content_kitchen

        # Collect existing sandwiches and their GF status from current state
        existing_sandwiches = set()
        gf_sandwiches_in_state = set()
        sandwiches_on_trays = {} # sandwich -> tray
        trays_at_places = {} # tray -> place

        for fact in state:
            if match(fact, "at_kitchen_sandwich", "?s"):
                s = get_parts(fact)[1]
                existing_sandwiches.add(s)
            elif match(fact, "ontray", "?s", "?t"):
                 s, t = get_parts(fact)[1:3]
                 existing_sandwiches.add(s)
                 sandwiches_on_trays[s] = t
            elif match(fact, "at", "?t", "?p"):
                 t, p = get_parts(fact)[1:3]
                 trays_at_places[t] = p

        # Determine GF status for existing sandwiches by checking state
        for s in existing_sandwiches:
             if f"(no_gluten_sandwich {s})" in state:
                 gf_sandwiches_in_state.add(s)

        # Calculate cost for each unserved child
        for child in self.children:
            if f"(served {child})" not in state:
                place = self.waiting_places.get(child)
                if place is None:
                    # Child is waiting but place not found? Should not happen in valid PDDL.
                    # Treat as unsolvable for this child.
                    h += penalty
                    continue

                is_allergic = child in self.allergic_children
                min_cost_c = penalty # Initialize with penalty

                # Check 1: Suitable sandwich on a tray at the child's location ?p
                found_suitable_at_p = False
                for s in existing_sandwiches:
                    if s in sandwiches_on_trays:
                        t = sandwiches_on_trays[s]
                        if trays_at_places.get(t) == place:
                            is_gf_sandwich = s in gf_sandwiches_in_state
                            if (is_allergic and is_gf_sandwich) or (not is_allergic):
                                min_cost_c = 1 # serve
                                found_suitable_at_p = True
                                break # Found the best case for this child
                if found_suitable_at_p:
                    h += min_cost_c
                    continue # Move to the next child

                # Check 2: Suitable sandwich on a tray at a different location ?p'
                found_suitable_elsewhere = False
                for s in existing_sandwiches:
                    if s in sandwiches_on_trays:
                        t = sandwiches_on_trays[s]
                        tray_place = trays_at_places.get(t)
                        if tray_place is not None and tray_place != place:
                            is_gf_sandwich = s in gf_sandwiches_in_state
                            if (is_allergic and is_gf_sandwich) or (not is_allergic):
                                min_cost_c = 1 + 1 # move_tray + serve
                                found_suitable_elsewhere = True
                                break # Found the next best case
                if found_suitable_elsewhere:
                    h += min_cost_c
                    continue # Move to the next child

                # Check 3: Suitable sandwich in the kitchen
                found_suitable_in_kitchen = False
                for s in existing_sandwiches:
                    # Check if sandwich is at_kitchen_sandwich
                    if f"(at_kitchen_sandwich {s})" in state:
                        is_gf_sandwich = s in gf_sandwiches_in_state
                        if (is_allergic and is_gf_sandwich) or (not is_allergic):
                            cost = 1 # put_on_tray
                            cost += (1 if place != "kitchen" else 0) # move_tray
                            cost += 1 # serve
                            min_cost_c = cost
                            found_suitable_in_kitchen = True
                            break # Found the next best case
                if found_suitable_in_kitchen:
                    h += min_cost_c
                    continue # Move to the next child

                # Check 4: Can a suitable sandwich be made from ingredients in the kitchen?
                ingredients_available_to_make = False
                if is_allergic:
                    # Need GF bread and GF content
                    if num_gf_bread_kitchen > 0 and num_gf_content_kitchen > 0:
                        ingredients_available_to_make = True
                else:
                    # Need any bread and any content
                    if (num_gf_bread_kitchen + num_reg_bread_kitchen > 0) and \
                       (num_gf_content_kitchen + num_reg_content_kitchen > 0):
                        ingredients_available_to_make = True

                if ingredients_available_to_make:
                    cost = 1 # make_sandwich
                    cost += 1 # put_on_tray
                    cost += (1 if place != "kitchen" else 0) # move_tray
                    cost += 1 # serve
                    min_cost_c = cost
                    # Add the cost for this child
                    h += min_cost_c
                    continue # Move to the next child
                else:
                    # If we reach here, min_cost_c is still the initial penalty value.
                    # This child cannot be served from this state based on our simple checks.
                    h += min_cost_c # Add the penalty

        return h
