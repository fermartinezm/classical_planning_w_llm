# Assuming Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic
import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty facts or malformed strings gracefully
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Basic check for arity match
    if len(parts) != len(args):
        return False
    return all(fnmatch.fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the cost to serve all children by summing the estimated
    cost for each unserved child. The cost for an unserved child is estimated based
    on the current state of a suitable sandwich (gluten-free if needed) relative
    to the child's waiting location:
    - 1 if a suitable sandwich is on a tray at the child's location (needs serve).
    - 2 if a suitable sandwich is on a tray elsewhere (needs move + serve).
    - 3 if a suitable sandwich is in the kitchen (needs put on tray + move + serve).
    - 4 if a suitable sandwich needs to be made (needs make + put on tray + move + serve).
    If a suitable sandwich cannot be made due to lack of ingredients, the state is
    considered unsolvable (infinity).

    # Assumptions
    - All children in the goal must be served.
    - A sandwich is "suitable" for a child if it is gluten-free for an allergic child,
      or any sandwich for a non-allergic child.
    - There are always enough ingredients (bread, content) in the kitchen to make
      a required sandwich type, provided the corresponding `at_kitchen_bread` and
      `at_kitchen_content` facts exist for ingredients of the correct gluten status.
    - There is always at least one tray available at the kitchen when needed for `put_on_tray`.
    - Moving a tray from any location to any other location takes 1 action.
    - The heuristic sums the costs for each unserved child independently, ignoring
      potential resource contention (e.g., multiple children needing the same tray
      or specific ingredients). This makes it potentially non-admissible but possibly
      more informative for greedy search.
    - All potential sandwich objects that can be made are represented by `notexist`
      facts in the initial state, or are already present in the initial state.

    # Heuristic Initialization
    - Extracts the set of children that need to be served from the task goals.
    - Extracts static facts like `allergic_gluten`, `not_allergic_gluten`,
      `waiting`, `no_gluten_bread`, and `no_gluten_content` to determine child
      requirements, locations, and ingredient properties.
    - Extracts the set of all possible sandwich objects from the initial state
      (those that initially `notexist` or are already made).

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Initialize total heuristic cost to 0.
    2. Identify the set of children that are goals (i.e., need to be served) from `self.goal_children`.
    3. Pre-process the current state for efficient lookups:
       - Set of served children.
       - Set of sandwiches in the kitchen.
       - Dictionary mapping sandwiches on trays to their trays.
       - Dictionary mapping trays to their locations (using the `at` predicate).
       - Set of gluten-free sandwiches.
       - Set of sandwiches that `notexist`.
       - Set of bread and content objects currently in the kitchen.
    4. For each child `C` in `self.goal_children`:
        a. Check if `C` is in the set of served children from the state. If yes, the cost for this child is 0; continue to the next child.
        b. If `C` is not served, calculate the minimum estimated cost to serve this child:
            i. Get the child's waiting location `P` from `self.child_locations`.
            ii. Determine if the child is gluten-allergic from `self.child_allergies`.
            iii. Evaluate the state of the *best* suitable sandwich found (the one closest to being served), checking in this order:
                - **Check 1 (Ready at location):** Is there any sandwich `S` in the state such that `(ontray S T)` and `(at T P)` are true for some tray `T`, AND `S` is suitable for child `C` (i.e., if `C` is allergic, `(no_gluten_sandwich S)` is true in the state; otherwise, any `S` is suitable)?
                    - If yes, estimated cost for this child is 1 (serve).
                - **Check 2 (On tray elsewhere):** If Check 1 failed, is there any sandwich `S` in the state such that `(ontray S T)` is true for some tray `T`, AND `(at T P)` is NOT true, AND `S` is suitable for child `C`?
                    - If yes, estimated cost for this child is 2 (move + serve).
                - **Check 3 (In kitchen):** If Check 2 failed, is there any sandwich `S` in the state such that `(at_kitchen_sandwich S)` is true, AND `S` is suitable for child `C`?
                    - If yes, estimated cost for this child is 3 (put on tray + move + serve). (Assumes a tray is available at the kitchen).
                - **Check 4 (Needs making):** If Check 3 failed, is there any sandwich object `S_obj` from `self.all_sandwich_objects` that is currently `notexist` (`(notexist S_obj)` is true in the state)?
                    - If yes, check if suitable ingredients are available in the kitchen.
                        - If child is allergic, check if any bread in kitchen (`breads_in_kitchen_state`) is also in `self.no_gluten_breads` AND if any content in kitchen (`contents_in_kitchen_state`) is also in `self.no_gluten_contents`.
                        - If child is not allergic, check if any bread is in kitchen AND any content is in kitchen.
                    - If a `notexist` sandwich object exists AND suitable ingredients are in the kitchen, estimated cost for this child is 4 (make + put on tray + move + serve).
                    - Otherwise (cannot make suitable sandwich), the state is likely unsolvable from this point for this child. Return `float('inf')` for the total heuristic.
                - **Unsolvable/Edge Case:** If none of the above conditions are met for any sandwich (made or notexist), it implies a suitable sandwich object exists but is not in any expected state (kitchen, on tray, notexist) or no suitable sandwich object exists at all. This suggests an unsolvable state or problem definition issue. Return `float('inf')`.
        c. Add the estimated cost for child `C` to the total heuristic cost.
    5. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal children and static facts."""
        # The set of children that must be served.
        self.goal_children = {
            get_parts(goal)[1]
            for goal in task.goals
            if match(goal, "served", "*")
        }

        # Extract static facts about children (allergies, waiting locations)
        self.child_allergies = {} # child_name -> True if allergic, False otherwise
        self.child_locations = {} # child_name -> place_name

        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == "allergic_gluten":
                self.child_allergies[parts[1]] = True
            elif parts[0] == "not_allergic_gluten":
                 self.child_allergies[parts[1]] = False
            elif parts[0] == "waiting":
                 self.child_locations[parts[1]] = parts[2]

        # Extract static facts about ingredient properties
        self.no_gluten_breads = {
            get_parts(fact)[1]
            for fact in task.static
            if match(fact, "no_gluten_bread", "*")
        }
        self.no_gluten_contents = {
            get_parts(fact)[1]
            for fact in task.static
            if match(fact, "no_gluten_content", "*")
        }

        # Store all sandwich objects that exist in the problem (initially notexist or already made)
        # This set represents all sandwich instances that can ever exist.
        self.all_sandwich_objects = set()
        # Add sandwiches that initially notexist
        self.all_sandwich_objects.update({
             get_parts(fact)[1]
             for fact in task.initial_state
             if match(fact, "notexist", "*")
        })
        # Add sandwiches that might be pre-made in the initial state
        self.all_sandwich_objects.update({
             get_parts(fact)[1]
             for fact in task.initial_state
             if match(fact, "at_kitchen_sandwich", "*") or match(fact, "ontray", "*", "*")
        })


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        total_cost = 0

        # Pre-process state for quick lookups
        served_children_in_state = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        sandwiches_in_kitchen = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_sandwich", "*")}
        sandwiches_on_trays = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "ontray", "*", "*")} # Map sandwich -> tray
        # Tray locations: Filter out ingredient locations which also use 'at'
        tray_locations = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*") and get_parts(fact)[0] == 'at'} # Map tray -> place
        gluten_free_sandwiches_in_state = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}
        not_exist_sandwiches_in_state = {get_parts(fact)[1] for fact in state if match(fact, "notexist", "*")}
        breads_in_kitchen_state = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_bread", "*")}
        contents_in_kitchen_state = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_content", "*")}


        for child in self.goal_children:
            # If child is already served, no cost for this child
            if child in served_children_in_state:
                continue

            # Child needs to be served. Calculate cost for this child.
            child_cost = 0
            child_location = self.child_locations.get(child)
            is_allergic = self.child_allergies.get(child, False) # Default to not allergic if info missing

            # --- Check 1: Suitable sandwich ready at child's location? ---
            found_ready_sandwich = False
            for s in sandwiches_on_trays: # Iterate through sandwiches currently on trays
                tray = sandwiches_on_trays[s]
                if tray in tray_locations and tray_locations[tray] == child_location:
                    # Found a sandwich on a tray at the child's location
                    is_gf_sandwich = s in gluten_free_sandwiches_in_state
                    if (is_allergic and is_gf_sandwich) or (not is_allergic):
                        found_ready_sandwich = True
                        break # Found the best case for this child

            if found_ready_sandwich:
                child_cost = 1 # Needs 1: serve
            else:
                # --- Check 2: Suitable sandwich on tray elsewhere? ---
                found_on_tray_elsewhere = False
                for s in sandwiches_on_trays: # Iterate through sandwiches currently on trays
                     tray = sandwiches_on_trays[s]
                     if tray in tray_locations and tray_locations[tray] != child_location:
                         # Found a sandwich on a tray elsewhere
                         is_gf_sandwich = s in gluten_free_sandwiches_in_state
                         if (is_allergic and is_gf_sandwich) or (not is_allergic):
                             found_on_tray_elsewhere = True
                             break # Found the next best case

                if found_on_tray_elsewhere:
                     child_cost = 2 # Needs 2: move + serve
                else:
                    # --- Check 3: Suitable sandwich in kitchen? ---
                    found_in_kitchen = False
                    for s in sandwiches_in_kitchen: # Iterate through sandwiches in kitchen
                        is_gf_sandwich = s in gluten_free_sandwiches_in_state
                        if (is_allergic and is_gf_sandwich) or (not is_allergic):
                            found_in_kitchen = True
                            break # Found the next best case

                    if found_in_kitchen:
                         child_cost = 3 # Needs 3: put on tray + move + serve
                         # Assumes a tray is available at the kitchen.
                    else:
                        # --- Check 4: Suitable sandwich needs to be made? ---
                        # Check if there is any sandwich object that hasn't been made yet
                        # AND if we have suitable ingredients to make the required type.
                        can_make_any_sandwich_object = any(s_obj in not_exist_sandwiches_in_state for s_obj in self.all_sandwich_objects)

                        if can_make_any_sandwich_object:
                            # Check if suitable ingredients are available in the kitchen
                            has_suitable_bread_in_kitchen = False
                            has_suitable_content_in_kitchen = False

                            if is_allergic:
                                # Need GF bread and GF content in the kitchen
                                if any(b in self.no_gluten_breads for b in breads_in_kitchen_state):
                                    has_suitable_bread_in_kitchen = True
                                if any(c in self.no_gluten_contents for c in contents_in_kitchen_state):
                                    has_suitable_content_in_kitchen = True
                            else:
                                # Need any bread and any content in the kitchen
                                if breads_in_kitchen_state:
                                    has_suitable_bread_in_kitchen = True
                                if contents_in_kitchen_state:
                                    has_suitable_content_in_kitchen = True

                            if has_suitable_bread_in_kitchen and has_suitable_content_in_kitchen:
                                child_cost = 4 # Needs 4: make + put on tray + move + serve
                            else:
                                # Cannot make the sandwich due to lack of suitable ingredients
                                return float('inf') # Unsolvable from this state

                        else:
                            # No sandwich object exists that can be made (all are already made or don't exist as objects)
                            # This implies either all suitable sandwiches are already made and are not in kitchen/on tray (lost?),
                            # or there are no sandwich objects defined in the problem that are currently notexist.
                            # In a well-formed problem, if a child needs a sandwich and none exist, there should be a notexist object.
                            # If we reach here, and cannot make one, it's likely unsolvable.
                            return float('inf') # Unsolvable from this state


            total_cost += child_cost

        return total_cost
