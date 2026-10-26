from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove surrounding parentheses and split by whitespace
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all unserved children.
    It calculates an independent cost for each unserved child based on the "closest"
    available suitable sandwich, summing these individual costs. It does not explicitly
    model resource contention (like limited trays or ingredients) beyond checking
    for the *existence* of resources needed to make a sandwich.

    # Assumptions
    - The goal is to serve all children.
    - Children are static (do not move).
    - Allergy status of children is static.
    - Gluten status of bread and content portions is static.
    - Waiting locations of children are static.
    - Once a sandwich is made, its gluten status is fixed.
    - Solvable problems have sufficient initial resources (bread, content, notexist sandwiches)
      to make all necessary sandwiches.
    - The cost of actions is 1.
    - The constant place 'kitchen' is named 'kitchen'.

    # Heuristic Initialization
    - Extract static-like information from initial state and static facts:
        - Allergy status for each child.
        - Waiting location for each child.
        - Gluten status for each bread and content portion.
        - The initial count of 'notexist' sandwich objects (limits total sandwiches).
    - Extract lists of all objects by type.
    - Identify the 'kitchen' place object (assumed name 'kitchen').

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify which children are already served.
    2. For each child that is *not* served:
       a. Determine the child's waiting location and allergy status (from initialization).
       b. Determine the type of sandwich needed (gluten-free if allergic, any if not).
       c. Find the "closest" available suitable sandwich for this child by checking in stages, from closest to furthest/most expensive to prepare:
          i.  **Stage 1 (Cost 1):** Is there a suitable sandwich currently on *any* tray that is located at the child's waiting place? (Requires 1 'serve' action).
          ii. **Stage 2 (Cost 2):** If not Stage 1, is there a suitable sandwich currently on *any* tray that is located at a *different* place? (Requires 1 'move_tray' + 1 'serve' action).
          iii. **Stage 3 (Cost 3):** If not Stage 1 or 2, is there a suitable sandwich currently in the kitchen? (Requires 1 'put_on_tray' + 1 'move_tray' + 1 'serve' action).
          iv. **Stage 4 (Cost 4):** If not Stage 1, 2, or 3, can a suitable sandwich be made from available ingredients and a 'notexist' sandwich object in the kitchen? (Requires 1 'make_sandwich' + 1 'put_on_tray' + 1 'move_tray' + 1 'serve' action). This check is simplified: it only verifies if *any* required ingredients and a 'notexist' object exist, not if there are enough for *all* children needing this stage.
          v. **Stage 5 (Cost 5):** If a suitable sandwich cannot be found or made (based on the simplified check). This represents a difficult or potentially unsolvable situation for this child.
       d. The minimum cost found across these stages is the estimated cost for this child.
    3. The total heuristic value is the sum of the estimated minimum costs for all unserved children.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static and initial state information."""
        self.goals = task.goals # Not strictly needed for this heuristic logic

        # Extract all objects by type
        self.all_children = {obj for obj, obj_type in task.objects if obj_type == 'child'}
        self.all_sandwiches = {obj for obj, obj_type in task.objects if obj_type == 'sandwich'}
        self.all_trays = {obj for obj, obj_type in task.objects if obj_type == 'tray'}
        self.all_places = {obj for obj, obj_type in task.objects if obj_type == 'place'}
        self.all_bread = {obj for obj, obj_type in task.objects if obj_type == 'bread-portion'}
        self.all_content = {obj for obj, obj_type in task.objects if obj_type == 'content-portion'}

        # Extract static-like information from initial state and static facts
        # Combine facts from task.init and task.static as they both contain static-like info
        static_like_facts = set(task.init) | set(task.static)

        self.child_is_allergic = {}
        self.child_waiting_loc = {}
        self.bread_is_gf_static = {}
        self.content_is_gf_static = {}
        self.initial_notexist_count = 0
        self.kitchen_place = 'kitchen' # Assume 'kitchen' is the name of the constant place

        for fact in static_like_facts:
             parts = get_parts(fact)
             if not parts: continue # Skip empty or malformed facts
             predicate = parts[0]

             if predicate == 'allergic_gluten' and len(parts) > 1:
                 self.child_is_allergic[parts[1]] = True
             elif predicate == 'not_allergic_gluten' and len(parts) > 1:
                 self.child_is_allergic[parts[1]] = False
             elif predicate == 'waiting' and len(parts) > 2:
                  child, place = parts[1], parts[2]
                  self.child_waiting_loc[child] = place
             elif predicate == 'no_gluten_bread' and len(parts) > 1:
                 self.bread_is_gf_static[parts[1]] = True
             elif predicate == 'no_gluten_content' and len(parts) > 1:
                 self.content_is_gf_static[parts[1]] = True
             elif predicate == 'notexist' and len(parts) > 1:
                  self.initial_notexist_count += 1


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings)

        # Dynamic state information
        served_children = set()
        sandwich_current_loc = {} # {s: 'kitchen' or ('ontray', t)}
        tray_current_loc = {}     # {t: p}
        sandwich_is_gf_dynamic = {} # {s: True} (only if no_gluten_sandwich is present)
        notexist_sandwiches_dynamic = set()
        kitchen_bread_dynamic = set()
        kitchen_content_dynamic = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts
            predicate = parts[0]

            if predicate == 'served' and len(parts) > 1:
                served_children.add(parts[1])
            elif predicate == 'at_kitchen_sandwich' and len(parts) > 1:
                sandwich_current_loc[parts[1]] = 'kitchen'
            elif predicate == 'ontray' and len(parts) > 2:
                sandwich, tray = parts[1], parts[2]
                sandwich_current_loc[sandwich] = ('ontray', tray)
            elif predicate == 'at' and len(parts) > 2:
                obj, place = parts[1], parts[2]
                # Only track tray locations, ignore other 'at' predicates if any
                if obj in self.all_trays:
                    tray_current_loc[obj] = place
            elif predicate == 'no_gluten_sandwich' and len(parts) > 1:
                sandwich_is_gf_dynamic[parts[1]] = True
            elif predicate == 'notexist' and len(parts) > 1:
                notexist_sandwiches_dynamic.add(parts[1])
            elif predicate == 'at_kitchen_bread' and len(parts) > 1:
                kitchen_bread_dynamic.add(parts[1])
            elif predicate == 'at_kitchen_content' and len(parts) > 1:
                kitchen_content_dynamic.add(parts[1])

        # Calculate dynamic ingredient counts
        dynamic_gf_bread_count = sum(1 for b in kitchen_bread_dynamic if self.bread_is_gf_static.get(b, False))
        dynamic_gf_content_count = sum(1 for c in kitchen_content_dynamic if self.content_is_gf_static.get(c, False))
        dynamic_reg_bread_count = len(kitchen_bread_dynamic) - dynamic_gf_bread_count
        dynamic_reg_content_count = len(kitchen_content_dynamic) - dynamic_gf_content_count
        dynamic_notexist_count = len(notexist_sandwiches_dynamic)

        # Simplified makeable checks
        # Can make GF if there's a notexist sandwich, GF bread, and GF content in the kitchen
        can_make_gf = dynamic_notexist_count > 0 and dynamic_gf_bread_count > 0 and dynamic_gf_content_count > 0

        # Can make Any sandwich if there's a notexist sandwich, any bread, and any content in the kitchen
        total_bread_count = dynamic_gf_bread_count + dynamic_reg_bread_count
        total_content_count = dynamic_gf_content_count + dynamic_reg_content_count
        can_make_any_sandwich = dynamic_notexist_count > 0 and total_bread_count > 0 and total_content_count > 0


        total_heuristic_cost = 0

        # Iterate through all children to find unserved ones
        for child in self.all_children:
            if child not in served_children:
                # This child needs to be served
                child_loc = self.child_waiting_loc.get(child)
                # If child_loc is None, something is wrong with initial state parsing or problem definition
                if child_loc is None:
                     # Cannot serve a child without a waiting location. Treat as high cost.
                     total_heuristic_cost += 5
                     continue

                is_allergic = self.child_is_allergic.get(child, False) # Default to not allergic if status missing

                min_child_cost = 5 # Initialize with a cost higher than max possible stage cost (4)

                # Check for suitable sandwiches in stages
                found_suitable_sandwich_option = False

                # Stage 1 (Cost 1): On tray at child's location
                for s in self.all_sandwiches:
                    s_loc_info = sandwich_current_loc.get(s)
                    if s_loc_info and s_loc_info[0] == 'ontray':
                        tray = s_loc_info[1]
                        tray_loc = tray_current_loc.get(tray)
                        if tray_loc == child_loc:
                            # Check suitability
                            s_is_gf = sandwich_is_gf_dynamic.get(s, False)
                            if (is_allergic and s_is_gf) or (not is_allergic):
                                min_child_cost = 1
                                found_suitable_sandwich_option = True
                                break # Found best stage for this child

                if found_suitable_sandwich_option:
                    total_heuristic_cost += min_child_cost
                    continue # Move to the next unserved child

                # Stage 2 (Cost 2): On tray elsewhere
                for s in self.all_sandwiches:
                    s_loc_info = sandwich_current_loc.get(s)
                    if s_loc_info and s_loc_info[0] == 'ontray':
                        tray = s_loc_info[1]
                        tray_loc = tray_current_loc.get(tray)
                        # Check if tray is at a known location, but not the child's location
                        if tray_loc is not None and tray_loc != child_loc:
                            # Check suitability
                            s_is_gf = sandwich_is_gf_dynamic.get(s, False)
                            if (is_allergic and s_is_gf) or (not is_allergic):
                                min_child_cost = 2
                                found_suitable_sandwich_option = True
                                break # Found best stage for this child

                if found_suitable_sandwich_option:
                    total_heuristic_cost += min_child_cost
                    continue # Move to the next unserved child

                # Stage 3 (Cost 3): In kitchen
                for s in self.all_sandwiches:
                    s_loc_info = sandwich_current_loc.get(s)
                    if s_loc_info == 'kitchen':
                        # Check suitability
                        s_is_gf = sandwich_is_gf_dynamic.get(s, False)
                        if (is_allergic and s_is_gf) or (not is_allergic):
                            min_child_cost = 3
                            found_suitable_sandwich_option = True
                            break # Found best stage for this child

                if found_suitable_sandwich_option:
                    total_heuristic_cost += min_child_cost
                    continue # Move to the next unserved child

                # Stage 4 (Cost 4): Makeable
                can_make_suitable = False
                if is_allergic: # Needs GF
                    if can_make_gf:
                        can_make_suitable = True
                else: # Needs Any
                    if can_make_any_sandwich: # Can use GF or Reg ingredients
                         can_make_suitable = True

                if can_make_suitable:
                    min_child_cost = 4
                    found_suitable_sandwich_option = True # Found a way to get a sandwich

                # If a suitable sandwich was found or can be made, add its cost
                # Otherwise, min_child_cost remains 5 (or higher)
                total_heuristic_cost += min_child_cost

        return total_heuristic_cost
