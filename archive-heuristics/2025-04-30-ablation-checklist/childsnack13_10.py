from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings defensively
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by spaces
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding pattern argument
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class ChildsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Childsnack domain.

    # Summary
    This heuristic estimates the number of actions required to serve all unserved
    children. It counts the remaining "tasks" at different stages of the
    snack delivery pipeline: making sandwiches, putting them on trays,
    moving trays to children's locations, and finally serving the children.

    # Assumptions
    - Each unserved child requires one suitable sandwich.
    - Each sandwich needs to be made (if not already), put on a tray (if not already),
      and the tray needs to be at the child's location (if not already).
    - The heuristic sums the deficit at each stage, providing a non-admissible estimate.
    - Resource limits (bread, content, notexist sandwich objects) are considered
      only for the "make sandwich" stage. Tray availability for "put on tray"
      and "move tray" is simplified.

    # Heuristic Initialization
    - Extracts goal children and their allergy status and waiting locations
      from the task's goals and static facts.
    - Identifies static gluten-free bread/content types.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic estimates the remaining cost as the sum of:
    1.  **Serve Actions:** The number of unserved children. Each needs one serve action.
    2.  **Tray Movement Actions:** The number of unique locations where unserved
        children are waiting, but no tray is currently present. Each such location
        requires at least one tray movement action.
    3.  **Put on Tray Actions:** The number of sandwiches that still need to be
        put on trays to satisfy the demand from unserved children, minus those
        already on trays.
    4.  **Make Sandwich Actions:** The number of sandwiches that still need to be
        made to satisfy the demand from unserved children, minus those already made.
        This count is limited by the available bread, content, and `notexist`
        sandwich objects in the kitchen.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal information and static facts.
        """
        self.goals = task.goals
        self.static = task.static

        # Pre-process static facts for quick lookup
        self.child_info = {}
        self.gluten_free_bread_types = set()
        self.gluten_free_content_types = set()

        for fact in self.static:
            parts = get_parts(fact)
            if not parts:
                continue # Skip malformed facts

            predicate = parts[0]

            if predicate == "waiting" and len(parts) == 3:
                child, place = parts[1], parts[2]
                if child not in self.child_info:
                    self.child_info[child] = {'place': place, 'allergic': False} # Default
                else:
                     self.child_info[child]['place'] = place # Update if child already seen

            elif predicate == "allergic_gluten" and len(parts) == 2:
                 child = parts[1]
                 if child not in self.child_info:
                     self.child_info[child] = {'place': None, 'allergic': True} # Default place None
                 else:
                     self.child_info[child]['allergic'] = True

            elif predicate == "not_allergic_gluten" and len(parts) == 2:
                 child = parts[1]
                 if child not in self.child_info:
                     self.child_info[child] = {'place': None, 'allergic': False} # Default place None
                 else:
                     self.child_info[child]['allergic'] = False

            elif predicate == "no_gluten_bread" and len(parts) == 2:
                self.gluten_free_bread_types.add(parts[1])

            elif predicate == "no_gluten_content" and len(parts) == 2:
                self.gluten_free_content_types.add(parts[1])

        # Identify all children who are goals (need to be served)
        self.goal_children = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}


    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state
        cost = 0

        # --- Step 1: Count unserved children (cost for serve actions) ---
        served_children_in_state = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children = {
            child for child in self.goal_children if child not in served_children_in_state
        }
        num_unserved_total = len(unserved_children)

        if num_unserved_total == 0:
            return 0 # Goal reached

        cost += num_unserved_total # Each unserved child needs a 'serve' action

        # Separate unserved children by allergy status
        unserved_allergy = {c for c in unserved_children if self.child_info.get(c, {}).get('allergic', False)}
        unserved_nonallergy = unserved_children - unserved_allergy
        num_unserved_allergy = len(unserved_allergy)
        num_unserved_nonallergy = len(unserved_nonallergy)


        # --- Step 2: Count locations needing trays (cost for move_tray actions) ---
        locations_with_unserved = {self.child_info.get(child, {}).get('place') for child in unserved_children if self.child_info.get(child, {}).get('place') is not None}
        trays_at_locations = {get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*") and get_parts(fact)[1].startswith("tray")}
        locations_needing_tray = locations_with_unserved - trays_at_locations
        # Don't count 'kitchen' as needing a tray move if children aren't waiting there
        locations_needing_tray.discard('kitchen')

        cost += len(locations_needing_tray) # Each location needs a tray moved there


        # --- Step 3: Count sandwiches to put on trays (cost for put_on_tray actions) ---
        # We need num_unserved_total sandwiches on trays eventually.
        # Count how many are already on trays, respecting allergy type if possible.
        sandwiches_on_trays = {}
        for fact in state:
            if match(fact, "ontray", "*", "*"):
                s_name = get_parts(fact)[1]
                # Check if this sandwich is gluten-free
                is_gf = "(no_gluten_sandwich " + s_name + ")" in state
                sandwiches_on_trays[s_name] = is_gf

        sandwiches_on_trays_gf = sum(1 for is_gf in sandwiches_on_trays.values() if is_gf)
        sandwiches_on_trays_reg = len(sandwiches_on_trays) - sandwiches_on_trays_gf

        # How many GF/Reg sandwiches do we still need to put on trays?
        needed_on_tray_gf = num_unserved_allergy
        needed_on_tray_reg = num_unserved_nonallergy

        to_put_on_tray_gf = max(0, needed_on_tray_gf - sandwiches_on_trays_gf)
        to_put_on_tray_reg = max(0, needed_on_tray_reg - sandwiches_on_trays_reg)

        cost += to_put_on_tray_gf + to_put_on_tray_reg # Each needs a 'put_on_tray' action


        # --- Step 4: Count sandwiches to make (cost for make_sandwich actions), limited by resources ---
        # We need num_unserved_total sandwiches made eventually.
        # Count how many are already made (either in kitchen or on tray).
        sandwiches_made = {}
        for fact in state:
             if match(fact, "at_kitchen_sandwich", "*"):
                 s_name = get_parts(fact)[1]
                 is_gf = "(no_gluten_sandwich " + s_name + ")" in state
                 sandwiches_made[s_name] = is_gf
             elif match(fact, "ontray", "*", "*"):
                 s_name = get_parts(fact)[1]
                 is_gf = "(no_gluten_sandwich " + s_name + ")" in state
                 sandwiches_made[s_name] = is_gf # Already counted if on tray

        sandwiches_made_gf_total = sum(1 for is_gf in sandwiches_made.values() if is_gf)
        sandwiches_made_reg_total = len(sandwiches_made) - sandwiches_made_gf_total

        # How many GF/Reg sandwiches do we still need to make?
        needed_made_gf = num_unserved_allergy
        needed_made_reg = num_unserved_nonallergy

        to_make_gf_deficit = max(0, needed_made_gf - sandwiches_made_gf_total)
        to_make_reg_deficit = max(0, needed_made_reg - sandwiches_made_reg_total)

        # Limit 'to_make' by available resources in the kitchen
        bread_kitchen = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_bread", "*")}
        content_kitchen = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_content", "*")}
        sandwich_objects_notexist = {get_parts(fact)[1] for fact in state if match(fact, "notexist", "*")}

        bread_gf_kitchen = {b for b in bread_kitchen if b in self.gluten_free_bread_types}
        content_gf_kitchen = {c for c in content_kitchen if c in self.gluten_free_content_types}

        num_bread_gf_k = len(bread_gf_kitchen)
        num_content_gf_k = len(content_gf_kitchen)
        num_bread_reg_k = len(bread_kitchen) - num_bread_gf_k
        num_content_reg_k = len(content_kitchen) - num_content_gf_k
        num_s_notexist = len(sandwich_objects_notexist)

        # Calculate how many GF sandwiches we can actually make
        can_make_gf = min(num_bread_gf_k, num_content_gf_k, num_s_notexist)
        actual_to_make_gf = min(to_make_gf_deficit, can_make_gf)

        # Calculate how many Regular sandwiches we can actually make with remaining resources
        remaining_s_notexist = num_s_notexist - actual_to_make_gf
        can_make_reg = min(num_bread_reg_k, num_content_reg_k, remaining_s_notexist)
        actual_to_make_reg = min(to_make_reg_deficit, can_make_reg)

        cost += actual_to_make_gf + actual_to_make_reg # Each needs a 'make_sandwich' action


        return cost

