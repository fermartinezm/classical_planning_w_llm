from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
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

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    Summary
    This heuristic estimates the number of actions required to serve all waiting children.
    It counts the necessary 'make', 'put_on_tray', 'move_tray', and 'serve' actions
    based on the current state of sandwiches, ingredients, trays, and children's needs.

    Assumptions
    - Each unserved child requires exactly one suitable sandwich.
    - Gluten-allergic children require gluten-free sandwiches; others can take any.
    - Sandwiches needed but not on trays must come from the kitchen (initial or newly made).
    - Trays can carry multiple sandwiches (no capacity limit).
    - A tray move is needed for each distinct location (other than the kitchen) that has
      unserved children but no tray currently present.
    - Sufficient ingredients and sandwich objects exist to make needed sandwiches
      (heuristic does not return infinity if ingredients are short, assuming solvability).

    Heuristic Initialization
    - Extracts static information: child allergy status and the set of all possible places.
    - Identifies the set of children who are initially waiting (these are the children who need to be served).

    Step-By-Step Thinking for Computing Heuristic
    1. Identify all children who are initially waiting and are not yet served in the current state.
    2. If no children are unserved, the heuristic is 0 (goal state).
    3. Count the total number of unserved children (`N_unserved`).
    4. Count the number of unserved children who are allergic (`N_allergic`) and non-allergic (`N_non_allergic`).
    5. Count available sandwiches currently `at_kitchen_sandwich` (GF and regular).
    6. Count available sandwiches currently `ontray` anywhere (GF and regular).
    7. Count available ingredients (`at_kitchen_bread`, `at_kitchen_content` - GF and regular) and `notexist` sandwich objects.
    8. Calculate the number of GF and regular sandwiches that *must* be made to satisfy the total demand (`Make_GF`, `Make_Reg`). This is the deficit of available sandwiches (anywhere) compared to the total needed, respecting allergy types. Cap the number of makes by the available ingredients and sandwich objects.
    9. The cost for 'make' actions is `Make_GF + Make_Reg`.
    10. Count the total number of sandwiches currently `ontray` (`S_ontray_total`).
    11. Count the number of sandwiches that are needed by children but are *not* currently on trays (`Needed_From_Kitchen = max(0, N_unserved - S_ontray_total)`). These must come from the kitchen and be put on trays.
    12. The cost for 'put_on_tray' actions is `Needed_From_Kitchen`.
    13. Identify all places where unserved children are waiting (`Waiting_places_set`).
    14. Identify all places where there is at least one tray (`Place_has_tray_set`).
    15. Count the number of distinct places in `Waiting_places_set` (excluding 'kitchen') that are *not* in `Place_has_tray_set`. Each such place requires at least one tray movement action to receive sandwiches.
    16. The cost for 'move_tray' actions is this count.
    17. The cost for 'serve' actions is `N_unserved`.
    18. The total heuristic value is the sum of costs for 'make', 'put_on_tray', 'move_tray', and 'serve'.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.
        """
        super().__init__(task)
        self.goals = task.goals # Goal conditions are (served ?c) for initially waiting children

        # Extract child allergy status and initial waiting places from static facts
        self.child_allergy = {} # Map child name to boolean (True if allergic)
        self.initial_waiting_places = {} # Map child name to initial waiting place
        self.all_places = {'kitchen'} # Start with kitchen constant

        # Static facts include initial waiting status and allergy info
        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'allergic_gluten':
                if len(parts) == 2:
                    child_name = parts[1]
                    self.child_allergy[child_name] = True
            elif predicate == 'not_allergic_gluten':
                 if len(parts) == 2:
                    child_name = parts[1]
                    self.child_allergy[child_name] = False
            elif predicate == 'waiting':
                 if len(parts) == 3:
                    child_name = parts[1]
                    place_name = parts[2]
                    self.initial_waiting_places[child_name] = place_name
                    self.all_places.add(place_name)
            elif predicate == 'at': # Find places mentioned in initial tray locations (if static)
                 if len(parts) == 3 and parts[1].startswith('tray'):
                     self.all_places.add(parts[2])

        # Also check initial state for places where trays or children might be
        # This ensures we capture all places mentioned in the problem instance
        for fact in task.initial_state:
             parts = get_parts(fact)
             if not parts: continue
             predicate = parts[0]
             if predicate == 'at':
                 if len(parts) == 3 and parts[1].startswith('tray'):
                     self.all_places.add(parts[2])
             elif predicate == 'waiting':
                 if len(parts) == 3 and parts[1].startswith('child'):
                      self.all_places.add(parts[2])


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # 1. Identify unserved children
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children_set = {
            child for child in self.initial_waiting_places
            if child not in served_children
        }

        N_unserved = len(unserved_children_set)

        # If all children are served, goal reached
        if N_unserved == 0:
            return 0

        # 2. Count unserved children by allergy status and identify waiting places
        unserved_allergic_count = 0
        unserved_non_allergic_count = 0
        waiting_places_set = set()

        for child in unserved_children_set:
            # Use .get() with default False in case a child is waiting but allergy status is missing (shouldn't happen in valid PDDL)
            if self.child_allergy.get(child, False):
                unserved_allergic_count += 1
            else:
                unserved_non_allergic_count += 1
            waiting_places_set.add(self.initial_waiting_places[child])


        # 3. Count available sandwiches and ingredients in the current state
        at_kitchen_bread_gf = 0
        at_kitchen_bread_reg = 0
        at_kitchen_content_gf = 0
        at_kitchen_content_reg = 0
        notexist_count = 0
        at_kitchen_sandwich_gf = 0
        at_kitchen_sandwich_reg = 0
        ontray_gf_total = 0
        ontray_reg_total = 0
        place_has_tray = set() # Places with at least one tray

        # Helper sets for quick lookup of GF status and sandwich locations
        is_gf_bread = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_bread", "*")}
        is_gf_content = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_content", "*")}
        is_gf_sandwich = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'at_kitchen_bread':
                if len(parts) == 2:
                    bread_name = parts[1]
                    if bread_name in is_gf_bread:
                        at_kitchen_bread_gf += 1
                    else:
                        at_kitchen_bread_reg += 1
            elif predicate == 'at_kitchen_content':
                 if len(parts) == 2:
                    content_name = parts[1]
                    if content_name in is_gf_content:
                        at_kitchen_content_gf += 1
                    else:
                        at_kitchen_content_reg += 1
            elif predicate == 'notexist':
                 if len(parts) == 2:
                    notexist_count += 1
            elif predicate == 'at_kitchen_sandwich':
                 if len(parts) == 2:
                    sandwich_name = parts[1]
                    if sandwich_name in is_gf_sandwich:
                        at_kitchen_sandwich_gf += 1
                    else:
                        at_kitchen_sandwich_reg += 1
            elif predicate == 'ontray':
                 if len(parts) == 3:
                     sandwich_name = parts[1]
                     # Tray name is parts[2], but we only need total count and location later
                     if sandwich_name in is_gf_sandwich:
                         ontray_gf_total += 1
                     else:
                         ontray_reg_total += 1
            elif predicate == 'at':
                 if len(parts) == 3:
                     obj_name = parts[1]
                     place_name = parts[2]
                     if obj_name.startswith('tray'):
                         place_has_tray.add(place_name)


        # 4. Calculate cost for 'make' actions
        Avail_GF_S_total = at_kitchen_sandwich_gf + ontray_gf_total
        Avail_Reg_S_total = at_kitchen_sandwich_reg + ontray_reg_total

        Can_Make_GF = min(at_kitchen_bread_gf, at_kitchen_content_gf, notexist_count)
        # Remaining notexist objects after potentially making GF sandwiches
        remaining_notexist = max(0, notexist_count - Can_Make_GF)
        Can_Make_Reg = min(at_kitchen_bread_reg, at_kitchen_content_reg, remaining_notexist)

        # Number of GF sandwiches that MUST be made to meet demand
        Make_GF_needed = max(0, unserved_allergic_count - Avail_GF_S_total)
        # Number of Reg sandwiches that MUST be made (after using available Reg and surplus GF)
        Surplus_Avail_GF = max(0, Avail_GF_S_total - unserved_allergic_count)
        Make_Reg_needed = max(0, unserved_non_allergic_count - (Avail_Reg_S_total + Surplus_Avail_GF))

        # Cap the number of makes by what's possible with ingredients/objects
        Make_GF = min(Make_GF_needed, Can_Make_GF)
        Make_Reg = min(Make_Reg_needed, Can_Make_Reg)

        cost_make = Make_GF + Make_Reg

        # 5. Calculate cost for 'put_on_tray' actions
        # These are needed for sandwiches that are in the kitchen (initial or made) and are needed by children.
        # Total sandwiches needed by children = N_unserved.
        # Sandwiches already on trays = S_ontray_total.
        # Sandwiches that must come from the kitchen = max(0, N_unserved - S_ontray_total).
        # These must be put on trays.
        S_ontray_total = ontray_gf_total + ontray_reg_total
        Needed_From_Kitchen = max(0, N_unserved - S_ontray_total)
        cost_put_on_tray = Needed_From_Kitchen # Assumes needed sandwiches are available in kitchen or can be made

        # 6. Calculate cost for 'move_tray' actions
        # Count distinct waiting places (excluding kitchen) that do not have a tray.
        waiting_places_needing_delivery_count = 0
        for place in waiting_places_set:
            if place != 'kitchen' and place not in place_has_tray:
                 waiting_places_needing_delivery_count += 1

        cost_move_tray = waiting_places_needing_delivery_count

        # 7. Calculate cost for 'serve' actions
        cost_serve = N_unserved

        # Total heuristic is the sum of estimated actions
        total_heuristic = cost_make + cost_put_on_tray + cost_move_tray + cost_serve

        return total_heuristic
