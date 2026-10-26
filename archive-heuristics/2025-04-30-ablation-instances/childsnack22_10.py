from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Define helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
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
    # The number of parts in the fact must match the number of arguments in the pattern.
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting children.
    It sums up the estimated costs for four main stages for the collective set of unserved children:
    1. Making necessary sandwiches.
    2. Putting sandwiches onto trays.
    3. Moving trays to the children's locations.
    4. Serving the children.

    # Assumptions
    - Each child needs one sandwich.
    - Gluten-allergic children require gluten-free sandwiches.
    - Sandwiches must be on a tray at the child's location to be served.
    - A tray can hold at least one sandwich needed for a child at its location.
    - Ingredients and 'notexist' sandwich instances are sufficient to make needed sandwiches if they are not currently available.
    - Trays are available to be moved to the kitchen or child locations when needed for putting sandwiches on or for delivery.
    - The cost of each 'unit' of action (making one sandwich, putting one sandwich on a tray, moving one tray for one delivery, serving one child) is 1.

    # Heuristic Initialization
    The heuristic extracts static information from the task:
    - The set of children who need to be served (from task goals).
    - The initial waiting location for each child (from static facts).
    - The allergy status (allergic_gluten) for each child (from static facts).
    - Note: The gluten-free status of sandwiches is a state property, not static, and is checked in __call__.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:

    1.  **Cost_serve:** Count the number of children who were initially waiting but are not yet marked as 'served' in the current state. This is the number of 'serve' actions remaining.

    2.  **Cost_make:**
        - Count the number of unserved allergic children (who need GF sandwiches) and unserved non-allergic children (who need Regular sandwiches).
        - Count the total number of suitable sandwiches (GF for allergic, Regular for non-allergic) that already exist in the current state (either in the kitchen or on trays).
        - The number of GF sandwiches that still need to be made is `max(0, Num_unserved_allergic - Num_available_GF_sandwiches)`.
        - The number of Regular sandwiches that still need to be made is `max(0, Num_unserved_non_allergic - Num_available_Regular_sandwiches)`.
        - The cost for making sandwiches is the sum of these two counts.

    3.  **Cost_put_on_tray:**
        - The total number of sandwiches that ultimately need to be on trays is the total number of unserved children.
        - Count the number of sandwiches that are already on trays in the current state.
        - The number of sandwiches that still need to be put on trays is `max(0, Total_unserved_children - Total_sandwiches_on_trays)`. This estimates the number of 'put_on_tray' actions needed, assuming trays are available in the kitchen.

    4.  **Cost_move_tray:**
        - For each unserved child at their waiting location `p`, check if there is already a suitable sandwich on a tray located at `p` in the current state.
        - Count the number of unserved children for whom no such suitable sandwich on a tray exists at their location.
        - Each such child represents a requirement for a sandwich-carrying tray to arrive at their location. This count estimates the number of 'move_tray' actions needed for these deliveries.

    The total heuristic value is the sum of Cost_serve + Cost_make + Cost_put_on_tray + Cost_move_tray.
    If the state is a goal state (all children served), the heuristic is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.task_goals = task.goals
        self.static_facts = task.static

        # Identify all children who need serving (those mentioned in goals)
        self.children_to_serve = {get_parts(goal)[1] for goal in self.task_goals if match(goal, "served", "*")}

        # Map children to their initial waiting locations and allergy status
        self.child_info = {} # {child_name: {'location': place, 'allergic': bool}}
        # Get all waiting facts from static
        waiting_facts = {fact for fact in self.static_facts if match(fact, "waiting", "*", "*")}
        for fact in waiting_facts:
            child, place = get_parts(fact)[1:]
            if child in self.children_to_serve: # Only track info for children who need serving
                self.child_info[child] = {'location': place}

        # Get allergy info from static
        allergic_facts = {fact for fact in self.static_facts if match(fact, "allergic_gluten", "*")}
        not_allergic_facts = {fact for fact in self.static_facts if match(fact, "not_allergic_gluten", "*")}

        for child in self.children_to_serve:
             if child in self.child_info: # Ensure we only process children we care about
                if f"(allergic_gluten {child})" in allergic_facts:
                    self.child_info[child]['allergic'] = True
                elif f"(not_allergic_gluten {child})" in not_allergic_facts:
                    self.child_info[child]['allergic'] = False
                # else: allergy status is unknown/not relevant for goal? (Shouldn't happen in valid problems)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Cost_serve: Count Unserved Children
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children = [c for c in self.children_to_serve if c not in served_children]
        n_unserved = len(unserved_children)

        # If no children are unserved, the goal is reached.
        if n_unserved == 0:
            return 0

        cost_serve = n_unserved
        cost_make = 0
        cost_put_on_tray = 0
        cost_move_tray = 0

        # Get all GF sandwiches in the current state
        all_gf_sandwiches_in_state = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}

        # 2. Cost_make: Count Sandwiches That Need Making
        n_allergic_unserved = sum(1 for c in unserved_children if self.child_info.get(c, {}).get('allergic', False))
        n_non_allergic_unserved = n_unserved - n_allergic_unserved

        # Count available sandwiches (kitchen or on tray) in the current state
        available_sandwiches_in_state = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_sandwich", "*") or match(fact, "ontray", "*", "*")}
        n_gf_avail_total = sum(1 for s in available_sandwiches_in_state if s in all_gf_sandwiches_in_state)
        n_reg_avail_total = sum(1 for s in available_sandwiches_in_state if s not in all_gf_sandwiches_in_state)

        make_gf = max(0, n_allergic_unserved - n_gf_avail_total)
        make_reg = max(0, n_non_allergic_unserved - n_reg_avail_total)
        cost_make = make_gf + make_reg

        # 3. Cost_put_on_tray: Count Sandwiches That Need Putting on Trays
        n_sandwiches_ontray_total = sum(1 for fact in state if match(fact, "ontray", "*", "*"))
        total_sandwiches_needed_on_trays = n_unserved # Each unserved child needs one sandwich on a tray
        cost_put_on_tray = max(0, total_sandwiches_needed_on_trays - n_sandwiches_ontray_total)

        # 4. Cost_move_tray: Count Tray Movements Needed
        n_children_need_delivery = 0

        # Build map of trays to locations in the current state
        tray_loc = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*")}
        # Build map of sandwiches to trays in the current state
        sandwich_on_tray = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "ontray", "*", "*")}

        # Check each unserved child
        for child in unserved_children:
            child_info = self.child_info.get(child)
            if child_info is None:
                 # Should not happen if child_info is correctly populated
                 continue

            child_location = child_info.get('location')
            needs_gf = child_info.get('allergic', False)

            if child_location is None:
                 # Should not happen
                 continue

            suitable_sandwich_on_tray_at_p = False
            # Check if any sandwich on any tray at the child's location is suitable
            for sandwich, tray in sandwich_on_tray.items():
                if tray in tray_loc and tray_loc[tray] == child_location:
                    is_gf_sandwich = sandwich in all_gf_sandwiches_in_state
                    if (needs_gf and is_gf_sandwich) or (not needs_gf and not is_gf_sandwich):
                        suitable_sandwich_on_tray_at_p = True
                        break # Found a suitable sandwich/tray at location for this child

            if not suitable_sandwich_on_tray_at_p:
                n_children_need_delivery += 1

        cost_move_tray = n_children_need_delivery

        # Total heuristic is the sum of estimated costs for each stage
        total_heuristic = cost_serve + cost_make + cost_put_on_tray + cost_move_tray

        return total_heuristic
