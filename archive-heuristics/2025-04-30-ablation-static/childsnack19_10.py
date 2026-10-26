from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all children
    by summing up the estimated costs for:
    1. Serving each unserved child.
    2. Making sandwiches that are not available.
    3. Putting sandwiches onto trays if they are in the kitchen or newly made
       and needed on trays.
    4. Moving trays to locations where unserved children are waiting.

    # Assumptions
    - Each child requires exactly one sandwich.
    - Gluten requirements must be met (allergic children need gluten-free sandwiches).
    - Trays can hold multiple sandwiches (tray capacity is not a bottleneck).
    - Bread, content, and 'notexist' sandwich objects are available in sufficient quantities
      in the kitchen to make any required sandwiches.
    - Trays are available somewhere to be moved to target locations.

    # Heuristic Initialization
    - Stores the goal conditions and static facts from the task.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  Identify unserved children: Iterate through the goal conditions (which are assumed
        to be `(served ?c)` facts) and check if the corresponding fact is present in the
        current state. Count the total number of unserved children (`N_unserved`).
        This contributes `N_unserved` to the total heuristic cost, representing the
        minimum number of `serve` actions required.

    2.  Separate unserved children by allergy status: For each unserved child, check
        if `(allergic_gluten ?c)` or `(not_allergic_gluten ?c)` is true in the static facts.
        Count unserved regular children (`N_reg_unserved`) and unserved gluten-free
        children (`N_gf_unserved`).

    3.  Count available sandwiches: Iterate through the current state to find facts
        `(at_kitchen_sandwich ?s)` or `(ontray ?s ?t)`. For each such sandwich `?s`,
        check if `(no_gluten_sandwich ?s)` is true in the state. Count available
        regular sandwiches (`Avail_reg_any`) and available gluten-free sandwiches
        (`Avail_gf_any`).

    4.  Calculate sandwiches to make: Determine how many regular and gluten-free
        sandwiches are needed but not available. `To_Make_reg = max(0, N_reg_unserved - Avail_reg_any)`
        and `To_Make_gf = max(0, N_gf_unserved - Avail_gf_any)`. The total number
        of sandwiches to make is `To_Make = To_Make_reg + To_Make_gf`. This contributes
        `To_Make` to the total heuristic cost (for `make_sandwich` actions).

    5.  Count sandwiches already on trays: Iterate through the state and count facts
        `(ontray ?s ?t)`. Let this be `S_ontray_any`.

    6.  Calculate sandwiches that need to be put on trays: We need `N_unserved`
        sandwiches to end up on trays. `S_to_put_on_tray = max(0, N_unserved - S_ontray_any)`.

    7.  Count sandwiches currently in the kitchen: Iterate through the state and count
        facts `(at_kitchen_sandwich ?s)`. Let this be `S_kitchen`.

    8.  Calculate `put_on_tray` actions needed: The number of sandwiches that can be
        put on trays is limited by those in the kitchen or newly made (`S_kitchen + To_Make`).
        The number of `put_on_tray` actions needed is the minimum of the demand
        (`S_to_put_on_tray`) and the supply (`S_kitchen + To_Make`).
        `Put_Actions = min(S_to_put_on_tray, S_kitchen + To_Make)`. This contributes
        `Put_Actions` to the total heuristic cost.

    9.  Identify target locations: Find the locations where unserved children are waiting
        by checking `(waiting ?c ?p)` facts for unserved children `?c`. Collect the
        unique places `?p` into a set `Target_Places`.

    10. Count trays at target locations: Iterate through the state and count facts
        `(at ?t ?p)` where `?p` is in `Target_Places`. Let this be `Trays_at_Target_Places`.

    11. Calculate tray movements needed: Estimate the number of `move_tray` actions
        required to get trays to the target locations. This is the number of target
        locations that do not currently have a tray, assuming one tray can serve
        all children at a location. `Tray_Moves = max(0, len(Target_Places) - Trays_at_Target_Places)`.
        This contributes `Tray_Moves` to the total heuristic cost.

    12. Sum the costs: The total heuristic value is the sum of the costs from steps 1, 4, 8, and 11:
        `N_unserved + To_Make + Put_Actions + Tray_Moves`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static_facts = task.static

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unserved children
        unserved_children_list = [get_parts(goal)[1] for goal in self.goals if goal not in state]
        n_unserved = len(unserved_children_list)

        # If no children are unserved, the goal is reached.
        if n_unserved == 0:
            return 0

        # 2. Separate unserved children by allergy status
        n_reg_unserved = sum(1 for c in unserved_children_list if f"(not_allergic_gluten {c})" in self.static_facts)
        n_gf_unserved = sum(1 for c in unserved_children_list if f"(allergic_gluten {c})" in self.static_facts)

        # 3. Count available sandwiches by type (anywhere: kitchen or ontray)
        available_sandwich_objects = set()
        for fact in state:
            if match(fact, "at_kitchen_sandwich", "*"):
                available_sandwich_objects.add(get_parts(fact)[1])
            elif match(fact, "ontray", "*", "*"):
                available_sandwich_objects.add(get_parts(fact)[1])

        avail_reg_any = 0
        avail_gf_any = 0
        for s in available_sandwich_objects:
            if f"(no_gluten_sandwich {s})" in state:
                avail_gf_any += 1
            else:
                avail_reg_any += 1

        # 4. Calculate sandwiches to make
        to_make_reg = max(0, n_reg_unserved - avail_reg_any)
        to_make_gf = max(0, n_gf_unserved - avail_gf_any)
        to_make = to_make_reg + to_make_gf

        # 5. Count sandwiches already on trays
        s_ontray_any = sum(1 for fact in state if match(fact, "ontray", "*", "*"))

        # 6. Calculate sandwiches that need to be put on trays
        s_to_put_on_tray = max(0, n_unserved - s_ontray_any)

        # 7. Count sandwiches currently in the kitchen
        s_kitchen = sum(1 for fact in state if match(fact, "at_kitchen_sandwich", "*"))

        # 8. Calculate put_on_tray actions needed
        # We need s_to_put_on_tray sandwiches put on trays.
        # We can use sandwiches from the kitchen (s_kitchen) or newly made ones (to_make).
        put_actions = min(s_to_put_on_tray, s_kitchen + to_make)

        # 9. Identify target locations (where unserved children are waiting)
        waiting_children_locations = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "waiting", "*", "*")}
        target_places = set(location for child, location in waiting_children_locations.items() if f"(served {child})" not in state)

        # 10. Count trays at target locations
        trays_at_target_places = sum(1 for fact in state if match(fact, "at", "*", "*") and get_parts(fact)[2] in target_places)

        # 11. Calculate tray movements needed
        # We need at least one tray at each target place.
        tray_moves = max(0, len(target_places) - trays_at_target_places)

        # 12. Sum the costs
        total_cost = n_unserved + to_make + put_actions + tray_moves

        return total_cost
