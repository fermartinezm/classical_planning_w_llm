from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
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
    This heuristic estimates the number of actions required to serve all unserved children.
    It models the process as a pipeline: Make Sandwich -> Put on Tray -> Move Tray -> Serve Child.
    The heuristic sums the number of items (sandwiches/children) that need to pass through each stage,
    minus those already past that stage, plus the final serve action for each unserved child.
    It considers sandwich types (gluten/no-gluten) for the final delivery stage.

    # Assumptions
    - Solvable problems have sufficient ingredients and 'notexist' sandwich slots to make needed sandwiches.
    - Solvable problems have sufficient trays to transport sandwiches.
    - Each 'make', 'put', 'move', 'serve' action costs 1.
    - A tray can carry multiple sandwiches. A single 'move_tray' action can satisfy the location requirement for all sandwiches on that tray. The heuristic simplifies this by counting needed *sandwiches* to be moved, not tray moves. This is a relaxation.
    - The kitchen is not a waiting place for children.

    # Heuristic Initialization
    - Extracts goal children from the task.
    - Maps children to their allergy status and waiting places from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify Unserved Children: Determine the set of children present in the task's goal list who are not currently marked as 'served' in the state.
    2. Goal Check: If there are no unserved children, the current state is a goal state, and the heuristic value is 0.
    3. Count Unserved Children: Calculate the total number of unserved children (`N_unserved`). This represents the number of final 'serve' actions required, contributing `N_unserved` to the total heuristic cost.
    4. Categorize Unserved Children: Count unserved children who are allergic (`N_unserved_NG` for allergic, `N_unserved_G` for non-allergic). The total number of suitable sandwiches needed is `Total_needed_sandwiches = N_unserved_NG + N_unserved_G`.
    5. Count Available Sandwiches: Iterate through the current state to count sandwiches that are already made. Distinguish between 'no_gluten_sandwich' (NG) and other sandwiches (G). Count those that are `at_kitchen_sandwich` and those that are `ontray` anywhere.
       - `Avail_NG_made`, `Avail_G_made`: Total sandwiches of each type that exist.
       - `Avail_NG_ontray_any`, `Avail_G_ontray_any`: Sandwiches of each type that are currently on any tray.
    6. Estimate Cost to Make: Calculate the number of sandwiches that still need to be made: `Cost_make = max(0, Total_needed_sandwiches - (Avail_NG_made + Avail_G_made))`. Each such sandwich needs a 'make_sandwich' action.
    7. Estimate Cost to Put on Tray: Calculate the number of sandwiches that need to be put on a tray: `Cost_put = max(0, Total_needed_sandwiches - (Avail_NG_ontray_any + Avail_G_ontray_any))`. Each such sandwich needs a 'put_on_tray' action.
    8. Estimate Cost to Move Tray: Calculate how many sandwiches need to be moved to the correct location.
       - For each location `p` where unserved children are waiting:
         - Count the number of unserved allergic (`Needed_NG_at_p`) and non-allergic (`Needed_G_at_p`) children waiting at `p`.
         - Count the number of available NG (`Avail_NG_ontray_at_p`) and G (`Avail_G_ontray_at_p`) sandwiches that are already `ontray` at location `p`.
         - Determine how many children at `p` can be served by the sandwiches already present on trays at `p` (`Ready_at_p`), considering type compatibility (NG can serve both, G only non-allergic). Prioritize serving allergic children with NG sandwiches first.
       - Sum `Ready_at_p` over all locations to get `Ready_at_location`. This is the number of needed sandwiches already at their destination location on a tray.
       - The number of sandwiches that still need to be moved is `Cost_move = max(0, Total_needed_sandwiches - Ready_at_location)`. Each such sandwich needs to be on a tray that is moved to the correct location.
        (Note: This counts needed *sandwiches* to move, not tray moves, which is a simplification/relaxation).
    9. Calculate Total Heuristic: Sum the costs from each stage and the final serve actions: `Total heuristic = Cost_make + Cost_put + Cost_move + N_unserved`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal children, allergy status,
        and waiting places from static facts.
        """
        # Extract goal children
        self.goal_children = {get_parts(goal)[1] for goal in task.goals if match(goal, "served", "*")}

        # Extract static information about children
        self.child_allergy = {}
        self.child_waiting_place = {}
        for fact in task.static:
            if match(fact, "allergic_gluten", "*"):
                self.child_allergy[get_parts(fact)[1]] = 'NG' # Needs No-Gluten
            elif match(fact, "not_allergic_gluten", "*"):
                self.child_allergy[get_parts(fact)[1]] = 'G' # Can have Gluten
            elif match(fact, "waiting", "*", "*"):
                child, place = get_parts(fact)[1:]
                self.child_waiting_place[child] = place

        # Note: Assumes all children in goal are also in static facts with allergy and waiting info.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unserved children
        served_children_in_state = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children = self.goal_children - served_children_in_state

        # 2. Goal Check
        if not unserved_children:
            return 0

        # 3. Count unserved children by type and total
        N_unserved = len(unserved_children)
        N_unserved_NG = sum(1 for child in unserved_children if self.child_allergy.get(child) == 'NG')
        N_unserved_G = sum(1 for child in unserved_children if self.child_allergy.get(child) == 'G')
        Total_needed_sandwiches = N_unserved_NG + N_unserved_G

        # 5. Count available sandwiches at different stages
        Avail_NG_made = 0
        Avail_G_made = 0
        Avail_NG_ontray_any = 0
        Avail_G_ontray_any = 0
        Avail_NG_kitchen = 0
        Avail_G_kitchen = 0

        sandwiches_ontray = {} # Map sandwich -> tray
        tray_locations = {} # Map tray -> place

        # First pass to find tray locations and on-tray sandwiches
        for fact in state:
             if match(fact, "at", "*", "*"):
                  obj, place = get_parts(fact)[1:]
                  # Simple check based on naming convention for trays
                  if obj.startswith('tray'):
                      tray_locations[obj] = place
             elif match(fact, "ontray", "*", "*"):
                  s, t = get_parts(fact)[1:]
                  sandwiches_ontray[s] = t

        # Second pass to count sandwiches by type and location status
        all_sandwiches_in_state = set()
        for fact in state:
             if match(fact, "at_kitchen_sandwich", "*"):
                 s = get_parts(fact)[1]
                 all_sandwiches_in_state.add(s)
             elif match(fact, "ontray", "*", "*"):
                 s = get_parts(fact)[1]
                 all_sandwiches_in_state.add(s)

        sandwich_is_ng = {}
        for s in all_sandwiches_in_state:
             sandwich_is_ng[s] = any(match(f, "no_gluten_sandwich", s) for f in state)

        for s in all_sandwiches_in_state:
             is_ng = sandwich_is_ng[s]
             is_ontray = s in sandwiches_ontray
             is_kitchen = any(match(f, "at_kitchen_sandwich", s) for f in state) # Check if it's at kitchen

             if is_ng:
                 Avail_NG_made += 1
                 if is_ontray:
                     Avail_NG_ontray_any += 1
                 if is_kitchen:
                     Avail_NG_kitchen += 1
             else:
                 Avail_G_made += 1
                 if is_ontray:
                     Avail_G_ontray_any += 1
                 if is_kitchen:
                     Avail_G_kitchen += 1


        # 6. Estimate Cost to Make
        Cost_make = max(0, Total_needed_sandwiches - (Avail_NG_made + Avail_G_made))

        # 7. Estimate Cost to Put on Tray
        Cost_put = max(0, Total_needed_sandwiches - (Avail_NG_ontray_any + Avail_G_ontray_any))

        # 8. Estimate Cost to Move Tray (and calculate Ready_at_location)
        Ready_at_location = 0
        unserved_by_place = {}
        for child in unserved_children:
            place = self.child_waiting_place.get(child)
            if place:
                if place not in unserved_by_place:
                    unserved_by_place[place] = {'NG': 0, 'G': 0}
                if self.child_allergy.get(child) == 'NG':
                    unserved_by_place[place]['NG'] += 1
                else:
                    unserved_by_place[place]['G'] += 1

        ontray_by_place = {}
        for s, t in sandwiches_ontray.items():
             place = tray_locations.get(t)
             if place:
                 if place not in ontray_by_place:
                     ontray_by_place[place] = {'NG': 0, 'G': 0}
                 is_ng = sandwich_is_ng.get(s, False) # Default to False if sandwich type unknown (shouldn't happen)
                 if is_ng:
                     ontray_by_place[place]['NG'] += 1
                 else:
                     ontray_by_place[place]['G'] += 1

        # Calculate Ready_at_location by matching needs at each place with available sandwiches on trays at that place
        for place in unserved_by_place:
            Needed_NG_at_p = unserved_by_place[place]['NG']
            Needed_G_at_p = unserved_by_place[place]['G']
            Avail_NG_ontray_at_p = ontray_by_place.get(place, {}).get('NG', 0)
            Avail_G_ontray_at_p = ontray_by_place.get(place, {}).get('G', 0)

            # Prioritize NG sandwiches for NG needs
            Served_by_NG_at_p = min(Needed_NG_at_p, Avail_NG_ontray_at_p)
            Remaining_Needed_NG_at_p = Needed_NG_at_p - Served_by_NG_at_p
            Remaining_Avail_NG_ontray_at_p = Avail_NG_ontray_at_p - Served_by_NG_at_p

            # Serve G needs with G sandwiches
            Served_by_G_at_p = min(Needed_G_at_p, Avail_G_ontray_at_p)
            Remaining_Needed_G_at_p = Needed_G_at_p - Served_by_G_at_p

            # Serve remaining G needs with surplus NG sandwiches
            Served_by_surplus_NG_at_p = min(Remaining_Needed_G_at_p, Remaining_Avail_NG_ontray_at_p)

            Ready_at_p = Served_by_NG_at_p + Served_by_G_at_p + Served_by_surplus_NG_at_p
            Ready_at_location += Ready_at_p

        Cost_move = max(0, Total_needed_sandwiches - Ready_at_location)

        # 10. Total heuristic
        total_heuristic = Cost_make + Cost_put + Cost_move + N_unserved

        return total_heuristic
