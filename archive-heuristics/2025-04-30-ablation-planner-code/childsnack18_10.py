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
    This heuristic estimates the number of actions required to serve all waiting
    children. It sums up the estimated costs for:
    1. Serving each unserved child.
    2. Making enough suitable sandwiches.
    3. Putting made sandwiches onto trays.
    4. Moving trays with sandwiches to the children's locations.

    It attempts to account for resources already in place (sandwiches on trays
    at the correct locations) using a greedy matching approach.

    # Assumptions
    - The primary goal is to serve all children.
    - Each child requires one suitable sandwich.
    - Gluten-allergic children require gluten-free sandwiches.
    - Non-allergic children can accept any sandwich.
    - Sandwiches must be on a tray at the child's location to be served.
    - Making a sandwich requires available bread, content, and a 'notexist' sandwich object.
    - Putting a sandwich on a tray requires the sandwich in the kitchen and a tray in the kitchen.
    - Moving a tray requires the tray at a location.
    - The heuristic assumes sufficient trays exist overall to deliver needed sandwiches.
    - The heuristic uses a greedy approach to match existing resources (sandwiches on trays at location) to children's needs, which is an approximation.

    # Heuristic Initialization
    The heuristic extracts static information from the task:
    - Which children are allergic to gluten.
    - The initial waiting location for each child. This location is assumed to be
      where the child remains until served.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all unserved children and their waiting locations (from initialization)
       and allergy status (from initialization). Count the total number of unserved children (`N_unserved`).
       If `N_unserved` is 0, the state is a goal state, return 0.
    2. Count available resources in the kitchen: bread, content, gluten-free bread,
       gluten-free content, and 'notexist' sandwich objects. Calculate the total
       number of sandwiches that can be made (`N_makable`).
    3. Count sandwiches that are already made: those `at_kitchen_sandwich` or `ontray`.
       Distinguish between gluten-free and regular made sandwiches.
       Count total made sandwiches (`N_sandwiches_made`) and those already `ontray` (`N_ontray`).
    4. Determine how many unserved children can be served immediately by matching
       them with suitable sandwiches already on trays at their waiting locations.
       Use a greedy matching approach: prioritize matching allergic children with
       available gluten-free sandwiches at their location, then match remaining
       children with any remaining suitable sandwiches at their location.
       Count the number of children matched (`N_ready_to_serve`).
    5. Calculate the number of sandwiches that still need to be delivered to children's
       locations on trays (`N_need_delivery = N_unserved - N_ready_to_serve`).
    6. Estimate the cost for each stage of the process:
       - `Cost_Serve`: Each unserved child needs one serve action. Cost = `N_unserved`.
       - `Cost_Make`: Estimate the number of sandwiches that need to be made. This is
         the total needed (`N_unserved`) minus those already made (`N_sandwiches_made`),
         capped by the number that can be made (`N_makable`). Cost = `min(max(0, N_unserved - N_sandwiches_made), N_makable)`.
       - `Cost_PutOnTray`: Estimate the number of sandwiches that need to be put on trays.
         This is the total needed on trays (`N_unserved`) minus those already on trays (`N_ontray`),
         capped by the number of sandwiches available to be put on trays (those `at_kitchen_sandwich`
         plus those that will be newly made, estimated by `Cost_Make`). Cost = `min(max(0, N_unserved - N_ontray), N_kitchen_sandwich + Cost_Make)`.
       - `Cost_MoveTray`: Estimate the number of tray movements needed. A simplified
         estimate is that each sandwich needing delivery (`N_need_delivery`) requires a tray move. Cost = `N_need_delivery`.
    7. The total heuristic value is the sum of these estimated costs:
       `H = Cost_Serve + Cost_Make + Cost_PutOnTray + Cost_MoveTray`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract static information: allergy status and initial waiting locations
        self.is_allergic = {} # {child_name: bool}
        self.waiting_loc_init = {} # {child_name: location}

        for fact_str in self.static:
            parts = get_parts(fact_str)
            if parts[0] == 'allergic_gluten':
                self.is_allergic[parts[1]] = True
            elif parts[0] == 'not_allergic_gluten':
                self.is_allergic[parts[1]] = False
            elif parts[0] == 'waiting':
                child, loc = parts[1], parts[2]
                self.waiting_loc_init[child] = loc

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unserved children
        served_children_in_state = {get_parts(fact)[1] for fact in state if match(fact, 'served', '*')}
        unserved_children_set = {c for c in self.is_allergic.keys() if c not in served_children_in_state}

        N_unserved = len(unserved_children_set)

        if N_unserved == 0:
            return 0 # Goal state

        # Map unserved children to their waiting location (from initial state)
        unserved_children_waiting_loc = {c: self.waiting_loc_init[c] for c in unserved_children_set}

        # 2. Count available resources (ingredients, notexist sandwiches)
        N_bread = sum(1 for fact in state if match(fact, 'at_kitchen_bread', '*'))
        N_content = sum(1 for fact in state if match(fact, 'at_kitchen_content', '*'))
        N_gf_bread = sum(1 for fact in state if match(fact, 'at_kitchen_bread', '*') and match('(no_gluten_bread ' + get_parts(fact)[1] + ')', 'no_gluten_bread', '*'))
        N_gf_content = sum(1 for fact in state if match(fact, 'at_kitchen_content', '*') and match('(no_gluten_content ' + get_parts(fact)[1] + ')', 'no_gluten_content', '*'))
        N_notexist = sum(1 for fact in state if match(fact, 'notexist', '*'))

        N_makable = min(N_bread, N_content, N_notexist)
        # N_gf_makable = min(N_gf_bread, N_gf_content, N_notexist) # Not directly used in simplified make cost

        # 3. Count sandwiches already made and their locations/types
        sandwiches_ontray = set() # set of sandwich names
        sandwiches_kitchen = set() # set of sandwich names
        gf_sandwiches_made = set() # set of GF sandwich names (kitchen or ontray)
        sandwich_ontray_map = {} # {sandwich_name: tray_name}

        for fact in state:
            parts = get_parts(fact)
            if match(fact, 'ontray', '*', '*'):
                s, t = parts[1], parts[2]
                sandwiches_ontray.add(s)
                sandwich_ontray_map[s] = t
            elif match(fact, 'at_kitchen_sandwich', '*'):
                sandwiches_kitchen.add(parts[1])
            elif match(fact, 'no_gluten_sandwich', '*'):
                gf_sandwiches_made.add(parts[1])

        N_ontray = len(sandwiches_ontray)
        N_kitchen_sandwich = len(sandwiches_kitchen)
        N_sandwiches_made = N_ontray + N_kitchen_sandwich
        # N_gf_sandwiches_made = len(gf_sandwiches_made.intersection(sandwiches_ontray.union(sandwiches_kitchen))) # Not directly used in simplified make cost

        # 4. Calculate N_ready_to_serve using greedy matching
        N_ready_to_serve = 0
        UsedSandwiches = set()

        # Group suitable sandwiches on trays by location
        available_gf_ontray_at_loc = {} # {loc: set of GF sandwiches on trays at loc}
        available_reg_ontray_at_loc = {} # {loc: set of Reg-only sandwiches on trays at loc}

        # Find locations of trays
        tray_locations = {} # {tray_name: location}
        for fact in state:
             if match(fact, 'at', 'tray*', '*'):
                 t, loc = get_parts(fact)[1], get_parts(fact)[2]
                 tray_locations[t] = loc

        for s in sandwiches_ontray:
            t = sandwich_ontray_map[s]
            is_gf = s in gf_sandwiches_made
            tray_loc = tray_locations.get(t)

            if tray_loc in unserved_children_waiting_loc.values(): # Tray is at a location where children are waiting
                # Check suitability for *any* child at this location
                is_suitable_for_any_child_at_loc = False
                for child in unserved_children_set:
                    if unserved_children_waiting_loc[child] == tray_loc:
                        is_allergic = self.is_allergic.get(child, False)
                        if (is_allergic and is_gf) or (not is_allergic):
                            is_suitable_for_any_child_at_loc = True
                            break
                if is_suitable_for_any_child_at_loc:
                     if is_gf:
                         available_gf_ontray_at_loc.setdefault(tray_loc, set()).add(s)
                     else:
                         available_reg_ontray_at_loc.setdefault(tray_loc, set()).add(s)

        matched_children = set()

        # 1. Match allergic children with available GF sandwiches at their location
        for child in unserved_children_set:
            if child in matched_children: continue
            loc = unserved_children_waiting_loc[child]
            is_allergic = self.is_allergic.get(child, False)

            if is_allergic and loc in available_gf_ontray_at_loc:
                 # Find an unused GF sandwich at this location
                 found_match_s = None
                 for s in available_gf_ontray_at_loc[loc]:
                     if s not in UsedSandwiches:
                         found_match_s = s
                         break
                 if found_match_s:
                     N_ready_to_serve += 1
                     UsedSandwiches.add(found_match_s)
                     matched_children.add(child)
                     # No need to remove from available_gf_ontray_at_loc if we use UsedSandwiches set

        # 2. Match remaining children (allergic or not) with any remaining suitable sandwiches at their location
        for child in unserved_children_set:
            if child in matched_children: continue
            loc = unserved_children_waiting_loc[child]
            is_allergic = self.is_allergic.get(child, False)

            found_match_s = None

            # Try remaining GF sandwiches first
            if loc in available_gf_ontray_at_loc:
                for s in available_gf_ontray_at_loc[loc]:
                     if s not in UsedSandwiches:
                         # This GF sandwich is suitable for any remaining child
                         found_match_s = s
                         break

            if found_match_s:
                N_ready_to_serve += 1
                UsedSandwiches.add(found_match_s)
                matched_children.add(child)
                continue # Move to next child

            # Try remaining Reg sandwiches (only for non-allergic)
            if not is_allergic and loc in available_reg_ontray_at_loc:
                for s in available_reg_ontray_at_loc[loc]:
                     if s not in UsedSandwiches:
                         found_match_s = s
                         break

            if found_match_s:
                N_ready_to_serve += 1
                UsedSandwiches.add(found_match_s)
                matched_children.add(child)
                # continue # Move to next child (already at end of loop)


        N_need_delivery = N_unserved - N_ready_to_serve

        # 6. Estimate costs for each stage
        Cost_Serve = N_unserved
        Cost_Make = min(max(0, N_unserved - N_sandwiches_made), N_makable)
        Cost_PutOnTray = min(max(0, N_unserved - N_ontray), N_kitchen_sandwich + Cost_Make)
        Cost_MoveTray = N_need_delivery # Simplified: one move per sandwich needing delivery

        # 7. Total heuristic value
        total_heuristic = Cost_Serve + Cost_Make + Cost_PutOnTray + Cost_MoveTray

        return total_heuristic
