from fnmatch import fnmatch
from collections import defaultdict
from heuristics.heuristic_base import Heuristic

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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the total number of actions required to serve all
    children who are currently waiting and not yet served. It counts the number
    of 'serve' actions needed and adds estimated costs for preparing and
    delivering the necessary sandwiches based on their current state.

    # Assumptions
    - Each unserved child requires one suitable sandwich.
    - A suitable sandwich for an allergic child must be gluten-free.
    - A suitable sandwich for a non-allergic child can be any sandwich (gluten-free or regular).
    - There are enough trays available at the kitchen when needed for 'put_on_tray'.
    - Ingredients (bread and content) and 'notexist' sandwich objects are sufficient to make any required sandwich type if one is not already made (ingredient counts are used to determine how many of each type *can* be made from current stock).
    - Getting a sandwich from the kitchen to a child's location requires putting it on a tray (at the kitchen) and then moving the tray.
    - Getting a sandwich already on a tray but at the wrong location requires moving the tray.
    - The heuristic assumes an optimal assignment of available sandwiches to needed children based on the sandwich's current state, prioritizing those closest to being served.

    # Heuristic Initialization
    The heuristic extracts static information from the task:
    - The allergy status of each child (`allergic_gluten`, `not_allergic_gluten`).
    - The waiting location for each child (`waiting ?c ?p`).
    - Which bread and content portions are gluten-free (`no_gluten_bread`, `no_gluten_content`).
    - The set of all children in the problem.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:

    1.  **Identify Unserved Children:** Determine which children in the goal list are not yet served. For each unserved child, note their waiting location and allergy status.
    2.  **Count Total Sandwich Needs:** Sum the number of unserved allergic children (needs GF sandwiches) and unserved non-allergic children (needs Any sandwiches).
    3.  **Count Available Sandwiches by State and Type:**
        -   Iterate through the current state facts to find all existing sandwiches (`at_kitchen_sandwich`, `ontray`). Determine if each existing sandwich is gluten-free (`no_gluten_sandwich`).
        -   Find the current location of each tray (`at ?t ?p`).
        -   Categorize existing sandwiches based on type (GF/Regular) and state relative to needed locations:
            -   On a tray at a location where that type is needed.
            -   On a tray elsewhere (not at a needed location).
            -   In the kitchen.
        -   Count available ingredients (`at_kitchen_bread`, `at_kitchen_content`, `no_gluten_bread`, `no_gluten_content`) and `notexist` sandwich objects to determine the maximum number of GF and Regular sandwiches that can still be made.
    4.  **Calculate Heuristic Cost:**
        -   Initialize the total cost with the number of unserved children (each requires one 'serve' action).
        -   Calculate the deficit of GF and Any sandwiches needed (total needed minus those already on trays at needed locations).
        -   Greedily cover these deficits using available sandwiches from the cheapest delivery states first:
            -   Use sandwiches on trays elsewhere (cost: 1 action per sandwich for 'move_tray').
            -   Use sandwiches in the kitchen (cost: 2 actions per sandwich for 'put_on_tray' + 'move_tray').
            -   Use sandwiches that need making (cost: 3 actions per sandwich for 'make_sandwich' + 'put_on_tray' + 'move_tray').
        -   Prioritize using GF sandwiches for GF needs. After GF needs are covered, use any remaining GF sandwiches to cover Any needs before using Regular sandwiches.
        -   Add the accumulated costs for 'move_tray', 'put_on_tray', and 'make_sandwich' actions to the initial count of 'serve' actions.
    5.  **Return Total Cost:** The final sum is the heuristic estimate.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information:
        - Allergy status of children.
        - Waiting locations of children.
        - Gluten-free status of bread and content portions.
        - Set of all children in the problem.
        """
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Facts that are not affected by actions.

        # Extract static information
        self.is_allergic = {}
        self.waiting_loc = {}
        self.is_gf_bread = {}
        self.is_gf_content = {}
        self.all_children = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "allergic_gluten":
                child = parts[1]
                self.is_allergic[child] = True
                self.all_children.add(child)
            elif parts[0] == "not_allergic_gluten":
                child = parts[1]
                self.is_allergic[child] = False
                self.all_children.add(child)
            elif parts[0] == "waiting":
                child, place = parts[1], parts[2]
                self.waiting_loc[child] = place
            elif parts[0] == "no_gluten_bread":
                bread = parts[1]
                self.is_gf_bread[bread] = True
            elif parts[0] == "no_gluten_content":
                content = parts[1]
                self.is_gf_content[content] = True

        # Identify children that are part of the goal (must be served)
        self.goal_children = set()
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == "served":
                 self.goal_children.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state

        # 1. Identify unserved children and their needs (child, place, is_allergic).
        unserved_needs = []
        needed_gf_locs = set()
        needed_any_locs = set()

        for child in self.goal_children:
            if f"(served {child})" not in state:
                place = self.waiting_loc.get(child) # Get waiting location
                is_allergic = self.is_allergic.get(child, False) # Default to not allergic if info missing
                unserved_needs.append((child, place, is_allergic))
                if place: # Only consider children with a known waiting place
                    if is_allergic:
                        needed_gf_locs.add(place)
                    needed_any_locs.add(place) # Non-allergic needs any sandwich at their location

        # If no unserved children, the goal is reached.
        if not unserved_needs:
            return 0

        # 2. Count available suitable sandwiches by state and type (GF/Regular).
        avail_gf = {'ontray_at_needed': 0, 'ontray_elsewhere': 0, 'kitchen': 0, 'to_make': 0}
        avail_reg = {'ontray_at_needed': 0, 'ontray_elsewhere': 0, 'kitchen': 0, 'to_make': 0}

        # Map sandwich object to its GF status
        sandwich_is_gf = {}
        for fact in state:
            if match(fact, "no_gluten_sandwich", "*"):
                sandwich = get_parts(fact)[1]
                sandwich_is_gf[sandwich] = True

        # Map tray object to its location
        tray_loc = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                tray, loc = get_parts(fact)[1], get_parts(fact)[2]
                tray_loc[tray] = loc

        # Count sandwiches on trays
        for fact in state:
            if match(fact, "ontray", "*", "*"):
                sandwich, tray = get_parts(fact)[1], get_parts(fact)[2]
                loc = tray_loc.get(tray)
                if loc is None: continue # Should not happen in valid states

                is_gf = sandwich_is_gf.get(sandwich, False)

                if is_gf:
                    if loc in needed_gf_locs:
                        avail_gf['ontray_at_needed'] += 1
                    else:
                        avail_gf['ontray_elsewhere'] += 1
                # Any sandwich can satisfy an 'any' need
                if loc in needed_any_locs:
                     # Avoid double counting if a GF sandwich is needed at a location
                     # where a non-allergic child also waits.
                     # We count GF sandwiches separately for GF needs.
                     # For Any needs, we count all suitable sandwiches.
                     # A GF sandwich can satisfy a GF need OR an Any need.
                     # A Regular sandwich can satisfy only an Any need.
                     # Let's count available sandwiches by type (GF/Reg) and then assign.
                     pass # Handled below

        # Recount avail_reg['ontray_at_needed'] and avail_reg['ontray_elsewhere']
        # This counts regular sandwiches on trays
        for fact in state:
            if match(fact, "ontray", "*", "*"):
                sandwich, tray = get_parts(fact)[1], get_parts(fact)[2]
                loc = tray_loc.get(tray)
                if loc is None: continue
                is_gf = sandwich_is_gf.get(sandwich, False)
                if not is_gf: # It's a regular sandwich
                    if loc in needed_any_locs:
                        avail_reg['ontray_at_needed'] += 1
                    else:
                        avail_reg['ontray_elsewhere'] += 1


        # Count sandwiches in kitchen
        for fact in state:
            if match(fact, "at_kitchen_sandwich", "*"):
                sandwich = get_parts(fact)[1]
                is_gf = sandwich_is_gf.get(sandwich, False)
                if is_gf:
                    avail_gf['kitchen'] += 1
                else:
                    avail_reg['kitchen'] += 1

        # Count ingredients and notexist sandwiches to determine 'to_make' potential
        current_gf_bread = sum(1 for f in state if match(f, "at_kitchen_bread", "*") and self.is_gf_bread.get(get_parts(f)[1], False))
        current_gf_content = sum(1 for f in state if match(f, "at_kitchen_content", "*") and self.is_gf_content.get(get_parts(f)[1], False))
        current_bread = sum(1 for f in state if match(f, "at_kitchen_bread", "*"))
        current_content = sum(1 for f in state if match(f, "at_kitchen_content", "*"))
        current_notexist = sum(1 for f in state if match(f, "notexist", "*"))

        # Calculate how many GF and Regular sandwiches can be made
        can_make_gf = min(current_gf_bread, current_gf_content, current_notexist)
        # Use remaining notexist slots for regular sandwiches, limited by remaining ingredients
        remaining_notexist = current_notexist - can_make_gf
        remaining_bread = current_bread - current_gf_bread
        remaining_content = current_content - current_gf_content
        can_make_reg = min(remaining_bread, remaining_content, remaining_notexist)

        avail_gf['to_make'] = can_make_gf
        avail_reg['to_make'] = can_make_reg

        # 3. Calculate total GF and Any sandwiches needed
        total_gf_needed = sum(1 for _, _, allergic in unserved_needs if allergic)
        total_any_needed = sum(1 for _, _, allergic in unserved_needs if not allergic)

        # 4. Initialize heuristic cost with the number of unserved children (for 'serve' actions)
        h = len(unserved_needs)

        # Calculate deficits
        gf_deficit = total_gf_needed
        any_deficit = total_any_needed

        # 5. Greedily cover deficits using available sandwiches from cheapest states

        # Use GF ontrays at needed locations (cost 0 for delivery)
        use_gf_ontray_at_needed = min(gf_deficit, avail_gf['ontray_at_needed'])
        gf_deficit -= use_gf_ontray_at_needed
        avail_gf['ontray_at_needed'] -= use_gf_ontray_at_needed # Consume available

        # Use Regular ontrays at needed locations for Any needs (cost 0 for delivery)
        use_reg_ontray_at_needed = min(any_deficit, avail_reg['ontray_at_needed'])
        any_deficit -= use_reg_ontray_at_needed
        avail_reg['ontray_at_needed'] -= use_reg_ontray_at_needed # Consume available

        # Use GF ontrays elsewhere (cost 1 for move)
        use_gf_ontray_elsewhere = min(gf_deficit, avail_gf['ontray_elsewhere'])
        gf_deficit -= use_gf_ontray_elsewhere
        h += use_gf_ontray_elsewhere * 1
        avail_gf['ontray_elsewhere'] -= use_gf_ontray_elsewhere # Consume available

        # Use Regular ontrays elsewhere for Any needs (cost 1 for move)
        use_reg_ontray_elsewhere = min(any_deficit, avail_reg['ontray_elsewhere'])
        any_deficit -= use_reg_ontray_elsewhere
        h += use_reg_ontray_elsewhere * 1
        avail_reg['ontray_elsewhere'] -= use_reg_ontray_elsewhere # Consume available

        # Use GF in kitchen (cost 2 for put+move)
        use_gf_kitchen = min(gf_deficit, avail_gf['kitchen'])
        gf_deficit -= use_gf_kitchen
        h += use_gf_kitchen * 2
        avail_gf['kitchen'] -= use_gf_kitchen # Consume available

        # Use Regular in kitchen for Any needs (cost 2 for put+move)
        use_reg_kitchen = min(any_deficit, avail_reg['kitchen'])
        any_deficit -= use_reg_kitchen
        h += use_reg_kitchen * 2
        avail_reg['kitchen'] -= use_reg_kitchen # Consume available

        # Use GF to make (cost 3 for make+put+move)
        use_gf_to_make = min(gf_deficit, avail_gf['to_make'])
        gf_deficit -= use_gf_to_make
        h += use_gf_to_make * 3
        avail_gf['to_make'] -= use_gf_to_make # Consume available

        # Use Regular to make for Any needs (cost 3 for make+put+move)
        use_reg_to_make = min(any_deficit, avail_reg['to_make'])
        any_deficit -= use_reg_to_make
        h += use_reg_to_make * 3
        avail_reg['to_make'] -= use_reg_to_make # Consume available

        # Use remaining GF (from kitchen and to_make) for Any needs
        # Remaining GF in kitchen after covering GF deficit
        remaining_gf_kitchen = avail_gf['kitchen']
        use_rem_gf_kitchen_for_any = min(any_deficit, remaining_gf_kitchen)
        any_deficit -= use_rem_gf_kitchen_for_any
        h += use_rem_gf_kitchen_for_any * 2 # Cost is still put+move

        # Remaining GF to make after covering GF deficit
        remaining_gf_to_make = avail_gf['to_make']
        use_rem_gf_to_make_for_any = min(any_deficit, remaining_gf_to_make)
        any_deficit -= use_rem_gf_to_make_for_any
        h += use_rem_gf_to_make_for_any * 3 # Cost is still make+put+move

        # If gf_deficit > 0 or any_deficit > 0 here, it means there aren't enough
        # suitable sandwiches in total. For solvable problems, this shouldn't happen.
        # We can add a large penalty or return infinity, but for greedy search
        # on solvable problems, the calculated finite value is sufficient.
        # assert gf_deficit == 0 and any_deficit == 0, "Problem likely unsolvable or heuristic logic error"

        # 6. Return Total Cost
        return h
