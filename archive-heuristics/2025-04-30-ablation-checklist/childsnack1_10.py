from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions from examples
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we have enough parts to match the pattern args
    if len(parts) < len(args):
        return False

    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    Estimates the number of actions needed to serve all unserved children.
    The heuristic sums the minimum estimated cost for each unserved child
    independently, based on the availability and location of suitable sandwiches
    or the possibility of creating one.

    Costs considered per child:
    1: Serve (suitable sandwich on tray at child's location)
    2: Move tray + Serve (suitable sandwich on tray in kitchen)
    3: Put on tray + Move tray + Serve (suitable sandwich at kitchen, not on tray)
    4: Make + Put on tray + Move tray + Serve (suitable sandwich needs to be made)
    Large penalty if no suitable sandwich exists or can be made.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about children.
        """
        self.goals = task.goals # Goal conditions are (served ?c) for all children
        self.static_facts = task.static

        # Pre-process static facts to get child info
        self.child_allergy = {}
        self.child_waiting_place = {}
        self.all_children = set()

        for fact in self.static_facts:
            if match(fact, "allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.child_allergy[child] = 'gluten'
                self.all_children.add(child)
            elif match(fact, "not_allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.child_allergy[child] = 'none'
                self.all_children.add(child)
            # Waiting place is also static in this domain based on the PDDL
            elif match(fact, "waiting", "*", "*"):
                 child, place = get_parts(fact)[1:3]
                 self.child_waiting_place[child] = place
                 self.all_children.add(child) # Ensure children mentioned in waiting are included

        # Ensure all children from goals are in our list (they should be)
        # This loop might be redundant if all goal children are in static waiting/allergy facts
        # but adds robustness.
        for goal in self.goals:
             if match(goal, "served", "*"):
                 child = get_parts(goal)[1]
                 self.all_children.add(child)


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # 1. Identify unserved children
        unserved_children = [] # List of (child, place, allergy) tuples
        for child in self.all_children:
            if f"(served {child})" not in state:
                # Child must be waiting to need serving. Get place and allergy from static info.
                waiting_place = self.child_waiting_place.get(child)
                allergy = self.child_allergy.get(child, 'none') # Default to not allergic
                if waiting_place:
                     unserved_children.append((child, waiting_place, allergy))

        if not unserved_children:
            return 0 # Goal reached

        # 2. Count available sandwiches and resources in the current state
        sandwiches_at_loc_on_tray = {} # {place: {'gf': count, 'reg': count}}
        sandwiches_on_kitchen_tray = {'gf': 0, 'reg': 0}
        sandwiches_at_kitchen = {'gf': 0, 'reg': 0}
        count_notexist = 0
        count_bread_gf = 0
        count_bread_reg = 0
        count_content_gf = 0
        count_content_reg = 0

        # Find tray locations first
        tray_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                # Check if the first argument is a tray object.
                # Assuming tray objects contain the substring 'tray'
                parts = get_parts(fact)
                if len(parts) == 3 and parts[0] == 'at':
                    obj, place = parts[1:3]
                    if 'tray' in obj.lower(): # Case-insensitive check
                         tray_locations[obj] = place


        # Determine GF status of sandwiches currently in the state
        sandwich_is_gf = {}
        for fact in state:
             if match(fact, "no_gluten_sandwich", "*"):
                 sandwich = get_parts(fact)[1]
                 sandwich_is_gf[sandwich] = True

        # Count sandwiches based on their state and location
        for fact in state:
            if match(fact, "ontray", "*", "*"):
                sandwich, tray = get_parts(fact)[1:3]
                is_gf = sandwich_is_gf.get(sandwich, False) # Default to not GF if predicate not present
                tray_loc = tray_locations.get(tray)

                if tray_loc == 'kitchen':
                    if is_gf:
                        sandwiches_on_kitchen_tray['gf'] += 1
                    else:
                        sandwiches_on_kitchen_tray['reg'] += 1
                elif tray_loc: # On tray at some other location
                    if tray_loc not in sandwiches_at_loc_on_tray:
                        sandwiches_at_loc_on_tray[tray_loc] = {'gf': 0, 'reg': 0}
                    if is_gf:
                        sandwiches_at_loc_on_tray[tray_loc]['gf'] += 1
                    else:
                        sandwiches_at_loc_on_tray[tray_loc]['reg'] += 1

            elif match(fact, "at_kitchen_sandwich", "*"):
                sandwich = get_parts(fact)[1]
                is_gf = sandwich_is_gf.get(sandwich, False)
                if is_gf:
                    sandwiches_at_kitchen['gf'] += 1
                else:
                    sandwiches_at_kitchen['reg'] += 1

            elif match(fact, "notexist", "*"):
                count_notexist += 1

            # Count ingredients in kitchen
            elif match(fact, "at_kitchen_bread", "*"):
                 bread = get_parts(fact)[1]
                 # Check if the no_gluten_bread predicate for this specific bread is in the state
                 if f"(no_gluten_bread {bread})" in state:
                     count_bread_gf += 1
                 else:
                     count_bread_reg += 1
            elif match(fact, "at_kitchen_content", "*"):
                 content = get_parts(fact)[1]
                 # Check if the no_gluten_content predicate for this specific content is in the state
                 if f"(no_gluten_content {content})" in state:
                     count_content_gf += 1
                 else:
                     count_content_reg += 1


        # 3. Determine if new sandwiches can potentially be made (simplistic check based on counts)
        # Can make a new GF sandwich if there's a notexist object, GF bread, and GF content
        can_make_new_gf = (count_notexist > 0 and count_bread_gf > 0 and count_content_gf > 0)
        # Can make a new Regular sandwich if there's a notexist object, any bread, and any content
        can_make_new_reg = (count_notexist > 0 and (count_bread_gf + count_bread_reg > 0) and (count_content_gf + count_content_reg > 0))


        # 4. Calculate total heuristic cost by summing minimum costs per child
        total_cost = 0
        LARGE_PENALTY = 1000 # Penalty for children who cannot be served with available/creatable sandwiches

        for child, place, allergy in unserved_children:
            min_child_cost = LARGE_PENALTY # Initialize with penalty

            is_allergic = (allergy == 'gluten')

            # Check Cost 1 (Serve): Suitable sandwich on tray at child's location
            cost1_possible = False
            if is_allergic:
                if sandwiches_at_loc_on_tray.get(place, {}).get('gf', 0) > 0:
                    cost1_possible = True
            else: # Not allergic, can take Reg or GF
                 if sandwiches_at_loc_on_tray.get(place, {}).get('reg', 0) > 0 or \
                    sandwiches_at_loc_on_tray.get(place, {}).get('gf', 0) > 0:
                    cost1_possible = True

            if cost1_possible:
                min_child_cost = 1
            else:
                # Check Cost 2 (Move & Serve): Suitable sandwich on tray in kitchen
                cost2_possible = False
                if is_allergic:
                    if sandwiches_on_kitchen_tray.get('gf', 0) > 0:
                        cost2_possible = True
                else: # Not allergic
                    if sandwiches_on_kitchen_tray.get('reg', 0) > 0 or \
                       sandwiches_on_kitchen_tray.get('gf', 0) > 0:
                        cost2_possible = True

                if cost2_possible:
                    min_child_cost = 2
                else:
                    # Check Cost 3 (Put, Move & Serve): Suitable sandwich at kitchen, not on tray
                    cost3_possible = False
                    if is_allergic:
                        if sandwiches_at_kitchen.get('gf', 0) > 0:
                            cost3_possible = True
                    else: # Not allergic
                        if sandwiches_at_kitchen.get('reg', 0) > 0 or \
                           sandwiches_at_kitchen.get('gf', 0) > 0:
                            cost3_possible = True

                    if cost3_possible:
                        min_child_cost = 3
                    else:
                        # Check Cost 4 (Make, Put, Move & Serve): Suitable sandwich needs to be made
                        cost4_possible = False
                        if is_allergic:
                            if can_make_new_gf:
                                cost4_possible = True
                        else: # Not allergic
                            if can_make_new_reg:
                                cost4_possible = True

                        if cost4_possible:
                            min_child_cost = 4
                        # Else: min_child_cost remains LARGE_PENALTY

            total_cost += min_child_cost

        return total_cost

