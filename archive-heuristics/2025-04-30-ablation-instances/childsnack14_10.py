from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or non-string input gracefully
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
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

    # Summary
    This heuristic estimates the number of actions required to serve all unserved children.
    It counts the number of 'serve' actions needed and adds the estimated cost
    to get the necessary sandwiches (of the correct type) onto trays at the
    children's waiting locations. The cost to get a sandwich to a location
    depends on its current state: needs making (cost 3), at kitchen (cost 2),
    or on a tray elsewhere (cost 1).

    # Assumptions
    - Each unserved child requires one suitable sandwich.
    - All necessary ingredients for making sandwiches are available at the kitchen.
    - Trays can be moved between any two places in one 'move_tray' action.
    - There are always enough trays available to move sandwiches.
    - The number of sandwiches that can be made is limited by the number of
      '(notexist ?s)' facts in the current state.

    # Heuristic Initialization
    The heuristic extracts static information:
    - Which children are waiting where (`waiting_info`).
    - Which children are allergic (`allergy_info`).

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify all unserved children. The number of unserved children is the base cost (each needs a 'serve' action). If this count is zero, the goal is reached, and the heuristic is 0.
    2. For each unserved child, determine the type of sandwich needed (no-gluten if allergic, any otherwise) and the required location (the child's waiting place). Count the total number of no-gluten and 'any' sandwiches required at each waiting place.
    3. Count the number of no-gluten and regular sandwiches currently available at each place (including the kitchen and any place where a tray is located). Also, count the number of available '(notexist ?s)' slots.
    4. Calculate the deficit of no-gluten and regular sandwiches at each waiting place (i.e., how many more are needed than are currently available at that specific place).
    5. Sum the deficits across all waiting places to get the total number of no-gluten and 'any' sandwiches needed from *elsewhere* (kitchen, other trays, or needing to be made).
    6. Count the total number of no-gluten and regular sandwiches available *not* at the waiting places (i.e., at the kitchen or on trays at non-waiting places).
    7. Determine how many sandwiches need to be *made* to cover the total deficit from elsewhere, considering the limit imposed by `(notexist ?s)` facts. Each sandwich made costs 3 actions (make, put on tray, move tray).
    8. Determine how many sandwiches needed from elsewhere can be supplied by existing sandwiches not at the waiting places.
    9. Allocate the needed sandwiches from existing sources, prioritizing those on trays elsewhere (cost 1: move tray) over those at the kitchen (cost 2: put on tray, move tray).
    10. Sum the costs: (number of unserved children * 1) + (number to make * 3) + (number from kitchen * 2) + (number from other trays * 1).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.
        """
        self.goals = task.goals # Goal conditions (used implicitly by checking served children)
        static_facts = task.static

        self.waiting_info = {} # child -> place
        self.allergy_info = {} # child -> is_allergic (bool)

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == 'waiting' and len(parts) == 3:
                child, place = parts[1], parts[2]
                self.waiting_info[child] = place
            elif parts[0] == 'allergic_gluten' and len(parts) == 2:
                child = parts[1]
                self.allergy_info[child] = True
            elif parts[0] == 'not_allergic_gluten' and len(parts) == 2:
                child = parts[1]
                self.allergy_info[child] = False


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unserved children and their requirements
        unserved_children = set()
        # Initialize required counts only for actual waiting places
        waiting_places_set = set(self.waiting_info.values())
        required_ng_at_p = {p: 0 for p in waiting_places_set}
        required_any_at_p = {p: 0 for p in waiting_places_set}

        served_children_in_state = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        for child, place in self.waiting_info.items():
            if child not in served_children_in_state:
                unserved_children.add(child)
                if self.allergy_info.get(child, False): # Default to not allergic if info missing
                    required_ng_at_p[place] += 1
                else:
                    required_any_at_p[place] += 1

        # Base cost: one serve action per unserved child
        serve_cost = len(unserved_children)

        # If no children unserved, goal reached, heuristic is 0
        if serve_cost == 0:
            return 0

        # 2. Count available sandwiches by type and location, and notexist slots
        tray_locations = {} # tray -> place
        current_notexist_count = 0

        sandwich_is_ng = {} # sandwich -> bool (is_no_gluten)
        sandwich_loc_raw = {} # sandwich -> 'kitchen' or tray_id

        # Collect raw locations and types first
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == 'at' and len(parts) == 3:
                obj, place = parts[1], parts[2]
                # Assuming only trays are 'at' places other than kitchen constant
                if obj.startswith('tray'):
                     tray_locations[obj] = place
            elif parts[0] == 'at_kitchen_sandwich' and len(parts) == 2:
                sandwich = parts[1]
                sandwich_loc_raw[sandwich] = 'kitchen'
            elif parts[0] == 'ontray' and len(parts) == 3:
                 sandwich, tray = parts[1], parts[2]
                 sandwich_loc_raw[sandwich] = tray
            elif parts[0] == 'no_gluten_sandwich' and len(parts) == 2:
                 sandwich_is_ng[parts[1]] = True
            elif parts[0] == 'notexist' and len(parts) == 2:
                 current_notexist_count += 1

        # Identify all places where sandwiches might be located (kitchen + tray locations)
        all_places_in_state = {'kitchen'} | set(tray_locations.values())

        # Populate supply counts based on raw locations and tray locations
        supply_ng_at_place = {p: 0 for p in all_places_in_state}
        supply_reg_at_place = {p: 0 for p in all_places_in_state}

        for sandwich, raw_loc in sandwich_loc_raw.items():
            is_ng = sandwich_is_ng.get(sandwich, False) # Default to regular if type unknown

            place = None
            if raw_loc == 'kitchen':
                place = 'kitchen'
            elif raw_loc.startswith('tray'):
                tray_id = raw_loc
                place = tray_locations.get(tray_id) # Get place from tray_locations

            if place and place in all_places_in_state: # Only count if at a known relevant place
                 if is_ng:
                     supply_ng_at_place[place] += 1
                 else:
                     supply_reg_at_place[place] += 1

        # 3. Calculate deficits at waiting places
        total_deficit_ng = 0
        total_deficit_any = 0 # This represents the need for *any* sandwich by non-allergic children

        for place in waiting_places_set: # Iterate through actual waiting places
            needed_ng = required_ng_at_p.get(place, 0)
            needed_any = required_any_at_p.get(place, 0)
            available_ng_at_p = supply_ng_at_place.get(place, 0)
            available_reg_at_place = supply_reg_at_place.get(place, 0)

            # Deficit of NG sandwiches needed for allergic children
            deficit_ng_at_p = max(0, needed_ng - available_ng_at_p)
            total_deficit_ng += deficit_ng_at_p

            # Deficit of Any sandwiches needed for non-allergic children
            # This deficit must be covered by sandwiches not already at this place.
            # We calculate the deficit of regular sandwiches needed.
            deficit_any_at_p = max(0, needed_any - available_reg_at_place)
            total_deficit_any += deficit_any_at_p


        # 4. Calculate available sandwiches not at waiting places
        avail_ng_kitchen = supply_ng_at_place.get('kitchen', 0)
        avail_reg_kitchen = supply_reg_at_place.get('kitchen', 0)

        avail_ng_other_trays = 0
        avail_reg_other_trays = 0
        for place in all_places_in_state:
            # If place is not kitchen AND not one of the waiting places
            if place != 'kitchen' and place not in waiting_places_set:
                 avail_ng_other_trays += supply_ng_at_place.get(place, 0)
                 avail_reg_other_trays += supply_reg_at_place.get(place, 0)

        # 5. Calculate total items needed from elsewhere and available not at destination
        # Total items needed from elsewhere = total deficit of NG + total deficit of Any
        # (NG sandwiches can fulfill Any needs)
        total_needed_from_elsewhere = total_deficit_ng + total_deficit_any

        # Total available items not at destination
        total_avail_not_at_dest = avail_ng_kitchen + avail_reg_kitchen + avail_ng_other_trays + avail_reg_other_trays

        # 6. Calculate items to make and their cost
        # We need to make items if the total needed from elsewhere exceeds what's available elsewhere.
        # The number to make is capped by the available notexist slots.
        make_count = min(current_notexist_count, max(0, total_needed_from_elsewhere - total_avail_not_at_dest))
        cost_make = make_count * 3 # make (1) + put_on_tray (1) + move_tray (1) = 3

        # 7. Calculate items from elsewhere that will be used and their cost
        # These are the items needed from elsewhere that don't need to be made.
        used_from_elsewhere = total_needed_from_elsewhere - make_count

        # Allocate used_from_elsewhere from cheapest sources first (other trays < kitchen)
        used_from_other_trays = min(used_from_elsewhere, avail_ng_other_trays + avail_reg_other_trays)
        cost_other_trays = used_from_other_trays * 1 # move_tray (1)

        used_from_kitchen = used_from_elsewhere - used_from_other_trays
        cost_kitchen = used_from_kitchen * 2 # put_on_tray (1) + move_tray (1) = 2

        # Total heuristic = Serve cost + Make cost + Other trays cost + Kitchen cost
        total_cost = serve_cost + cost_make + cost_other_trays + cost_kitchen

        return total_cost
