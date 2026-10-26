from fnmatch import fnmatch
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

    Summary:
    This heuristic estimates the number of actions required to serve all children.
    It counts the number of sandwiches that need to be made, put on trays,
    moved to the children's locations, and finally served. It aggregates these
    counts across all unserved children, assuming resources (like trays) can be
    shared efficiently where possible (e.g., one tray move can potentially serve
    multiple children at the same location).

    Assumptions:
    - Each unserved child requires one sandwich of the correct type (gluten-free or regular).
    - Sandwiches must be made in the kitchen, put on a tray in the kitchen,
      the tray moved to the child's location, and then the child served.
    - Enough ingredients and 'notexist' sandwich slots are available to make needed sandwiches.
    - Enough trays are available in the kitchen to put sandwiches on.
    - A single tray movement to a location is sufficient to bring all sandwiches needed at that location
      that are not already there on trays.
    - The cost of each action (make, put_on_tray, move_tray, serve) is 1.

    Heuristic Initialization:
    - The heuristic does not require specific initialization from task.goals or task.static
      within __init__ for its core logic, as all necessary information (goal state structure,
      static predicates like allergies) is derived from the state during the heuristic calculation.
      However, task.goals and task.static are stored as per the base class requirement.

    Step-By-Step Thinking for Computing Heuristic:
    1. Parse the current state to identify unserved children, their locations, and allergies,
       as well as the locations and types of existing sandwiches and trays.
    2. Count the total number of unserved children. If zero, the heuristic is 0 (goal state).
    3. Determine the total number of gluten-free and regular sandwiches required (equal to the counts of unserved allergic and non-allergic children).
    4. Count the number of existing sandwiches of each type currently in the state (anywhere).
    5. Calculate the number of sandwiches of each type that still need to be made (`cost_make`) based on the deficit between required and existing sandwiches.
    6. Count the number of sandwiches currently in the kitchen that are *not* on trays.
    7. Calculate the number of sandwiches that need the `put_on_tray` action. This includes sandwiches currently in the kitchen (not on trays) plus those calculated in step 5 that need to be made (`cost_put_on_tray`).
    8. For each location (other than the kitchen) where unserved children are waiting, calculate the deficit of suitable sandwiches on trays at that location (needed for children vs. available on trays at that location).
    9. Identify the set of locations (other than kitchen) that have a deficit from step 8 (i.e., need *any* sandwich delivery).
    10. From the set in step 9, identify the subset of locations that do *not* currently have a tray present. The size of this subset is the estimated number of tray movements needed (`cost_move_tray`).
    11. The number of `serve` actions needed is simply the total number of unserved children (`cost_serve`).
    12. The total heuristic value is the sum of `cost_make`, `cost_put_on_tray`, `cost_move_tray`, and `cost_serve`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic.
        task: The planning task object.
        """
        # Store task information as required by the base class.
        self.goals = task.goals
        self.static_facts = task.static

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        node: The search node containing the state.
        Returns: The estimated number of actions to reach the goal.
        """
        state = node.state

        # Collect facts by predicate for easier access
        facts_by_predicate = {}
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts if any
            predicate = parts[0]
            if predicate not in facts_by_predicate:
                facts_by_predicate[predicate] = []
            facts_by_predicate[predicate].append(parts)

        # --- Step 1: Extract relevant information ---
        unserved_children = set()
        child_info = {} # child -> {'place': place, 'allergy': allergy}
        sandwich_info = {} # sandwich -> {'type': 'gf' or 'reg', 'location': 'kitchen' or tray_name}
        tray_location = {} # tray -> place
        all_places = {'kitchen'} # Start with the constant kitchen

        # Identify all children and their properties
        all_children_waiting = set()
        for parts in facts_by_predicate.get('waiting', []):
            child, place = parts[1], parts[2]
            child_info[child] = {'place': place}
            all_children_waiting.add(child)
            all_places.add(place)

        for parts in facts_by_predicate.get('allergic_gluten', []):
            child = parts[1]
            if child in child_info: # Only track children who are waiting
                child_info[child]['allergy'] = 'allergic'

        for parts in facts_by_predicate.get('not_allergic_gluten', []):
            child = parts[1]
            if child in child_info: # Only track children who are waiting
                child_info[child]['allergy'] = 'non_allergic'

        # Identify served children
        served_children_set = set(parts[1] for parts in facts_by_predicate.get('served', []))

        # Determine unserved children (those waiting but not served)
        unserved_children = all_children_waiting - served_children_set

        # If no unserved children, goal reached
        if not unserved_children:
            return 0

        # Identify sandwich locations and types
        for parts in facts_by_predicate.get('at_kitchen_sandwich', []):
            sandwich = parts[1]
            sandwich_info[sandwich] = {'location': 'kitchen'}

        for parts in facts_by_predicate.get('ontray', []):
            sandwich, tray = parts[1], parts[2]
            sandwich_info[sandwich] = {'location': tray} # Location is the tray name

        for parts in facts_by_predicate.get('no_gluten_sandwich', []):
            sandwich = parts[1]
            # Ensure sandwich was seen in at_kitchen_sandwich or ontray, or create entry
            if sandwich not in sandwich_info:
                 sandwich_info[sandwich] = {} # Create entry if not seen yet
            sandwich_info[sandwich]['type'] = 'gf'

        # Ensure all sandwiches found have a type (default to regular if not marked gf)
        for s_info in sandwich_info.values():
            if 'type' not in s_info:
                s_info['type'] = 'reg'

        # Identify tray locations
        for parts in facts_by_predicate.get('at', []):
            item, place = parts[1], parts[2] # 'at' is only for trays in this domain
            tray_location[item] = place
            all_places.add(place) # Add places from tray locations too

        # notexist, ingredients predicates are ignored for this heuristic's assumptions

        # --- Step 2-11: Heuristic Calculation ---

        # 2. Count sandwiches needed by type
        needed_gf = sum(1 for c in unserved_children if child_info[c].get('allergy') == 'allergic')
        needed_reg = sum(1 for c in unserved_children if child_info[c].get('allergy') == 'non_allergic')

        # 3. Count available sandwiches by type anywhere
        available_gf_made = sum(1 for s_info in sandwich_info.values() if s_info['type'] == 'gf')
        available_reg_made = sum(1 for s_info in sandwich_info.values() if s_info['type'] == 'reg')

        # 4. Calculate sandwiches to make
        to_make_gf = max(0, needed_gf - available_gf_made)
        to_make_reg = max(0, needed_reg - available_reg_made)
        cost_make = to_make_gf + to_make_reg

        # 5. Count sandwiches in kitchen not on trays
        available_gf_kitchen = sum(1 for s, s_info in sandwich_info.items() if s_info['type'] == 'gf' and s_info['location'] == 'kitchen')
        available_reg_kitchen = sum(1 for s, s_info in sandwich_info.items() if s_info['type'] == 'reg' and s_info['location'] == 'kitchen')

        # 6. Calculate sandwiches needing put_on_tray
        # These are sandwiches currently in the kitchen (not on trays) plus those that need to be made.
        to_put_on_tray = available_gf_kitchen + available_reg_kitchen + to_make_gf + to_make_reg
        cost_put_on_tray = to_put_on_tray # Assumes trays available in kitchen

        # 7. Calculate missing sandwiches on trays at places P != kitchen
        # This determines which places need deliveries and what type/count of sandwiches they need.
        missing_at_place = {} # place -> {'gf': count, 'reg': count}
        for place in all_places:
            if place == 'kitchen': continue

            # Count unserved children at this place needing each type
            needed_gf_at_P = sum(1 for c in unserved_children if child_info[c].get('place') == place and child_info[c].get('allergy') == 'allergic')
            needed_reg_at_P = sum(1 for c in unserved_children if child_info[c].get('place') == place and child_info[c].get('allergy') == 'non_allergic')

            # Count suitable sandwiches already on trays at this place
            avail_gf_at_P = sum(1 for s, s_info in sandwich_info.items() if s_info['type'] == 'gf' and s_info['location'] in tray_location and tray_location[s_info['location']] == place)
            avail_reg_at_P = sum(1 for s, s_info in sandwich_info.items() if s_info['type'] == 'reg' and s_info['location'] in tray_location and tray_location[s_info['location']] == place)

            # Calculate deficit at this place
            missing_gf_at_P = max(0, needed_gf_at_P - avail_gf_at_P)
            missing_reg_at_P = max(0, needed_reg_at_P - avail_reg_at_P)

            if missing_gf_at_P > 0 or missing_reg_at_P > 0:
                 missing_at_place[place] = {'gf': missing_gf_at_P, 'reg': missing_reg_at_P}


        # 8. Calculate tray movements needed
        # A tray move is needed for a place if it needs sandwiches delivered AND no tray is currently there.
        places_needing_tray_move = set()
        trays_at_place_set = set(tray_location[t] for t in tray_location if tray_location[t] != 'kitchen')

        for place in missing_at_place: # Iterate only places that need deliveries
            if place not in trays_at_place_set:
                 places_needing_tray_move.add(place)

        cost_move_tray = len(places_needing_tray_move)

        # 9. Calculate serve actions
        cost_serve = len(unserved_children)

        # Total heuristic
        total_cost = cost_make + cost_put_on_tray + cost_move_tray + cost_serve

        return total_cost
