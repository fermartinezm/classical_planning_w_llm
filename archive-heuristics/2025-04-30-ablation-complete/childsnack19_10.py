from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def parse_fact(fact_string):
    """Extract the components of a PDDL fact string."""
    # Remove parentheses and split by space
    # Handle cases where fact_string might be empty or malformed, though unlikely with planner output
    if not fact_string or not fact_string.startswith('(') or not fact_string.endswith(')'):
        return tuple()
    parts = fact_string[1:-1].split()
    return tuple(parts)

def match(fact_parts, *args):
    """
    Check if parsed PDDL fact parts match a given pattern.
    - `fact_parts`: Tuple of fact components (e.g., ('at', 'ball1', 'rooma')).
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact parts match the pattern, else `False`.
    """
    if len(fact_parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(fact_parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children.
    It counts the necessary steps: serving each unserved child, making sandwiches
    to meet the demand not covered by existing ones, putting needed sandwiches
    onto trays at the kitchen, and moving trays to locations where unserved
    children are waiting and no tray is currently present.

    # Assumptions
    - Each unserved child requires one 'serve' action.
    - Sandwiches needed are counted based on deficits at locations after accounting
      for suitable sandwiches already on trays at those specific locations.
    - Sandwiches are made in the kitchen if needed to cover the total deficit
      across all locations not met by existing sandwiches (at kitchen or on trays at kitchen).
    - Sandwiches newly made or already `at_kitchen_sandwich` that are needed
      to fulfill deficits must be put on a tray at the kitchen ('put_on_tray' action).
    - A 'move_tray' action is needed for each location (other than kitchen)
      with a sandwich deficit that does not currently have a tray.
    - Enough bread, content, and sandwich objects exist to make needed sandwiches.
    - Trays can hold all sandwiches needed for a location.
    - Moving a tray from kitchen to any other location takes 1 action.
    - Allergy information and child waiting locations are static.

    # Heuristic Initialization
    - Extracts goal children (those who need to be served).
    - Extracts static facts about child allergy status and child waiting locations.
    - Extracts static facts about gluten-free bread and content (though not directly used in the main heuristic logic, they define what makes a sandwich GF).

    # Step-By-Step Thinking for Computing Heuristic
    1.  Initialize total heuristic cost to 0.
    2.  Identify all children who are not yet served in the current state by comparing the state's 'served' facts against the goal children identified during initialization.
    3.  If there are no unserved children, the goal is reached, return 0.
    4.  Add the number of unserved children to the cost. This accounts for the final 'serve' action for each child.
    5.  Group the unserved children by their waiting location. For each location, count the number of allergic children (who need a GF sandwich) and non-allergic children (who need any sandwich).
    6.  Identify sandwiches currently on trays in the state and their locations. Count available GF and Regular sandwiches already on trays at each location.
    7.  For each location with unserved children, calculate the *deficit* of needed GF and Regular sandwiches. This deficit is the number needed minus the number of suitable sandwiches already available on trays at that specific location. Prioritize using available GF sandwiches for allergic children first.
    8.  Sum the GF deficits and Regular deficits across *all* locations (including kitchen if children wait there) to get the total number of GF and Regular sandwiches that must be sourced from the kitchen (either made or already present there).
    9.  Count available GF and Regular sandwiches currently at the kitchen (either `at_kitchen_sandwich` or `ontray` at kitchen).
    10. Calculate the number of GF and Regular sandwiches that need to be *made*. This is the total deficit from the kitchen minus the available sandwiches at the kitchen. Prioritize using available sandwiches already on trays at the kitchen, then those `at_kitchen_sandwich`. Add the total count of sandwiches to be made to the heuristic cost.
    11. Calculate the number of sandwiches that need to be *put on a tray* at the kitchen. These are the sandwiches needed from the kitchen that are currently `at_kitchen_sandwich` (either initially or newly made). Add this count to the heuristic cost.
    12. Identify locations (other than kitchen) that have a non-zero sandwich deficit (meaning sandwiches need to be brought there) and do not currently have *any* tray present. Add the count of such locations to the heuristic cost. This accounts for the 'move_tray' action needed to get a tray to that location.
    13. Return the total calculated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information from the task.
        """
        self.goals = task.goals
        self.static = task.static

        self.all_children = set()
        self.goal_children = set()
        self.child_allergy = {} # {child: True/False}
        self.child_location = {} # {child: place}
        self.gf_bread = set() # Not strictly needed for this heuristic logic, but good to parse
        self.gf_content = set() # Not strictly needed

        # Extract goal children
        for goal in self.goals:
             parts = parse_fact(goal)
             if match(parts, 'served', '*'):
                 self.goal_children.add(parts[1])

        # Extract static facts
        for fact in self.static:
            parts = parse_fact(fact)
            if match(parts, 'allergic_gluten', '*'):
                child = parts[1]
                self.all_children.add(child)
                self.child_allergy[child] = True
            elif match(parts, 'not_allergic_gluten', '*'):
                child = parts[1]
                self.all_children.add(child)
                self.child_allergy[child] = False
            elif match(parts, 'waiting', '*', '*'):
                child, place = parts[1], parts[2]
                self.child_location[child] = place
            elif match(parts, 'no_gluten_bread', '*'):
                self.gf_bread.add(parts[1])
            elif match(parts, 'no_gluten_content', '*'):
                self.gf_content.add(parts[1])

        # Ensure all children in goals are accounted for in static (standard PDDL assumption)
        # If a child is in goals but not in static allergy/waiting, heuristic might be inaccurate.
        # We assume valid PDDL structure where all relevant objects/predicates are defined.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        cost = 0

        # --- Parse current state facts ---
        served_children = set()
        at_kitchen_sandwiches = set()
        ontray_sandwiches = set() # {(sandwich, tray)}
        tray_locations = {} # {tray: place}
        gf_sandwiches = set()

        state_facts_parsed = [parse_fact(f) for f in state]

        for parts in state_facts_parsed:
            if not parts: continue # Skip empty or malformed facts

            predicate = parts[0]
            if predicate == 'served' and len(parts) == 2:
                served_children.add(parts[1])
            elif predicate == 'at_kitchen_sandwich' and len(parts) == 2:
                at_kitchen_sandwiches.add(parts[1])
            elif predicate == 'ontray' and len(parts) == 3:
                ontray_sandwiches.add((parts[1], parts[2]))
            elif predicate == 'at' and len(parts) == 3:
                tray_locations[parts[1]] = parts[2]
            elif predicate == 'no_gluten_sandwich' and len(parts) == 2:
                gf_sandwiches.add(parts[1])

        # --- Step 1: Cost for serving ---
        unserved_children = {c for c in self.goal_children if c not in served_children}

        if not unserved_children:
            return 0 # Goal state reached

        cost += len(unserved_children)

        # --- Step 2: Calculate sandwich deficits at each location ---
        needed_at_loc = {} # {location: {'gf': count, 'reg': count}}
        for child in unserved_children:
            loc = self.child_location.get(child)
            if loc is None: continue # Should not happen in valid PDDL

            if loc not in needed_at_loc:
                needed_at_loc[loc] = {'gf': 0, 'reg': 0}

            if self.child_allergy.get(child, False): # Default to not allergic if status unknown
                needed_at_loc[loc]['gf'] += 1
            else:
                needed_at_loc[loc]['reg'] += 1 # Non-allergic needs any sandwich

        # Count available sandwiches already on trays at each location
        available_ontray_at_loc = {} # {location: {'gf': count, 'reg': count}}
        for s, t in ontray_sandwiches:
            loc = tray_locations.get(t)
            if loc is None: continue # Skip if tray location is unknown

            if loc not in available_ontray_at_loc:
                 available_ontray_at_loc[loc] = {'gf': 0, 'reg': 0}

            is_gf = s in gf_sandwiches
            if is_gf:
                available_ontray_at_loc[loc]['gf'] += 1
            else:
                available_ontray_at_loc[loc]['reg'] += 1

        # Calculate deficit at each location
        deficit_at_loc = {} # {location: {'gf': count, 'reg': count}}
        for loc, needs in needed_at_loc.items():
            avail_ontray = available_ontray_at_loc.get(loc, {'gf': 0, 'reg': 0})

            # Prioritize using available GF for allergic needs
            still_needed_gf = max(0, needs['gf'] - avail_ontray['gf'])
            remaining_avail_ontray_gf = max(0, avail_ontray['gf'] - needs['gf'])

            # Non-allergic can use remaining available GF or available Regular
            still_needed_reg = max(0, needs['reg'] - (avail_ontray['reg'] + remaining_avail_ontray_gf))

            if still_needed_gf > 0 or still_needed_reg > 0:
                 deficit_at_loc[loc] = {'gf': still_needed_gf, 'reg': still_needed_reg}


        # --- Step 3: Calculate total sandwich deficit from kitchen ---
        total_deficit_gf = sum(d['gf'] for d in deficit_at_loc.values())
        total_deficit_reg = sum(d['reg'] for d in deficit_at_loc.values())

        # --- Step 4: Count available sandwiches at kitchen ---
        available_at_kitchen_only_gf = sum(1 for s in at_kitchen_sandwiches if s in gf_sandwiches)
        available_at_kitchen_only_reg = sum(1 for s in at_kitchen_sandwiches if s not in gf_sandwiches)

        available_ontray_kitchen_gf = available_ontray_at_loc.get('kitchen', {}).get('gf', 0)
        available_ontray_kitchen_reg = available_ontray_at_loc.get('kitchen', {}).get('reg', 0)

        # --- Step 5 & 6: Calculate sandwiches to make and needing put_on_tray at kitchen ---
        # Sandwiches needed from kitchen are the total deficits
        needed_from_kitchen_gf = total_deficit_gf
        needed_from_kitchen_reg = total_deficit_reg

        # Use available sandwiches at kitchen (ontray first, then at_kitchen_sandwich)
        use_ontray_kitchen_gf = min(needed_from_kitchen_gf, available_ontray_kitchen_gf)
        needed_from_kitchen_gf -= use_ontray_kitchen_gf

        use_ontray_kitchen_reg = min(needed_from_kitchen_reg, available_ontray_kitchen_reg)
        needed_from_kitchen_reg -= use_ontray_kitchen_reg

        use_at_kitchen_gf = min(needed_from_kitchen_gf, available_at_kitchen_only_gf)
        needed_from_kitchen_gf -= use_at_kitchen_gf

        use_at_kitchen_reg = min(needed_from_kitchen_reg, available_at_kitchen_only_reg)
        needed_from_kitchen_reg -= use_at_kitchen_reg

        # Remaining needed must be made
        make_gf = needed_from_kitchen_gf
        make_reg = needed_from_kitchen_reg

        cost += make_gf + make_reg # Cost for make_sandwich / make_sandwich_no_gluten

        # Sandwiches needing put_on_tray at kitchen are those used from at_kitchen_sandwich plus those newly made
        needed_put_on_tray = use_at_kitchen_gf + use_at_kitchen_reg + make_gf + make_reg
        cost += needed_put_on_tray # Cost for put_on_tray

        # --- Step 7: Calculate tray moves ---
        # A tray move is needed for each location (not kitchen) that has a deficit
        # AND does not currently have a tray.
        locations_with_deficit = {loc for loc, deficit in deficit_at_loc.items() if loc != 'kitchen'}
        trays_at_locations = {loc for tray, loc in tray_locations.items() if loc != 'kitchen'}
        locations_needing_tray_move = locations_with_deficit - trays_at_locations

        cost += len(locations_needing_tray_move) # Cost for move_tray

        return cost
