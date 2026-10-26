from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Utility functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all unserved
    children. It counts the number of children still needing service, the number
    of sandwiches that need to be made, the number of sandwiches that need to be
    put on trays at the kitchen, and the number of tray movements required to
    reach locations where children are waiting.

    # Assumptions
    - Children wait at fixed places.
    - Trays can hold multiple sandwiches.
    - Sandwiches are made at the kitchen, put on trays at the kitchen, and then
      trays are moved to the children's locations for serving.
    - Gluten-free constraints are considered for making and serving.
    - Ingredient availability (bread/content) and 'notexist' sandwich slots
      are implicitly assumed to be sufficient for the 'make' actions counted.

    # Heuristic Initialization
    - Extract static information: which children are allergic, where each child
      is waiting, and identify all possible children and places (including kitchen).

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic sums up estimated costs for different stages of satisfying
    the goal (serving all children):

    1.  **Cost of Serving:** Count the number of children who are in the goal
        state as 'served' but are not yet 'served' in the current state. Each
        such child requires one 'serve' action.

    2.  **Cost of Making Sandwiches:** Determine how many suitable sandwiches
        (gluten-free for allergic children, any for non-allergic) are needed
        from the kitchen supply chain (made or already at kitchen) to satisfy
        the demand from unserved children at their waiting places. Then, count
        how many of these needed sandwiches are *not* already available at the
        kitchen (either `at_kitchen_sandwich` or `ontray` at `kitchen`),
        considering GF needs first. This deficit represents the number of
        sandwiches that must be made from ingredients.

    3.  **Cost of Putting Sandwiches on Trays:** Any sandwich that is needed
        from the kitchen supply chain (i.e., needed at a place other than the
        kitchen) must be put on a tray at the kitchen before the tray can be
        moved. The number of such sandwiches is the total demand from places
        other than the kitchen that isn't met by sandwiches already on trays
        at those places.

    4.  **Cost of Moving Trays:** For each distinct place (other than the kitchen)
        where unserved children are waiting and need sandwiches, if there is
        currently no tray located at that place, one 'move_tray' action is
        estimated to be needed to bring a tray there.

    The total heuristic value is the sum of these four cost components.

    Detailed Calculation Steps:
    a. Identify all unserved children based on the goal and current state.
    b. For each place P (excluding kitchen), count unserved allergic and non-allergic
       children waiting there (`needed_gf_P`, `needed_any_P`). This requires
       looking up the waiting place for each unserved child using static facts.
    c. For each place P, count available GF and Any sandwiches already on trays
       at that place (`avail_ontray_P_gf`, `avail_ontray_P_any`) by parsing the
       current state facts about `ontray` and `at` tray locations, and `no_gluten_sandwich`.
    d. Calculate the deficit at each place P: `still_need_P_gf = max(0, needed_gf_P - avail_ontray_P_gf)`,
       `still_need_P_any = max(0, needed_any_P - (avail_ontray_P_any + avail_ontray_P_gf - min(needed_gf_P, avail_ontray_P_gf)))`.
       These represent sandwiches needed from the kitchen supply chain for place P.
    e. Sum deficits across all places P != kitchen to get `total_still_need_gf` and `total_still_need_any`.
    f. Count available GF and Any sandwiches at the kitchen (either `at_kitchen_sandwich`
       or `ontray` at `kitchen`) (`avail_kitchen_gf`, `avail_kitchen_any`) by parsing
       current state facts.
    g. Calculate sandwiches to make: `make_gf = max(0, total_still_need_gf - avail_kitchen_gf)`,
       `rem_avail_kitchen_gf = avail_kitchen_gf - min(total_still_need_gf, avail_kitchen_gf)`,
       `make_any = max(0, total_still_need_any - (avail_kitchen_any + rem_avail_kitchen_gf))`.
       Total make cost = `make_gf + make_any`.
    h. Total sandwiches needed from kitchen chain = `total_still_need_gf + total_still_need_any`.
       Total put_on_tray cost = `total_still_need_gf + total_still_need_any`.
    i. Count distinct places P != kitchen where `(still_need_P_gf + still_need_P_any) > 0`
       and no tray is present (`trays_at_place[P] == 0`). Total move_tray cost = this count.
       Tray locations (`trays_at_place`) are parsed from current state facts.
    j. Total heuristic = Cost of Serving + Cost of Making + Cost of Putting on Tray + Cost of Moving Tray.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.allergic_children = set()
        self.waiting_places_static = {} # child -> place
        self.all_children = set()
        self.all_places = set() # Includes kitchen

        # Extract static information from task.static and task.goals
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "allergic_gluten":
                self.allergic_children.add(parts[1])
            elif parts[0] == "waiting":
                child, place = parts[1], parts[2]
                self.waiting_places_static[child] = place
                self.all_children.add(child)
                self.all_places.add(place) # Add all places mentioned in waiting
            elif parts[0] == "not_allergic_gluten":
                 self.all_children.add(parts[1])
            # We don't strictly need no_gluten_bread/content for this heuristic's calculation logic,
            # as we only count the *need* to make GF/Any sandwiches, assuming ingredients exist.
            # We also don't need to parse all objects explicitly if we derive places/children from facts.

        # Add children from goals if they weren't in static (e.g., if not waiting initially)
        for goal in self.goals:
             if match(goal, "served", "*"):
                 self.all_children.add(get_parts(goal)[1])

        # Ensure 'kitchen' is in the set of places
        self.all_places.add('kitchen')


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Step a: Identify unserved children ---
        served_children_in_state = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        goal_children = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}
        unserved_children = goal_children - served_children_in_state
        num_unserved = len(unserved_children)

        # If all children are served, the goal is reached.
        if num_unserved == 0:
            return 0

        # --- Parse current state for relevant facts ---
        ontray_map = {} # sandwich -> tray
        trays_at_map = {} # tray -> place
        sandwiches_kitchen_set = set() # sandwiches at kitchen (not on tray)
        sandwiches_no_gluten_set = set() # gluten-free sandwiches

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "ontray":
                ontray_map[parts[1]] = parts[2]
            elif parts[0] == "at" and parts[1].startswith("tray"): # Ensure it's a tray location
                 trays_at_map[parts[1]] = parts[2]
            elif parts[0] == "at_kitchen_sandwich":
                sandwiches_kitchen_set.add(parts[1])
            elif parts[0] == "no_gluten_sandwich":
                sandwiches_no_gluten_set.add(parts[1])

        # --- Step b, c: Count needs and available supply at each place ---
        needed_gf_at_place = {p: 0 for p in self.all_places}
        needed_any_at_place = {p: 0 for p in self.all_places}
        avail_ontray_at_place_gf = {p: 0 for p in self.all_places}
        avail_ontray_at_place_any = {p: 0 for p in self.all_places}
        trays_at_place = {p: 0 for p in self.all_places}

        # Count unserved needs per place
        for child in unserved_children:
            place = self.waiting_places_static.get(child) # Get waiting place from static facts
            if place: # Child might not have a waiting fact if problem is malformed, but assume valid PDDL
                if child in self.allergic_children:
                    needed_gf_at_place[place] += 1
                else:
                    needed_any_at_place[place] += 1

        # Count available sandwiches on trays at each place
        for sandwich, tray in ontray_map.items():
            place = trays_at_map.get(tray)
            if place:
                if sandwich in sandwiches_no_gluten_set:
                    avail_ontray_at_place_gf[place] += 1
                avail_ontray_at_place_any[place] += 1 # GF sandwiches also count as 'any'

        # Count trays at each place
        for tray, place in trays_at_map.items():
             trays_at_place[place] += 1


        # --- Step d, e: Calculate deficit at each place and total needed from kitchen chain ---
        still_need_gf_at_place = {p: 0 for p in self.all_places}
        still_need_any_at_place = {p: 0 for p in self.all_places}
        total_still_need_gf = 0
        total_still_need_any = 0

        for place in self.all_places:
            if place == 'kitchen': continue # Children don't wait at kitchen

            needed_gf_P = needed_gf_at_place[place]
            needed_any_P = needed_any_at_place[place]
            avail_ontray_P_gf = avail_ontray_at_place_gf[place]
            avail_ontray_P_any = avail_ontray_at_place_any[place]

            # How many children at P can be served by current sandwiches at P?
            served_P_gf = min(needed_gf_P, avail_ontray_P_gf)
            remaining_avail_ontray_P_gf = avail_ontray_P_gf - served_P_gf
            served_P_any = min(needed_any_P, avail_ontray_P_any - served_P_gf + remaining_avail_ontray_P_gf) # Use remaining GF for any needs

            # Children still needing sandwiches at P (must come from kitchen chain)
            still_need_P_gf = needed_gf_P - served_P_gf
            still_need_P_any = needed_any_P - served_P_any

            still_need_gf_at_place[place] = still_need_P_gf
            still_need_any_at_place[place] = still_need_P_any

            total_still_need_gf += still_need_P_gf
            total_still_need_any += still_need_P_any

        # --- Step f: Count available sandwiches at kitchen (ready to leave) ---
        avail_kitchen_gf = 0
        avail_kitchen_any = 0

        # Sandwiches at kitchen (not on tray)
        for s in sandwiches_kitchen_set:
            if s in sandwiches_no_gluten_set:
                avail_kitchen_gf += 1
            avail_kitchen_any += 1

        # Sandwiches on tray at kitchen
        for sandwich, tray in ontray_map.items():
            if trays_at_map.get(tray) == 'kitchen':
                 if sandwich in sandwiches_no_gluten_set:
                     avail_kitchen_gf += 1
                 avail_kitchen_any += 1


        # --- Step g: Calculate sandwiches to make ---
        # How many GF sandwiches must be made?
        make_gf = max(0, total_still_need_gf - avail_kitchen_gf)

        # How many Any sandwiches must be made?
        # Available kitchen GF can cover GF needs first, then Any needs.
        # Available kitchen Any can cover Any needs.
        served_by_kitchen_gf = min(total_still_need_gf, avail_kitchen_gf)
        remaining_avail_kitchen_gf = avail_kitchen_gf - served_by_kitchen_gf
        total_avail_kitchen_for_any = avail_kitchen_any + remaining_avail_kitchen_gf
        make_any = max(0, total_still_need_any - total_avail_kitchen_for_any)

        cost_make = make_gf + make_any

        # --- Step h: Calculate cost of putting on tray ---
        # Each sandwich needed from the kitchen chain must be put on a tray at the kitchen
        total_needed_from_kitchen_chain = total_still_need_gf + total_still_need_any
        cost_put_on_tray = total_needed_from_kitchen_chain

        # --- Step i: Calculate cost of moving trays ---
        cost_move_tray = 0
        for place in self.all_places:
            if place == 'kitchen': continue # Children don't wait at kitchen

            # If this place needs any sandwiches from the kitchen chain
            if still_need_gf_at_place[place] > 0 or still_need_any_at_place[place] > 0:
                # And there is no tray currently at this place
                if trays_at_place[place] == 0:
                    cost_move_tray += 1 # Estimate one tray move is needed

        # --- Step j: Total heuristic ---
        cost_serve = num_unserved # Each unserved child needs one serve action

        total_heuristic = cost_serve + cost_make + cost_put_on_tray + cost_move_tray

        return total_heuristic
