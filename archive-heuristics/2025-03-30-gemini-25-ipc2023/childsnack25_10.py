import math
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(at tray1 kitchen)" -> ["at", "tray1", "kitchen"]
    return fact[1:-1].split()

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain 'childsnacks'.

    # Summary
    Estimates the number of actions required to reach a goal state where all specified children are served.
    The heuristic calculates the cost by summing the estimated minimum actions (make_sandwich, put_on_tray, move_tray, serve_sandwich)
    needed for each unserved child. It uses a greedy approach to assign available resources (ingredients, existing sandwiches)
    to children's needs and estimates tray movement costs based on the distinct locations that require deliveries.

    # Assumptions
    - Ingredients and sandwiches are consumed once assigned in the heuristic calculation.
    - Trays can implicitly carry multiple sandwiches needed for a location in one trip (optimistic).
    - A greedy strategy is used for resource allocation (assigning existing sandwiches/ingredients), which might not be globally optimal but is efficient.
    - Tray movement cost is estimated as the number of distinct child locations needing a delivery, plus one potential extra move if a tray must be brought to the kitchen for `put_on_tray` actions and none is currently there.
    - Object names are informative (e.g., objects with 'at' predicates that aren't ingredients/sandwiches are assumed to be trays).

    # Heuristic Initialization
    - The constructor (`__init__`) pre-processes the static information from the task definition (`task.static`).
    - It stores which children need serving based on the goal (`task.goals`).
    - It builds dictionaries/sets for quick lookups:
        - `child_allergy`: Maps each child to their required sandwich type ('gf' for gluten-free, 'regular' otherwise).
        - `child_waiting_location`: Maps each child to the place where they are waiting.
        - `gf_bread`: A set containing the names of all gluten-free bread portions.
        - `gf_content`: A set containing the names of all gluten-free content portions.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Identify Unserved Children:** Determine the set of children specified in the goal (`(served c)`) that are not yet served in the current state (`node.state`). If this set is empty, the goal is reached, and the heuristic value is 0.
    2.  **Parse Current State:** Extract dynamic information from the current state:
        - Location and type ('gf' or 'regular') of all existing sandwiches (distinguishing between those `at_kitchen` and those `ontray`).
        - Current location (`at`) of all trays.
        - Available ingredients (`at_kitchen_bread`, `at_kitchen_content`) categorized by gluten-free status.
        - Available sandwich 'slots' (names of sandwiches currently satisfying `(notexist ?s)`).
    3.  **Initialize Action Counts:** Set up counters for the estimated number of actions needed: `make_count = 0`, `put_count = 0`, `serve_count = number_of_unserved_children`. Also, initialize `locations_needing_delivery = set()` and `puts_needed = False`.
    4.  **Resource Tracking:** Create mutable copies of the available resources (lists/dictionaries of sandwiches, ingredients, slots) so they can be "consumed" as they are assigned greedily.
    5.  **Greedy Assignment per Child:** Iterate through each unserved child (e.g., in alphabetical order for deterministic results):
        a. Identify the child's required sandwich type (`required_type`) and waiting location (`target_location`).
        b. **Find Sandwich Source (Greedy Order):** Try to satisfy the child's need using the cheapest available option first:
            i.   **On Tray at Target Location:** Look for a suitable sandwich (matching type) already on a tray at the `target_location`. If found, consume it from the available resources. This requires only the `serve` action (already counted).
            ii.  **On Tray Elsewhere:** If not found, look for a suitable sandwich on a tray at a *different* location. If found, consume it. This implies a `move_tray` action is needed. Add `target_location` to `locations_needing_delivery`.
            iii. **At Kitchen:** If not found, look for a suitable sandwich at the kitchen. If found, consume it. This implies `put_on_tray` and `move_tray`. Increment `put_count`, set `puts_needed = True`, and add `target_location` to `locations_needing_delivery`.
            iv.  **Make New Sandwich:** If no existing sandwich is suitable/available, attempt to make one. Check for required ingredients (considering GF status) and an available sandwich slot (`notexist` name). If resources allow, consume them. This implies `make_sandwich`, `put_on_tray`, and `move_tray`. Increment `make_count`, increment `put_count`, set `puts_needed = True`, and add `target_location` to `locations_needing_delivery`.
            v.   **Impossible:** If a sandwich cannot be found or made for the child (e.g., lack of ingredients/slots), the state is deemed unsolvable by this heuristic path. Return `float('inf')`.
    6.  **Calculate Total Move Cost:**
        - The primary move cost is the number of unique locations that need a delivery: `len(locations_needing_delivery)`.
        - An additional move cost of 1 is added if `put_on_tray` actions are required (`puts_needed == True`), no tray is currently at the kitchen, but at least one tray exists elsewhere in the state (implying a tray must be moved to the kitchen).
    7.  **Final Heuristic Value:** Sum the estimated counts for each action type: `h = make_count + put_count + serve_count + move_cost`. This sum represents the estimated total number of actions to reach the goal.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static_facts = task.static

        # Pre-process static facts for efficiency
        self.child_allergy = {} # child -> 'gf' or 'regular'
        self.child_waiting_location = {} # child -> place
        self.gf_bread = set() # set of GF bread portions
        self.gf_content = set() # set of GF content portions

        for fact in self.static_facts:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'allergic_gluten':
                self.child_allergy[parts[1]] = 'gf'
            elif predicate == 'not_allergic_gluten':
                self.child_allergy[parts[1]] = 'regular'
            elif predicate == 'waiting':
                # Ensure child exists in allergy map, default to regular if somehow missing
                if parts[1] not in self.child_allergy:
                     self.child_allergy[parts[1]] = 'regular'
                self.child_waiting_location[parts[1]] = parts[2]
            elif predicate == 'no_gluten_bread':
                self.gf_bread.add(parts[1])
            elif predicate == 'no_gluten_content':
                self.gf_content.add(parts[1])

        self.target_children = {
            get_parts(g)[1] for g in self.goals if get_parts(g)[0] == 'served'
        }
        
        # Ensure all target children have allergy info (default to regular if missing)
        for child in self.target_children:
            if child not in self.child_allergy:
                self.child_allergy[child] = 'regular'


    def __call__(self, node):
        state = node.state

        # 1. Identify unserved children
        served_children = {
            get_parts(f)[1] for f in state if get_parts(f)[0] == 'served'
        }
        unserved_children = self.target_children - served_children

        if not unserved_children:
            return 0 # Goal state reached

        # 2. Parse current state
        sandwiches_kitchen_gf = []
        sandwiches_kitchen_reg = []
        # s -> {'tray': t, 'location': p, 'type': 'gf'/'regular'}
        sandwiches_tray_details = {}
        tray_locations = {} # t -> location
        notexist_sandwiches = []
        bread_kitchen_gf = []
        bread_kitchen_reg = []
        content_kitchen_gf = []
        content_kitchen_reg = []

        # Temporary structures to resolve dependencies (sandwich type/location)
        temp_ontray_sandwiches = {} # s -> t
        temp_kitchen_sandwiches = set()
        temp_gf_sandwiches = set()

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]

            # Basic type/location parsing
            if predicate == 'at':
                obj, p = parts[1], parts[2]
                # Simple check: if it's not an ingredient/sandwich/place constant, assume it's a tray
                if obj != 'kitchen' and not obj.startswith('sandw') and \
                   not obj.startswith('bread') and not obj.startswith('content') and \
                   not obj.startswith('child') and p != 'place': # Avoid matching type declarations if any
                     tray_locations[obj] = p
            elif predicate == 'ontray':
                temp_ontray_sandwiches[parts[1]] = parts[2]
            elif predicate == 'at_kitchen_sandwich':
                temp_kitchen_sandwiches.add(parts[1])
            elif predicate == 'no_gluten_sandwich':
                temp_gf_sandwiches.add(parts[1])
            elif predicate == 'notexist':
                notexist_sandwiches.append(parts[1])
            elif predicate == 'at_kitchen_bread':
                b = parts[1]
                if b in self.gf_bread: bread_kitchen_gf.append(b)
                else: bread_kitchen_reg.append(b)
            elif predicate == 'at_kitchen_content':
                c = parts[1]
                if c in self.gf_content: content_kitchen_gf.append(c)
                else: content_kitchen_reg.append(c)

        # Consolidate parsed info into usable structures
        for s in temp_kitchen_sandwiches:
            if s in temp_gf_sandwiches: sandwiches_kitchen_gf.append(s)
            else: sandwiches_kitchen_reg.append(s)

        for s, t in temp_ontray_sandwiches.items():
            if t in tray_locations:
                loc = tray_locations[t]
                s_type = 'gf' if s in temp_gf_sandwiches else 'regular'
                sandwiches_tray_details[s] = {'tray': t, 'location': loc, 'type': s_type}
            # else: Ignore sandwich on tray with unknown location

        # 3. Initialize counts and resource tracking
        make_count = 0
        put_count = 0
        serve_count = len(unserved_children)
        locations_needing_delivery = set()
        puts_needed = False

        # Mutable copies for greedy assignment
        avail_sandwiches_kitchen_gf = list(sandwiches_kitchen_gf)
        avail_sandwiches_kitchen_reg = list(sandwiches_kitchen_reg)
        avail_sandwiches_tray = sandwiches_tray_details.copy() # dict: s -> info
        avail_bread_gf = list(bread_kitchen_gf)
        avail_bread_reg = list(bread_kitchen_reg)
        avail_content_gf = list(content_kitchen_gf)
        avail_content_reg = list(content_kitchen_reg)
        avail_slots = list(notexist_sandwiches)

        # 4. Iterate through unserved children (sorted for determinism)
        children_list = sorted(list(unserved_children))

        for child in children_list:
            # Ensure static info exists for the child
            if child not in self.child_allergy or child not in self.child_waiting_location:
                 # Should not happen if initialization is correct, but handle defensively
                 return float('inf')

            required_type = self.child_allergy[child]
            target_location = self.child_waiting_location[child]
            found_source = False

            # 5a. Check Tray at Target Location
            found_s_tray_at_loc = None
            for s, info in list(avail_sandwiches_tray.items()):
                # Type check: GF child needs GF sandwich. Regular child accepts any.
                type_match = (info['type'] == 'gf' and required_type == 'gf') or \
                             (required_type == 'regular')
                if info['location'] == target_location and type_match:
                    found_s_tray_at_loc = s
                    break
            if found_s_tray_at_loc:
                del avail_sandwiches_tray[found_s_tray_at_loc]
                found_source = True
                continue # Only serve action needed (already counted)

            # 5b. Check Tray Elsewhere
            if not found_source:
                found_s_tray_elsewhere = None
                for s, info in list(avail_sandwiches_tray.items()):
                    type_match = (info['type'] == 'gf' and required_type == 'gf') or \
                                 (required_type == 'regular')
                    if type_match:
                        found_s_tray_elsewhere = s
                        break
                if found_s_tray_elsewhere:
                    del avail_sandwiches_tray[found_s_tray_elsewhere]
                    found_source = True
                    locations_needing_delivery.add(target_location) # Implies move
                    continue

            # 5c. Check Kitchen
            if not found_source:
                found_s_kitchen = None
                if required_type == 'gf':
                    if avail_sandwiches_kitchen_gf:
                        found_s_kitchen = avail_sandwiches_kitchen_gf.pop(0)
                else: # Regular required - prefer regular sandwich, then GF
                    if avail_sandwiches_kitchen_reg:
                        found_s_kitchen = avail_sandwiches_kitchen_reg.pop(0)
                    elif avail_sandwiches_kitchen_gf:
                        found_s_kitchen = avail_sandwiches_kitchen_gf.pop(0)

                if found_s_kitchen:
                    found_source = True
                    put_count += 1      # Implies put_on_tray
                    puts_needed = True
                    locations_needing_delivery.add(target_location) # Implies move
                    continue

            # 5d. Make New Sandwich
            if not found_source:
                can_make = False
                bread_used = None
                content_used = None
                slot_used = None

                if not avail_slots: return float('inf') # No names left for new sandwiches
                slot_used = avail_slots.pop(0) # Tentatively consume a slot

                if required_type == 'gf':
                    if avail_bread_gf and avail_content_gf:
                        bread_used = avail_bread_gf.pop(0)
                        content_used = avail_content_gf.pop(0)
                        can_make = True
                else: # Regular sandwich needed
                    # Use any available bread
                    if avail_bread_reg: bread_used = avail_bread_reg.pop(0)
                    elif avail_bread_gf: bread_used = avail_bread_gf.pop(0)
                    # Use any available content
                    if avail_content_reg: content_used = avail_content_reg.pop(0)
                    elif avail_content_gf: content_used = avail_content_gf.pop(0)

                    if bread_used and content_used:
                        can_make = True
                    else: # Failed: put back any ingredient taken
                        if bread_used:
                            if bread_used in self.gf_bread: avail_bread_gf.insert(0, bread_used)
                            else: avail_bread_reg.insert(0, bread_used)
                        if content_used:
                             if content_used in self.gf_content: avail_content_gf.insert(0, content_used)
                             else: avail_content_reg.insert(0, content_used)

                if can_make:
                    make_count += 1     # Implies make_sandwich
                    put_count += 1      # Implies put_on_tray
                    puts_needed = True
                    locations_needing_delivery.add(target_location) # Implies move
                    found_source = True
                    continue
                else:
                    avail_slots.insert(0, slot_used) # Put back slot if making failed

            # 5e. Impossible
            if not found_source:
                # Cannot satisfy this child's need with current resources/state
                return float('inf')

        # 6. Calculate Total Move Cost
        move_cost = len(locations_needing_delivery)
        is_tray_at_kitchen = any(loc == 'kitchen' for loc in tray_locations.values())

        # Add cost for moving a tray to kitchen if needed for putting,
        # none is there, but trays exist elsewhere.
        if puts_needed and not is_tray_at_kitchen and tray_locations:
            move_cost += 1

        # 7. Final Heuristic Value
        h = make_count + put_count + serve_count + move_cost
        return h
