from collections import Counter
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper to split a PDDL fact string into predicate and arguments."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

def match(fact, *args):
    """Helper to check if a fact matches a pattern using fnmatch."""
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if fact has fewer parts than args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    The heuristic estimates the cost to reach the goal (all children served)
    by summing the cost of serving each unserved child. The cost for serving
    a child is broken down into:
    1. The 'serve' action itself (cost 1 per child).
    2. The cost to get a suitable sandwich onto a tray and moved to the child's
       waiting location. This cost depends on the current state of available
       sandwiches (on trays at wrong locations, at the kitchen, or makeable)
       and their type (gluten-free or not), prioritizing cheaper options.

    Assumptions:
    - The problem is solvable (enough resources exist in total). If not, the
      heuristic returns infinity.
    - Trays are primarily used for transport and serving; enough trays are
      implicitly available for putting sandwiches on at the kitchen when needed.
    - The cost of moving a tray is 1, putting a sandwich on a tray is 1,
      making a sandwich is 1, and serving is 1.
    - Gluten-free sandwiches can satisfy the needs of both allergic and
      non-allergic children. Non-gluten-free sandwiches can only satisfy
      non-allergic children.

    Heuristic Initialization:
    The constructor extracts static information from the task:
    - Which children are allergic and non-allergic.
    - Where each child is waiting.
    - Which bread and content portions are gluten-free.
    - The name of the kitchen place (inferred from static or state facts).
    - A set of all possible places mentioned in the problem (from waiting facts).

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify all unserved children and their waiting places, separating them
       into allergic and non-allergic groups.
    2. Calculate the base heuristic cost as the total number of unserved children
       (each requires one 'serve' action).
    3. For each place, determine the number of gluten-free (GF) sandwiches and
       any sandwiches (non-GF or GF) that are still needed on trays at that
       location to serve the waiting children, considering sandwiches already
       present on trays at that location.
    4. Sum the needed GF sandwiches across all locations to get the total GF
       sandwiches that need to be brought.
    5. Sum the needed Any sandwiches across all locations to get the total Any
       sandwiches that need to be brought.
    6. Count the total available GF and Non-GF sandwiches based on their current
       "readiness" stage:
       - Stage 1 (Cost 1 to bring): Already on a tray (any location).
       - Stage 2 (Cost 2 to bring): At the kitchen (need put_on_tray + move_tray).
       - Stage 3 (Cost 3 to bring): Makeable from ingredients at the kitchen
         (need make + put_on_tray + move_tray).
    7. Calculate the cost to bring the needed GF sandwiches: Satisfy the total
       GF bringing demand using available GF sources, prioritizing sources
       from Stage 1, then Stage 2, then Stage 3. Add the cost (1, 2, or 3)
       multiplied by the number of sandwiches taken from each stage. Track
       any remaining GF sources at each stage.
    8. Calculate the cost to bring the needed Any sandwiches: Satisfy the total
       Any bringing demand using available Non-GF sources (Stage 1, 2, 3) and
       any remaining GF sources (from Step 7), prioritizing sources from
       Stage 1, then Stage 2, then Stage 3. Add the cost (1, 2, or 3)
       multiplied by the number of sandwiches taken from each stage.
    9. If, after exhausting all available sources of the required type(s), there
       are still sandwiches needed (either GF or Any), the problem is likely
       unsolvable in this state, and the heuristic returns infinity.
    10. The total heuristic value is the sum of the base cost (serve actions)
        and the calculated costs for bringing GF and Any sandwiches.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        self.allergic_children = set()
        self.non_allergic_children = set()
        self.waiting_place = {} # child -> place
        self.gluten_free_bread_types = set()
        self.gluten_free_content_types = set()
        self.all_places = set()
        self.kitchen_place = None # Will try to find this later

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'allergic_gluten':
                self.allergic_children.add(parts[1])
            elif parts[0] == 'not_allergic_gluten':
                self.non_allergic_children.add(parts[1])
            elif parts[0] == 'waiting':
                child, place = parts[1], parts[2]
                self.waiting_place[child] = place
                self.all_places.add(place)
            elif parts[0] == 'no_gluten_bread':
                self.gluten_free_bread_types.add(parts[1])
            elif parts[0] == 'no_gluten_content':
                self.gluten_free_content_types.add(parts[1])
            # Attempt to find kitchen from static facts like (at tray kitchen)
            elif parts[0] == 'at' and len(parts) == 3 and parts[1].endswith('tray'):
                 self.kitchen_place = parts[2] # Assume the place a tray starts at is kitchen
                 self.all_places.add(self.kitchen_place)


    def __call__(self, node):
        state = node.state

        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        unserved_allergic_children = {c for c in self.allergic_children if c not in served_children}
        unserved_non_allergic_children = {c for c in self.non_allergic_children if c not in served_children}

        # If all children are served, goal reached, heuristic is 0
        if not unserved_allergic_children and not unserved_non_allergic_children:
            return 0

        # Base cost: one 'serve' action per unserved child
        heuristic_cost = len(unserved_allergic_children) + len(unserved_non_allergic_children)

        # --- Step 1: Identify needed sandwiches at each place ---
        needed_gf_at_place = Counter()
        needed_any_at_place = Counter()
        # Start with places from static waiting facts
        all_current_places = set(self.all_places)

        for child in unserved_allergic_children:
            place = self.waiting_place.get(child)
            if place:
                needed_gf_at_place[place] += 1
                all_current_places.add(place) # Ensure place is in our set

        for child in unserved_non_allergic_children:
            place = self.waiting_place.get(child)
            if place:
                needed_any_at_place[place] += 1
                all_current_places.add(place) # Ensure place is in our set

        # --- Step 2: Count available sandwiches on trays at each place ---
        avail_gf_ontray_at_place = Counter()
        avail_nongf_ontray_at_place = Counter()
        sandwich_is_gf = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}
        tray_location = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*")}

        # Add places from current tray locations
        all_current_places.update(tray_location.values())

        # If kitchen wasn't found in static, try finding it from tray locations in state
        if self.kitchen_place is None:
             for loc in tray_location.values():
                 if loc == 'kitchen': # Assuming 'kitchen' is the string name
                     self.kitchen_place = 'kitchen'
                     all_current_places.add(self.kitchen_place)
                     break
        elif self.kitchen_place not in all_current_places:
             # If kitchen was found in static but not in current tray locations (e.g. no trays at kitchen), add it
             all_current_places.add(self.kitchen_place)


        for fact in state:
            if match(fact, "ontray", "*", "*"):
                s, t = get_parts(fact)[1], get_parts(fact)[2]
                place = tray_location.get(t)
                if place: # Only count if the tray location is known
                    if s in sandwich_is_gf:
                        avail_gf_ontray_at_place[place] += 1
                    else:
                        avail_nongf_ontray_at_place[place] += 1

        # --- Step 3: Calculate sandwiches that need to be brought to each location ---
        # These are sandwiches needed *in addition* to what's already on trays there
        bring_gf_to_place = Counter()
        bring_any_to_place = Counter()

        for place in all_current_places:
            needed_gf = needed_gf_at_place[place]
            needed_any = needed_any_at_place[place]
            avail_gf_here = avail_gf_ontray_at_place[place]
            avail_nongf_here = avail_nongf_ontray_at_place[place]

            # GF sandwiches needed at this place that are not already here on trays
            gf_to_bring = max(0, needed_gf - avail_gf_here)
            bring_gf_to_place[place] = gf_to_bring

            # Remaining GF sandwiches already here after meeting GF needs
            rem_gf_here = max(0, avail_gf_here - needed_gf)

            # Any sandwiches needed at this place that are not already here on trays
            # (using non-GF available here and remaining GF available here)
            any_to_bring = max(0, needed_any - avail_nongf_here - rem_gf_here)
            bring_any_to_place[place] = any_to_bring

        total_bring_gf = sum(bring_gf_to_place.values())
        total_bring_any = sum(bring_any_to_place.values())

        # --- Step 4: Count available sandwiches by type and readiness stage ---
        # Stage 1 (Cost 1): On tray (any location)
        total_gf_ontray = sum(avail_gf_ontray_at_place.values())
        total_nongf_ontray = sum(avail_nongf_ontray_at_place.values())
        avail_gf_c1 = total_gf_ontray
        avail_nongf_c1 = total_nongf_ontray

        # Stage 2 (Cost 2): At kitchen
        avail_gf_kitchen = 0
        avail_nongf_kitchen = 0
        for fact in state:
            if match(fact, "at_kitchen_sandwich", "*"):
                s = get_parts(fact)[1]
                if s in sandwich_is_gf:
                    avail_gf_kitchen += 1
                else:
                    avail_nongf_kitchen += 1
        avail_gf_c2 = avail_gf_kitchen
        avail_nongf_c2 = avail_nongf_kitchen

        # Stage 3 (Cost 3): Makeable at kitchen
        bread_at_kitchen = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_bread", "*")}
        content_at_kitchen = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_content", "*")}
        notexist_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "notexist", "*")}

        num_gf_bread_kitchen = len(bread_at_kitchen.intersection(self.gluten_free_bread_types))
        num_gf_content_kitchen = len(content_at_kitchen.intersection(self.gluten_free_content_types))
        num_notexist_sandwich = len(notexist_sandwiches)

        avail_makeable_gf = min(num_gf_bread_kitchen, num_gf_content_kitchen, num_notexist_sandwich)

        # Remaining notexist sandwiches after making GF ones
        notexist_remaining = num_notexist_sandwich - avail_makeable_gf
        num_nongf_bread_kitchen = len(bread_at_kitchen) - num_gf_bread_kitchen
        num_nongf_content_kitchen = len(content_at_kitchen) - num_gf_content_kitchen

        avail_makeable_nongf = min(num_nongf_bread_kitchen, num_nongf_content_kitchen, notexist_remaining)

        avail_gf_c3 = avail_makeable_gf
        avail_nongf_c3 = avail_makeable_nongf

        # Check if enough potential sandwiches exist in total
        total_potential_gf = avail_gf_c1 + avail_gf_c2 + avail_gf_c3
        total_potential_nongf = avail_nongf_c1 + avail_nongf_c2 + avail_nongf_c3

        total_needed_gf_overall = sum(needed_gf_at_place.values())
        total_needed_any_overall = sum(needed_any_at_place.values())

        if total_potential_gf < total_needed_gf_overall:
             return float('inf') # Not enough GF sandwiches possible

        # Remaining GF sandwiches after meeting all GF needs can serve Any needs
        remaining_potential_gf_for_any = max(0, total_potential_gf - total_needed_gf_overall)

        # Any sandwiches needed that must be met by Non-GF supply
        needed_from_nongf_supply = max(0, total_needed_any_overall - remaining_potential_gf_for_any)

        if total_potential_nongf < needed_from_nongf_supply:
             return float('inf') # Not enough Non-GF sandwiches possible for remaining Any needs


        # --- Step 5: Calculate cost for bringing GF sandwiches ---
        cost_gf_bring = 0
        needed = total_bring_gf

        # Use Cost 1 GF sources
        use = min(needed, avail_gf_c1)
        cost_gf_bring += use * 1
        needed -= use
        rem_gf_c1 = avail_gf_c1 - use

        # Use Cost 2 GF sources
        use = min(needed, avail_gf_c2)
        cost_gf_bring += use * 2
        needed -= use
        rem_gf_c2 = avail_gf_c2 - use

        # Use Cost 3 GF sources
        use = min(needed, avail_gf_c3)
        cost_gf_bring += use * 3
        needed -= use
        rem_gf_c3 = avail_gf_c3 - use

        if needed > 0:
             # This should not happen if the total potential check passed, but as a safeguard
             return float('inf')

        # --- Step 6: Calculate cost for bringing Any sandwiches ---
        cost_any_bring = 0
        needed = total_bring_any

        # Available Any sources by cost (including remaining GF)
        any_src_c1 = avail_nongf_c1 + rem_gf_c1
        any_src_c2 = avail_nongf_c2 + rem_gf_c2
        any_src_c3 = avail_nongf_c3 + rem_gf_c3

        # Use Cost 1 Any sources
        use = min(needed, any_src_c1)
        cost_any_bring += use * 1
        needed -= use

        # Use Cost 2 Any sources
        use = min(needed, any_src_c2)
        cost_any_bring += use * 2
        needed -= use

        # Use Cost 3 Any sources
        use = min(needed, any_src_c3)
        cost_any_bring += use * 3
        needed -= use

        if needed > 0:
             # This should not happen if the total potential check passed, but as a safeguard
             return float('inf')

        # Total heuristic is base cost (serve actions) + cost to bring sandwiches
        return heuristic_cost + cost_gf_bring + cost_any_bring
