from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Remove surrounding parentheses and split fact into tokens."""
    return fact[1:-1].split()

def match(fact, *pattern):
    """Return True when fact tokens match the pattern (supports '*' wildcard)."""
    parts = get_parts(fact)
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the childsnack domain. It estimates the number
    of actions required to serve all remaining children that are not yet served.

    Assumptions
    -----------
    - Each child goal is a (served <child>) fact in the task goals.
    - A child can be either allergic to gluten (requires a no-gluten sandwich) or
      not allergic (accepts any sandwich).
    - Sandwiches can be in one of several states:
        * on a tray (ontray)
        * at the kitchen (at_kitchen_sandwich)
        * non-existing (notexist) meaning it can be made
    - Making a no-gluten sandwich requires both a gluten-free bread and a
      gluten-free content portion available at the kitchen; making a regular
      sandwich requires any bread and any content portion at the kitchen.
    - Trays can be at the kitchen or at arbitrary places; put_on_tray requires
      the tray to be at the kitchen.
    - move_tray moves a single tray from any place to any other place in one
      action. put_on_tray, make_sandwich*, and serve_sandwich* each cost one
      action in this STRIPS encoding.

    Heuristic Initialization
    ------------------------
    During construction, the heuristic extracts:
    - The set of goal children (from task.goals).
    - Static properties of children: whether they are allergic and their waiting
      place (from static facts). If a child's allergy or waiting place is not
      present in static facts, the heuristic assumes the child is not allergic
      and has no specific waiting place (in which case serving will require a
      tray move).
    The initialization does not assume dynamic facts (tray positions,
    existing sandwiches, breads, etc.) which are read from a node state in
    __call__.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given state, the heuristic proceeds as follows:

    1. Quick goal check:
       - If all goals are already satisfied in the given state, return 0.

    2. Extract dynamic facts from the state:
       - Locations of trays (maps place -> set of tray ids).
       - Which sandwiches are on which trays (ontray mapping).
       - Which sandwiches are at the kitchen (at_kitchen_sandwich set).
       - Which sandwiches are gluten-free (no_gluten_sandwich set).
       - Counts of breads and contents at the kitchen and how many of those
         are gluten-free (if those properties are present).
       - Which sandwich names are still `notexist` (i.e., available to be created).

    3. Determine the set of children still needing service (goal children not
       yet 'served' in the state). For each such child we also get:
       - Their waiting place.
       - Whether they are allergic (need no-gluten sandwich).

    4. Greedy matching and cost estimate (resource-aware):
       - Serve children who already have a suitable sandwich on their tray at
         their place: cost += 1 (serve).
       - If a suitable sandwich exists on any other tray: move that tray to the
         child's place (cost +1) and serve (cost +1) -> cost += 2.
       - If a suitable sandwich exists at the kitchen:
           * If there is a tray at the kitchen: put_on_tray (1), move (if
             needed) (1), serve (1) -> cost += 3 (typical).
           * If no tray at the kitchen but trays exist elsewhere: move one tray
             to the kitchen first (1), then put_on_tray (1), move to child
             (1), serve (1) -> cost += 4.
       - If no suitable sandwich exists, but there are kitchen bread/content and
         a `notexist` sandwich id available, the heuristic models making a
         sandwich (1 action) followed by the steps for at_kitchen_sandwich
         above (1..4 actions depending on tray positions). If the child is
         allergic, making requires gluten-free bread and content.
       - The heuristic consumes the simulated resources as it assigns sandwiches
         to children so resources won't be double-counted (breads, contents,
         notexist slots, trays, sandwiches).

    5. Resource insufficiency:
       - If there are insufficient resources to create required sandwiches
         (e.g., notexist slots or bread/content), the heuristic returns a
         large but finite penalty per unserviceable child to ensure a finite
         heuristic value. This keeps the heuristic finite while strongly
         discouraging states that look impossible under the present facts.

    6. Efficiency:
       - Computation is linear in the number of facts and children; only simple
         set/dict operations are used, so the heuristic is cheap to compute and
         provides a fairly tight estimate that accounts for trays, existing
         sandwiches, and the need to create sandwiches.

    Note: This heuristic is not admissible (it may underestimate or overestimate
    in complex interactions), but it is designed to be informative for greedy
    best-first search in typical childsnack instances.
    """

    def __init__(self, task):
        # store goal (served child) facts and extract child names from them
        self.goals = task.goals  # set of goal fact strings like "(served child1)"

        # Extract the children that are expected in goals
        self.goal_children = []
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "served" and len(parts) >= 2:
                self.goal_children.append(parts[1])

        # static facts: allergy and waiting info (these do not change in the domain)
        self.child_allergic = {}     # child -> True/False (True if allergic_gluten)
        self.child_waiting_place = {}  # child -> place

        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "allergic_gluten" and len(parts) >= 2:
                self.child_allergic[parts[1]] = True
            elif parts[0] == "not_allergic_gluten" and len(parts) >= 2:
                # explicit not_allergic_gluten -> treat as not allergic
                self.child_allergic.setdefault(parts[1], False)
            elif parts[0] == "waiting" and len(parts) >= 3:
                child = parts[1]
                place = parts[2]
                self.child_waiting_place[child] = place

        # Precompute useful static sets in case they are present statically for
        # breads/content gluten-free properties (some instances may include them
        # in static set). Default: none are gluten-free unless stated.
        self.static_no_gluten_bread = set()
        self.static_no_gluten_content = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "no_gluten_bread" and len(parts) >= 2:
                self.static_no_gluten_bread.add(parts[1])
            if parts[0] == "no_gluten_content" and len(parts) >= 2:
                self.static_no_gluten_content.add(parts[1])

    def __call__(self, node):
        state = node.state  # frozenset of fact strings

        # Quick goal check: heuristic is 0 if all goals satisfied
        if self.goals <= state:
            return 0

        # Parse dynamic facts
        trays_at_place = {}  # place -> set of trays
        ontray_of = {}       # sandwich -> tray
        sandwiches_ontray = {}  # tray -> sandwich (inverse)
        at_kitchen_sandwich = set()
        no_gluten_sandwich = set()
        at_kitchen_breads = set()
        at_kitchen_contents = set()
        notexist_sandwiches = set()

        served_children = set()
        # Parse all facts in state
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]

            if pred == "at" and len(parts) >= 3:
                obj = parts[1]
                place = parts[2]
                # only keep trays locations (we only need trays)
                # sanity: tray names are of type tray in domain; we can't know type here,
                # so detect by later references (ontray uses tray names). We'll record all.
                trays_at_place.setdefault(place, set()).add(obj)

            elif pred == "ontray" and len(parts) >= 3:
                sandw = parts[1]
                tray = parts[2]
                ontray_of[sandw] = tray
                sandwiches_ontray[tray] = sandw

            elif pred == "at_kitchen_sandwich" and len(parts) >= 2:
                at_kitchen_sandwich.add(parts[1])

            elif pred == "no_gluten_sandwich" and len(parts) >= 2:
                no_gluten_sandwich.add(parts[1])

            elif pred == "at_kitchen_bread" and len(parts) >= 2:
                at_kitchen_breads.add(parts[1])

            elif pred == "at_kitchen_content" and len(parts) >= 2:
                at_kitchen_contents.add(parts[1])

            elif pred == "notexist" and len(parts) >= 2:
                notexist_sandwiches.add(parts[1])

            elif pred == "served" and len(parts) >= 2:
                served_children.add(parts[1])

        # For breads/contents gluten-free flags, check static sets too
        # Count gluten-free breads and contents available at kitchen
        gf_bread_count = sum(1 for b in at_kitchen_breads if b in self.static_no_gluten_bread)
        gf_content_count = sum(1 for c in at_kitchen_contents if c in self.static_no_gluten_content)

        # Counts of generic breads/contents available
        bread_count = len(at_kitchen_breads)
        content_count = len(at_kitchen_contents)

        # Useful counts and convenience data
        trays_total = sum(len(s) for s in trays_at_place.values())
        trays_in_kitchen = len(trays_at_place.get("kitchen", set()))
        trays_elsewhere = trays_total - trays_in_kitchen

        # Determine which children still need serving
        remaining_children = [c for c in self.goal_children if c not in served_children]

        # If no remaining, return 0 (caught earlier, but safe)
        if not remaining_children:
            return 0

        # Sort children so that allergic children are served first (they are constrained)
        def child_priority(c):
            return 0 if self.child_allergic.get(c, False) else 1
        remaining_children.sort(key=child_priority)

        # Make mutable copies of resources to simulate greedy assignment
        at_kitchen_sandwich = set(at_kitchen_sandwich)
        ontray_of = dict(ontray_of)
        sandwiches_ontray = dict(sandwiches_ontray)
        no_gluten_sandwich = set(no_gluten_sandwich)
        notexist_list = set(notexist_sandwiches)
        trays_at_place = {p: set(trays_at_place[p]) for p in trays_at_place}
        bread_count = int(bread_count)
        content_count = int(content_count)
        gf_bread_count = int(gf_bread_count)
        gf_content_count = int(gf_content_count)

        total_cost = 0
        LARGE_PENALTY = 1000  # finite but large penalty for resource impossibility

        for child in remaining_children:
            # Determine child's waiting place and allergy
            place = self.child_waiting_place.get(child, None)
            allergic = self.child_allergic.get(child, False)

            # Helper to check if sandwich name is suitable for this child
            def sandwich_suitable(sandw):
                if allergic:
                    return sandw in no_gluten_sandwich
                else:
                    return True

            # 1) Check for ontray sandwich already at child's place and suitable
            found = False
            if place is not None:
                trays_at_child_place = trays_at_place.get(place, set())
            else:
                trays_at_child_place = set()

            # Any sandwich on a tray at child's place?
            for tray in list(trays_at_child_place):
                sandw = sandwiches_ontray.get(tray)
                if sandw and sandwich_suitable(sandw):
                    # We can serve with one action
                    total_cost += 1  # serve_sandwich*
                    # simulate: sandwich consumed (ontray removed), no change to tray location
                    if sandw in ontray_of:
                        del ontray_of[sandw]
                    if tray in sandwiches_ontray:
                        del sandwiches_ontray[tray]
                    if sandw in no_gluten_sandwich:
                        no_gluten_sandwich.discard(sandw)
                    found = True
                    break
            if found:
                continue

            # 2) Check for ontray sandwich anywhere else (move + serve)
            candidate_tray = None
            candidate_sandw = None
            for sandw, tray in list(ontray_of.items()):
                if sandwich_suitable(sandw):
                    candidate_tray = tray
                    candidate_sandw = sandw
                    break
            if candidate_tray is not None:
                # Move tray to child's place (1) + serve (1)
                total_cost += 2
                # Update tray location: find its current location and move it
                # Remove from old place
                old_place = None
                for p, trays in trays_at_place.items():
                    if candidate_tray in trays:
                        old_place = p
                        trays.remove(candidate_tray)
                        break
                # place it at child's place
                if place is not None:
                    trays_at_place.setdefault(place, set()).add(candidate_tray)
                else:
                    # if no waiting place known, place arbitrarily (kitchen)
                    trays_at_place.setdefault("kitchen", set()).add(candidate_tray)
                # Serve consumes sandwich on the tray
                if candidate_sandw in ontray_of:
                    del ontray_of[candidate_sandw]
                if candidate_tray in sandwiches_ontray:
                    del sandwiches_ontray[candidate_tray]
                if candidate_sandw in no_gluten_sandwich:
                    no_gluten_sandwich.discard(candidate_sandw)
                continue

            # 3) Check for a suitable at-kitchen-sandwich
            chosen_k_sandw = None
            for sandw in list(at_kitchen_sandwich):
                if sandwich_suitable(sandw):
                    chosen_k_sandw = sandw
                    break
            if chosen_k_sandw is not None:
                # Compute cost depending on tray positions
                if trays_in_kitchen > 0:
                    # put_on_tray (1), possibly move to child place (1 if place != kitchen), serve (1)
                    move_needed = 0
                    if place is not None and place != "kitchen":
                        move_needed = 1
                    total_cost += 1 + move_needed + 1  # put + move(if any) + serve
                    # Simulate: remove at_kitchen_sandwich, consume it for serving
                    at_kitchen_sandwich.discard(chosen_k_sandw)
                    if chosen_k_sandw in no_gluten_sandwich:
                        no_gluten_sandwich.discard(chosen_k_sandw)
                    # Tray used: a tray from kitchen ends up at child's place
                    # pick one tray from kitchen
                    tray_used = next(iter(trays_at_place.get("kitchen", set())))
                    trays_at_place["kitchen"].remove(tray_used)
                    if place is not None:
                        trays_at_place.setdefault(place, set()).add(tray_used)
                    else:
                        trays_at_place.setdefault("kitchen", set()).add(tray_used)
                    # counts updated
                    trays_in_kitchen = len(trays_at_place.get("kitchen", set()))
                    trays_elsewhere = sum(len(s) for p, s in trays_at_place.items() if p != "kitchen")
                    continue
                else:
                    # No tray in kitchen: need to bring one from elsewhere (if any)
                    if trays_elsewhere > 0:
                        # move one tray to kitchen (1), put (1), move to child (if needed 1), serve (1)
                        move_to_kitchen_cost = 1
                        move_to_child_cost = 0 if (place == "kitchen") else 1
                        total_cost += move_to_kitchen_cost + 1 + move_to_child_cost + 1
                        # Simulate: take a tray from some other place
                        tray_taken = None
                        for p, trays in trays_at_place.items():
                            if p != "kitchen" and trays:
                                tray_taken = next(iter(trays))
                                trays.remove(tray_taken)
                                break
                        # tray moved to kitchen then to child's place
                        if place is not None:
                            trays_at_place.setdefault(place, set()).add(tray_taken)
                        else:
                            trays_at_place.setdefault("kitchen", set()).add(tray_taken)
                        # Update counts and consume sandwich
                        trays_in_kitchen = len(trays_at_place.get("kitchen", set()))
                        trays_elsewhere = sum(len(s) for p, s in trays_at_place.items() if p != "kitchen")
                        at_kitchen_sandwich.discard(chosen_k_sandw)
                        if chosen_k_sandw in no_gluten_sandwich:
                            no_gluten_sandwich.discard(chosen_k_sandw)
                        continue
                    else:
                        # No tray at all -> impossible under normal domain, penalize
                        total_cost += LARGE_PENALTY
                        # We cannot simulate further changes here
                        continue

            # 4) Need to make a sandwich and then proceed (make_sandwich* costs 1)
            # Check resource availability to make appropriate sandwich
            can_make = False
            needs_gf = allergic
            if needs_gf:
                # require gf bread and gf content and at least one notexist id
                if gf_bread_count >= 1 and gf_content_count >= 1 and len(notexist_list) >= 1:
                    can_make = True
            else:
                # require any bread and content and a notexist sandwich id
                if bread_count >= 1 and content_count >= 1 and len(notexist_list) >= 1:
                    can_make = True

            if can_make:
                # consume resources to make sandwich
                total_cost += 1  # make_sandwich or make_sandwich_no_gluten
                # allocate a sandwich id from notexist_list
                sandw_id = next(iter(notexist_list))
                notexist_list.remove(sandw_id)
                # after making, sandwich is at kitchen
                at_kitchen_sandwich.add(sandw_id)
                if needs_gf:
                    no_gluten_sandwich.add(sandw_id)
                    # consume gf bread/content
                    gf_bread_count -= 1
                    gf_content_count -= 1
                    bread_count = max(0, bread_count - 1)
                    content_count = max(0, content_count - 1)
                else:
                    # consume any bread/content
                    bread_count = max(0, bread_count - 1)
                    content_count = max(0, content_count - 1)
                # Now treat this freshly-made at_kitchen_sandwich as in step 3:
                # Re-run the logic for at_kitchen_sandwich (we can inline minimal cost)
                # Prefer trays in kitchen
                if trays_in_kitchen > 0:
                    move_needed = 0
                    if place is not None and place != "kitchen":
                        move_needed = 1
                    total_cost += 1 + move_needed + 1  # put + move(if any) + serve
                    # Simulate consumption and tray movement
                    at_kitchen_sandwich.discard(sandw_id)
                    if sandw_id in no_gluten_sandwich:
                        no_gluten_sandwich.discard(sandw_id)
                    tray_used = next(iter(trays_at_place.get("kitchen", set())))
                    trays_at_place["kitchen"].remove(tray_used)
                    if place is not None:
                        trays_at_place.setdefault(place, set()).add(tray_used)
                    else:
                        trays_at_place.setdefault("kitchen", set()).add(tray_used)
                    trays_in_kitchen = len(trays_at_place.get("kitchen", set()))
                    trays_elsewhere = sum(len(s) for p, s in trays_at_place.items() if p != "kitchen")
                    continue
                else:
                    if trays_elsewhere > 0:
                        # move tray from elsewhere to kitchen then put then move to child then serve
                        total_cost += 1 + 1 + (0 if place == "kitchen" else 1) + 1  # move_to_kitchen + put + move_to_child + serve
                        # simulate: move tray from elsewhere to child's place final
                        tray_taken = None
                        for p, trays in trays_at_place.items():
                            if p != "kitchen" and trays:
                                tray_taken = next(iter(trays))
                                trays.remove(tray_taken)
                                break
                        if place is not None:
                            trays_at_place.setdefault(place, set()).add(tray_taken)
                        else:
                            trays_at_place.setdefault("kitchen", set()).add(tray_taken)
                        at_kitchen_sandwich.discard(sandw_id)
                        if sandw_id in no_gluten_sandwich:
                            no_gluten_sandwich.discard(sandw_id)
                        trays_in_kitchen = len(trays_at_place.get("kitchen", set()))
                        trays_elsewhere = sum(len(s) for p, s in trays_at_place.items() if p != "kitchen")
                        continue
                    else:
                        # No trays at all -> penalize
                        total_cost += LARGE_PENALTY
                        continue
            else:
                # Cannot make sandwich for this child due to missing resources
                # Return a large but finite penalty for this child
                total_cost += LARGE_PENALTY
                # do not simulate resource consumption; continue with other children
                continue

        # Final heuristic value
        return int(total_cost)