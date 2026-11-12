from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split into tokens."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the childsnack domain.

    The heuristic estimates how many actions are required to serve all
    children that are not yet marked as (served X). It reasons about
    the following resources and steps:
      - existing sandwiches (at_kitchen_sandwich and ontray),
      - availability of gluten-free ingredients (no_gluten_bread and no_gluten_content)
        located at the kitchen (at_kitchen_bread / at_kitchen_content),
      - trays and their positions (at tray place and ontray s tray relationships),
      - creation of sandwiches (make_sandwich / make_sandwich_no_gluten),
      - placing sandwiches on trays (put_on_tray),
      - moving trays (move_tray),
      - serving (serve_sandwich / serve_sandwich_no_gluten).

    Assumptions
    -----------
    - Each sandwich serves exactly one child.
    - A sandwich that is on a tray and located at the child's waiting place can be served with 1 action.
    - Moving a tray from any place to any place is one action (move_tray).
    - Putting a kitchen sandwich on a tray (put_on_tray) requires that the tray be at the kitchen.
    - Making a sandwich at the kitchen costs 1 action and consumes one bread and one content at the kitchen.
    - For allergic children, only sandwiches built from gluten-free bread AND gluten-free content are acceptable.
    - Trays are reusable: after a serve action the tray remains at the child's place and becomes empty.
    - The heuristic uses a greedy assignment of available resources -> may be inadmissible but is efficient.
    - If required resources are missing for a particular child (e.g., no gluten-free ingredients for an allergic child),
      the heuristic assigns a large finite penalty to reflect infeasibility.

    Heuristic Initialization
    ------------------------
    In the constructor we extract relevant static information from the task:
      - Which children are allergic (allergic_gluten) and which are not (not_allergic_gluten).
      - The waiting place of each child (waiting child place).
    This static information is used to decide whether a sandwich must be gluten-free for a given child.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given node/state the heuristic performs the following steps:

    1. Quick goal check:
       - If all goal facts (served ...) are already present in the state, return 0.

    2. Extract dynamic resources from the state:
       - Count how many bread and content portions are available at the kitchen.
       - Count how many gluten-free bread/content portions are available at the kitchen by intersecting
         static "no_gluten_bread"/"no_gluten_content" with dynamic "at_kitchen_bread"/"at_kitchen_content".
       - Collect lists of sandwiches:
         * at_kitchen_sandwich (located at kitchen)
         * ontray sandwiches with the tray they are on and the tray position
       - Track which sandwiches are flagged as no_gluten_sandwich.
       - Identify trays, their current positions, and whether they are carrying a sandwich (occupied).
       - Identify already served children.

    3. Pre-serve matches:
       - Count children who are unserved but already have a compatible ontray sandwich located at their waiting place.
         Each such child needs exactly 1 action (serve). Those sandwiches are removed from the pool and their trays become empty.

    4. Greedy assignment for the remaining unserved children:
       - Process allergic children first (they have stricter resource needs), then non-allergic children.
       - For each child, consider a set of feasible options and their estimated action costs:
         a) Use an ontray sandwich elsewhere: move tray (1) + serve (1) = 2
         b) Use a sandwich at kitchen: (put_on_tray + move_tray + serve) = 3 if a free tray is at kitchen,
            otherwise 4 (move free tray to kitchen + put_on_tray + move to child + serve).
         c) Make a new sandwich at kitchen (1) then same as (b), but only allowed if required ingredients exist.
       - Choose the minimum-cost feasible option, update resource counts (consume bread/content/sandwich,
         update tray occupancy/positions) and add its cost to the heuristic total.
       - If no feasible option exists for a child (e.g., no gluten-free ingredients for an allergic child),
         assign a large finite penalty (1000) for that child to keep the heuristic finite while indicating
         strong infeasibility.

    5. Sum the selected costs for all remaining children and return that value.

    Notes
    -----
    - The heuristic is intentionally optimistic and greedy; it does not try to solve complex scheduling
      interactions optimally but provides a focused and efficient estimate that helps guiding greedy
      best-first search effectively in practice.
    """

    def __init__(self, task):
        # Goals to check for goal-state quick return.
        self.goals = task.goals

        # Extract from static facts:
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.waiting_place = dict()  # child -> place

        for fact in task.static:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "allergic_gluten":
                self.allergic_children.add(parts[1])
            elif pred == "not_allergic_gluten":
                self.not_allergic_children.add(parts[1])
            elif pred == "waiting":
                # waiting child place
                # predicate form: (waiting child place)
                if len(parts) >= 3:
                    self.waiting_place[parts[1]] = parts[2]

    def __call__(self, node):
        state = node.state

        # 1) Goal check
        if self.goals <= state:
            return 0

        # Helper sets and counters extracted from the state
        at_kitchen_breads = set()  # names of bread portions at kitchen
        at_kitchen_contents = set()  # names of content portions at kitchen
        at_kitchen_sandwiches = set()  # sandwich ids at kitchen
        no_gluten_sandwiches = set()  # sandwich ids flagged as no gluten
        ontray = {}  # sandwich -> tray
        tray_pos = {}  # tray -> place
        notexist_sandwiches = set()  # sandwich names available to create
        served_children = set()

        # Parse state facts
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "at_kitchen_bread":
                at_kitchen_breads.add(parts[1])
            elif pred == "at_kitchen_content":
                at_kitchen_contents.add(parts[1])
            elif pred == "at_kitchen_sandwich":
                at_kitchen_sandwiches.add(parts[1])
            elif pred == "no_gluten_sandwich":
                no_gluten_sandwiches.add(parts[1])
            elif pred == "ontray":
                # (ontray sandwich tray)
                if len(parts) >= 3:
                    s, t = parts[1], parts[2]
                    ontray[s] = t
            elif pred == "at":
                # could be (at tray place)
                if len(parts) >= 3:
                    obj, place = parts[1], parts[2]
                    # A tray object is anything that appears in ontray values or objects with 'tray' prefix.
                    # We conservatively treat any object that appears in an (at X P) where X is referenced as a tray elsewhere
                    # or starts with "tray" as a tray. We'll just store all 'at' facts and later use trays that we know.
                    tray_pos[parts[1]] = parts[2]
            elif pred == "notexist":
                notexist_sandwiches.add(parts[1])
            elif pred == "served":
                served_children.add(parts[1])

        # Determine trays: trays are objects that appear in tray_pos and/or in ontray values.
        trays = set(tray_pos.keys()) | set(ontray.values())
        # Build a mapping tray -> occupied sandwich or None
        tray_occupied = {}
        for tr in trays:
            tray_occupied[tr] = None
        for s, t in ontray.items():
            if t in tray_occupied:
                tray_occupied[t] = s
            else:
                # in case tray wasn't detected in tray_pos
                tray_occupied[t] = s
                tray_pos.setdefault(t, None)

        # classify trays: free at kitchen, free elsewhere
        free_trays_at_kitchen = 0
        free_trays_elsewhere = 0
        for tr in trays:
            occ = tray_occupied.get(tr)
            pos = tray_pos.get(tr)
            if occ is None:
                if pos == "kitchen":
                    free_trays_at_kitchen += 1
                else:
                    free_trays_elsewhere += 1

        # Count gluten-free ingredient portions available at kitchen.
        # To know which breads/contents are gluten-free we should look into static facts:
        # static contains "(no_gluten_bread breadX)" and "(no_gluten_content contentY)" usually.
        # We'll find those in node.task.static from constructor but we don't store breads/content names in constructor.
        # Instead we detect 'no_gluten_bread' and 'no_gluten_content' from the static set in self by scanning again.
        # Because constructor only recorded allergies and waiting places, we need to ask the planner about static facts.
        # However, task.static is not stored beyond constructor; but we can rely on that static facts for names appear in the node.state
        # or remained in self.waiting_place sets. To robustly get no_gluten bread/content, we inspect self via the node's state and static mapping:
        # To keep it simple and correct, we detect gluten-free bread/content by checking for explicit flags in the state OR in
        # the static facts that were used in the constructor: since the constructor had task.static but didn't store these,
        # we try to recover by looking at node.state and self.waiting_place is not relevant. So we will look in both node.state and node.static if available.
        # Note: The environment used to call heuristics provides static facts through the Task (constructor) only; we lacked storing no_gluten_bread/content.
        # To be robust, we also look for these predicates inside the current state's facts (they might be present there as well).
        # As fallback, assume none gluten-free if not detected.
        gluten_free_breads = set()
        gluten_free_contents = set()
        # Check in state (some instances put no_gluten_bread/content in initial which becomes part of state)
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "no_gluten_bread" and len(parts) >= 2:
                gluten_free_breads.add(parts[1])
            elif pred == "no_gluten_content" and len(parts) >= 2:
                gluten_free_contents.add(parts[1])

        # Also, try to glean gluten-free flags from self.waiting_place keys if any static is available there (no, not needed).
        # Now compute counts of gluten-free ingredients that are actually at the kitchen
        gf_bread_at_kitchen = len([b for b in at_kitchen_breads if b in gluten_free_breads])
        gf_content_at_kitchen = len([c for c in at_kitchen_contents if c in gluten_free_contents])
        total_bread_at_kitchen = len(at_kitchen_breads)
        total_content_at_kitchen = len(at_kitchen_contents)

        # Build list of unserved children and split by allergy
        all_children = set(list(self.allergic_children) + list(self.not_allergic_children))
        # It's possible some children are neither in static lists; also some children's served facts exist in state.
        # We will infer children to consider from goals (goals often are (served childX) for each child). Use goals to find children in goal.
        children_in_goals = []
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "served" and len(parts) >= 2:
                children_in_goals.append(parts[1])
        # final list of children to serve is those in goals
        children_to_consider = children_in_goals

        unserved_children = [c for c in children_to_consider if c not in served_children]

        # Separate allergic and non-allergic (prefer to serve allergic first)
        allergic_unserved = [c for c in unserved_children if c in self.allergic_children]
        nonallergic_unserved = [c for c in unserved_children if c in self.not_allergic_children or c not in self.allergic_children]

        # Resources describing sandwiches:
        # Build set of sandwiches and their attributes
        sandwiches_ontray = {}  # s -> (tray, tray_place)
        for s, t in ontray.items():
            place = tray_pos.get(t)
            sandwiches_ontray[s] = (t, place)

        sandwiches_at_kitchen = set(at_kitchen_sandwiches)
        sandwiches_no_gluten = set(no_gluten_sandwiches)

        # For quicker lookup, lists of available sandwich ids by gluten property
        ontray_gf = {s for s in sandwiches_ontray if s in sandwiches_no_gluten}
        ontray_non_gf = {s for s in sandwiches_ontray if s not in sandwiches_no_gluten}
        ak_gf = {s for s in sandwiches_at_kitchen if s in sandwiches_no_gluten}
        ak_non_gf = {s for s in sandwiches_at_kitchen if s not in sandwiches_no_gluten}

        # Pre-serve immediate matches: sandwich ontray and tray at child's waiting place and gluten match.
        total_cost = 0

        def serve_ready_matches(children_list):
            nonlocal total_cost, sandwiches_ontray, sandwiches_no_gluten, tray_occupied, tray_pos, trays
            served_now = []
            for child in children_list:
                place = self.waiting_place.get(child)
                if place is None:
                    continue
                # Find any ontray sandwich at that place with matching gluten property
                found_s = None
                for s, (t, p) in list(sandwiches_ontray.items()):
                    if p == place:
                        # check compatibility
                        gluten_flag = s in sandwiches_no_gluten
                        if child in self.allergic_children and not gluten_flag:
                            continue
                        # non-allergic accepts both
                        found_s = s
                        found_tray = t
                        break
                if found_s:
                    # It will cost 1 action to serve
                    total_cost += 1
                    served_now.append((child, found_s, found_tray))
                    # update resources: sandwich removed, tray becomes free at that place
                    sandwiches_ontray.pop(found_s, None)
                    sandwiches_no_gluten.discard(found_s)
                    if found_tray is not None:
                        tray_occupied[found_tray] = None
                        tray_pos[found_tray] = place
            return served_now

        # First pass: exact ontray matches at child's place for allergic and nonallergic
        serve_ready_matches(allergic_unserved)
        serve_ready_matches(nonallergic_unserved)

        # Refresh resource groups after removals
        ontray_gf = {s for s in sandwiches_ontray if s in sandwiches_no_gluten}
        ontray_non_gf = {s for s in sandwiches_ontray if s not in sandwiches_no_gluten}
        ak_gf = {s for s in sandwiches_at_kitchen if s in sandwiches_no_gluten}
        ak_non_gf = {s for s in sandwiches_at_kitchen if s not in sandwiches_no_gluten}

        # Recompute counts of free trays (they may have changed)
        free_trays_at_kitchen = sum(1 for tr in trays if tray_occupied.get(tr) is None and tray_pos.get(tr) == "kitchen")
        free_trays_elsewhere = sum(1 for tr in trays if tray_occupied.get(tr) is None and tray_pos.get(tr) != "kitchen")

        # Helper: assign resource to one child greedily
        PENALTY = 1000  # large finite penalty for infeasible child

        def assign_child(child, require_gluten_free):
            """
            Try to assign the cheapest available sequence of actions to serve 'child'.
            Update resources accordingly and return estimated cost (int).
            """
            nonlocal total_cost
            nonlocal ontray_gf, ontray_non_gf, ak_gf, ak_non_gf
            nonlocal sandwiches_ontray, sandwiches_at_kitchen, sandwiches_no_gluten
            nonlocal free_trays_at_kitchen, free_trays_elsewhere
            nonlocal gf_bread_at_kitchen, gf_content_at_kitchen, total_bread_at_kitchen, total_content_at_kitchen

            place = self.waiting_place.get(child, None)

            # Build candidate options (cost, action_label, resource_details)
            candidates = []

            # Option A: use an ontray sandwich somewhere (move tray if necessary)
            # find any ontray matching gluten requirement
            ontray_candidates = ontray_gf if require_gluten_free else (ontray_gf | ontray_non_gf)
            if ontray_candidates:
                # choose cheapest among them (they are equivalent except for tray positions)
                best = None
                best_cost = None
                best_s = None
                best_tray = None
                for s in ontray_candidates:
                    t, p = sandwiches_ontray[s]
                    # If tray already at child's place: cost 1 to serve (we already handled exact place earlier)
                    if p == place:
                        c = 1
                    else:
                        # move_tray (1) + serve (1)
                        c = 2
                    if best_cost is None or c < best_cost:
                        best_cost = c
                        best = (s, t, p)
                if best is not None:
                    s, t, p = best
                    candidates.append((best_cost, "use_ontray", (s, t, p)))

            # Option B: use an at_kitchen_sandwich
            ak_candidates = ak_gf if require_gluten_free else (ak_gf | ak_non_gf)
            if ak_candidates:
                # If a free tray at kitchen exists => put_on_tray(1) + move_tray(1) + serve(1) = 3
                if free_trays_at_kitchen > 0:
                    # pick any sandwich
                    s = next(iter(ak_candidates))
                    candidates.append((3, "use_at_kitchen_with_tray_at_kitchen", (s,)))
                elif free_trays_elsewhere > 0:
                    # move tray to kitchen (1) + put_on_tray(1) + move to child(1) + serve(1) = 4
                    s = next(iter(ak_candidates))
                    candidates.append((4, "use_at_kitchen_with_tray_elsewhere", (s,)))
                else:
                    # no free trays at all -> cannot use at_kitchen_sandwich in our simplified model
                    pass

            # Option C: make a new sandwich at kitchen then use it
            # Check ingredient availability
            # For gluten-free requirement both gf bread and gf content are required
            possible_make = False
            if require_gluten_free:
                if gf_bread_at_kitchen >= 1 and gf_content_at_kitchen >= 1:
                    possible_make = True
            else:
                if total_bread_at_kitchen >= 1 and total_content_at_kitchen >= 1:
                    possible_make = True
            if possible_make:
                # cost of make is 1 + then as in Option B the cost to put on tray/move/serve
                if free_trays_at_kitchen > 0:
                    candidates.append((1 + 3, "make_then_use_tray_at_kitchen", None))
                elif free_trays_elsewhere > 0:
                    candidates.append((1 + 4, "make_then_use_tray_elsewhere", None))
                else:
                    # if no trays, can't put sandwich anywhere
                    pass

            # Choose minimal cost candidate
            if not candidates:
                # no feasible way to serve this child in our model -> penalty
                total_cost += PENALTY
                return PENALTY

            candidates.sort(key=lambda x: x[0])
            cost, label, details = candidates[0]
            total_cost += cost

            # Update resources according to chosen label
            if label == "use_ontray":
                s, t, p = details
                # Remove sandwich from ontray
                sandwiches_ontray.pop(s, None)
                sandwiches_no_gluten.discard(s)
                # Mark tray now at child's place and free (serve removes ontray)
                tray_occupied[t] = None
                tray_pos[t] = place
                # If the tray was previously counted as free somewhere, adjust counts: it becomes free at child's place
                # We do not need exact per-place bookkeeping except kitchen vs elsewhere
                # If previously free at kitchen or elsewhere, adjust counts conservatively:
                # If previous p was kitchen and it was free, we should have earlier counted it. Rather than recomputing,
                # we update free_trays_at_kitchen/elsewhere to reflect new state:
                # We will recompute counts more conservatively: decrement whichever category p belonged to if tray was free before.
                # But we can't know whether it was free before since it carried sandwich; if it carried sandwich it wasn't free.
                # So only handle states where it was free before (not the case). For simplicity, if it carried a sandwich, free counts increase by 1 at child's place.
                if p == "kitchen":
                    # it was occupied at kitchen previously, now free at child's place -> free_trays_at_kitchen unaffected (it was occupied), increase free_trays_elsewhere
                    free_trays_elsewhere += 1
                else:
                    # previously occupied elsewhere, now free elsewhere (at child's place)
                    free_trays_elsewhere += 1

            elif label == "use_at_kitchen_with_tray_at_kitchen":
                s = details[0]
                # consume the at_kitchen_sandwich
                sandwiches_at_kitchen.discard(s)
                sandwiches_no_gluten.discard(s)
                if s in ak_gf:
                    ak_gf.discard(s)
                if s in ak_non_gf:
                    ak_non_gf.discard(s)
                # Use one free tray at kitchen; after serve, tray ends at child's place and is free
                free_trays_at_kitchen -= 1
                free_trays_elsewhere += 1

            elif label == "use_at_kitchen_with_tray_elsewhere":
                s = details[0]
                sandwiches_at_kitchen.discard(s)
                sandwiches_no_gluten.discard(s)
                if s in ak_gf:
                    ak_gf.discard(s)
                if s in ak_non_gf:
                    ak_non_gf.discard(s)
                # Move one free tray from elsewhere to kitchen and then after serve it ends at child's place and is free
                free_trays_elsewhere -= 1
                free_trays_elsewhere += 1  # net unchanged (moved from elsewhere to kitchen then to child place -> still elsewhere)
                # but to be conservative we leave counts unchanged except ensure non-negative
                free_trays_elsewhere = max(free_trays_elsewhere, 0)

            elif label == "make_then_use_tray_at_kitchen":
                # consume appropriate ingredients
                if require_gluten_free:
                    gf_bread_at_kitchen -= 1
                    gf_content_at_kitchen -= 1
                    total_bread_at_kitchen -= 1
                    total_content_at_kitchen -= 1
                else:
                    total_bread_at_kitchen -= 1
                    total_content_at_kitchen -= 1
                    # if gluten-free specific ingredient was used we don't track here; assume worst-case none
                    if gf_bread_at_kitchen > 0:
                        gf_bread_at_kitchen = max(gf_bread_at_kitchen - 1, 0)
                    if gf_content_at_kitchen > 0:
                        gf_content_at_kitchen = max(gf_content_at_kitchen - 1, 0)
                # After making we would have an at_kitchen_sandwich which we immediately use with free tray at kitchen:
                free_trays_at_kitchen -= 1
                free_trays_elsewhere += 1

            elif label == "make_then_use_tray_elsewhere":
                # consume ingredients
                if require_gluten_free:
                    gf_bread_at_kitchen -= 1
                    gf_content_at_kitchen -= 1
                    total_bread_at_kitchen -= 1
                    total_content_at_kitchen -= 1
                else:
                    total_bread_at_kitchen -= 1
                    total_content_at_kitchen -= 1
                    if gf_bread_at_kitchen > 0:
                        gf_bread_at_kitchen = max(gf_bread_at_kitchen - 1, 0)
                    if gf_content_at_kitchen > 0:
                        gf_content_at_kitchen = max(gf_content_at_kitchen - 1, 0)
                # tray movement leaves counts roughly unchanged for elsewhere
                free_trays_elsewhere = max(free_trays_elsewhere, 0)

            return cost

        # Assign allergic children first
        for child in allergic_unserved:
            # protect against having already been served by earlier exact matches
            if child in served_children:
                continue
            assign_child(child, require_gluten_free=True)

        # Assign non-allergic children
        for child in nonallergic_unserved:
            if child in served_children:
                continue
            assign_child(child, require_gluten_free=False)

        return total_cost