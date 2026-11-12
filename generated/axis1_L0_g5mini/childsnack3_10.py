from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a fact matches a pattern with wildcards.
    Example: match("(waiting child1 table1)", "waiting", "*", "*") -> True
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the childsnack domain that estimates the
    number of actions required to serve all children. The heuristic reasons about:
      - sandwiches that already exist (at kitchen or on trays),
      - trays and their locations,
      - gluten constraints (allergic vs non-allergic children),
      - kitchen resources (bread/content, and which are gluten-free),
      - the ability to move trays and serve multiple children with one tray move.

    The heuristic is not necessarily admissible; it aims to be informative and
    computationally cheap so greedy best-first search expands fewer nodes.

    Assumptions
    -----------
    - Each unserved child requires exactly one sandwich that matches their
      gluten requirement (no_gluten for allergic children).
    - Trays have unlimited capacity (a tray can hold many sandwiches).
    - Putting each sandwich on a tray requires a put_on_tray action.
    - Serving a child costs one serve action (serve_sandwich or
      serve_sandwich_no_gluten).
    - Moving a tray between places costs one move_tray action.
    - Creating a sandwich (make_sandwich or make_sandwich_no_gluten) costs one action
      and consumes one bread and one content item from kitchen.
    - If no tray is at the kitchen and the planner needs to put sandwiches on a
      tray at the kitchen, the heuristic assumes moving one tray to the kitchen
      costs one action (this is counted once, not per sandwich).
    - The heuristic favors using already-made sandwiches and already-positioned
      trays. It batches operations when reasonable (one tray move can deliver
      multiple sandwiches to the same place).

    Heuristic Initialization
    ------------------------
    In the constructor the heuristic extracts static information from the task:
      - waiting_map: maps each child to the place they are waiting at (from
        "(waiting child place)").
      - allergy_map: maps each child to True if allergic_gluten, False if
        not_allergic_gluten (if neither is present, defaults to False).
    These static structures are used to quickly determine which sandwich type a
    child requires and where they must be served.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a node/state the heuristic follows these steps:

    1. Goal check:
       - If all goal facts are satisfied, return 0.

    2. Gather current state information:
       - Identify which children are already served.
       - Build lists/dictionaries for:
         * trays and their current places,
         * sandwiches and their status:
           - notexist (still available to be created),
           - at_kitchen_sandwich,
           - ontray (and which tray),
           - whether each sandwich is no_gluten_sandwich.
         * kitchen resources:
           - counts of at_kitchen_bread, at_kitchen_content,
           - counts of gluten-free breads and contents (no_gluten_bread,
             no_gluten_content).
       - Count number of trays currently at the kitchen.

    3. Fast assignment of already-ready sandwiches:
       - For each unserved child, try to assign a sandwich already on a tray at
         the child's waiting place that satisfies the gluten need.
         Each such assignment costs 1 (the serve action).
       - Remove assigned sandwiches from the available pool and mark children as
         heuristically served.

    4. Reuse trays that can be moved:
       - For each place with unserved children, greedily select trays that are
         not at that place but that carry suitable sandwiches. Moving a tray
         costs 1 and then each sandwich on that tray used to serve children
         costs 1 per child. We choose trays that maximize the number of
         suitable sandwiches for the place first (greedy packing).

    5. Use sandwiches already at the kitchen:
       - If there are suitable at_kitchen_sandwiches, allocate them to remaining
         children. For this we:
           * If no tray is at the kitchen and we haven't moved one there yet,
             account for a one-time cost of moving one tray to the kitchen.
           * Each sandwich moved from kitchen to ontray costs 1 (put_on_tray).
           * For each place receiving any kitchen sandwiches, moving the tray
             from kitchen to that place costs 1 (move_tray) once for that place.
           * Each serve costs 1.
       - These costs are computed in a place-aware manner so multiple sandwiches
         for the same place share a single move_tray from kitchen -> place.

    6. Create new sandwiches if needed:
       - For remaining unserved children (after consuming all existing sandwiches
         that can be used), create sandwiches at the kitchen:
           * Allergic children must be created using no_gluten_bread and
             no_gluten_content. If there are insufficient gluten-free
             bread/content pairs, the heuristic returns a large penalty (to
             reflect impossibility / high cost).
           * Non-allergic children can use any remaining bread/content.
           * Each created sandwich costs 1 (make), plus 1 (put_on_tray), plus 1
             to move a tray to the child's place (shared per place as above),
             plus 1 to serve.
       - As with kitchen sandwiches, the first time we need to put sandwiches
         on a tray and no tray is at the kitchen, we count one move of a tray
         to the kitchen.

    7. Return the summed action estimate.

    Notes on design choices
    -----------------------
    - The heuristic is intentionally greedy and approximate (not admissible).
      It tries to capture reuse of trays and reuse of kitchen sandwiches to
      reduce estimated move costs, which makes it helpful for greedy search.
    - If the domain resources make serving impossible (not enough gluten-free
      pairs for allergic children), the heuristic returns a large finite value
      rather than infinity.
    """

    def __init__(self, task):
        # Goals are kept for the zero-test (but usually checked by the planner).
        self.goals = task.goals

        # Extract static waiting and allergy info (these are typically static).
        self.waiting_map = {}  # child -> place
        self.allergy_map = {}  # child -> True if allergic_gluten, False if not_allergic

        static_facts = task.static if task.static is not None else frozenset()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "waiting" and len(parts) == 3:
                child, place = parts[1], parts[2]
                self.waiting_map[child] = place
            elif parts[0] == "allergic_gluten" and len(parts) == 2:
                self.allergy_map[parts[1]] = True
            elif parts[0] == "not_allergic_gluten" and len(parts) == 2:
                self.allergy_map[parts[1]] = False

    def __call__(self, node):
        state = node.state

        # 1) If goal reached, heuristic is 0
        if self.goals <= state:
            return 0

        # Helper: get waiting place for a child (static or from state)
        waiting = dict(self.waiting_map)  # copy
        # If some waiting facts were not static, try to find them in state
        for fact in state:
            if match(fact, "waiting", "*", "*"):
                _, child, place = get_parts(fact)
                waiting[child] = place

        # Allergy info: default to False if unknown
        allergy = dict(self.allergy_map)
        for fact in state:
            if match(fact, "allergic_gluten", "*"):
                _, child = get_parts(fact)
                allergy[child] = True
            elif match(fact, "not_allergic_gluten", "*"):
                _, child = get_parts(fact)
                allergy[child] = False

        # 2) Identify which children are unserved and their requirements
        served_children = {get_parts(f)[1] for f in state if match(f, "served", "*")}
        # Build list of children that appear either in waiting or in goals
        # Use goals to identify which children must be served
        goal_children = []
        for g in self.goals:
            if match(g, "served", "*"):
                _, child = get_parts(g)
                goal_children.append(child)

        unserved_children = [c for c in goal_children if c not in served_children]

        # Group unserved by place and by allergy requirement
        children_by_place = {}
        for c in unserved_children:
            place = waiting.get(c, None)
            if place is None:
                # If waiting place unknown, assume worst-case unique place
                place = "unknown_place_for_" + c
            children_by_place.setdefault(place, []).append(c)

        # 3) Parse trays, sandwiches, kitchen resources
        trays_at = {}  # tray -> place
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, tray, place = get_parts(fact)
                trays_at[tray] = place

        # Sandwich info
        sandwiches = {}  # sandw -> dict(status: 'notexist'/'kitchen'/'ontray', tray, no_gluten)
        for fact in state:
            if match(fact, "notexist", "*"):
                _, s = get_parts(fact)
                sandwiches.setdefault(s, {})["notexist"] = True
            if match(fact, "at_kitchen_sandwich", "*"):
                _, s = get_parts(fact)
                sandwiches.setdefault(s, {})["at_kitchen_sandwich"] = True
            if match(fact, "ontray", "*", "*"):
                _, s, t = get_parts(fact)
                sandwiches.setdefault(s, {})["ontray"] = t
            if match(fact, "no_gluten_sandwich", "*"):
                _, s = get_parts(fact)
                sandwiches.setdefault(s, {})["no_gluten"] = True

        # Normalize sandwich dicts into a list of sandwich objects
        sandwich_objs = {}
        for s, info in sandwiches.items():
            if info.get("notexist", False):
                status = "notexist"
            elif info.get("ontray", None) is not None:
                status = "ontray"
            elif info.get("at_kitchen_sandwich", False):
                status = "kitchen"
            else:
                # Unknown means not present in state (treat as notexist)
                status = "notexist"
            sandwich_objs[s] = {
                "status": status,
                "tray": info.get("ontray", None),
                "no_gluten": info.get("no_gluten", False),
            }

        # Count kitchen bread/content resources
        kitchen_breads = 0
        kitchen_contents = 0
        kitchen_no_gluten_breads = 0
        kitchen_no_gluten_contents = 0
        for fact in state:
            if match(fact, "at_kitchen_bread", "*"):
                _, b = get_parts(fact)
                kitchen_breads += 1
            if match(fact, "at_kitchen_content", "*"):
                _, c = get_parts(fact)
                kitchen_contents += 1
            if match(fact, "no_gluten_bread", "*"):
                _, b = get_parts(fact)
                kitchen_no_gluten_breads += 1
            if match(fact, "no_gluten_content", "*"):
                _, c = get_parts(fact)
                kitchen_no_gluten_contents += 1

        # Count trays at kitchen
        trays_at_kitchen = [t for t, p in trays_at.items() if p == "kitchen"]
        num_trays_at_kitchen = len(trays_at_kitchen)

        # Organize trays with their sandwiches (ontray)
        trays_with_sandwiches = {}
        for s, info in sandwich_objs.items():
            if info["status"] == "ontray" and info["tray"] is not None:
                t = info["tray"]
                trays_with_sandwiches.setdefault(t, []).append((s, info["no_gluten"]))

        # Count total notexist sandwiches available for creation
        notexist_sandwiches = [s for s, info in sandwich_objs.items() if info["status"] == "notexist"]

        # 4) Heuristic simulation: allocate sandwiches to children greedily
        cost = 0

        # Track which sandwiches are used/consumed by our heuristic allocation
        used_sandwiches = set()

        # Track which children we've simulated as served
        simulated_served = set()

        # Helper to check allergy requirement for a child
        def requires_no_gluten(child):
            return allergy.get(child, False)

        # 4.1 Assign sandwiches already on trays at the child's place (fast serves)
        for place, children in list(children_by_place.items()):
            # Build lists of available ontray sandwiches at this place (group by gluten)
            ontray_at_place = []
            for tray, tray_place in trays_at.items():
                if tray_place != place:
                    continue
                for (s, no_gluten) in trays_with_sandwiches.get(tray, []):
                    if s in used_sandwiches:
                        continue
                    ontray_at_place.append((s, no_gluten, tray))
            # Assign for allergic children first (they need no_gluten)
            allergic_children = [c for c in children if requires_no_gluten(c) and c not in simulated_served]
            non_allergic_children = [c for c in children if (not requires_no_gluten(c)) and c not in simulated_served]

            # Assign gluten-free ontray to allergic children
            for c in allergic_children:
                assigned = False
                for s, no_gluten, t in ontray_at_place:
                    if s in used_sandwiches:
                        continue
                    if no_gluten:
                        used_sandwiches.add(s)
                        cost += 1  # serve action
                        simulated_served.add(c)
                        assigned = True
                        break
                if assigned:
                    # remove used sandwich from list
                    ontray_at_place = [(s,ng,t) for (s,ng,t) in ontray_at_place if s not in used_sandwiches]
            # Assign any ontray (including gluten-free) to non-allergic children
            for c in non_allergic_children:
                assigned = False
                for s, no_gluten, t in ontray_at_place:
                    if s in used_sandwiches:
                        continue
                    # non-allergic can take any sandwich
                    used_sandwiches.add(s)
                    cost += 1  # serve
                    simulated_served.add(c)
                    assigned = True
                    break
                if assigned:
                    ontray_at_place = [(s,ng,t) for (s,ng,t) in ontray_at_place if s not in used_sandwiches]

        # 4.2 For each place, consider moving trays that carry suitable sandwiches (tray move + serves)
        # We'll greedily move trays that maximize serves for that place.
        # Keep track of trays we've moved so we don't double count their move cost.
        moved_trays = set()
        for place, children in list(children_by_place.items()):
            remaining_children = [c for c in children if c not in simulated_served]
            if not remaining_children:
                continue
            # Split remaining into allergic and non-allergic
            rem_allergic = [c for c in remaining_children if requires_no_gluten(c)]
            rem_non_allergic = [c for c in remaining_children if not requires_no_gluten(c)]

            # For each tray not already at 'place' and not yet moved, consider moving it
            # Greedily pick trays providing the most suitable sandwiches
            candidate_trays = [t for t in trays_with_sandwiches.keys() if trays_at.get(t, None) != place]
            # Build per-tray counts of suitable sandwiches
            tray_suitability = {}
            for t in candidate_trays:
                gluten_free_count = sum(1 for (s, ng) in trays_with_sandwiches.get(t, []) if (not s in used_sandwiches) and ng)
                any_count = sum(1 for (s, ng) in trays_with_sandwiches.get(t, []) if (not s in used_sandwiches))
                tray_suitability[t] = (gluten_free_count, any_count)

            # Greedy loop: while there are remaining children and trays with suitable sandwiches
            while (rem_allergic or rem_non_allergic) and any(tray_suitability.values()):
                # Choose tray maximizing number of gluten-free sandwiches first (to help allergic)
                best_tray = None
                best_score = -1
                for t, (gf, anyc) in tray_suitability.items():
                    # score prioritized: gluten_free_count then total any_count
                    score = (gf * 10) + anyc
                    if score > best_score:
                        best_score = score
                        best_tray = t
                if best_tray is None:
                    break
                gf, anyc = tray_suitability[best_tray]
                if gf == 0 and anyc == 0:
                    break  # nothing usable on this tray
                # Move the tray (cost 1) if we haven't moved it yet
                if best_tray not in moved_trays:
                    cost += 1  # move_tray
                    moved_trays.add(best_tray)
                    # After moving, the tray is considered at 'place'; update trays_at
                    trays_at[best_tray] = place
                # Serve as many allergic children as possible using gluten-free sandwiches on that tray
                serve_count = 0
                # Use gluten-free sandwiches first for allergic children
                for (s, ng) in list(trays_with_sandwiches.get(best_tray, [])):
                    if s in used_sandwiches:
                        continue
                    if rem_allergic and ng:
                        # serve one allergic child
                        used_sandwiches.add(s)
                        cost += 1  # serve
                        simulated_served.add(rem_allergic.pop(0))
                        serve_count += 1
                # Then serve non-allergic children with any remaining sandwiches on the tray
                for (s, ng) in list(trays_with_sandwiches.get(best_tray, [])):
                    if s in used_sandwiches:
                        continue
                    if rem_non_allergic:
                        used_sandwiches.add(s)
                        cost += 1
                        simulated_served.add(rem_non_allergic.pop(0))
                        serve_count += 1
                # Update remaining children lists
                remaining_children = [c for c in children if c not in simulated_served]
                rem_allergic = [c for c in remaining_children if requires_no_gluten(c)]
                rem_non_allergic = [c for c in remaining_children if not requires_no_gluten(c)]
                # Update tray_suitability for this tray because some sandwiches consumed
                gf_new = sum(1 for (s, ng) in trays_with_sandwiches.get(best_tray, []) if (not s in used_sandwiches) and ng)
                any_new = sum(1 for (s, ng) in trays_with_sandwiches.get(best_tray, []) if (not s in used_sandwiches))
                tray_suitability[best_tray] = (gf_new, any_new)
                # Remove trays that have no more usable sandwiches
                if tray_suitability[best_tray] == (0, 0):
                    del tray_suitability[best_tray]

        # 4.3 Use sandwiches that are at the kitchen already (put_on_tray + move + serve)
        # Build lists of available kitchen sandwiches split by gluten
        kitchen_sandwiches = [s for s, info in sandwich_objs.items() if info["status"] == "kitchen" and s not in used_sandwiches]
        kitchen_gf = [s for s in kitchen_sandwiches if sandwich_objs[s]["no_gluten"]]
        kitchen_any = [s for s in kitchen_sandwiches if not sandwich_objs[s]["no_gluten"]]

        # Flag: whether we've paid the one-time cost to move a tray to kitchen (if needed)
        moved_tray_to_kitchen = False
        if num_trays_at_kitchen > 0:
            moved_tray_to_kitchen = True  # already a tray at kitchen, no move needed

        # For each place, allocate kitchen sandwiches
        for place, children in list(children_by_place.items()):
            remaining_children = [c for c in children if c not in simulated_served]
            if not remaining_children:
                continue
            # Allergic first
            rem_allergic = [c for c in remaining_children if requires_no_gluten(c)]
            rem_non_allergic = [c for c in remaining_children if not requires_no_gluten(c)]

            used_from_kitchen = 0  # number of sandwiches moved from kitchen for this place

            # Serve allergic children from kitchen gluten-free sandwiches
            while rem_allergic and kitchen_gf:
                s = kitchen_gf.pop()  # take one gluten-free kitchen sandwich
                used_sandwiches.add(s)
                kitchen_any = [x for x in kitchen_any if x != s]  # remove if present
                cost += 1  # put_on_tray
                used_from_kitchen += 1
                cost += 1  # serve
                simulated_served.add(rem_allergic.pop(0))

            # Serve non-allergic children, can use any kitchen sandwich including gf
            while rem_non_allergic and (kitchen_any or kitchen_gf):
                # prefer non-gf if present to save gf for allergic
                if kitchen_any:
                    s = kitchen_any.pop()
                else:
                    s = kitchen_gf.pop()
                used_sandwiches.add(s)
                cost += 1  # put_on_tray
                used_from_kitchen += 1
                cost += 1  # serve
                simulated_served.add(rem_non_allergic.pop(0))

            if used_from_kitchen > 0:
                # If there was no tray at kitchen and we haven't moved one yet, move one to kitchen
                if not moved_tray_to_kitchen:
                    # If there exists at least one tray somewhere, moving one to kitchen costs 1
                    # If there are no trays at all, we fallback to a large penalty (rare)
                    if trays_at:
                        cost += 1
                        moved_tray_to_kitchen = True
                    else:
                        # No trays available: very bad, large penalty to discourage this path
                        cost += 200
                        moved_tray_to_kitchen = True
                # Now we must move the tray from kitchen to the place: assume one move per place used
                cost += 1  # move_tray kitchen -> place

        # 4.4 Create new sandwiches for remaining children (use bread & content resources)
        # Count remaining unserved children
        remaining_unserved = [c for c in unserved_children if c not in simulated_served]
        if remaining_unserved:
            # First satisfy allergic ones using gluten-free resources
            needed_gf = sum(1 for c in remaining_unserved if requires_no_gluten(c))
            needed_any = sum(1 for c in remaining_unserved if not requires_no_gluten(c))

            # Available GF resources are pairs of (no_gluten_bread, no_gluten_content)
            available_gf_pairs = min(kitchen_no_gluten_breads, kitchen_no_gluten_contents)
            # Available general pairs (after GF assigned) come from remaining breads & contents
            # total pairs possible with remaining resources:
            total_pairs = min(kitchen_breads, kitchen_contents)
            # After using GF pairs, remaining general pairs are:
            remaining_pairs_after_gf = max(0, total_pairs - min(total_pairs, available_gf_pairs))

            # Check feasibility: need enough GF pairs for allergic children
            if needed_gf > available_gf_pairs:
                # Not enough GF resources - penalize heavily but keep finite
                # Make a big penalty proportional to missing count
                missing = needed_gf - available_gf_pairs
                cost += 1000 * missing
                # We still attempt to estimate cost for the rest as if we could create them
                needed_gf = available_gf_pairs

            # Now estimate creation costs:
            # For each new sandwich created: make (1), put_on_tray (1), serve (1)
            # We'll charge move_tray per place used similarly to kitchen handling above.
            # We'll assume that created sandwiches are produced at kitchen, so we may need to move a tray to kitchen once (if not already)
            if remaining_unserved:
                if not moved_tray_to_kitchen:
                    if trays_at:
                        cost += 1
                        moved_tray_to_kitchen = True
                    else:
                        cost += 200
                        moved_tray_to_kitchen = True

            # Allocate created sandwiches per place, batching moves per place.
            # Build per-place remaining lists again
            for place, children in list(children_by_place.items()):
                rem_children = [c for c in children if c not in simulated_served]
                if not rem_children:
                    continue
                # For each place, we'll create sandwiches for its remaining children
                # For allergic first, ensure GF resources used
                create_count = 0
                for c in list(rem_children):
                    if requires_no_gluten(c):
                        if available_gf_pairs <= 0:
                            # no GF left, heavy penalty already added above; treat as create using general resources implicitly
                            pass
                        else:
                            # create GF sandwich
                            available_gf_pairs -= 1
                            # consume overall kitchen resources counts (conservative)
                            kitchen_no_gluten_breads = max(0, kitchen_no_gluten_breads - 1)
                            kitchen_no_gluten_contents = max(0, kitchen_no_gluten_contents - 1)
                            kitchen_breads = max(0, kitchen_breads - 1)
                            kitchen_contents = max(0, kitchen_contents - 1)
                            create_count += 1
                            simulated_served.add(c)
                    else:
                        # non-allergic: try to use remaining general pairs (including remaining GF if any)
                        if total_pairs > 0:
                            total_pairs -= 1
                            kitchen_breads = max(0, kitchen_breads - 1)
                            kitchen_contents = max(0, kitchen_contents - 1)
                            create_count += 1
                            simulated_served.add(c)
                        else:
                            # no resources left - heavy penalty
                            cost += 500
                            simulated_served.add(c)
                            create_count += 1
                if create_count > 0:
                    # For created sandwiches: each costs make (1) + put_on_tray (1) + serve (1)
                    cost += create_count * 3
                    # For this place, moving tray from kitchen -> place once
                    cost += 1

        # Final safety: heuristic must be finite - ensure it's an integer
        if cost is None or cost == float("inf"):
            return 10 ** 6
        return int(cost)