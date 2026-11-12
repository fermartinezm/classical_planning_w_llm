from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split a fact string into parts."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the childsnacks domain. The heuristic
    estimates the (approximate) minimum number of actions required to serve
    all children listed in the goal (served childX). It reasons about:
      - existing sandwiches (at kitchen or on trays),
      - trays and their locations,
      - available bread and content portions at the kitchen,
      - gluten requirements for allergic children (must receive a no-gluten sandwich).

    Assumptions
    -----------
    - Each PDDL action has unit cost:
        make_sandwich / make_sandwich_no_gluten = 1
        put_on_tray = 1
        move_tray = 1
        serve_sandwich / serve_sandwich_no_gluten = 1
      (This matches the intuitive action-count cost model used by many domain
       heuristics and keeps computations simple.)
    - A sandwich compatible with an allergic child must be explicitly
      marked with (no_gluten_sandwich ?s). To create such a sandwich the
      bread and content used must both be labelled no_gluten_* in the static
      facts / initial state.
    - Waiting locations of children are static (they do not change during
      planning). These are typically provided as static facts "(waiting childX placeY)".
    - Trays persist and are reusable; moving a tray changes its location and
      is modeled explicitly.
    - The heuristic is not required to be admissible; it should be informative
      and cheaply computable to guide greedy best-first search.

    Heuristic Initialization
    ------------------------
    The constructor extracts:
      - the set of goal children from task.goals (goals are of the form "(served childX)"),
      - static facts that are useful for allocation:
        * waiting_map: maps child -> place
        * allergic_gluten set and not_allergic_gluten set
        * no_gluten_bread and no_gluten_content sets (to decide whether a
          gluten-free sandwich can be made using particular ingredients)
    These static facts are used repeatedly in the heuristic computation.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given state, the heuristic proceeds in a greedy resource-allocation
    manner (allergic children are prioritized to avoid wasting gluten-free
    ingredients on non-allergic children):

    1. Parse the current state to obtain:
       - Which children are already served (so they can be ignored).
       - Sandwich facts:
         * sandwiches currently on trays (with the tray id),
         * sandwiches currently at the kitchen (not on any tray),
         * which sandwiches are labeled no_gluten_sandwich.
       - Tray locations (mapping tray -> place and counts per place).
       - Bread and content portions at the kitchen (available ingredients).
       - Count of notexist sandwich IDs (available sandwich "slots" to create new sandwiches).
       - The places where children are waiting (if not found in static, a missing entry will force a conservative estimate).
    2. Build a list of unserved goal children. Sort it so allergic children are considered first
       (they have stricter resource requirements).
    3. For each unserved child (in that order), compute the cheapest feasible way
       to get a compatible sandwich served, considering current resources:
         - If a compatible sandwich is already on a tray at the child's place: cost = 1 (serve).
         - If a compatible sandwich is on a tray elsewhere: cost = 1 (move_tray) + 1 (serve) = 2.
         - If a compatible sandwich is at the kitchen (not on tray):
             * If a tray is already at the kitchen: cost = put_on_tray (1) + move_tray? (1 if child not at kitchen) + serve (1).
             * If no tray at kitchen but a tray exists somewhere: we must first move a tray to kitchen (1), then put_on_tray (1), then move to child if needed (1), then serve (1).
         - If no compatible sandwich exists but the kitchen has compatible bread & content and an unused sandwich id:
             * Cost is make (1) + same placement costs as for an at_kitchen_sandwich.
         - If none of the above are feasible, the child is considered currently unreachable and a large penalty is added for this child (to keep heuristic finite but large).
       The heuristic picks the cheapest feasible option and then updates a local, mutable view
       of the world (consuming the used sandwich, bread/content, notexist slot, moving trays, etc.)
       so subsequent children are estimated with the remaining resources.
    4. The heuristic value is the sum of costs chosen for all unserved children.
       If the state already satisfies all goals, the heuristic returns 0.
    5. The heuristic always returns a finite integer. For children that are
       impossible to satisfy with current static constraints (e.g., an allergic
       child but no gluten-free bread/content exist anywhere), a large fixed
       penalty is used to reflect infeasibility while keeping the heuristic
       numeric and finite.

    The overall strategy is deliberately greedy and approximate: it aims to
    capture the main resource constraints (tray availability, ingredient counts,
    and gluten compatibility) which are the dominant causes of branching in
    this domain, leading to informative guidance for greedy best-first search.
    """

    # Large penalty for infeasible allocations (keeps heuristic finite).
    INFEASIBLE_PENALTY = 1000

    def __init__(self, task):
        # Goals are strings like "(served child1)"; we extract the child ids.
        self.goals = set(task.goals)
        self.goal_children = []
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "served" and len(parts) >= 2:
                self.goal_children.append(parts[1])

        # Extract static facts for allergies, waiting places, and gluten-free ingredient tags.
        static = task.static if hasattr(task, "static") else set()

        self.waiting_map = {}  # child -> place
        self.allergic = set()
        self.not_allergic = set()
        self.no_gluten_bread_static = set()
        self.no_gluten_content_static = set()

        for fact in static:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "waiting" and len(parts) >= 3:
                child, place = parts[1], parts[2]
                self.waiting_map[child] = place
            elif pred == "allergic_gluten" and len(parts) >= 2:
                self.allergic.add(parts[1])
            elif pred == "not_allergic_gluten" and len(parts) >= 2:
                self.not_allergic.add(parts[1])
            elif pred == "no_gluten_bread" and len(parts) >= 2:
                self.no_gluten_bread_static.add(parts[1])
            elif pred == "no_gluten_content" and len(parts) >= 2:
                self.no_gluten_content_static.add(parts[1])

    def __call__(self, node):
        state = node.state

        # Fast check: if goals are already satisfied return 0.
        if self.goals <= state:
            return 0

        # Parse dynamic facts from the current state.
        # 1) Served children (to ignore)
        served_children = set()
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "served" and len(parts) >= 2:
                served_children.add(parts[1])

        # 2) Trays and their current locations.
        tray_at = {}  # tray -> place
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at" and len(parts) >= 3:
                obj, place = parts[1], parts[2]
                # We only care about trays' locations here; other "at" facts exist but we treat by types by naming convention.
                # In practice the object names for trays begin with "tray" in instances.
                tray_at[obj] = place

        # Build a place -> tray count mapping for convenience
        tray_counts = {}
        for tr, pl in tray_at.items():
            tray_counts[pl] = tray_counts.get(pl, 0) + 1
        total_trays = len(tray_at)

        # 3) Sandwich facts: ontray, at_kitchen_sandwich, and gluten labels.
        ontray = {}  # sandwich -> tray
        at_kitchen_sandwiches = set()
        no_gluten_sandwiches = set()
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "ontray" and len(parts) >= 3:
                sandw, tray = parts[1], parts[2]
                ontray[sandw] = tray
            elif pred == "at_kitchen_sandwich" and len(parts) >= 2:
                at_kitchen_sandwiches.add(parts[1])
            elif pred == "no_gluten_sandwich" and len(parts) >= 2:
                no_gluten_sandwiches.add(parts[1])

        # 4) Bread and content portions currently at kitchen (available ingredients).
        breads_at_kitchen = []
        contents_at_kitchen = []
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "at_kitchen_bread" and len(parts) >= 2:
                breads_at_kitchen.append(parts[1])
            elif pred == "at_kitchen_content" and len(parts) >= 2:
                contents_at_kitchen.append(parts[1])

        # Count notexist sandwich ids (available to create new sandwiches).
        notexist_sandw = set()
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "notexist" and len(parts) >= 2:
                notexist_sandw.add(parts[1])
        notexist_count = len(notexist_sandw)

        # 5) waiting places for all children: prefer static waiting_map, but if missing use any waiting facts from state.
        waiting_map = dict(self.waiting_map)  # start from static
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "waiting" and len(parts) >= 3:
                waiting_map[parts[1]] = parts[2]

        # Build list of children to serve (in goals but not yet served)
        to_serve = [c for c in self.goal_children if c not in served_children]

        # Prioritize allergic children first (they require gluten-free sandwiches)
        def child_priority(c):
            return 0 if c in self.allergic else 1

        to_serve.sort(key=child_priority)

        # Helper: check if a sandwich id is gluten-free
        def is_sandwich_gluten_free(sid):
            return sid in no_gluten_sandwiches

        # Helper: find an available bread-content pair for a given gluten_free requirement
        # We will pick and consume one bread and one content (represented by list entries)
        def has_and_consume_ingredients(require_gluten_free):
            # For gluten-free requirement, the chosen bread must be in no_gluten_bread_static and also at kitchen list.
            # For non gluten-free, any bread/content at kitchen is fine.
            # Implementation picks the first matching pair.
            if require_gluten_free:
                # find bread at kitchen that is gluten free
                b_index = None
                for i, b in enumerate(breads_at_kitchen):
                    if b in self.no_gluten_bread_static:
                        b_index = i
                        break
                c_index = None
                for j, c in enumerate(contents_at_kitchen):
                    if c in self.no_gluten_content_static:
                        c_index = j
                        break
                if b_index is not None and c_index is not None:
                    # consume them
                    breads_at_kitchen.pop(b_index)
                    # adjust content index if needed
                    contents_at_kitchen.pop(c_index)
                    return True
                else:
                    return False
            else:
                if breads_at_kitchen and contents_at_kitchen:
                    # consume first of each
                    breads_at_kitchen.pop()
                    contents_at_kitchen.pop()
                    return True
                else:
                    return False

        total_cost = 0

        # Build quick structures mapping trays to sandwiches on them
        # and mapping sandwich -> gluten flag and current place if ontray (via tray_at)
        sandwich_ontray_map = dict(ontray)  # sandw -> tray
        # sandwiches at kitchen set already: at_kitchen_sandwiches
        # We'll mutate sandwich_ontray_map and at_kitchen_sandwiches as we allocate.

        # For selecting trays to move to kitchen when none at kitchen: we will pick any tray that is not at kitchen.
        # Keep a mapping place -> list of trays for more accurate movement updates.
        place_to_trays = {}
        for tr, pl in tray_at.items():
            place_to_trays.setdefault(pl, []).append(tr)

        # Start allocating for each child greedily
        for child in to_serve:
            need_gluten_free = child in self.allergic
            child_place = waiting_map.get(child, None)
            if child_place is None:
                # If we don't know where the child waits, pessimistically assume a move is required.
                # We'll still attempt to serve using kitchen as destination with extra moves.
                child_place = "UNKNOWN_PLACE"

            best_option = None
            best_cost = None
            best_data = None  # store data to apply the chosen option later

            # Option A: an ontray sandwich at child's place (cost 1)
            chosen_s = None
            for s, tr in sandwich_ontray_map.items():
                tr_place = tray_at.get(tr)
                if tr_place == child_place:
                    # check gluten compatibility
                    if need_gluten_free:
                        if is_sandwich_gluten_free(s):
                            chosen_s = s
                            break
                    else:
                        # any sandwich ok
                        chosen_s = s
                        break
            if chosen_s is not None:
                best_option = "ontray_same_place"
                best_cost = 1
                best_data = {"sandwich": chosen_s, "tray": sandwich_ontray_map[chosen_s]}

            # Option B: an ontray sandwich elsewhere: move + serve = 2
            if best_cost is None or 2 < best_cost:
                chosen_s = None
                chosen_tr = None
                for s, tr in sandwich_ontray_map.items():
                    tr_place = tray_at.get(tr)
                    if tr_place != child_place:
                        if need_gluten_free:
                            if is_sandwich_gluten_free(s):
                                chosen_s = s
                                chosen_tr = tr
                                break
                        else:
                            chosen_s = s
                            chosen_tr = tr
                            break
                if chosen_s is not None:
                    best_option = "ontray_other_place"
                    best_cost = 2
                    best_data = {"sandwich": chosen_s, "tray": chosen_tr, "from_place": tray_at.get(chosen_tr)}

            # Option C: a sandwich at kitchen that can be used (put_on_tray + maybe move + serve)
            if at_kitchen_sandwiches:
                # check if any at kitchen sandwich matches gluten requirement
                chosen_s = None
                for s in at_kitchen_sandwiches:
                    if need_gluten_free:
                        if is_sandwich_gluten_free(s):
                            chosen_s = s
                            break
                    else:
                        chosen_s = s
                        break
                if chosen_s is not None:
                    # compute extra move to get a tray to kitchen if none there
                    need_move_tray_to_kitchen = 0
                    if tray_counts.get("kitchen", 0) == 0:
                        # if there is at least one tray somewhere else, we need one move to move a tray to kitchen
                        if total_trays > 0:
                            need_move_tray_to_kitchen = 1
                        else:
                            # no trays at all -- infeasible unless serving at kitchen without tray (not possible)
                            need_move_tray_to_kitchen = None  # mark infeasible
                    if need_move_tray_to_kitchen is not None:
                        move_from_kitchen_to_child = 0 if child_place == "kitchen" else 1
                        cost = need_move_tray_to_kitchen + 1 + move_from_kitchen_to_child + 1
                        # (move tray to kitchen if needed) + put_on_tray (1) + move to child if needed + serve (1)
                        if best_cost is None or cost < best_cost:
                            best_option = "at_kitchen_sandwich"
                            best_cost = cost
                            best_data = {"sandwich": chosen_s, "need_move_tray_to_kitchen": need_move_tray_to_kitchen}
            # Option D: make a new sandwich using at-kitchen ingredients (and notexist slot)
            if not (breads_at_kitchen == [] or contents_at_kitchen == []) and notexist_count > 0:
                # Check if we can create gluten-free sandwich when required
                can_make = None
                if need_gluten_free:
                    # check if there exists at least one bread in breads_at_kitchen that is also in static no_gluten set,
                    # and one content in contents_at_kitchen that is in static no_gluten set.
                    has_b = any((b in self.no_gluten_bread_static) for b in breads_at_kitchen)
                    has_c = any((c in self.no_gluten_content_static) for c in contents_at_kitchen)
                    can_make = has_b and has_c and notexist_count > 0
                else:
                    # any available bread and content suffice
                    can_make = bool(breads_at_kitchen) and bool(contents_at_kitchen) and notexist_count > 0

                if can_make:
                    # making adds cost 1, then same placement costs as an at_kitchen_sandwich
                    need_move_tray_to_kitchen = 0
                    if tray_counts.get("kitchen", 0) == 0:
                        if total_trays > 0:
                            need_move_tray_to_kitchen = 1
                        else:
                            need_move_tray_to_kitchen = None
                    if need_move_tray_to_kitchen is not None:
                        move_from_kitchen_to_child = 0 if child_place == "kitchen" else 1
                        cost = 1 + need_move_tray_to_kitchen + 1 + move_from_kitchen_to_child + 1
                        # make (1) + (move tray to kitchen if necessary) + put_on_tray (1) + move to child + serve (1)
                        if best_cost is None or cost < best_cost:
                            best_option = "make_and_place"
                            best_cost = cost
                            best_data = {"need_move_tray_to_kitchen": need_move_tray_to_kitchen, "gluten_free": need_gluten_free}

            # If none of the options are feasible, mark infeasible (large penalty)
            if best_option is None:
                total_cost += self.INFEASIBLE_PENALTY
                # No resource changes possible; continue to next child.
                continue

            # Apply the chosen option's resource consumption in our local model and accumulate cost.
            total_cost += best_cost

            if best_option == "ontray_same_place":
                s = best_data["sandwich"]
                # remove sandwich from ontray mapping (it's consumed by serving)
                if s in sandwich_ontray_map:
                    tr = sandwich_ontray_map.pop(s)
                    # the tray remains at child's place; nothing to change except remove sandwich
                    # The sandwich no longer on a tray and we treat it as consumed.
                    # also ensure that we don't consider this sandwich later
                # also if it was in at_kitchen_sandwiches (shouldn't), remove
                at_kitchen_sandwiches.discard(s)
                no_gluten_sandwiches.discard(s)
            elif best_option == "ontray_other_place":
                s = best_data["sandwich"]
                tr = best_data["tray"]
                from_place = best_data.get("from_place", tray_at.get(tr))
                # move the tray to child's place (1) and then serve (1)
                # update tray_counts and tray_at
                if from_place is not None:
                    tray_counts[from_place] = tray_counts.get(from_place, 1) - 1
                    if tray_counts[from_place] <= 0:
                        tray_counts.pop(from_place, None)
                tray_at[tr] = child_place
                tray_counts[child_place] = tray_counts.get(child_place, 0) + 1
                # remove sandwich from ontray mapping (consumed)
                if s in sandwich_ontray_map:
                    sandwich_ontray_map.pop(s)
                at_kitchen_sandwiches.discard(s)
                no_gluten_sandwiches.discard(s)
            elif best_option == "at_kitchen_sandwich":
                s = best_data["sandwich"]
                need_move = best_data.get("need_move_tray_to_kitchen", 0)
                # If we need to move a tray to kitchen, pick some tray from other place
                if need_move == 1:
                    # pick any place with a tray (prefer not child's place to avoid extra back-and-forth)
                    donor_place = None
                    for place, trays in place_to_trays.items():
                        if place != "kitchen" and trays:
                            donor_place = place
                            break
                    # fallback to any place
                    if donor_place is None:
                        for place, trays in place_to_trays.items():
                            if trays:
                                donor_place = place
                                break
                    if donor_place is not None:
                        # pick a tray id from donor_place
                        tr = place_to_trays[donor_place].pop()
                        # update tray_at and counts
                        tray_at[tr] = "kitchen"
                        tray_counts[donor_place] = tray_counts.get(donor_place, 1) - 1
                        if tray_counts[donor_place] <= 0:
                            tray_counts.pop(donor_place, None)
                        tray_counts["kitchen"] = tray_counts.get("kitchen", 0) + 1
                        # also register this in place_to_trays
                        place_to_trays.setdefault("kitchen", []).append(tr)
                # Now there is at least one tray at kitchen; pick one tray from kitchen to put sandwich on
                # choose a tray id from place_to_trays["kitchen"]
                pick_tr = None
                if place_to_trays.get("kitchen"):
                    pick_tr = place_to_trays["kitchen"].pop()
                    # this tray will end up at child's place after moving (unless child's place == kitchen)
                else:
                    # If no tray id tracked in place_to_trays but tray_counts shows some, attempt to pick any tray id
                    # (we fall back to generic handling by inventing a tray name; safe for heuristic accounting)
                    # Find any tray in tray_at whose place is kitchen
                    for tr, pl in tray_at.items():
                        if pl == "kitchen":
                            pick_tr = tr
                            break
                if pick_tr is not None:
                    # remove from kitchen list and update counts
                    tray_counts["kitchen"] = tray_counts.get("kitchen", 1) - 1
                    if tray_counts["kitchen"] <= 0:
                        tray_counts.pop("kitchen", None)
                # After put_on_tray and potential move to child:
                final_place = child_place
                if pick_tr is not None:
                    tray_at[pick_tr] = final_place
                    place_to_trays.setdefault(final_place, []).append(pick_tr)
                    tray_counts[final_place] = tray_counts.get(final_place, 0) + 1
                # consume the sandwich s
                at_kitchen_sandwiches.discard(s)
                sandwich_ontray_map.pop(s, None)
                no_gluten_sandwiches.discard(s)
            elif best_option == "make_and_place":
                # consume ingredients and notexist slot
                success = has_and_consume_ingredients(best_data["gluten_free"])
                if success:
                    notexist_count -= 1
                    # After creation, treat this as an at_kitchen_sandwich and then do same steps as above
                    # create a synthetic sandwich id for bookkeeping (we don't need its actual name)
                    # but we must account for tray moves similarly
                    need_move = best_data.get("need_move_tray_to_kitchen", 0)
                    if need_move == 1:
                        donor_place = None
                        for place, trays in place_to_trays.items():
                            if place != "kitchen" and trays:
                                donor_place = place
                                break
                        if donor_place is None:
                            for place, trays in place_to_trays.items():
                                if trays:
                                    donor_place = place
                                    break
                        if donor_place is not None:
                            tr = place_to_trays[donor_place].pop()
                            tray_at[tr] = "kitchen"
                            tray_counts[donor_place] = tray_counts.get(donor_place, 1) - 1
                            if tray_counts[donor_place] <= 0:
                                tray_counts.pop(donor_place, None)
                            tray_counts["kitchen"] = tray_counts.get("kitchen", 0) + 1
                            place_to_trays.setdefault("kitchen", []).append(tr)
                    # pick a tray from kitchen to place the newly created sandwich
                    pick_tr = None
                    if place_to_trays.get("kitchen"):
                        pick_tr = place_to_trays["kitchen"].pop()
                        tray_counts["kitchen"] = tray_counts.get("kitchen", 1) - 1
                        if tray_counts["kitchen"] <= 0:
                            tray_counts.pop("kitchen", None)
                    else:
                        for tr, pl in tray_at.items():
                            if pl == "kitchen":
                                pick_tr = tr
                                break
                    if pick_tr is not None:
                        final_place = child_place
                        tray_at[pick_tr] = final_place
                        place_to_trays.setdefault(final_place, []).append(pick_tr)
                        tray_counts[final_place] = tray_counts.get(final_place, 0) + 1
                    # we consumed a notexist slot and the ingredients; new sandwich is then served
                else:
                    # If ingredients could not actually be consumed (unexpected), treat as infeasible
                    total_cost += self.INFEASIBLE_PENALTY
                    continue
            else:
                # Should not happen, but safe guard
                total_cost += self.INFEASIBLE_PENALTY
                continue

        # Ensure heuristic is at least 0
        if total_cost < 0:
            total_cost = 0

        return total_cost