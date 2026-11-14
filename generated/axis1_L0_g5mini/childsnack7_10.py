from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split fact into parts."""
    return fact[1:-1].split()


def match(fact, *args):
    """Check whether a fact matches a pattern with possible '*' wildcards."""
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A fast, greedy domain-dependent heuristic for the childsnacks domain.
    The heuristic estimates the number of actions required to serve all
    children that are not yet served. It reasons about currently available
    sandwiches (on trays or in the kitchen), breads and contents at the
    kitchen, gluten-free resources, and trays at locations. The estimate is
    computed by greedily assigning available resources to unserved children
    using costed action sequences that reflect the domain operators:
      - make_sandwich / make_sandwich_no_gluten
      - put_on_tray
      - move_tray
      - serve_sandwich / serve_sandwich_no_gluten

    Assumptions
    -----------
    - Each child requires exactly one serve action to be satisfied.
    - A gluten-allergic child requires a no-gluten sandwich (made with
      no_gluten_bread and no_gluten_content; also marked as no_gluten_sandwich).
    - Non-allergic children accept any sandwich (including gluten-free ones).
    - One tray can carry one sandwich at a time (ontray).
    - move_tray between any two places costs 1 action.
    - put_on_tray, make_sandwich, and serve_sandwich cost 1 each.
    - If trays are nowhere at the kitchen when a sandwich must be put on tray,
      the heuristic accounts for a 1-action move to bring any available tray
      to the kitchen first (if any trays exist elsewhere).
    - The heuristic is greedy (not admissible) but aims to minimize node
      expansions by preferring low-cost uses of already prepared sandwiches
      and by conserving gluten-free resources for allergic children.

    Heuristic Initialization
    ------------------------
    The constructor extracts static facts from the task:
    - The set of children that are allergic to gluten (allergic_gluten).
    - The set of children explicitly marked not_allergic_gluten.
    - The waiting location for each child (waiting child place) when available.
    - The set of breads and contents that are marked no_gluten (when present
      in static). These are used to determine whether a gluten-free sandwich
      can be produced.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state the heuristic performs the following steps:

    1. Determine which children still need to be served by comparing the goal
       "served" facts against the facts present in the current state. If
       zero children remain, return 0.

    2. Parse the state to collect current resources:
       - trays_at_place: counts of trays at each place (including 'kitchen').
       - ontray_sandwiches_by_place: sandwiches currently on trays, grouped by
         tray location, with their gluten-free flags.
       - at_kitchen_sandwiches: sandwiches prepared in the kitchen (not on a tray),
         with GF flags.
       - at_kitchen_breads / at_kitchen_contents: counts of bread and content
         portions available at the kitchen (and their GF flags).
       - no_gluten_sandwich: sandwiches already labeled GF.
       - notexist_sandwich_count: how many sandwich names are still available to
         be created (facts of the form (notexist sandwX)).

    3. Use a greedy assignment procedure that, for every unserved child,
       selects the cheapest feasible way to obtain and deliver an appropriate
       sandwich. The chooser prefers:
         a) an ontray sandwich already at the child's place (cost 1: serve)
         b) an ontray sandwich elsewhere (cost 2: move_tray + serve)
         c) an at-kitchen sandwich (cost 3: put_on_tray + move_tray + serve;
            +1 extra if a tray must first be moved into the kitchen)
         d) make a sandwich and then put/move/serve (cost 4, or 5 if a tray must
            first be moved into the kitchen)
       For non-allergic children non-GF sandwiches are preferred to conserve
       GF resources for allergic children. When a chosen option is used the
       corresponding resource counts are decremented so subsequent assignments
       respect resource consumption.

    4. If an allergic child requires a GF sandwich but the heuristic finds no
       available GF resources (prepared GF sandwich, GF bread+content pair,
       and an available notexist sandwich name), the heuristic returns a large
       finite penalty value to reflect difficulty (but remains finite).

    5. The total heuristic value is the sum of the action costs chosen for all
       unserved children.

    Notes
    -----
    - The heuristic is deliberately greedy (not exact) to maintain efficiency
      and to reduce node expansions in greedy best-first search.
    - The heuristic always returns 0 on goal states, and returns finite values
      for states judged solvable by the greedy resource accounting (or a large
      finite penalty otherwise).
    """

    def __init__(self, task):
        # store goal facts
        self.goals = task.goals
        static_facts = set(task.static)

        # Extract allergic / non-allergic information from static facts
        self.allergic = {
            get_parts(fact)[1]
            for fact in static_facts
            if match(fact, "allergic_gluten", "*")
        }
        self.not_allergic = {
            get_parts(fact)[1]
            for fact in static_facts
            if match(fact, "not_allergic_gluten", "*")
        }

        # Extract waiting place for children if available in static
        self.waiting_place = {}
        for fact in static_facts:
            if match(fact, "waiting", "*", "*"):
                _, child, place = get_parts(fact)
                self.waiting_place[child] = place

        # Extract which breads and contents are marked as no_gluten (static)
        self.static_no_gluten_breads = {
            get_parts(fact)[1]
            for fact in static_facts
            if match(fact, "no_gluten_bread", "*")
        }
        self.static_no_gluten_contents = {
            get_parts(fact)[1]
            for fact in static_facts
            if match(fact, "no_gluten_content", "*")
        }

    def __call__(self, node):
        state = set(node.state)

        # Quick goal check: if all goals already satisfied, return 0
        if self.goals <= state:
            return 0

        # Helper to parse facts
        def parts(f):
            return get_parts(f)

        # Determine which children still need to be served
        unserved_children = []
        for goal in self.goals:
            if match(goal, "served", "*"):
                child = parts(goal)[1]
                served_fact = f"(served {child})"
                if served_fact not in state:
                    unserved_children.append(child)

        # Build resource inventories from the current state.

        # Trays at places
        trays_at_place = {}
        total_trays = 0
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, place = parts(fact)
                # identify trays by the fact that their name probably starts with 'tray'
                # but we cannot rely on naming; instead infer from domain constants by
                # checking that the object is used as a tray in ontray facts or in task facts.
                # For simplicity, treat any object referenced by (at <obj> <place>) that also appears
                # in an ontray fact as a tray. We will also treat any object that appears in ontray as tray.
                # However, to be robust, count all objects that appear in ontray as trays and their (at ...) facts.
                # Build trays set via ontray facts first.
                pass

        # First find all trays by scanning ontray facts and at facts together
        trays_set = set()
        for fact in state:
            if match(fact, "ontray", "*", "*"):
                _, sand, tray = parts(fact)
                trays_set.add(tray)
        # Also include objects that appear in 'at' facts but which are named like 'tray' by pattern
        # (fallback)
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, place = parts(fact)
                if obj.startswith("tray"):
                    trays_set.add(obj)

        # Now compute trays_at_place
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, place = parts(fact)
                if obj in trays_set:
                    trays_at_place[place] = trays_at_place.get(place, 0) + 1
                    total_trays += 1

        # Sandwiches on trays, grouped by place
        ontray_by_place = {}
        sandwich_ontray_origin = {}  # sandwich -> place
        for fact in state:
            if match(fact, "ontray", "*", "*"):
                _, sandwich, tray = parts(fact)
                # find where tray is
                tray_place = None
                for f in state:
                    if match(f, "at", tray, "*"):
                        tray_place = parts(f)[2]
                        break
                if tray_place is None:
                    # Shouldn't happen in well-formed states; assume kitchen fallback
                    tray_place = "kitchen"
                sandwich_is_gf = f"(no_gluten_sandwich {sandwich})" in state
                ontray_by_place.setdefault(tray_place, []).append((sandwich, sandwich_is_gf))
                sandwich_ontray_origin[sandwich] = tray_place

        # Sandwiches in kitchen (ready but not on tray)
        at_kitchen_sandwiches = []
        for fact in state:
            if match(fact, "at_kitchen_sandwich", "*"):
                sandwich = parts(fact)[1]
                sandwich_is_gf = f"(no_gluten_sandwich {sandwich})" in state
                at_kitchen_sandwiches.append((sandwich, sandwich_is_gf))

        # Breads and contents at kitchen (and GF flags)
        breads_at_kitchen = []
        contents_at_kitchen = []
        for fact in state:
            if match(fact, "at_kitchen_bread", "*"):
                bread = parts(fact)[1]
                is_gf = (bread in self.static_no_gluten_breads) or (f"(no_gluten_bread {bread})" in state)
                breads_at_kitchen.append((bread, is_gf))
            if match(fact, "at_kitchen_content", "*"):
                content = parts(fact)[1]
                is_gf = (content in self.static_no_gluten_contents) or (f"(no_gluten_content {content})" in state)
                contents_at_kitchen.append((content, is_gf))

        # Which sandwiches are labeled GF
        gf_sandwiches = {parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}

        # Count notexist sandwich objects (possible new sandwich names we can use)
        notexist_sandwiches = [parts(fact)[1] for fact in state if match(fact, "notexist", "*")]
        notexist_count = len(notexist_sandwiches)

        # Helper counters for bread/content counts
        total_breads = len(breads_at_kitchen)
        total_contents = len(contents_at_kitchen)
        gf_breads = sum(1 for _, gf in breads_at_kitchen if gf)
        gf_contents = sum(1 for _, gf in contents_at_kitchen if gf)

        # Helper: get child's waiting place (try static info, else guess 'table1' if unknown)
        def child_place(child):
            # check state as well for waiting facts
            for fact in state:
                if match(fact, "waiting", child, "*"):
                    return parts(fact)[2]
            # else use static if available
            if child in self.waiting_place:
                return self.waiting_place[child]
            # fallback: choose some non-kitchen place if possible, else 'kitchen'
            # gather any place objects from trays or ontray facts
            places = set()
            for fact in state:
                if match(fact, "at", "*", "*"):
                    _, place = parts(fact)
                    places.add(place)
                if match(fact, "waiting", "*", "*"):
                    _, _, p = parts(fact)
                    places.add(p)
            # remove kitchen if others exist
            if places:
                if "kitchen" in places and len(places) > 1:
                    places.discard("kitchen")
                # pick an arbitrary place
                return next(iter(places))
            return "kitchen"

        # Prepare mutable structures for greedy consumption
        trays = dict(trays_at_place)  # place -> count
        ontray = {p: list(lst) for p, lst in ontray_by_place.items()}
        at_kitchen = list(at_kitchen_sandwiches)
        breads = list(breads_at_kitchen)  # list of (bread, is_gf)
        contents = list(contents_at_kitchen)  # list of (content, is_gf)
        notexist = notexist_sandwiches[:]  # names to create
        total_trays_available = total_trays

        LARGE_PENALTY = 1000  # finite but large for impossible resource situations

        total_cost = 0

        # Utility to move any tray from some place 'src' to 'dst'
        def move_tray_from(src, dst):
            nonlocal total_cost, trays
            # precondition: trays[src] >= 1
            trays[src] -= 1
            if trays[src] == 0:
                del trays[src]
            trays[dst] = trays.get(dst, 0) + 1
            total_cost += 1  # cost of move_tray

        # Serve one child greedily
        for child in unserved_children:
            required_gf = False
            # determine if child is allergic (check static and state)
            if child in self.allergic:
                required_gf = True
            else:
                # also check dynamic state facts
                if f"(allergic_gluten {child})" in state:
                    required_gf = True
            child_loc = child_place(child)

            # Attempt options in the preferred order and take the cheapest available.

            served = False

            # 1) ontray at child's place (prefer non-gf for non-allergic)
            place_list = ontray.get(child_loc, [])
            if place_list:
                # choose suitable sandwich
                chosen_index = None
                # for non-allergic, prefer non-gf to preserve gf
                if not required_gf:
                    for i, (_, is_gf) in enumerate(place_list):
                        if not is_gf:
                            chosen_index = i
                            break
                # otherwise pick any gf if required or any if none preferred
                if chosen_index is None:
                    for i, (_, is_gf) in enumerate(place_list):
                        if (required_gf and is_gf) or (not required_gf):
                            chosen_index = i
                            break
                if chosen_index is not None:
                    # serve it
                    sandwich_name, is_gf = place_list.pop(chosen_index)
                    # remove sandwich from origin -> now tray remains at child's place
                    # cost: serve action
                    total_cost += 1
                    served = True
                    # no change in trays location counts (tray remains at child_loc)
                    # If ontray list empty, remove key
                    if not place_list:
                        ontray.pop(child_loc, None)
                    continue

            # 2) ontray somewhere else (move + serve). Prefer nearest: we treat any as cost 2.
            found = False
            origin_place = None
            origin_index = None
            origin_is_gf = None
            origin_sandwich_name = None
            # Prefer non-gf for non-allergic
            if not required_gf:
                for p, lst in ontray.items():
                    if p == child_loc:
                        continue
                    for i, (_, is_gf) in enumerate(lst):
                        if not is_gf:
                            origin_place = p
                            origin_index = i
                            origin_is_gf = is_gf
                            origin_sandwich_name = lst[i][0]
                            found = True
                            break
                    if found:
                        break
            if not found:
                for p, lst in ontray.items():
                    if p == child_loc:
                        continue
                    for i, (_, is_gf) in enumerate(lst):
                        if (required_gf and is_gf) or (not required_gf):
                            origin_place = p
                            origin_index = i
                            origin_is_gf = is_gf
                            origin_sandwich_name = lst[i][0]
                            found = True
                            break
                    if found:
                        break
            if found:
                # move tray from origin_place to child_loc (cost 1) and serve (cost 1)
                # update trays counts
                trays[origin_place] = trays.get(origin_place, 0)
                # ensure correct tray accounting: there must be at least one tray at origin_place
                if trays.get(origin_place, 0) <= 0:
                    # this is a fallback to keep counts consistent (should not happen)
                    trays[origin_place] = 1
                move_tray_from(origin_place, child_loc)
                # remove the sandwich from ontray origin list
                ontray[origin_place].pop(origin_index)
                if not ontray[origin_place]:
                    ontray.pop(origin_place, None)
                # cost of serve
                total_cost += 1
                served = True
                continue

            # 3) at_kitchen_sandwich (put_on_tray + move + serve). Prefer non-gf for non-allergic.
            chosen_index = None
            if at_kitchen:
                if not required_gf:
                    for i, (_, is_gf) in enumerate(at_kitchen):
                        if not is_gf:
                            chosen_index = i
                            break
                if chosen_index is None:
                    for i, (_, is_gf) in enumerate(at_kitchen):
                        if (required_gf and is_gf) or (not required_gf):
                            chosen_index = i
                            break
            if chosen_index is not None:
                # ensure there is a tray at kitchen, otherwise move one tray to kitchen if possible
                if trays.get("kitchen", 0) >= 1:
                    # we can put_on_tray (1), then move tray to child (1), then serve (1)
                    # put_on_tray does not change tray count at kitchen until moved
                    total_cost += 1  # put_on_tray
                    # move tray from kitchen to child_loc
                    move_tray_from("kitchen", child_loc)
                    total_cost += 1  # serve
                else:
                    # need to bring one tray from elsewhere to kitchen first (if available)
                    # find any other place with a tray
                    donor = None
                    for p, cnt in trays.items():
                        if p != "kitchen" and cnt > 0:
                            donor = p
                            break
                    if donor is not None:
                        # move donor->kitchen (1)
                        move_tray_from(donor, "kitchen")
                        # now put, move, serve
                        total_cost += 1  # put_on_tray
                        move_tray_from("kitchen", child_loc)
                        total_cost += 1  # serve
                    else:
                        # no trays at all -- extremely unlikely; penalize heavily
                        total_cost += LARGE_PENALTY
                # consume the at_kitchen sandwich
                at_kitchen.pop(chosen_index)
                served = True
                continue

            # 4) Make a sandwich (possibly GF) then put_on_tray + move + serve
            # Check resource availability
            can_make_gf = (gf_breads >= 1 and gf_contents >= 1 and len(notexist) >= 1)
            can_make_any = (total_breads >= 1 and total_contents >= 1 and len(notexist) >= 1)
            if required_gf:
                if not can_make_gf:
                    # cannot make a GF sandwich for an allergic child -> penalize
                    total_cost += LARGE_PENALTY
                    served = True  # treat as handled (but costly)
                    # do not change resource counters (there were none)
                    continue
                # make GF sandwich: choose and consume gf bread and gf content preferentially
                # remove one gf bread and content from lists
                # find and remove one gf bread
                removed_bread_idx = None
                for i, (_, gf) in enumerate(breads):
                    if gf:
                        removed_bread_idx = i
                        break
                if removed_bread_idx is not None:
                    breads.pop(removed_bread_idx)
                    gf_breads -= 1
                    total_breads -= 1
                # remove one gf content
                removed_content_idx = None
                for i, (_, gf) in enumerate(contents):
                    if gf:
                        removed_content_idx = i
                        break
                if removed_content_idx is not None:
                    contents.pop(removed_content_idx)
                    gf_contents -= 1
                    total_contents -= 1
                # consume one notexist sandwich name
                notexist.pop()
                # cost of make + put + move + serve (plus potential tray move to kitchen first)
                # make_sandwich_no_gluten (1)
                total_cost += 1
                # then similar to at_kitchen: need tray at kitchen
                if trays.get("kitchen", 0) >= 1:
                    total_cost += 1  # put_on_tray
                    move_tray_from("kitchen", child_loc)
                    total_cost += 1  # serve
                else:
                    donor = None
                    for p, cnt in trays.items():
                        if p != "kitchen" and cnt > 0:
                            donor = p
                            break
                    if donor is not None:
                        move_tray_from(donor, "kitchen")
                        total_cost += 1  # put_on_tray
                        move_tray_from("kitchen", child_loc)
                        total_cost += 1  # serve
                    else:
                        # no tray available -> penalize
                        total_cost += LARGE_PENALTY
                served = True
                continue
            else:
                # non-allergic: prefer to make a non-GF sandwich if possible (to conserve GF)
                # try to find a non-GF bread & content to use
                non_gf_bread_idx = None
                for i, (_, gf) in enumerate(breads):
                    if not gf:
                        non_gf_bread_idx = i
                        break
                non_gf_content_idx = None
                for i, (_, gf) in enumerate(contents):
                    if not gf:
                        non_gf_content_idx = i
                        break
                if non_gf_bread_idx is not None and non_gf_content_idx is not None and len(notexist) >= 1:
                    # consume these
                    breads.pop(non_gf_bread_idx)
                    total_breads -= 1
                    contents.pop(non_gf_content_idx)
                    total_contents -= 1
                    notexist.pop()
                    # cost of make + put + move + serve
                    total_cost += 1  # make
                    if trays.get("kitchen", 0) >= 1:
                        total_cost += 1  # put_on_tray
                        move_tray_from("kitchen", child_loc)
                        total_cost += 1  # serve
                    else:
                        donor = None
                        for p, cnt in trays.items():
                            if p != "kitchen" and cnt > 0:
                                donor = p
                                break
                        if donor is not None:
                            move_tray_from(donor, "kitchen")
                            total_cost += 1  # put_on_tray
                            move_tray_from("kitchen", child_loc)
                            total_cost += 1  # serve
                        else:
                            total_cost += LARGE_PENALTY
                    served = True
                    continue
                # otherwise, if cannot make a non-GF, but can make any (including using GF resources), fall back
                if can_make_any:
                    # consume any bread and content (prefer non-gf bread/content if present was already attempted)
                    if breads:
                        breads.pop(0)
                        total_breads -= 1
                        # reduce gf counts if it was gf
                        # we don't need to detect here precisely because we used lists of tuples; just check gf flags
                    if contents:
                        contents.pop(0)
                        total_contents -= 1
                    if notexist:
                        notexist.pop()
                    total_cost += 1  # make
                    if trays.get("kitchen", 0) >= 1:
                        total_cost += 1  # put_on_tray
                        move_tray_from("kitchen", child_loc)
                        total_cost += 1  # serve
                    else:
                        donor = None
                        for p, cnt in trays.items():
                            if p != "kitchen" and cnt > 0:
                                donor = p
                                break
                        if donor is not None:
                            move_tray_from(donor, "kitchen")
                            total_cost += 1  # put_on_tray
                            move_tray_from("kitchen", child_loc)
                            total_cost += 1  # serve
                        else:
                            total_cost += LARGE_PENALTY
                    served = True
                    continue

            # If we reach here and still not served, impose a penalty (should not generally happen)
            if not served:
                total_cost += LARGE_PENALTY

        return total_cost