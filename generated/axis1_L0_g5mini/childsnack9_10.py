from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split a fact into its parts."""
    return fact[1:-1].split()


def match(fact, *pattern):
    """Return True if the fact matches the provided pattern with wildcards."""
    parts = get_parts(fact)
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the `childsnack` domain that estimates the
    number of actions required to serve all children listed in the goal.
    The heuristic greedily simulates serving children, taking into account
    available pre-made sandwiches (at kitchen or on trays), empty trays, trays'
    locations, and available kitchen resources (bread and content portions),
    including gluten-free resources for allergic children. The heuristic is
    not required to be admissible; it prioritizes realistic (and cheaply
    computable) estimates that help guide a greedy best-first search.

    Assumptions
    -----------
    - A single move_tray can move any tray between any two places in one action.
    - Every sandwich must be unique (must have a corresponding `notexist`
      fact before it is made). Making a sandwich consumes one bread and one
      content portion (and one `notexist` item).
    - put_on_tray requires the sandwich to be in the kitchen and the tray to
      be at the kitchen.
    - After serving, the tray becomes empty at the place where the serve
      occurred (the sandwich is removed).
    - Static facts (like `allergic_gluten`, `not_allergic_gluten`, and
      `waiting`) may reside in task.static or in the current state; both are
      consulted.
    - If the required gluten-free raw ingredients (bread/content) are not
      available and there is no pre-made matching sandwich, the problem may
      be unsolvable; in that case the heuristic returns a large finite value.

    Heuristic Initialization
    ------------------------
    - The constructor stores:
      * self.goals: the set of goal facts (used to extract which children must be served).
      * self.static: the static facts for the task (task.static) to consult for
        allergy and waiting/place information that might not be present in dynamic state.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    The heuristic computes an estimate by a greedy simulation that repeatedly
    picks the next child that can be served with the minimal marginal number
    of actions, updates an internal resource model, and accumulates the actions.
    The simulation uses the following resource model derived from the current
    state (plus static facts): which sandwiches exist (at kitchen or on trays),
    which sandwiches are gluten-free, how many breads/contents (and gluten-free
    ones) are in the kitchen, which trays exist and where they are located, and
    which trays are empty.

    For each not-yet-served child we consider these alternative ways to serve
    them and compute the minimal action cost among them:

    1) Serve using an already ontray sandwich that matches allergy:
       - If the ontray is at the child's waiting place: cost = 1 (serve).
       - Otherwise: cost = 1 (move_tray to child's place) + 1 (serve) = 2.

    2) Serve using a sandwich that is at the kitchen (pre-made):
       - Needs an empty tray at the kitchen to put the sandwich upon. If no empty
         tray is at the kitchen, a tray must be moved there (1 action).
       - Actions: possibly move a tray to kitchen (0 or 1), put_on_tray (1),
         possibly move tray to child's place (0 or 1), serve (1).
       - Typical cost = 3 (put + move + serve) if tray already at kitchen; +1
         if a tray must be fetched from elsewhere.

    3) Make a sandwich in the kitchen and then serve (when no suitable
       pre-made sandwich is available):
       - Requires at least one `notexist` sandwich id, one bread in kitchen,
         and one content in kitchen. For allergic children, both bread and
         content used must be gluten-free.
       - Actions: make_sandwich or make_sandwich_no_gluten (1),
         then follow the "at-kitchen sandwich" path above (put_on_tray, move,
         serve). Costs are therefore (1 + cost of option 2).

    The algorithm picks the child with smallest cost, applies the associated
    resource consumption to the internal model (consume bread/content or a
    prepared sandwich, update tray contents and locations), and repeats until
    all goal children are accounted for or until an impossibility is detected.
    If a child must be served gluten-free but no gluten-free sandwiche ingredient
    or pre-made sandwich exists, the heuristic returns a large finite value
    (representing that the state is effectively far from a solution).

    The heuristic returns 0 iff the current state already satisfies all goals.
    """

    def __init__(self, task):
        # Goal facts (e.g., "(served child1)")
        self.goals = task.goals
        # Static facts (may include waiting locations and allergy info)
        self.static = task.static

    def __call__(self, node):
        # Combine dynamic state with static facts for easier checks
        state = set(node.state)
        static = set(self.static)
        combined = state | static

        # Quick goal check: if all goals are in the dynamic state -> h = 0
        if self.goals <= state:
            return 0

        # Helper to get all facts matching a pattern from combined facts
        def all_matching(*pattern):
            return [fact for fact in combined if match(fact, *pattern)]

        # Extract goal children (served X) -> child ids
        goal_children = []
        for g in self.goals:
            parts = get_parts(g)
            if parts[0] == "served":
                goal_children.append(parts[1])

        # Current served children
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        # Children still to serve
        remaining_children = [c for c in goal_children if c not in served_children]

        # Extract waiting places and allergy info (prefer dynamic state, else static)
        waiting_place = {}
        for fact in combined:
            if match(fact, "waiting", "*", "*"):
                _, child, place = get_parts(fact)
                waiting_place[child] = place

        allergic = {}
        for fact in combined:
            if match(fact, "allergic_gluten", "*"):
                _, child = get_parts(fact)
                allergic[child] = True
            if match(fact, "not_allergic_gluten", "*"):
                _, child = get_parts(fact)
                # only set False if not previously marked allergic
                if allergic.get(child, None) is None:
                    allergic[child] = False

        # Sets of breads and contents currently at kitchen
        breads_at_k = {get_parts(fact)[1] for fact in combined if match(fact, "at_kitchen_bread", "*")}
        contents_at_k = {get_parts(fact)[1] for fact in combined if match(fact, "at_kitchen_content", "*")}

        # Gluten-free breads/contents (may appear in static or state)
        gluten_breads = {get_parts(fact)[1] for fact in combined if match(fact, "no_gluten_bread", "*")}
        gluten_contents = {get_parts(fact)[1] for fact in combined if match(fact, "no_gluten_content", "*")}

        gluten_breads_at_k = breads_at_k & gluten_breads
        gluten_contents_at_k = contents_at_k & gluten_contents

        # Sandwich facts
        at_kitchen_sandwichs = {get_parts(f)[1] for f in combined if match(f, "at_kitchen_sandwich", "*")}
        no_gluten_sandwichs = {get_parts(f)[1] for f in combined if match(f, "no_gluten_sandwich", "*")}
        notexist_sandwichs = {get_parts(f)[1] for f in combined if match(f, "notexist", "*")}

        # Trays and their locations
        trays = {}
        for fact in combined:
            if match(fact, "at", "*", "*"):
                _, t, place = get_parts(fact)
                # We only care about trays -> prefix 'tray' or domain typing isn't visible here,
                # so we treat any object that appears in an (at ?t ?place) and is also used in ontray as tray.
                trays[t] = place

        # Map trays that currently have a sandwich on them
        tray_with_sand = {}
        sand_on_tray = {}
        for fact in combined:
            if match(fact, "ontray", "*", "*"):
                _, s, t = get_parts(fact)
                tray_with_sand[t] = s
                sand_on_tray[s] = t

        # Build a list of empty trays and occupied trays with current places
        empty_trays = {t for t, place in trays.items() if t not in tray_with_sand}
        trays_at_place = {}
        for t, place in trays.items():
            trays_at_place.setdefault(place, set()).add(t)

        # Pre-made sandwiches on trays mapped to their place
        ontray_sandwich_places = {}
        for s, t in sand_on_tray.items():
            place = trays.get(t, None)
            if place is not None:
                ontray_sandwich_places[s] = place

        # Available counts / sets used by greedy simulation (mutable copies)
        breads_k = set(breads_at_k)
        contents_k = set(contents_at_k)
        gluten_breads_k = set(gluten_breads_at_k)
        gluten_contents_k = set(gluten_contents_at_k)
        at_k_sand = set(at_kitchen_sandwichs)
        no_gluten_sand = set(no_gluten_sandwichs)
        notexist_sand = set(notexist_sandwichs)

        # Tray model: mapping tray -> place and occupancy
        tray_loc = dict(trays)  # tray -> place
        tray_occupied = {t: (t in tray_with_sand) for t in tray_loc}
        # Map sandwich -> tray occupancy if applicable
        sand_location = {}  # sandwich -> place (kitchen or place from tray)
        for s in at_k_sand:
            sand_location[s] = "kitchen"
        for s, place in ontray_sandwich_places.items():
            sand_location[s] = place

        total_cost = 0
        LARGE = 9999

        # Helper to compute minimal cost to serve a single child under current simulation state
        def cost_to_serve(child):
            """Return (cost, plan_kind, details) where plan_kind indicates what resource used."""
            # plan_kind: "ontray_at_place", "ontray_move", "at_kitchen_with_empty_tray",
            # "at_kitchen_fetch_tray", "make_and_then_at_kitchen..." or ("impossible", reason)
            child_place = waiting_place.get(child, None)
            child_allergic = allergic.get(child, False)

            # Check ontray sandwiches that match allergy (and are not yet consumed)
            best = (LARGE, None, None)

            # Candidate sandwiches (pre-made) that satisfy allergy constraint
            def sandwich_matches(sand_id):
                if child_allergic:
                    return sand_id in no_gluten_sand
                else:
                    # non-allergic can accept any sandwich (gluten-free or not)
                    return True

            # 1) Use any ontray sandwich that matches allergy
            for s, place in list(sand_location.items()):
                # only consider sandwiches currently on trays (sand_location == some place not 'kitchen')
                # sand_location 'kitchen' means at_kitchen_sand; those are considered later.
                if place is None:
                    continue
                if s in at_k_sand:
                    continue
                if not sandwich_matches(s):
                    continue
                if place == child_place:
                    # serve directly
                    return (1, "ontray_at_place", {"sand": s, "tray": sand_on_tray.get(s)})
                else:
                    # move tray then serve
                    return (2, "ontray_move", {"sand": s, "from": place, "tray": sand_on_tray.get(s)})

            # 2) Use sandwich that is at the kitchen already
            # Need an empty tray at kitchen (or a tray brought there)
            if at_k_sand:
                # Look for any at_k kitchen sandwich that matches allergy
                for s in at_k_sand:
                    if not sandwich_matches(s):
                        continue
                    # check empty tray at kitchen
                    kitchen_place = "kitchen"
                    empty_trays_at_k = [t for t, p in tray_loc.items() if (not tray_occupied.get(t, False)) and p == kitchen_place]
                    if empty_trays_at_k:
                        # put_on_tray (1), maybe move_tray to child (1 if child place != kitchen), serve (1)
                        move_needed = 0 if child_place == kitchen_place else 1
                        cost = 1 + move_needed + 1
                        return (cost, "at_kitchen_with_empty_tray", {"sand": s, "tray": empty_trays_at_k[0]})
                    else:
                        # need to fetch a tray from elsewhere (if any tray exists)
                        if tray_loc:
                            # pick any tray not at kitchen (or even at kitchen but occupied - but occupied ones can't be used without serving first)
                            trays_not_available = [t for t in tray_loc if tray_loc[t] != kitchen_place]
                            if trays_not_available:
                                # fetch a tray (1), put_on_tray (1), move to child maybe (1), serve (1)
                                move_needed = 0 if child_place == kitchen_place else 1
                                cost = 1 + 1 + move_needed + 1
                                return (cost, "at_kitchen_fetch_tray", {"sand": s, "tray": trays_not_available[0]})
                            else:
                                # all trays are at kitchen but occupied; cannot put unless we free one by moving a tray (but moving occupied tray moves the sandwich as well - not useful)
                                # We do not simulate complex freeing; count a slightly higher cost by assuming we can free/bring an empty tray with +1
                                move_needed = 0 if child_place == kitchen_place else 1
                                cost = 1 + 1 + move_needed + 1 + 1  # +1 extra for freeing a tray
                                return (cost, "at_kitchen_assume_free_tray", {"sand": s})
                        else:
                            # no trays at all (very unlikely); treat as impossible
                            return (LARGE, "impossible", "no_trays")

            # 3) Make a sandwich then treat as at_kitchen_sandwich
            # Check availability of raw ingredients and a notexist sandwich id
            # For allergic child, need both gluten bread and gluten content
            if notexist_sand:
                if child_allergic:
                    if gluten_breads_k and gluten_contents_k:
                        # can make gluten-free sandwich
                        # cost: make (1) + put_on_tray (1) + maybe move (1) + serve (1)
                        # plus possibly fetching a tray if none empty at kitchen (+1)
                        kitchen_place = "kitchen"
                        empty_trays_at_k = [t for t, p in tray_loc.items() if (not tray_occupied.get(t, False)) and p == kitchen_place]
                        move_fetch = 0
                        if empty_trays_at_k:
                            move_fetch = 0
                        elif tray_loc:
                            move_fetch = 1
                        else:
                            return (LARGE, "impossible", "no_trays")

                        move_needed = 0 if child_place == kitchen_place else 1
                        cost = 1 + move_fetch + 1 + move_needed + 1
                        return (cost, "make_gluten", {"sand_new": next(iter(notexist_sand)), "use_gbread": next(iter(gluten_breads_k)), "use_gcontent": next(iter(gluten_contents_k))})
                    else:
                        # cannot make gluten-free sandwich from ingredients
                        pass
                else:
                    # non-allergic: any bread & content suffice
                    if breads_k and contents_k:
                        kitchen_place = "kitchen"
                        empty_trays_at_k = [t for t, p in tray_loc.items() if (not tray_occupied.get(t, False)) and p == kitchen_place]
                        move_fetch = 0
                        if empty_trays_at_k:
                            move_fetch = 0
                        elif tray_loc:
                            move_fetch = 1
                        else:
                            return (LARGE, "impossible", "no_trays")
                        move_needed = 0 if child_place == kitchen_place else 1
                        cost = 1 + move_fetch + 1 + move_needed + 1
                        return (cost, "make_regular", {"sand_new": next(iter(notexist_sand)), "use_bread": next(iter(breads_k)), "use_content": next(iter(contents_k))})
                    else:
                        # not enough raw ingredients to make a regular sandwich
                        pass

            # If no option found, impossible under current resources
            return (LARGE, "impossible", "no_resource")

        # Greedy simulation: repeatedly serve the child with minimal marginal cost and update resources
        # To prevent infinite loops, cap iterations by number of children
        while remaining_children:
            # compute cost for each remaining child
            costs = []
            for child in remaining_children:
                c = cost_to_serve(child)
                costs.append((c[0], child, c))
            # pick minimal cost
            costs.sort(key=lambda x: x[0])
            min_cost, chosen_child, chosen_info = costs[0]
            if min_cost >= LARGE:
                # unsolvable (or needs resources not present); return large finite value
                return LARGE
            # apply chosen plan and update our simulated resources
            total_cost += min_cost
            _, plan_kind, details = chosen_info

            # Apply effect of the chosen plan to our resource model
            if plan_kind == "ontray_at_place":
                sand = details["sand"]
                tray = sand_on_tray.get(sand)
                # serving removes sandwich from tray, tray becomes empty at child's place
                if tray is not None:
                    tray_occupied[tray] = False
                    place = tray_loc.get(tray, None)
                    # sandwich consumed
                    sand_location.pop(sand, None)
                    at_k_sand.discard(sand)
                    no_gluten_sand.discard(sand)
                    # the sandwich identity is removed; no need to add to notexist
                else:
                    # if sandwich was not mapped to tray for some reason, just remove sandwich
                    sand_location.pop(sand, None)
                    at_k_sand.discard(sand)
                    no_gluten_sand.discard(sand)

            elif plan_kind == "ontray_move":
                sand = details["sand"]
                tray = details.get("tray", None)
                # move tray to child's place then serve
                child_place = waiting_place.get(chosen_child, None)
                if tray is not None:
                    tray_loc[tray] = child_place
                    # then serve: becomes empty
                    tray_occupied[tray] = False
                # sandwich removed
                sand_location.pop(sand, None)
                at_k_sand.discard(sand)
                no_gluten_sand.discard(sand)

            elif plan_kind == "at_kitchen_with_empty_tray":
                sand = details["sand"]
                tray = details["tray"]
                # put_on_tray: sandwich moves from kitchen to tray at kitchen
                if tray is not None:
                    tray_occupied[tray] = True
                    sand_on_tray[sand] = tray
                    sand_location[sand] = tray_loc.get(tray, "kitchen")
                # then move to child (if needed) and serve: after serve tray is empty at child place
                child_place = waiting_place.get(chosen_child, None)
                if tray is not None:
                    tray_loc[tray] = child_place
                    tray_occupied[tray] = False
                sand_location.pop(sand, None)
                at_k_sand.discard(sand)
                no_gluten_sand.discard(sand)

            elif plan_kind == "at_kitchen_fetch_tray" or plan_kind == "at_kitchen_assume_free_tray":
                sand = details["sand"]
                tray = details.get("tray", None) or next(iter(tray_loc))  # pick some tray
                # bring tray to kitchen
                tray_loc[tray] = "kitchen"
                # put_on_tray
                tray_occupied[tray] = True
                sand_on_tray[sand] = tray
                sand_location[sand] = "kitchen"
                # move to child and serve
                child_place = waiting_place.get(chosen_child, None)
                tray_loc[tray] = child_place
                tray_occupied[tray] = False
                sand_location.pop(sand, None)
                at_k_sand.discard(sand)
                no_gluten_sand.discard(sand)

            elif plan_kind == "make_gluten":
                # consume one gluten bread and one gluten content, one notexist sandwich id
                sand_new = details["sand_new"]
                use_b = details["use_gbread"]
                use_c = details["use_gcontent"]
                # remove used resources
                if use_b in gluten_breads_k:
                    gluten_breads_k.discard(use_b)
                if use_b in breads_k:
                    breads_k.discard(use_b)
                if use_c in gluten_contents_k:
                    gluten_contents_k.discard(use_c)
                if use_c in contents_k:
                    contents_k.discard(use_c)
                notexist_sand.discard(sand_new)
                # created sandwich at kitchen then proceed as at_kitchen_with_empty_tray behavior:
                # attempt to find an empty tray at kitchen
                kitchen_place = "kitchen"
                empty_at_k = [t for t, p in tray_loc.items() if (not tray_occupied.get(t, False)) and p == kitchen_place]
                if empty_at_k:
                    tray = empty_at_k[0]
                    # put_on_tray
                    tray_occupied[tray] = True
                    sand_on_tray[sand_new] = tray
                    sand_location[sand_new] = "kitchen"
                    # move to child and serve
                    child_place = waiting_place.get(chosen_child, None)
                    tray_loc[tray] = child_place
                    tray_occupied[tray] = False
                    sand_location.pop(sand_new, None)
                else:
                    # fetch a tray
                    if tray_loc:
                        tray = next(iter(tray_loc))
                        tray_loc[tray] = kitchen_place
                        tray_occupied[tray] = True
                        sand_on_tray[sand_new] = tray
                        sand_location[sand_new] = "kitchen"
                        child_place = waiting_place.get(chosen_child, None)
                        tray_loc[tray] = child_place
                        tray_occupied[tray] = False
                        sand_location.pop(sand_new, None)
                    else:
                        # no tray -> impossible
                        return LARGE

            elif plan_kind == "make_regular":
                # consume one bread and one content, one notexist sandwich id
                sand_new = details["sand_new"]
                use_b = details["use_bread"]
                use_c = details["use_content"]
                if use_b in breads_k:
                    breads_k.discard(use_b)
                if use_c in contents_k:
                    contents_k.discard(use_c)
                notexist_sand.discard(sand_new)
                # put on a tray and serve similarly to make_gluten
                kitchen_place = "kitchen"
                empty_at_k = [t for t, p in tray_loc.items() if (not tray_occupied.get(t, False)) and p == kitchen_place]
                if empty_at_k:
                    tray = empty_at_k[0]
                    tray_occupied[tray] = True
                    sand_on_tray[sand_new] = tray
                    sand_location[sand_new] = "kitchen"
                    child_place = waiting_place.get(chosen_child, None)
                    tray_loc[tray] = child_place
                    tray_occupied[tray] = False
                    sand_location.pop(sand_new, None)
                else:
                    if tray_loc:
                        tray = next(iter(tray_loc))
                        tray_loc[tray] = kitchen_place
                        tray_occupied[tray] = True
                        sand_on_tray[sand_new] = tray
                        sand_location[sand_new] = "kitchen"
                        child_place = waiting_place.get(chosen_child, None)
                        tray_loc[tray] = child_place
                        tray_occupied[tray] = False
                        sand_location.pop(sand_new, None)
                    else:
                        return LARGE

            else:
                # Other plan kinds not explicitly handled; treat as impossible for safety
                return LARGE

            # After serving, mark child as served and remove from remaining list
            remaining_children.remove(chosen_child)

        # End simulation loop
        return total_cost