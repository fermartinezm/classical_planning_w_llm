from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the parts of a fact string like '(pred a b)' -> ['pred','a','b']"""
    return fact[1:-1].split()


def match(fact, *pattern):
    """
    Check whether a fact matches a pattern with optional wildcards.
    Example: match("(at tray1 kitchen)", "at", "*", "kitchen") -> True
    """
    parts = get_parts(fact)
    return all(fnmatch(p, pat) for p, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the childsnack domain.
    Estimates the remaining number of actions to serve all children by
    greedily accounting for available sandwiches (at kitchen / on trays),
    available kitchen resources (breads and contents), and trays locations.
    The heuristic is not admissible but aims to give a tight, quickly
    computable estimate to guide greedy best-first search.

    Assumptions
    -----------
    - Each child needs exactly one serving action (served predicate).
    - Serving requires a sandwich on a tray at the child's waiting place.
      Typical sequence to serve a child (if nothing already prepared):
        make_sandwich / make_sandwich_no_gluten (1)
        put_on_tray (1)
        move_tray (kitchen -> child place) (1)
        serve_sandwich / serve_sandwich_no_gluten (1)
      => up to 4 actions per child in the simple baseline case.
    - If a sandwich is already on a tray at the child's place (and matches
      the child's gluten requirement), only the serve action is needed.
    - If a sandwich is on a tray at the kitchen, moving the tray + serve
      suffices (2 actions).
    - Non-allergic children can accept any sandwich; allergic children
      require a no_gluten sandwich.
    - No attempt is made to optimally reorder tray movements across children;
      a greedy allocation is used (cheap to compute and effective in practice).
    - If essential resources for allergic children (no_gluten bread/content)
      are insufficient, the heuristic returns a large value (indicating
      likely unsolvable under current resources).

    Heuristic Initialization
    ------------------------
    - Extract goal facts from the task (expect goals of form '(served childN)').
    - Extract static facts:
      - waiting(child) -> place mapping
      - allergy information: allergic_gluten / not_allergic_gluten
    - The rest of the facts (tray locations, sandwiches present, kitchen
      resources) are read from the dynamic state during each heuristic call.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state:
    1. If all goal facts are already satisfied in the state, return 0.
    2. Parse the dynamic facts to count:
       - trays at kitchen and trays elsewhere (based on '(at ?t ?p)').
       - sandwiches at kitchen (at_kitchen_sandwich), distinguished by whether
         they are no_gluten_sandwich.
       - sandwiches on trays by place and gluten type (ontray + tray location).
       - available kitchen breads and contents (and which are marked no_gluten).
       - the number of remaining "notexist" sandwich names (to allow making new sandwiches).
    3. Build the list of children that still need serving, split into
       allergic and non-allergic (use static allergy info).
    4. Serve children in this priority:
       a) Children who already have an appropriate sandwich on a tray at
          their place: cost += 1 (serve). Use up that sandwich & tray.
       b) Children who have an appropriate sandwich on a tray at the kitchen:
          cost += 2 (move tray to child place + serve). Update tray counts.
       c) Children who have an appropriate sandwich at the kitchen (not on
          a tray): we need put_on_tray + move + serve. If no tray at kitchen
          we first move one there (cost +1). Typical cost: 3 (tray at kitchen)
          or 4 (tray must be moved to kitchen first).
       d) Otherwise, make a sandwich (make_sandwich or make_sandwich_no_gluten),
          consuming appropriate bread/content and one notexist name, and then
          follow step (c) to bring it to the child. This is 1 extra action
          for the make step (so 4 or 5 total depending on tray availability).
    5. Allergic children are processed first so that scarce no-gluten breads/
       contents are preferentially allocated to them.
    6. If at any point required resources (e.g., no_gluten bread/content for
       remaining allergic children, or notexist sandwich names) are missing,
       the heuristic returns a large number (indicating high estimated cost /
       likely impossibility).
    """

    def __init__(self, task):
        # Goal facts (set of strings)
        self.goals = task.goals
        # Static facts: waiting mapping and allergy information
        static_facts = task.static

        # waiting child -> place
        self.waiting_place = {}
        # allergy sets
        self.allergic = set()
        self.not_allergic = set()

        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "waiting":
                child, place = parts[1], parts[2]
                self.waiting_place[child] = place
            elif pred == "allergic_gluten":
                self.allergic.add(parts[1])
            elif pred == "not_allergic_gluten":
                self.not_allergic.add(parts[1])

    def __call__(self, node):
        state = node.state

        # If already goal, heuristic = 0
        if self.goals <= state:
            return 0

        # Parse dynamic facts in the state
        trays_at_place = {}  # place -> count of trays at that place
        trays_total = 0
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, place = get_parts(fact)[1], get_parts(fact)[2]
                # identify trays by checking 'tray' naming pattern is not strictly enforceable,
                # but we can detect presence of ontray facts to get tray names. To be safe, treat
                # any object that also appears in ontray facts as a tray; otherwise, count all 'at'
                # facts for unknown objects as potential trays if their name starts with 'tray'.
                if obj.startswith("tray"):
                    trays_at_place[place] = trays_at_place.get(place, 0) + 1
                    trays_total += 1

        trays_at_kitchen = trays_at_place.get("kitchen", 0)
        trays_elsewhere = trays_total - trays_at_kitchen

        # Sandwiches on trays: map (place, gluten_flag) -> count
        ontray_by_place_gluten = {}  # (place, is_no_gluten) -> count
        # Map sandwich -> whether it's no_gluten (present in state)
        sandwich_is_no_gluten = {}
        # Sandwich presence at kitchen (not on tray)
        at_kitchen_sandwiches = {"no_gluten": 0, "regular": 0}

        # available kitchen breads and contents (names)
        kitchen_breads = set()
        kitchen_contents = set()
        no_gluten_breads = set()
        no_gluten_contents = set()

        # notexist sandwich names count
        notexist_count = 0

        # served children set
        served_children = set()
        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "ontray":
                sandw, tray = parts[1], parts[2]
                # find tray location
                # locate "(at tray place)" in state
                tray_place = None
                for f in state:
                    if match(f, "at", tray, "*"):
                        tray_place = get_parts(f)[2]
                        break
                # determine gluten status for sandwich (check if there is a no_gluten_sandwich fact)
                is_no = ("(no_gluten_sandwich " + sandw + ")") in state
                key = (tray_place if tray_place is not None else "unknown", is_no)
                ontray_by_place_gluten[key] = ontray_by_place_gluten.get(key, 0) + 1
                sandwich_is_no_gluten[sandw] = is_no
            elif pred == "at_kitchen_sandwich":
                sandw = parts[1]
                is_no = ("(no_gluten_sandwich " + sandw + ")") in state
                if is_no:
                    at_kitchen_sandwiches["no_gluten"] += 1
                else:
                    at_kitchen_sandwiches["regular"] += 1
                sandwich_is_no_gluten[sandw] = is_no
            elif pred == "at_kitchen_bread":
                kitchen_breads.add(parts[1])
            elif pred == "at_kitchen_content":
                kitchen_contents.add(parts[1])
            elif pred == "no_gluten_bread":
                no_gluten_breads.add(parts[1])
            elif pred == "no_gluten_content":
                no_gluten_contents.add(parts[1])
            elif pred == "notexist":
                notexist_count += 1
            elif pred == "served":
                served_children.add(parts[1])

        # Counts of breads/contents available in kitchen that are no-gluten
        breads_no_gluten_in_kitchen = len(kitchen_breads & no_gluten_breads)
        contents_no_gluten_in_kitchen = len(kitchen_contents & no_gluten_contents)

        # Total breads and contents in kitchen for regular sandwiches
        total_breads_in_kitchen = len(kitchen_breads)
        total_contents_in_kitchen = len(kitchen_contents)

        # Determine children to serve
        # Use static info for list of children: union of allergy info and waiting mapping
        children = set(self.waiting_place.keys()) | self.allergic | self.not_allergic
        # If static doesn't list a child that appears in goals, include from goals too
        for g in self.goals:
            parts = get_parts(g)
            if parts[0] == "served":
                children.add(parts[1])

        # Build lists of remaining (not yet served) allergic and non-allergic children
        remaining_allergic = [c for c in children if c not in served_children and c in self.allergic]
        remaining_non_allergic = [c for c in children if c not in served_children and c in self.not_allergic]

        # Some children may be neither in allergic nor not_allergic in static (rare);
        # treat them as non-allergic (permissive).
        for c in children:
            if c not in served_children and c not in self.allergic and c not in self.not_allergic:
                remaining_non_allergic.append(c)

        total_cost = 0

        # Helper functions to allocate supplies and trays
        def use_ontray_at_place(place, need_no_gluten):
            """Try to use an ontray sandwich at exactly 'place' with gluten constraint.
               Return True if used (and decrement), False otherwise."""
            key = (place, need_no_gluten)
            if ontray_by_place_gluten.get(key, 0) > 0:
                ontray_by_place_gluten[key] -= 1
                # tray remains at place after serving; no change to trays_at_kitchen
                return True
            return False

        def use_ontray_at_kitchen(need_no_gluten):
            """Try to use an ontray sandwich at kitchen. If successful, move it to child's place
               (cost accounted at caller), reduce trays at kitchen by 1 and ontray count."""
            key = ("kitchen", need_no_gluten)
            if ontray_by_place_gluten.get(key, 0) > 0:
                ontray_by_place_gluten[key] -= 1
                nonlocal_trays = True
                # moving this tray reduces trays_at_kitchen by 1
                return True
            return False

        # Because closures with assignment are tricky, manipulate trays via mutable dict
        trays = {"at_kitchen": trays_at_kitchen, "elsewhere": trays_elsewhere}

        # Function to ensure a tray is available at kitchen; move one from elsewhere if needed.
        def ensure_tray_at_kitchen():
            """Return extra cost to bring a tray to kitchen (0 if already there, 1 if moved, -1 if none available)."""
            if trays["at_kitchen"] > 0:
                return 0
            if trays["elsewhere"] > 0:
                # move one tray to kitchen
                trays["elsewhere"] -= 1
                trays["at_kitchen"] += 1
                return 1
            # no trays anywhere (unlikely), indicate failure
            return -1

        # Process allergic children first (they require no_gluten sandwich)
        for child in remaining_allergic:
            place = self.waiting_place.get(child, None)
            need_no = True

            # 1) sandwich on tray at child's place?
            if place is not None and use_ontray_at_place(place, True):
                total_cost += 1  # serve
                # tray remains at place; no changes to trays counts
                continue

            # 2) sandwich on tray at kitchen?
            if ontray_by_place_gluten.get(("kitchen", True), 0) > 0:
                # Move tray kitchen->child place (1) and serve (1)
                ontray_by_place_gluten[("kitchen", True)] -= 1
                trays["at_kitchen"] -= 1  # tray moved out of kitchen
                total_cost += 2
                # after serving, tray is at child's place (not kitchen)
                continue

            # 3) sandwich at kitchen (not on tray)?
            if at_kitchen_sandwiches["no_gluten"] > 0:
                at_kitchen_sandwiches["no_gluten"] -= 1
                # need a tray at kitchen to put on
                extra = ensure_tray_at_kitchen()
                if extra == -1:
                    # no tray available anywhere -> we must move a tray from nowhere: treat as very costly
                    return 99999
                # put_on_tray(1) + move_tray(1) + serve(1) + extra (maybe 0 or 1)
                total_cost += 3 + extra
                # after this sequence, a tray is at child's place (so at_kitchen decreased by any moved tray)
                if extra == 0:
                    trays["at_kitchen"] -= 1
                else:
                    trays["at_kitchen"] -= 1  # moved then used
                continue

            # 4) need to make a no-gluten sandwich
            if breads_no_gluten_in_kitchen > 0 and contents_no_gluten_in_kitchen > 0 and notexist_count > 0:
                breads_no_gluten_in_kitchen -= 1
                contents_no_gluten_in_kitchen -= 1
                notexist_count -= 1
                # now we have an at_kitchen no_gluten sandwich -> follow step 3
                extra = ensure_tray_at_kitchen()
                if extra == -1:
                    return 99999
                total_cost += 1 + 3 + extra  # make + (put+move+serve) + possible move-of-tray
                if extra == 0:
                    trays["at_kitchen"] -= 1
                else:
                    trays["at_kitchen"] -= 1
                continue

            # If we cannot make a required no-gluten sandwich because resources missing:
            return 99999

        # Process non-allergic children
        # For non-allergic children, they can accept any sandwich; try to reuse regular ones first
        for child in remaining_non_allergic:
            place = self.waiting_place.get(child, None)

            # 1) sandwich on tray at child's place? Prefer regular, but accept no_gluten too.
            used = False
            if place is not None:
                # try regular first
                if use_ontray_at_place(place, False):
                    total_cost += 1
                    used = True
                else:
                    # try any no_gluten as fallback
                    if use_ontray_at_place(place, True):
                        total_cost += 1
                        used = True
            if used:
                continue

            # 2) sandwich on tray at kitchen? prefer regular then no_gluten
            if ontray_by_place_gluten.get(("kitchen", False), 0) > 0:
                ontray_by_place_gluten[("kitchen", False)] -= 1
                trays["at_kitchen"] -= 1
                total_cost += 2  # move + serve
                continue
            if ontray_by_place_gluten.get(("kitchen", True), 0) > 0:
                # use no_gluten ontray if no regular available
                ontray_by_place_gluten[("kitchen", True)] -= 1
                trays["at_kitchen"] -= 1
                total_cost += 2
                continue

            # 3) sandwich at kitchen (not on tray): prefer regular then no_gluten
            if at_kitchen_sandwiches["regular"] > 0:
                at_kitchen_sandwiches["regular"] -= 1
                extra = ensure_tray_at_kitchen()
                if extra == -1:
                    return 99999
                total_cost += 3 + extra
                trays["at_kitchen"] -= 1
                continue
            if at_kitchen_sandwiches["no_gluten"] > 0:
                # use no_gluten sandwich if no regular ones
                at_kitchen_sandwiches["no_gluten"] -= 1
                extra = ensure_tray_at_kitchen()
                if extra == -1:
                    return 99999
                total_cost += 3 + extra
                trays["at_kitchen"] -= 1
                continue

            # 4) make a sandwich (regular) if possible
            if total_breads_in_kitchen > 0 and total_contents_in_kitchen > 0 and notexist_count > 0:
                # consume any bread/content (prefer non-no-gluten to preserve no-gluten resources)
                # reduce counts accordingly (we don't track individual names here beyond totals)
                # prefer to use non-no-gluten bread/content if possible
                used_ng_bread = False
                used_ng_content = False
                # try to use non-no-gluten first
                non_ng_breads = total_breads_in_kitchen - breads_no_gluten_in_kitchen
                non_ng_contents = total_contents_in_kitchen - contents_no_gluten_in_kitchen

                if non_ng_breads > 0:
                    total_breads_in_kitchen -= 1
                elif breads_no_gluten_in_kitchen > 0:
                    breads_no_gluten_in_kitchen -= 1
                    total_breads_in_kitchen -= 1

                if non_ng_contents > 0:
                    total_contents_in_kitchen -= 1
                elif contents_no_gluten_in_kitchen > 0:
                    contents_no_gluten_in_kitchen -= 1
                    total_contents_in_kitchen -= 1

                notexist_count -= 1
                # now follow step 3 for a kitchen sandwich
                extra = ensure_tray_at_kitchen()
                if extra == -1:
                    return 99999
                total_cost += 1 + 3 + extra  # make + put + move + serve + possible tray move
                trays["at_kitchen"] -= 1
                continue

            # If cannot make or find a sandwich, treat as very costly / unsolvable
            return 99999

        # Final guard: total_cost should be finite integer
        return int(total_cost)