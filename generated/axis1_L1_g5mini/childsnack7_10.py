from collections import Counter
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Split a PDDL fact string '(pred arg1 arg2)' into a list ['pred','arg1','arg2']."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (biqhd61z). The domain models
    preparing sandwiches, putting sandwiches on trays, moving trays between tables,
    and serving sandwiches to children. Goals require the predicate (ivo59apj child)
    for each child in the goal set.

    The heuristic estimates the number of actions (operators) required to reach
    a state where all goal children have their (ivo59apj child) facts. It is
    designed to be informative for greedy best-first search (not necessarily
    admissible) while being efficiently computable.

    Assumptions
    -----------
    - A child is served by either operator `nfarwtet` (requires that the sandwich
      has predicate l2thmd6e) or `xtn2vmez` (used for other child types).
    - A sandwich is prepared (becomes i9fq3vit) by one of two operators:
      - kdzjcqwj: produces i9fq3vit + l2thmd6e (for children requiring l2thmd6e).
      - h0xkysus: produces i9fq3vit without l2thmd6e.
    - To put a prepared sandwich onto a tray, jhzk46zy must be used and requires
      the tray to be at the special constant table `bupp7mhz`.
    - Trays can be moved between tables with whgyhnl0 (one action per move).
    - The heuristic treats trays, prepared sandwiches, and resources (bread/content)
      as consumable resources and greedily assigns them to pending children.
    - If multiple equally-good resources exist, the heuristic picks one arbitrarily.
    - The heuristic ignores fine-grained matching of which specific bread/content
      object must pair with which sandwich beyond the basic static-type compatibility
      (i.e., it uses set intersections to check availability).

    Heuristic Initialization
    ------------------------
    From task.static the heuristic extracts:
    - which children are of type `n2ij1fah` (these require l2thmd6e),
      and which are `dr25ifx3`.
    - the mapping eyker22y child -> table (where the child should be served).
    The constructor also stores the goal facts (task.goals) for quick goal checks.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given node/state the heuristic proceeds as follows:

    1. Quick check: if all goals are already satisfied in the state, return 0.

    2. Parse the current state into useful collections:
       - trays: mapping tray -> current_table (from y0rxzzmc facts).
       - mpv: mapping sandwich -> tray (mpvvhjpt facts).
       - prepared sandwiches: set of sandw with i9fq3vit.
       - special-prepared sandwiches: set of sandw with l2thmd6e.
       - raw sandwiches available (zogo0zhl).
       - bread/content resources available (go9lczo2, e6jz04yd, fqungbps, odd6wyd1).
       - children static types and their required table (from task.static).

    3. Build a list of pending children (those with a goal ivo59apj child missing from state).
       Sort the pending children by requirement: children requiring l2thmd6e are handled first,
       as they are more constrained.

    4. For each pending child, greedily pick the cheapest achievable sequence of actions given
       the current (remaining) resources. The considered options (from cheapest to more expensive)
       are:
       - Use an existing mpvvhjpt sandwich that is already on a tray at the child's table
         (cost 1: apply nfarwtet/xtn2vmez).
       - Use an existing mpvvhjpt sandwich on some other table: move tray once to child's table
         (whgyhnl0) and serve (cost 2).
       - Use a prepared i9fq3vit sandwich (that meets l2thmd6e requirement if needed):
         (a) ensure a tray is at `bupp7mhz`: if none exists, move any tray there (cost 1),
         (b) jhzk46zy to put sandwich onto the tray at `bupp7mhz` (cost 1),
         (c) if child's table != `bupp7mhz`, move tray to child's table (cost 1),
         (d) serve with nfarwtet/xtn2vmez (cost 1).
         Total cost depends on whether a tray at `bupp7mhz` exists and whether child's table
         equals `bupp7mhz` (cost 2..4).
       - Prepare a new sandwich (kdzjcqwj for l2 children, h0xkysus otherwise) when resources
         exist: preparation (cost 1) followed by the same pipeline as using a prepared sandwich.
       - If none of the above is possible due to resource absence, a large finite penalty is
         assigned for that child (to keep heuristic finite); in solvable states the earlier
         options should succeed.

    5. When the heuristic "commits" to an option for a child, it updates the local view of
       resources (consumes the sandwich/tray/bread/content used and updates tray positions),
       so subsequent children are evaluated with the updated resource availability. This
       greedy, resource-aware assignment helps the heuristic prefer reusing existing ready
       sandwiches and trays and yields a more informative estimate.

    6. The heuristic returns the sum of the chosen costs for all pending children.

    Rationale
    ---------
    The heuristic tries to reflect realistic sequences of domain operators, accounts for
    constraints (like the need for a tray at `bupp7mhz` to execute jhzk46zy), and is greedy
    about reusing already prepared/placed sandwiches and trays. This keeps computation cheap
    but yields informative guidance to the search.
    """

    def __init__(self, task):
        # Store goal conditions and static facts.
        self.goals = task.goals
        self.static = task.static

        # Extract static child types and child->table relations.
        self.n2_children = set()
        self.dr_children = set()
        self.child_table = {}

        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "n2ij1fah":
                self.n2_children.add(parts[1])
            elif parts[0] == "dr25ifx3":
                self.dr_children.add(parts[1])
            elif parts[0] == "eyker22y":
                # eyker22y child table
                self.child_table[parts[1]] = parts[2]

    def __call__(self, node):
        state = node.state

        # If already solved, heuristic is 0.
        if self.goals <= state:
            return 0

        # Parse state facts into structures.
        trays = {}  # tray -> table
        mpv = {}  # sandwich -> tray (mpvvhjpt)
        prepared = set()  # i9fq3vit sandw
        special_prepared = set()  # l2thmd6e sandw
        raw_sandw = Counter()  # zogo0zhl sandw counts
        breads_go = set()  # go9lczo2 bread
        breads_e6 = set()  # e6jz04yd bread
        contents_fqu = set()  # fqungbps content
        contents_odd = set()  # odd6wyd1 content

        # chairs/tables constant name is 'bupp7mhz' (used in domain constant)
        BU_TABLE = "bupp7mhz"

        # Fill structures from state facts
        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "y0rxzzmc":
                # (y0rxzzmc tray table)
                tray, table = parts[1], parts[2]
                trays[tray] = table
            elif pred == "mpvvhjpt":
                # (mpvvhjpt sandw tray)
                sandw, tray = parts[1], parts[2]
                mpv[sandw] = tray
            elif pred == "i9fq3vit":
                prepared.add(parts[1])
            elif pred == "l2thmd6e":
                special_prepared.add(parts[1])
            elif pred == "zogo0zhl":
                raw_sandw[parts[1]] += 1
            elif pred == "go9lczo2":
                breads_go.add(parts[1])
            elif pred == "e6jz04yd":
                breads_e6.add(parts[1])
            elif pred == "fqungbps":
                contents_fqu.add(parts[1])
            elif pred == "odd6wyd1":
                contents_odd.add(parts[1])

        # Determine which children are missing their goal (ivo59apj)
        goal_children = []
        for g in self.goals:
            parts = get_parts(g)
            if parts[0] == "ivo59apj":
                child = parts[1]
                goal_children.append(child)

        pending_children = []
        for child in goal_children:
            if f"(ivo59apj {child})" not in state:
                pending_children.append(child)

        # If there are no trays at all, we cannot do jhzk46zy/placement; return
        # a conservative finite estimate (high cost per child).
        if not trays:
            # Fallback: assume at least 4 actions per child (prepare+put+move+serve)
            return len(pending_children) * 6

        total_cost = 0

        # Sort children: ones that need l2thmd6e (n2 children) first (more constrained).
        def needs_l2(child):
            return child in self.n2_children

        pending_children.sort(key=lambda c: (not needs_l2(c), c))

        # Helper utilities to update resources when committing to an action.
        def consume_mpv(sandw):
            # remove mpv mapping for sandw
            if sandw in mpv:
                del mpv[sandw]

        def consume_prepared(sandw):
            if sandw in prepared:
                prepared.remove(sandw)
            if sandw in special_prepared:
                special_prepared.discard(sandw)

        def consume_raw_sandwich():
            # consume any available raw sandwich (zogo0zhl)
            if not raw_sandw:
                return None
            sandw, cnt = next(iter(raw_sandw.items()))
            raw_sandw[sandw] -= 1
            if raw_sandw[sandw] <= 0:
                del raw_sandw[sandw]
            return sandw

        def find_mpv_for_child(child_table, require_l2):
            # Prefer mpv at child's table
            for sandw, tray in list(mpv.items()):
                if trays.get(tray) == child_table:
                    if (not require_l2) or (sandw in special_prepared):
                        return sandw, tray, trays[tray], True
            # fallback: any other mpv
            for sandw, tray in list(mpv.items()):
                if (not require_l2) or (sandw in special_prepared):
                    return sandw, tray, trays[tray], False
            return None

        # Choose an arbitrary tray (used when we need to move a tray to bupp7mhz)
        def any_tray():
            return next(iter(trays.keys()))

        # For each pending child, greedily compute cheapest option given remaining resources.
        for child in pending_children:
            child_tbl = self.child_table.get(child, None)
            require_l2 = child in self.n2_children

            # 1) Try existing mpvvhjpt on child's table
            best_option_cost = None
            best_option = None

            mpv_found = find_mpv_for_child(child_tbl, require_l2)
            if mpv_found:
                sandw, tray, tray_table, at_child_table = mpv_found
                if at_child_table:
                    # directly serve
                    best_option_cost = 1
                    best_option = ("use_mpv_here", sandw, tray)
                else:
                    # move tray then serve
                    best_option_cost = 2
                    best_option = ("move_mpv_then_serve", sandw, tray, child_tbl)

            # 2) Try using a prepared i9fq3vit sandwich (and then jhzk46zy + moves + serve).
            if best_option_cost is None or best_option_cost > 2:
                # find a prepared sandwich that meets l2 requirement if needed
                candidate_sandw = None
                for s in prepared:
                    if (not require_l2) or (s in special_prepared):
                        candidate_sandw = s
                        break
                if candidate_sandw:
                    # compute cost for pipeline using this prepared sandwich
                    # check if a tray is already at BU_TABLE
                    trays_at_bu = [t for t, tab in trays.items() if tab == BU_TABLE]
                    tmp_cost = 0
                    moved_tray = None
                    if trays_at_bu:
                        tray_to_use = trays_at_bu[0]
                    else:
                        # need to move some tray to BU_TABLE: cost 1
                        tray_to_use = any_tray()
                        tmp_cost += 1
                    # jhzk46zy to put sandwich on tray at BU_TABLE
                    tmp_cost += 1
                    # if child's table == BU_TABLE, no move needed; else move tray to child's table
                    if child_tbl != BU_TABLE:
                        tmp_cost += 1
                    # finally serve
                    tmp_cost += 1

                    if best_option_cost is None or tmp_cost < best_option_cost:
                        best_option_cost = tmp_cost
                        best_option = ("use_prepared_and_serve", candidate_sandw, tray_to_use)

            # 3) Try preparing a new sandwich then pipeline.
            if best_option_cost is None or best_option_cost > 3:
                # Check whether we can create the required prepared sandwich
                can_prepare = False
                prep_is_special = False  # kdzjcqwj produces special l2thmd6e
                if require_l2:
                    # need a bread in both go and e6, and content in both fqu and odd
                    if (breads_go & breads_e6) and (contents_fqu & contents_odd) and raw_sandw:
                        can_prepare = True
                        prep_is_special = True
                else:
                    # need any bread in go and any content in fqu and a raw sandwich
                    if breads_go and contents_fqu and raw_sandw:
                        can_prepare = True
                        prep_is_special = False

                if can_prepare:
                    # preparation cost 1 + pipeline as above for prepared sandwich
                    tmp_cost = 1  # for prepare
                    trays_at_bu = [t for t, tab in trays.items() if tab == BU_TABLE]
                    if trays_at_bu:
                        tray_to_use = trays_at_bu[0]
                    else:
                        tray_to_use = any_tray()
                        tmp_cost += 1  # move tray to BU_TABLE
                    tmp_cost += 1  # jhzk46zy
                    if child_tbl != BU_TABLE:
                        tmp_cost += 1  # move to child's table
                    tmp_cost += 1  # serve

                    if best_option_cost is None or tmp_cost < best_option_cost:
                        best_option_cost = tmp_cost
                        best_option = ("prepare_and_serve", prep_is_special, tray_to_use)

            # If still nothing found, assign a large finite penalty for this child.
            if best_option is None:
                # fallback high but finite cost
                total_cost += 10
                continue

            # Commit the chosen option: update resources and trays accordingly, and add cost.
            total_cost += best_option_cost

            typ = best_option[0]
            if typ == "use_mpv_here":
                _, sandw, tray = best_option
                consume_mpv(sandw)
                consume_prepared(sandw)
                # tray remains at same table
            elif typ == "move_mpv_then_serve":
                _, sandw, tray, dest_table = best_option
                # move tray to child's table (1 action was accounted)
                trays[tray] = dest_table
                # serve
                consume_mpv(sandw)
                consume_prepared(sandw)
            elif typ == "use_prepared_and_serve":
                _, sandw, tray_to_use = best_option
                # If tray_to_use wasn't at BU_TABLE and we accounted a move, set its pos to BU_TABLE
                if trays.get(tray_to_use) != BU_TABLE:
                    trays[tray_to_use] = BU_TABLE  # we moved it
                # jhzk46zy: sandwich goes to tray (mpv)
                mpv[sandw] = tray_to_use
                # prepared fact is consumed by jhzk46zy
                consume_prepared(sandw)
                # If child's table != BU_TABLE, we move tray to child's table (we accounted cost)
                if child_tbl != BU_TABLE:
                    trays[tray_to_use] = child_tbl
                # After moving, serving consumes mpv
                consume_mpv(sandw)
            elif typ == "prepare_and_serve":
                _, prep_special, tray_to_use = best_option
                # consume raw sandwich
                sandw_obj = consume_raw_sandwich()
                # consume one bread and one content from available sets.
                # For simplicity, remove arbitrary items from relevant sets.
                if prep_special:
                    # remove from intersection sets
                    bread_choice = next(iter(breads_go & breads_e6))
                    content_choice = next(iter(contents_fqu & contents_odd))
                    breads_go.discard(bread_choice)
                    breads_e6.discard(bread_choice)
                    contents_fqu.discard(content_choice)
                    contents_odd.discard(content_choice)
                else:
                    bread_choice = next(iter(breads_go))
                    content_choice = next(iter(contents_fqu))
                    breads_go.discard(bread_choice)
                    contents_fqu.discard(content_choice)
                # After preparation, a prepared sandwich is produced (i9fq3vit)
                # Represent it by sandw_obj (if we had a raw sandwich). If no explicit raw object,
                # we cannot create a new identifier; but we consumed raw_sandw earlier.
                sandw = sandw_obj if sandw_obj is not None else "prepared_sandw_fake"
                # If we created special prepared, mark it as special
                if prep_special:
                    special_prepared.add(sandw)
                prepared.add(sandw)
                # Now follow the prepared pipeline: ensure tray at BU_TABLE
                if trays.get(tray_to_use) != BU_TABLE:
                    trays[tray_to_use] = BU_TABLE  # moved
                # jhzk46zy: put sandwich onto tray
                mpv[sandw] = tray_to_use
                consume_prepared(sandw)
                # move to child table if needed
                if child_tbl != BU_TABLE:
                    trays[tray_to_use] = child_tbl
                # serve: remove mpv
                consume_mpv(sandw)

        return total_cost