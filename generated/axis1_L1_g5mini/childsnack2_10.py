from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Split a fact string like '(pred a b)' into ['pred', 'a', 'b']."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the secret_domain (biqhd61z) problem.
    The planning task in this domain constructs sandwiches, places them on trays,
    and delivers them to children. The goal facts are of the form (ivo59apj childX).
    The main steps to satisfy a child's goal are:
      1) create/prepare a sandwich (i9fq3vit),
      2) put the sandwich on a tray (mpvvhjpt),
      3) move the tray to the child's table (y0rxzzmc mapping), and
      4) use the delivery operator (nfarwtet or xtn2vmez) to obtain (ivo59apj child).

    Assumptions
    -----------
    - Each child goal needs exactly one delivered sandwich (one use of nfarwtet or xtn2vmez).
    - There are two kinds of children (distinguished by predicates n2ij1fah and dr25ifx3).
      Children in n2ij1fah require that the sandwich has the l2thmd6e property (i.e., the
      sandwich must be prepared using kdzjcqwj which requires special bread/content).
      Children in dr25ifx3 can be delivered without l2thmd6e (xtn2vmez).
    - Sandwich preparation consumes: a raw sandwich object (zogo0zhl), a bread (go9lczo2) and a
      content (fqungbps). kdzjcqwj additionally requires bread with e6jz04yd and content with
      odd6wyd1 to produce l2thmd6e.
    - Placing a sandwich on a tray uses jhzk46zy which requires the tray to be at the special
      location constant bupp7mhz (y0rxzzmc tray bupp7mhz). Moving a tray from one table to
      another requires whgyhnl0 (changes the y0rxzzmc mapping). We approximate each such move
      as costing one action.
    - The heuristic is not admissible; it aims to estimate action counts to guide greedy search.

    Heuristic Initialization
    ------------------------
    The constructor extracts static information from the task:
      - which children are labeled with n2ij1fah (need special sandwich property),
        and which are labeled with dr25ifx3;
      - eyker22y static relations mapping child -> table (if provided statically);
      - the set of goal facts (to know which children must be delivered).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state (node.state) we:
      1. Return 0 immediately if all goal facts are already satisfied.

      2. Parse the current state to get:
         - trays' current locations (y0rxzzmc tray loc),
         - which sandwiches are prepared (i9fq3vit facts),
         - which sandwiches are already on trays (mpvvhjpt sandwich tray),
         - which sandwiches carry the l2thmd6e property,
         - counts (and lists) of available raw resources:
             * raw sandwiches (zogo0zhl),
             * breads currently available (go9lczo2),
             * contents currently available (fqungbps),
             * breads marked e6jz04yd and contents marked odd6wyd1 (from static and state).
         - which children still need delivery (from goals not satisfied in the state),
         - the child's required table (from eyker22y facts present in static or in the state).

      3. Greedy matching of available deliverables:
         - Prefer using an mpvvhjpt that is already at the child's table and (if required)
           whose sandwich already has l2thmd6e. Using such an mpvvhjpt costs 1 action
           (the delivery action).
         - Then prefer mpvvhjpt located elsewhere: move the tray to the child's table
           (cost 1) and deliver (cost 1) → total 2 actions.
         - Next prefer an already-prepared sandwich i9fq3vit:
           place on a tray at bupp7mhz using jhzk46zy (cost 1) — if no tray is currently at
           bupp7mhz we conservatively assume we need to move some tray there first (cost 1).
           Then move tray to child's table (cost 1) and deliver (cost 1).
           If a tray at bupp7mhz exists the cost for prepared -> placed -> delivered is 3,
           otherwise 4.
         - Finally, if none of the above is available, prepare a sandwich from raw resources:
           * For n2ij1fah children we use kdzjcqwj which consumes a raw sandwich, a bread that
             must have e6jz04yd, and a content that must have odd6wyd1 (if those are missing the
             state is likely unsolvable; we still return a large finite cost estimate).
           * For dr25ifx3 children we use h0xkysus which consumes any raw sandwich, any bread
             and any content.
           Preparing costs 1 action; after preparing we assume the subsequent placement/move/deliver
           costs as for prepared sandwiches (3 or 4 depending on availability of tray at bupp7mhz).

      4. We deduct used resources as we greedily assign them to children (so resources are not
         double-counted). The sum of estimated actions for all undelivered children is returned.

    The heuristic therefore combines resource accounting, existing partial progress (i9fq3vit/mpvvhjpt),
    static child-type requirements, and simple tray-location reasoning to produce a finite and
    informative estimate of the number of remaining actions.
    """

    def __init__(self, task):
        # Goals (set of goal fact strings)
        self.goals = set(task.goals)

        # Extract static facts that classify children and map children -> table
        static = task.static if hasattr(task, "static") else set()

        self.n2_children = set()  # children that have predicate n2ij1fah
        self.dr_children = set()  # children that have predicate dr25ifx3
        self.eyker = {}  # child -> table mapping (from eyker22y facts in static)

        for fact in static:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "n2ij1fah" and len(parts) >= 2:
                self.n2_children.add(parts[1])
            elif pred == "dr25ifx3" and len(parts) >= 2:
                self.dr_children.add(parts[1])
            elif pred == "eyker22y" and len(parts) >= 3:
                child, table = parts[1], parts[2]
                self.eyker[child] = table

    def __call__(self, node):
        state = set(node.state)

        # 1) If goals already satisfied -> heuristic = 0
        if self.goals <= state:
            return 0

        # Helper extractors from current state (and we augment with static info where relevant)
        # Map trays to locations
        tray_loc = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "y0rxzzmc" and len(parts) >= 3:
                tray, loc = parts[1], parts[2]
                tray_loc[tray] = loc

        # mpvvhjpt: sandwich placed on tray -> list of tuples (sandwich, tray)
        mp_on_tray = []
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "mpvvhjpt" and len(parts) >= 3:
                sand, tray = parts[1], parts[2]
                mp_on_tray.append((sand, tray))

        # prepared sandwiches (i9fq3vit)
        prepared = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "i9fq3vit" and len(parts) >= 2:
                prepared.add(parts[1])

        # sandwiches that have l2thmd6e property
        l2_set = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "l2thmd6e" and len(parts) >= 2:
                l2_set.add(parts[1])

        # lists of raw resources in the state
        raw_sandwiches = [get_parts(f)[1] for f in state if get_parts(f)[0] == "zogo0zhl"]
        breads = [get_parts(f)[1] for f in state if get_parts(f)[0] == "go9lczo2"]
        contents = [get_parts(f)[1] for f in state if get_parts(f)[0] == "fqungbps"]

        # special breads and contents possibly coming from state or static
        breads_e6 = set()
        contents_odd = set()
        # from state
        for f in state:
            parts = get_parts(f)
            if parts[0] == "e6jz04yd" and len(parts) >= 2:
                breads_e6.add(parts[1])
            if parts[0] == "odd6wyd1" and len(parts) >= 2:
                contents_odd.add(parts[1])
            if parts[0] == "eyker22y" and len(parts) >= 3:
                # runtime eyker mapping can override/augment static eyker
                child, table = parts[1], parts[2]
                self.eyker[child] = table

        # incorporate static e6/odd if present in task.static (some problems may have them static)
        # (task.static not accessible here; we used constructor for child classification only)
        # but parts that are static would have been included in initial state's static set
        # so breads_e6/contents_odd already include them if they are in node.state

        # count presence of tray at bupp7mhz (special location required by jhzk46zy)
        tray_at_bupp = any(loc == "bupp7mhz" for loc in tray_loc.values())

        # parse goal children (from goals collected earlier)
        goal_children = []
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "ivo59apj" and len(parts) >= 2:
                goal_children.append(parts[1])

        # determine which children are already satisfied
        undelivered = [c for c in goal_children if f"(ivo59apj {c})" not in state]

        # resource pools that we will mutate as we allocate resources
        available_raw_sand = list(raw_sandwiches)
        available_breads = list(breads)
        available_contents = list(contents)
        available_prepared = set(prepared)  # prepared sandwiches not yet placed on tray
        available_mp = list(mp_on_tray)  # sandwiches currently on trays (sand, tray)
        available_l2 = set(l2_set)

        total_cost = 0

        # Helper to get child's table (eyker22y). If unknown, we conservatively
        # assume a move is needed (so table mismatch forces a tray move).
        def child_table(child):
            return self.eyker.get(child, None)

        # 1) Use mpvvhjpt already at child's table and suitable
        remaining_children = []
        used_mp_indices = set()
        for child in undelivered:
            table = child_table(child)
            satisfied = False
            for idx, (sand, tray) in enumerate(available_mp):
                if idx in used_mp_indices:
                    continue
                tray_location = tray_loc.get(tray, None)
                if tray_location is None or table is None:
                    # if we don't know either, fall back to later decisions
                    continue
                # check whether tray is already at child's table
                if tray_location == table:
                    # check l2 requirement if child in n2 category
                    if child in self.n2_children:
                        if sand in available_l2:
                            # deliverable directly
                            used_mp_indices.add(idx)
                            total_cost += 1  # delivery action
                            satisfied = True
                            break
                        else:
                            # not suitable, cannot use this mp for this child
                            continue
                    else:
                        # dr25 child: can deliver
                        used_mp_indices.add(idx)
                        total_cost += 1
                        satisfied = True
                        break
            if not satisfied:
                remaining_children.append(child)

        # remove used mp entries
        available_mp = [mp for i, mp in enumerate(available_mp) if i not in used_mp_indices]

        # 2) Use mpvvhjpt located elsewhere: move tray to child's table (1) + deliver (1)
        still_remaining = []
        used_mp_indices = set()
        for child in remaining_children:
            table = child_table(child)
            satisfied = False
            for idx, (sand, tray) in enumerate(available_mp):
                if idx in used_mp_indices:
                    continue
                # check l2 if needed
                if child in self.n2_children and sand not in available_l2:
                    continue
                # we assume we can move any tray to child's table with cost 1
                used_mp_indices.add(idx)
                total_cost += 2  # move tray + deliver
                satisfied = True
                break
            if not satisfied:
                still_remaining.append(child)

        available_mp = [mp for i, mp in enumerate(available_mp) if i not in used_mp_indices]

        # 3) Use prepared sandwiches (i9fq3vit). For each prepared:
        # place on tray at bupp7mhz (1, possibly +1 to move a tray to bupp7mhz),
        # move to child's table (1) and deliver (1)
        remaining_children = still_remaining
        still_remaining = []
        for child in remaining_children:
            table = child_table(child)
            satisfied = False
            # find a prepared sandwich that meets l2 requirement if necessary
            chosen_sand = None
            for sand in list(available_prepared):
                if child in self.n2_children:
                    if sand in available_l2:
                        chosen_sand = sand
                        break
                else:
                    chosen_sand = sand
                    break
            if chosen_sand is not None:
                # place on tray: if there's a tray at bupp7mhz, just jhzk46zy (1)
                # otherwise assume we must move some tray to bupp7mhz first (cost 1)
                place_cost = 1
                if not tray_at_bupp:
                    place_cost += 1  # move a tray to bupp7mhz so we can place sandwich
                    # after this hypothetical move, we can assume one tray is at bupp7mhz
                    tray_at_bupp = True
                # after placing, need to move tray to child's table (1) and deliver (1)
                move_and_deliver = 2
                total_cost += place_cost + move_and_deliver
                available_prepared.remove(chosen_sand)
                # the prepared sandwich is consumed (placed and delivered)
                satisfied = True
            if not satisfied:
                still_remaining.append(child)

        # 4) For remaining children, prepare new sandwiches from raw resources
        remaining_children = still_remaining
        for child in remaining_children:
            # Attempt to prepare appropriate sandwich
            if child in self.n2_children:
                # requires special bread and special content and a raw sandwich
                # check availability
                # find a bread that is in breads_e6 (or e6 in current state)
                candidate_bread = None
                for b in available_breads:
                    if b in breads_e6:
                        candidate_bread = b
                        break
                candidate_content = None
                for c in available_contents:
                    if c in contents_odd:
                        candidate_content = c
                        break
                if candidate_bread is None or candidate_content is None or not available_raw_sand:
                    # missing special resources: assume costly but finite fallback estimate
                    # add a fixed penalty (we use 6) to indicate effort / impossibility of using cheap ops
                    total_cost += 6
                    # still attempt to consume any available general resources to avoid blow-up
                    if available_raw_sand:
                        available_raw_sand.pop()
                    if candidate_bread and candidate_bread in available_breads:
                        available_breads.remove(candidate_bread)
                    elif available_breads:
                        available_breads.pop()
                    if candidate_content and candidate_content in available_contents:
                        available_contents.remove(candidate_content)
                    elif available_contents:
                        available_contents.pop()
                    # after this we assume we produced a sandwich with l2 property (for heuristic)
                    # placing & delivering as below
                    place_cost = 1
                    if not tray_at_bupp:
                        place_cost += 1
                        tray_at_bupp = True
                    total_cost += place_cost + 2  # move + deliver
                else:
                    # consume resources
                    available_raw_sand.pop()
                    available_breads.remove(candidate_bread)
                    available_contents.remove(candidate_content)
                    # produce sandwich with l2 property (cost 1)
                    total_cost += 1
                    # now place & deliver like above
                    place_cost = 1
                    if not tray_at_bupp:
                        place_cost += 1
                        tray_at_bupp = True
                    total_cost += place_cost + 2  # move + deliver
            else:
                # dr25 family: can use general prepare action
                if not available_raw_sand or not available_breads or not available_contents:
                    # missing general resources; penalize but remain finite
                    total_cost += 5
                    # consume any available to model progress
                    if available_raw_sand:
                        available_raw_sand.pop()
                    if available_breads:
                        available_breads.pop()
                    if available_contents:
                        available_contents.pop()
                    # place & deliver
                    place_cost = 1
                    if not tray_at_bupp:
                        place_cost += 1
                        tray_at_bupp = True
                    total_cost += place_cost + 2
                else:
                    # consume resources
                    available_raw_sand.pop()
                    available_breads.pop()
                    available_contents.pop()
                    # prepare
                    total_cost += 1
                    place_cost = 1
                    if not tray_at_bupp:
                        place_cost += 1
                        tray_at_bupp = True
                    total_cost += place_cost + 2

        # Final heuristic estimate
        return total_cost