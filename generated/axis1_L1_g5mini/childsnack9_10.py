from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the tokens of a PDDL fact string without surrounding parentheses."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent greedy heuristic for the `secret_domain` (domain id: biqhd61z).
    The task modeled by the domain is to deliver prepared sandwiches to children.
    Goals are predicates of the form (ivo59apj <child>) — a delivered child.
    The typical delivery pipeline in the domain (informally) is:
      1) prepare a sandwich (kdzjcqwj) -> produces a "prepared" sandwich fact l2thmd6e and i9fq3vit;
      2) place the sandwich on a tray at a special table position (jhzk46zy needs tray at bupp7mhz) -> mpvvhjpt;
      3) move the tray to the child's table if needed (whgyhnl0) -> updates y0rxzzmc for the tray;
      4) execute the child-specific deliver action (nfarwtet or xtn2vmez) -> produces the goal ivo59apj for the child.

    This heuristic estimates the number of actions required to reach the goal by
    analyzing which partial steps have already been completed in the given state,
    counting (and greedily consuming) available prepared sandwiches, sandwiches on
    trays, and available raw resources. It aims to be informative (distinguishes
    cases like "already on correct tray" vs "needs full preparation + moves") while
    remaining cheap to compute.

    Assumptions
    -----------
    - Each goal is one predicate (ivo59apj <child>); each child needs exactly one delivery.
    - Either nfarwtet or xtn2vmez will be used to finish a delivery (they both cost 1
      action and require the same set of non-changing static facts except the child's
      static type). The heuristic counts that as one final action.
    - Trays can be moved between tables with a single action (whgyhnl0).
    - jhzk46zy requires a tray at the special location `bupp7mhz` to put a sandwich on a tray.
      If no tray is currently at `bupp7mhz`, we assume we can move any tray there first using
      whgyhnl0 (cost 1) and then run jhzk46zy.
    - For speed, the heuristic uses simple integer counters for available resources
      and greedily assigns them to outstanding children. It is not admissible but
      yields compact, informative estimates.

    Heuristic Initialization
    ------------------------
    - Extract the set of goal children from task.goals (any goal of the form "(ivo59apj child)").
    - Extract static facts and build:
        * child_to_table: mapping from child -> required table (from eyker22y facts in static).
      Other static typing (n2ij1fah/dr25ifx3) is not needed for the cost model since both
      delivery operators cost the same in our estimate.
    - We do not rely on domain type declarations beyond the static eyker22y facts.
    - If the special table constant `bupp7mhz` is visible in static facts, we record it;
      otherwise we still assume the same name `bupp7mhz` as a domain constant fallback.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state (node.state), the heuristic performs the following steps:
    1) Quick goal check: if all goals are already in the state, return 0.
    2) Parse the state to collect:
       - prepared_sandwiches: set of sandwich IDs with l2thmd6e
       - sandwiches_i9: set of sandwich IDs with i9fq3vit (may overlap with prepared)
       - sandwiches_on_tray: dict mapping sandwich -> tray (from mpvvhjpt facts)
       - tray_position: dict mapping tray -> table (from y0rxzzmc facts)
       - counts of raw resources: breads (go9lczo2), contents (fqungbps), raw_sandwich_bases (zogo0zhl)
    3) Build a list of children that still need delivery (goals not yet satisfied).
    4) Greedily assign existing, most-useful resources to outstanding children:
       - Prefer sandwiches already on a tray AND prepared where the tray is already at the child's table:
         cost 1 (single delivery action).
       - If the sandwich is on a tray and prepared but the tray is at a different table:
         cost 2 (move tray + deliver).
       - If we have a prepared sandwich not on a tray:
         - If some tray is already at `bupp7mhz` then we can jhzk46zy (put on tray) and deliver:
           cost 2 if `bupp7mhz` is the child's table (jhzk + deliver), else cost 3 (jhzk + move + deliver).
         - If no tray at `bupp7mhz`: we assume we first move a tray to `bupp7mhz` (1), then jhzk
           (1), then possibly move to child's table (1) and deliver (1) — accounted below as 3 or 4.
       - If we have only raw resources (i9fq3vit present or raw bread/content/sandwich bases), we
         estimate the full pipeline: prepare (kdzjcqwj) + put-on-tray (jhzk46zy) + maybe move + deliver.
         That is counted as 3 or 4 actions, depending on whether the child's table is `bupp7mhz`.
    5) After assigning a resource to a child, the heuristic decrements the corresponding counters
       (prepared, on-tray, raw) so the next child cannot reuse the same physical sandwich.
    6) Sum the per-child minimal estimated action counts and return the total.

    Note: The heuristic is intentionally optimistic in some respects (e.g., it ignores
    potential contention for a specific tray object beyond simple counting), but it
    captures the main structure of the task and distinguishes partial progress states
    (prepared sandwich, sandwich on tray, tray already at correct table), which helps
    guide a greedy best-first search efficiently.
    """

    def __init__(self, task):
        # Goal facts (frozenset of strings) kept for the goal check.
        self.goals = task.goals

        # Extract static mapping child -> table from eyker22y static facts.
        self.child_to_table = {}
        self.bupp_constant = "bupp7mhz"  # domain constant fallback

        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "eyker22y" and len(parts) >= 3:
                child = parts[1]
                table = parts[2]
                self.child_to_table[child] = table
            # if static explicitly references bupp7mhz we remember it
            if "bupp7mhz" in parts:
                self.bupp_constant = "bupp7mhz"

    def __call__(self, node):
        state = node.state

        # 1) If goal reached, heuristic is zero.
        if self.goals <= state:
            return 0

        # 2) Parse state facts into useful structures.
        prepared = set()  # sandwiches with l2thmd6e
        i9_set = set()  # sandwiches with i9fq3vit
        on_tray = {}  # sandwich -> tray for mpvvhjpt
        tray_pos = {}  # tray -> table for y0rxzzmc
        breads = 0
        contents = 0
        raw_sandwich_bases = 0  # zogo0zhl facts

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "l2thmd6e":
                prepared.add(parts[1])
            elif pred == "i9fq3vit":
                i9_set.add(parts[1])
            elif pred == "mpvvhjpt" and len(parts) >= 3:
                sand = parts[1]
                tray = parts[2]
                on_tray[sand] = tray
            elif pred == "y0rxzzmc" and len(parts) >= 3:
                tray = parts[1]
                table = parts[2]
                tray_pos[tray] = table
            elif pred == "go9lczo2":
                breads += 1
            elif pred == "fqungbps":
                contents += 1
            elif pred == "zogo0zhl":
                raw_sandwich_bases += 1

        # 3) Determine which children still require delivery.
        goal_children = []
        for g in self.goals:
            parts = get_parts(g)
            if parts[0] == "ivo59apj" and len(parts) >= 2:
                child = parts[1]
                goal_children.append(child)

        unsatisfied_children = []
        for child in goal_children:
            goal_fact = f"(ivo59apj {child})"
            if goal_fact not in state:
                unsatisfied_children.append(child)

        # Early exit: no unsatisfied children
        if not unsatisfied_children:
            return 0

        # 4) Prepare candidate pools (counts & lists).
        # sandwiches that are both prepared and on tray
        prepared_on_tray = []
        # prepared but not on tray
        prepared_not_on_tray = []
        # sandwiches that are i9 only (prepared step present or partial) but not l2thmd6e
        i9_only = []

        # Build sets for all sandwiches mentioned
        all_sandwiches = set(prepared) | set(i9_set) | set(on_tray.keys())
        for s in all_sandwiches:
            on = s in on_tray
            pr = s in prepared
            i9 = s in i9_set
            if pr and on:
                prepared_on_tray.append(s)
            elif pr and not on:
                prepared_not_on_tray.append(s)
            elif i9 and not pr:
                i9_only.append(s)

        # Count available trays and whether any is at bupp7mhz
        trays = set(tray_pos.keys())
        any_tray = len(trays) > 0
        any_tray_at_bupp = any((table == self.bupp_constant) for table in tray_pos.values())

        # Greedy assignment
        total_cost = 0

        # We'll consume resources as we assign them
        prepared_on_tray_list = list(prepared_on_tray)
        prepared_not_on_tray_count = len(prepared_not_on_tray)
        i9_only_count = len(i9_only)
        breads_count = breads
        contents_count = contents
        raw_bases_count = raw_sandwich_bases
        trays_available = list(trays)  # for potential moves

        for child in unsatisfied_children:
            child_table = self.child_to_table.get(child, None)

            # 1) Prefer a prepared sandwich already on a tray.
            used = False
            for idx, s in enumerate(prepared_on_tray_list):
                tray = on_tray.get(s)
                tray_table = tray_pos.get(tray)
                if tray_table == child_table:
                    # can deliver directly using nfarwtet/xtn2vmez
                    total_cost += 1
                    # consume that sandwich
                    prepared_on_tray_list.pop(idx)
                    used = True
                    break
            if used:
                continue

            # 2) Use prepared sandwich on tray but not at child's table: move + deliver
            if prepared_on_tray_list:
                s = prepared_on_tray_list.pop(0)
                # move tray to child's table (whgyhnl0) + deliver
                total_cost += 2
                used = True
                continue

            # 3) Use a prepared sandwich not on a tray: need jhzk46zy (put on tray) and maybe move and deliver
            if prepared_not_on_tray_count > 0:
                prepared_not_on_tray_count -= 1
                # If any tray at bupp7mhz, jhzk46zy can be used immediately.
                if any_tray_at_bupp:
                    # If child's table already bupp7mhz, we only need jhzk and deliver => 2 actions.
                    if child_table == self.bupp_constant:
                        total_cost += 2  # jhzk + deliver
                    else:
                        total_cost += 3  # jhzk + move + deliver
                else:
                    # Need to move a tray to bupp7mhz (1), jhzk (1), possibly move to child table (1), deliver (1)
                    if child_table == self.bupp_constant:
                        total_cost += 3  # move tray->bupp, jhzk, deliver
                    else:
                        total_cost += 4  # move->bupp, jhzk, move->child, deliver
                used = True
                continue

            # 4) Use partially prepared (i9 only) or raw resources: full pipeline
            # Check if we have i9 (partial) or raw resources to perform kdzjcqwj
            if i9_only_count > 0:
                i9_only_count -= 1
                # Need jhzk and deliver etc. But because l2thmd6e is not set we must perform kdzjcqwj first:
                # kdzjcqwj (1) + jhzk (1) + maybe move + deliver (1) => 3 or 4
                if any_tray_at_bupp:
                    if child_table == self.bupp_constant:
                        total_cost += 3  # prepare (kdzjcqwj) + jhzk + deliver
                    else:
                        total_cost += 4  # prepare + jhzk + move + deliver
                else:
                    if child_table == self.bupp_constant:
                        total_cost += 4  # move->bupp + prepare + jhzk + deliver
                    else:
                        total_cost += 5  # move->bupp + prepare + jhzk + move + deliver
                used = True
                continue

            # 5) If we have raw ingredients, assume we can create a new sandwich (kdzjcqwj)
            if breads_count > 0 and contents_count > 0 and raw_bases_count > 0:
                breads_count -= 1
                contents_count -= 1
                raw_bases_count -= 1
                # Full pipeline: kdzjcqwj + jhzk + maybe move + deliver
                if any_tray_at_bupp:
                    if child_table == self.bupp_constant:
                        total_cost += 3  # prepare + jhzk + deliver
                    else:
                        total_cost += 4  # prepare + jhzk + move + deliver
                else:
                    if child_table == self.bupp_constant:
                        total_cost += 4  # move->bupp + prepare + jhzk + deliver
                    else:
                        total_cost += 5  # move->bupp + prepare + jhzk + move + deliver
                used = True
                continue

            # 6) No explicit resources visible: be optimistic but finite:
            # Assume we can create one sandwich by spending 1..4 actions depending on tray location.
            if any_tray_at_bupp:
                if child_table == self.bupp_constant:
                    total_cost += 3
                else:
                    total_cost += 4
            else:
                if child_table == self.bupp_constant:
                    total_cost += 4
                else:
                    total_cost += 5

        return total_cost