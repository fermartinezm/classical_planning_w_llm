from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Split a PDDL fact string like '(pred a b)' into ['pred', 'a', 'b']."""
    return fact[1:-1].split()


def match(fact, *pattern):
    """
    Match a fact against a pattern with support for '*' wildcards.
    Example: match('(at b r)', 'at', '*', 'r') -> True
    """
    parts = get_parts(fact)
    return all(fnmatch(p, pat) for p, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic tailored for the 'secret_domain' (rb7qr1fs).
    The main observable structure in this domain (from domain analysis)
    is that many desired goals are of the form:
      (ixsb7u9c <dbk5i2a5> <chxau8fy>)
    which are produced by a chained sequence of actions:
      - produce an r1frsf5x fact for the given rb7qr1fs, dbk5i2a5 and chxau8fy
        (dlx472ba creates r1frsf5x when prerequisites about tom0ncko and jckjsxca, cs7680d7, etc. hold),
      - then apply ymbljzxy which consumes r1frsf5x (plus jckjsxca and some static facts)
        to produce ixsb7u9c (the typical goal predicate).
    There are also other goal types in instances (for example y01m0fhm),
    created by a different chain (xmjnictn -> sn0ysqqm / ttp8j82e).
    This heuristic focuses on recognizing those chain dependencies and
    estimating the number of *actions* needed to satisfy each missing goal.

    Assumptions
    -----------
    - The planner represents states as frozensets of PDDL fact strings.
    - Static relations (task.static) are indeed static and can be used to
      check long-lived preconditions like compatibility relations (wx95011q),
      mapping facts (cs7680d7), and helper relations (rrhxwbyt, z6yvwl7n).
    - Typical plan to achieve an ixsb7u9c(goal) is:
         (potentially) p8i30ct9 -> dlx472ba -> ymbljzxy
      where p8i30ct9 creates a wfrx4exs fact required by dlx472ba.
    - Reasonable cost estimates:
      - If the terminal goal fact is already true -> 0.
      - If the final operator that directly creates the goal is applicable
        -> cost 1.
      - If an intermediate fact exists but some small enabling facts are missing
        -> cost 2.
      - If no intermediate facts exist but the necessary chain can be started
        with one or two operators available in state/static -> cost 3.
      - Otherwise we conservatively estimate a small finite cost (4) for a
        missing goal to ensure the heuristic remains finite.

    Heuristic Initialization
    ------------------------
    The constructor extracts and indexes static facts to accelerate lookups:
      - static sets/maps for predicates that appear often in preconditions:
        wx95011q, cs7680d7, rrhxwbyt, z6yvwl7n, iaub97vi, rycjvfag, p6u7tzz0,
        a8x3330w, ipivzc5v, a298chgk, bszmiquc, dfhfijzr and so on.
    The set of goal facts is also kept for quick checks.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    To compute the heuristic value for a node/state we:
      1. If all goals are satisfied in the current state return 0.
      2. Build quick indexes of facts present in the current state by predicate.
      3. For every missing goal fact estimate the minimal number of actions:
         - For ixsb7u9c (<dbk> <chx>):
             * If ixsb7u9c already in state -> 0.
             * If there is an r1frsf5x fact with matching dbk and chx:
                 - Check whether there exists a jckjsxca mapping for the same
                   rb7qr1fs and whether compatible wx95011q / rrhxwbyt facts
                   exist (often static). If yes, ymbljzxy can be applied -> cost 1.
                 - Otherwise assume small repair is needed (relink jckjsxca or
                   provide missing static-like facts) -> cost 2.
             * Else (no r1frsf5x present):
                 - Check whether dlx472ba is applicable or almost applicable:
                   dlx472ba requires a matching tom0ncko (efxvv9ax / rycjvfag),
                   cs7680d7 linking dbk->uw, and jckjsxca for the relevant rb7.
                   If all are present except the transient wfrx4exs, we assume
                   p8i30ct9 + dlx472ba + ymbljzxy (3 actions) suffice -> cost 3.
                 - If even more prerequisites are missing, conservatively set cost 4.
         - For y01m0fhm (<uw>):
             * Similar reasoning but following the chain xmjnictn -> sn0ysqqm,
               i.e., if rpzrvk8g exists and the rrhxwbyt / jckjsxca / wx95011q
               combination is available, the producing operator sn0ysqqm
               can be applied -> cost 1; otherwise estimate 2-3 depending on
               which supporting facts are missing.
         - For any other goal predicate not explicitly handled we fall back to
           a small finite estimated cost proportional to 1 + number of missing
           conjuncts (we use 2 as a conservative estimate to keep the heuristic
           informative but finite).
      4. Sum the per-goal estimates and return this integer value.

    The heuristic is deliberately non-admissible (it may under- or over-estimate)
    but attempts to capture the necessary operator-chain length to reach
    common goal predicates found in problem instances of this domain.
    """

    def __init__(self, task):
        # Goals and static facts for quick access.
        self.goals = set(task.goals)
        self.static = set(task.static)

        # Index static facts by predicate name for quick membership tests.
        self.static_by_pred = {}
        for fact in self.static:
            pred = get_parts(fact)[0]
            self.static_by_pred.setdefault(pred, []).append(get_parts(fact)[1:])

        # Useful static predicate quick-access sets (tuples) for membership checks
        # Common predicates used in action preconditions/effects in the domain:
        self.static_wx95011q = set((a, b) for a, b in self.static_by_pred.get("wx95011q", []))
        self.static_cs7680d7 = set((a, b) for a, b in self.static_by_pred.get("cs7680d7", []))
        self.static_rrhxwbyt = set((a, b) for a, b in self.static_by_pred.get("rrhxwbyt", []))
        self.static_z6yvwl7n = set((a, b) for a, b in self.static_by_pred.get("z6yvwl7n", []))
        self.static_iaub97vi = set((a, b) for a, b in self.static_by_pred.get("iaub97vi", []))
        self.static_rycjvfag = set((a, b) for a, b in self.static_by_pred.get("rycjvfag", []))
        self.static_p6u7tzz0 = set(a for (a,) in self.static_by_pred.get("p6u7tzz0", []))
        self.static_a8x3330w = set(a for (a,) in self.static_by_pred.get("a8x3330w", []))
        self.static_ipivzc5v = set(a for (a,) in self.static_by_pred.get("ipivzc5v", []))
        self.static_a298chgk = set(a for (a,) in self.static_by_pred.get("a298chgk", []))
        self.static_bszmiquc = set(a for (a,) in self.static_by_pred.get("bszmiquc", []))
        self.static_dfhfijzr = set(a for (a,) in self.static_by_pred.get("dfhfijzr", []))

    def __call__(self, node):
        state = set(node.state)  # frozenset -> mutable set for lookups

        # If goals already satisfied return 0
        if self.goals <= state:
            return 0

        # Index current facts by predicate
        facts_by_pred = {}
        for fact in state:
            parts = get_parts(fact)
            facts_by_pred.setdefault(parts[0], []).append(parts[1:])

        def fact_present(pred, *args):
            """Return True if a fact (pred args...) is present in current state."""
            return any(tuple(parts) == tuple(args) for parts in facts_by_pred.get(pred, []))

        def get_facts(pred):
            """Return list of argument-tuples for predicate pred in current state."""
            return [tuple(parts) for parts in facts_by_pred.get(pred, [])]

        estimate_sum = 0

        # For every goal, compute an estimated action cost to achieve it.
        for goal in self.goals:
            if goal in state:
                continue  # cost 0 for this goal
            parts = get_parts(goal)
            pred = parts[0]
            args = parts[1:]

            # Heuristic for ixsb7u9c goals (common in instances)
            if pred == "ixsb7u9c" and len(args) == 2:
                r6gw2, pmq6p = args

                # 1) direct presence check (already done above)
                # 2) check for r1frsf5x with matching dbk and chx
                r1_candidates = [
                    (jdovh, r6, pm)
                    for (jdovh, r6, pm) in get_facts("r1frsf5x")
                    if r6 == r6gw2 and pm == pmq6p
                ]

                if r1_candidates:
                    # For each candidate jdovh, check whether ymbljzxy preconditions
                    # (jckjsxca jdovh ?jehs5, wx95011q ?jehs5 ?ijb5b, rrhxwbyt ?stk6x ?ijb5b)
                    possible_cost = 10  # large initial
                    for (jdovh, _, _) in r1_candidates:
                        # all jckjsxca facts for jdovh in state:
                        jehs_list = [jeh for (jeh,) in get_facts("jckjsxca") if (jeh,) in [(x,) for x in [jeh]] and fact_present("jckjsxca", jdovh, jeh)]
                        # simpler: iterate jckjsxca facts that match jdovh:
                        jehs_list = [jeh for (jeh,) in []]  # placeholder to be replaced
                        # collect jehs properly:
                        jehs_list = [jeh for (jeh,) in [(parts[0],) for parts in []]]  # keep placeholder safe

                        # Instead of the above complicated attempt, simply iterate jckjsxca facts:
                        jehs_list = [jeh for (j, jeh) in get_facts("jckjsxca") if j == jdovh]

                        if not jehs_list:
                            # No jckjsxca mapping - we may need an extra operator to set it -> cost 2
                            possible_cost = min(possible_cost, 2)
                            continue

                        # For each candidate jehs5, check compatibility with static wx95011q and rrhxwbyt
                        for jehs5 in jehs_list:
                            # find any ijb5 (uw) such that wx95011q jehs5 ijb5 exists (static or state)
                            ijb_candidates = [ij for (a, ij) in self.static_wx95011q if a == jehs5] + [
                                ij for (a, ij) in get_facts("wx95011q") if a == jehs5
                            ]
                            # If no ijb candidates, assume one needs to be created/changed -> small penalty
                            if not ijb_candidates:
                                possible_cost = min(possible_cost, 2)
                                continue
                            # For each ijb, check rrhxwbyt exists with some stk6x (usually static)
                            for ij in ijb_candidates:
                                rr_exists = any(1 for (stk, b) in self.static_rrhxwbyt if b == ij) or any(
                                    1 for (stk, b) in get_facts("rrhxwbyt") if b == ij
                                )
                                if rr_exists:
                                    # All preconditions for ymbljzxy satisfied -> cost 1
                                    possible_cost = min(possible_cost, 1)
                                else:
                                    possible_cost = min(possible_cost, 2)
                    # If we found any viable candidate set:
                    if possible_cost != 10:
                        estimate_sum += possible_cost
                        continue
                    else:
                        # fallback if no candidate matched
                        estimate_sum += 3
                        continue

                # 3) no r1frsf5x exists: try to see if dlx472ba can be applied or nearly applied
                # dlx472ba pre: efxvv9ax eeuld jdovh, bszmiquc jdovh, wfrx4exs eeuld jdovh,
                #              rycjvfag eeuld pmq6p, jckjsxca jdovh jeh3h, cs7680d7 r6gw2 jeh3h
                # We search for jdovh and jeh3h that link r6gw2 via cs7680d7 and jckjsxca
                possible_cost = 10
                # find all jeh3h such that cs7680d7 r6gw2 jeh3h holds (static or in state)
                jeh3h_candidates = [jeh for (r, jeh) in self.static_cs7680d7 if r == r6gw2] + [
                    jeh for (r, jeh) in get_facts("cs7680d7") if r == r6gw2
                ]
                # For each such jeh3h, find jdovh such that jckjsxca jdovh jeh3h exists
                for jeh3h in jeh3h_candidates:
                    jdovh_list = [jd for (jd, jeh) in get_facts("jckjsxca") if jeh == jeh3h] + [
                        jd for (jd, jeh) in []  # static jckjsxca unlikely; keep placeholder empty
                    ]
                    # If no jdovh found in state, cost will be higher
                    if not jdovh_list:
                        continue
                    for jd in jdovh_list:
                        # check bszmiquc jdovh in state or static
                        bsz_ok = jd in self.static_bszmiquc or fact_present("bszmiquc", jd)
                        # try to find eeuld such that efxvv9ax eeuld jd and rycjvfag eeuld pmq6p
                        eeuld_candidates = [ee for (ee, j) in get_facts("efxvv9ax") if j == jd] + [
                            ee for (ee, j) in self.static_by_pred.get("efxvv9ax", []) if j == jd
                        ]
                        for ee in eeuld_candidates:
                            ryc_ok = any(1 for (t, c) in get_facts("rycjvfag") if t == ee and c == pmq6p) or any(
                                1 for (t, c) in self.static_rycjvfag if t == ee and c == pmq6p
                            )
                            # check wfrx4exs ee jd exists in state
                            wfr_exists = fact_present("wfrx4exs", ee, jd)
                            if bsz_ok and ryc_ok and wfr_exists:
                                # dlx472ba is applicable -> then ymbljzxy -> total 2
                                possible_cost = min(possible_cost, 2)
                            elif bsz_ok and ryc_ok and not wfr_exists:
                                # we can likely create wfrx4exs via p8i30ct9 if iaub97vi & cs7680d7 exist
                                # check whether iaub97vi ee ec and cs7680d7 ec jeh3h exist (ec is some dbk)
                                iaub_ecs = [ec for (t, ec) in get_facts("iaub97vi") if t == ee] + [
                                    ec for (t, ec) in self.static_iaub97vi if t == ee
                                ]
                                for ec in iaub_ecs:
                                    cs_ok = (ec, jeh3h) in self.static_cs7680d7 or fact_present("cs7680d7", ec, jeh3h)
                                    # also p8 needs jckjsxca jd jeh3h (we already have that) and bszmiquc jd
                                    if cs_ok and bsz_ok:
                                        # p8 + dlx + ymbl -> 3
                                        possible_cost = min(possible_cost, 3)
                        # end ee loop
                if possible_cost != 10:
                    estimate_sum += possible_cost
                else:
                    # Fallback conservative estimate for this ixsb7u9c goal.
                    estimate_sum += 4

            # Heuristic for y01m0fhm goals (also appear in instances)
            elif pred == "y01m0fhm" and len(args) == 1:
                lt0bb = args[0]
                # If already present handled above. Check whether sn0ysqqm/ttp8j82e chain can produce it.
                # sn0ysqqm pre: rrhxwbyt stk ijb, rpzrvk8g jd jeh3h, wx95011q jehs ijb, jckjsxca jd jehs
                # If rpzrvk8g already exists for some jd/jeh3 and jckjsxca/jck + wx + rrhx exist -> cost 1
                possible_cost = 10
                rpz_pairs = [(jd, jeh) for (jd, jeh) in get_facts("rpzrvk8g")]
                for (jd, jeh) in rpz_pairs:
                    # need jckjsxca jd jehs such that jehs has wx95011q jehs ijb and rrhxwbyt something ijb
                    jehs_list = [jehs for (j, jehs) in get_facts("jckjsxca") if j == jd]
                    for jehs in jehs_list:
                        # check wx95011q jehs ijb
                        ijb_candidates = [ij for (a, ij) in self.static_wx95011q if a == jehs] + [
                            ij for (a, ij) in get_facts("wx95011q") if a == jehs
                        ]
                        for ij in ijb_candidates:
                            rr_exists = any(1 for (stk, b) in self.static_rrhxwbyt if b == ij) or any(
                                1 for (stk, b) in get_facts("rrhxwbyt") if b == ij
                            )
                            if rr_exists:
                                possible_cost = min(possible_cost, 1)
                            else:
                                possible_cost = min(possible_cost, 2)
                if possible_cost != 10:
                    estimate_sum += possible_cost
                else:
                    # If no rpzrvk8g yet, see if xmjnictn can produce rpzrvk8g with small cost:
                    # xmjnictn pre: z6yvwl7n zmr jd, ipivzc5v jeh3h, p6u7tzz0 zmr, dfhfijzr jehs5, jckjsxca jehs5 jeh3h
                    # If many prerequisites present -> cost 2 else fallback 3
                    # quick heuristic: if there exists jeh3h in ipivzc5v and some zmr in p6u7tzz0 plus jckjsxca linking -> cost 2
                    ipiv = set(a for a in self.static_ipivzc5v) | set(a for (a,) in get_facts("ipivzc5v"))
                    p6u = set(self.static_p6u7tzz0) | set(a for (a,) in get_facts("p6u7tzz0"))
                    jck_pairs = get_facts("jckjsxca")
                    if ipiv and p6u and jck_pairs:
                        estimate_sum += 2
                    else:
                        estimate_sum += 3

            # Generic fallback for other goal predicates
            else:
                # Conservative small finite estimate (at least one action required)
                estimate_sum += 2

        # ensure heuristic returns positive integer for non-goal states
        if estimate_sum == 0:
            return 1
        return int(estimate_sum)