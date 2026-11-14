from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the parts of fact '(pred arg1 arg2 ...)' as a list of strings."""
    return fact[1:-1].split()


def match(fact, *pattern):
    """
    Test whether a fact (string) matches a pattern of tokens (wildcards '*' allowed).
    Example: match('(at x y)', 'at', '*', 'y') -> True
    """
    parts = get_parts(fact)
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic tailored to the "secret_domain" (rb7qr1fs).
    The main goals seen in example instances are of the form
      (ixsb7u9c <dbk5i2a5> <chxau8fy>)
    and sometimes goals of the form
      (y01m0fhm <uwq7ubjf>).
    The domain contains a short production pipeline:
      - create an intermediate fact r1frsf5x for a given <rb7qr1fs> and <dbk5i2a5>/<chxau8fy>,
        then use an operator (ymbljzxy) to produce the desired ixsb7u9c if supporting
        mappings (jckjsxca and suitable wx95011q/rrhxwbyt pairs) exist.
    The heuristic therefore focuses on detecting which goal facts are missing and
    estimating the minimal number of domain actions needed to produce each missing goal,
    taking into account the presence of helpful intermediate facts already present in
    the current state and static facts.

    Assumptions
    -----------
    - The primary atomic goals to estimate are ixsb7u9c(...) and y01m0fhm(...).
    - Creating an ixsb7u9c typically requires:
        * an r1frsf5x(...) fact (if already present, cheaper),
        * a suitable jckjsxca mapping for the same rb7qr1fs object,
        * and compatible wx95011q / rrhxwbyt pairs (many of which are static).
      If r1frsf5x is missing, the heuristic assumes we need to perform the
      sequence that produces it (p8i30ct9 then dlx472ba), so it assigns a larger
      base cost for creating r1frsf5x from scratch.
    - For y01m0fhm goals, the action sn0ysqqm produces them; if rpzrvk8g is already
      present the cost is lower; otherwise a small creation cost is assumed.
    - The heuristic is not required to be admissible; it trades admissibility for
      a more aggressive cost estimate that helps prioritize promising states.
    - The heuristic treats different goal facts as (mostly) independent and sums
      their estimated costs. Some actions in the domain can produce multiple goal
      facts or share subgoals; this interaction is ignored for efficiency.

    Heuristic Initialization
    ------------------------
    In the constructor we:
    - Keep the set of goal facts (self.goals).
    - Extract static relationships that are used repeatedly:
        * wx95011q pairs (uw -> uw2)
        * rrhxwbyt pairs (stk -> ijb)
        * cs7680d7 facts (dbk5i2a5-location compatibility)
        * jckjsxca facts that are static (but jckjsxca may appear also in state)
      These are stored in dictionary / set structures for quick lookup.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given node/state the heuristic computes an estimate as follows:

    1. If the current state already satisfies all goal facts, return 0.

    2. Parse the current state and construct quick-lookup tables for:
       - which ixsb7u9c facts are already true (so they need no work),
       - r1frsf5x facts (triples (jdovh, r6gw2, pmq6p)),
       - jckjsxca mappings present in the state (map from rb7qr1fs -> set(uwq)),
       - wx95011q pairs currently true (union of static and state),
       - rrhxwbyt pairs (union of static and state),
       - rpzrvk8g facts (for y01m0fhm support).

    3. For each missing goal fact (goals not in the current state) estimate a local cost:

       - Missing ixsb7u9c(r6, pm):
         a) If there exists r1frsf5x(jd, r6, pm) in the state: base_cost = 1
            (one application of ymbljzxy is typically sufficient provided the
             jckjsxca/wx/rrhxwbyt chain exists).
         b) Else: base_cost = 3 (conservative estimate: p8i30ct9 + dlx472ba to
            create r1frsf5x plus some overhead).
         c) Chain check (for the required jckjsxca/wx/rrhxwbyt triple):
            - For each candidate jdovh that either already has r1frsf5x for the
              (r6, pm) pair or (when r1frsf5x missing) for any jdovh that has
              jckjsxca mappings, examine:
                * whether there exists a jckjsxca(jd, uw) in the state,
                * whether there is a wx95011q(uw, uw2) (static or state),
                * whether there is an rrhxwbyt(stk, uw2) (static or state).
            - If the full chain exists for some jdovh, extra_cost = 0.
            - Else if jdovh has at least a jckjsxca but the chain misses one link, extra_cost = 1
              (we assume one reconfiguration action like xvldpwsb or a small fix).
            - Else extra_cost = 2 (no jckjsxca for jdovh; we assume creation or reconfiguration
              will cost extra).

         The estimated cost for this missing ixsb7u9c is base_cost + min_extra_cost_over_candidates.

       - Missing y01m0fhm(u):
         a) If rpzrvk8g(...) already exists that matches a usable pair -> base_cost = 1
            (sn0ysqqm can produce y01m0fhm if its other preconditions are met).
         b) Else base_cost = 2 (a small creation step for rpzrvk8g or related support).
         c) If jckjsxca linking a candidate rb7qr1fs for the required uw exists and the wx/rrhxwbyt
            pattern is satisfied -> extra_cost = 0; else extra_cost = 1.

         Cost for y01m0fhm is base_cost + extra_cost.

    4. Sum the estimated costs for all missing goal facts and return that sum.

    Notes on Design
    ---------------
    - The heuristic intentionally uses a small, domain-aware causal model rather than
      a full reachability graph (to remain efficient). It inspects the presence of
      the intermediate r1frsf5x and jckjsxca/wx/rrhxwbyt chains since these are the
      dominant structures to reach ixsb7u9c goals in the provided domain.
    - Because this heuristic is not required to be admissible, the numeric costs are
      tuned to bias the search towards states that already have useful intermediates.
    """

    def __init__(self, task):
        """
        Initialize the heuristic with static information and the goal list.

        Parameters:
        - task: the Task object (see Task in code-file-task)
        """
        self.goals = set(task.goals)
        # store static facts for quick lookup
        static = task.static

        # Build static lookups for wx95011q and rrhxwbyt (many instances contain these statically)
        self.static_wx = {}  # uw -> set(uw2)
        self.static_rrh = set()  # (stk, ijb)
        self.static_jck = {}  # jdovh -> set(uw) (jckjsxca that are static)
        self.static_rpz = set()  # rpzrvk8g facts (jdovh, uw)
        for fact in static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "wx95011q":
                _, a, b = parts
                self.static_wx.setdefault(a, set()).add(b)
            elif pred == "rrhxwbyt":
                _, stk, ijb = parts
                self.static_rrh.add((stk, ijb))
            elif pred == "jckjsxca":
                _, jd, uw = parts
                self.static_jck.setdefault(jd, set()).add(uw)
            elif pred == "rpzrvk8g":
                _, jd, uw = parts
                self.static_rpz.add((jd, uw))
        # We will also consult other static predicates if needed at call-time.

    def __call__(self, node):
        """
        Compute the heuristic estimate for a given search node.

        - node.state is a frozenset of fact-strings representing the current state.
        - The heuristic returns an integer score (smaller => more promising).
        """
        state = node.state

        # If the node already satisfies all goals, return 0.
        if self.goals <= state:
            return 0

        # Parse the current state into convenient lookup structures:
        current_ix = set()  # set of (r6, pm) that already have ixsb7u9c
        current_r1 = []  # list of tuples (jdovh, r6, pm)
        jck_map = {}  # jdovh -> set(uw) (jckjsxca facts present in state)
        wx_map = {}  # union of static and current wx95011q: uw -> set(uw2)
        rrh_set = set(self.static_rrh)  # union of static and current rrhxwbyt
        rpz_set = set(self.static_rpz)  # rpzrvk8g present (static + state)
        # collect rpzrvk8g in state too
        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "ixsb7u9c":
                _, r6, pm = parts
                current_ix.add((r6, pm))
            elif pred == "r1frsf5x":
                _, jd, r6, pm = parts
                current_r1.append((jd, r6, pm))
            elif pred == "jckjsxca":
                _, jd, uw = parts
                jck_map.setdefault(jd, set()).add(uw)
            elif pred == "wx95011q":
                _, a, b = parts
                wx_map.setdefault(a, set()).add(b)
            elif pred == "rrhxwbyt":
                _, stk, ijb = parts
                rrh_set.add((stk, ijb))
            elif pred == "rpzrvk8g":
                _, jd, uw = parts
                rpz_set.add((jd, uw))

        # Merge static wx into wx_map
        for a, bs in self.static_wx.items():
            wx_map.setdefault(a, set()).update(bs)

        # Build quick index: for a given (r6, pm) which jdovh already has r1frsf5x?
        r1_index = {}  # (r6, pm) -> set(jdovh)
        for (jd, r6, pm) in current_r1:
            r1_index.setdefault((r6, pm), set()).add(jd)

        # Collect all jdovh names that have some jckjsxca (static or state)
        all_jd_candidates = set(jck_map.keys()) | set(self.static_jck.keys())

        # Helper to check whether there is a usable jck/wx/rrhx chain for a jdovh
        def has_chain_for_jd(jd):
            """
            Return:
              0 if full chain (jckjsxca(jd, uw) and wx95011q(uw, uw2) and rrhxwbyt(stk, uw2))
                exists for some uw and some uw2 (rrhxwbyt may involve any stk).
              1 if partial chain: jd has jckjsxca(jd, uw) but no full chain
              2 if no jckjsxca known for jd
            """
            # get all uw known for jd (state or static)
            uws = set()
            if jd in jck_map:
                uws.update(jck_map[jd])
            if jd in self.static_jck:
                uws.update(self.static_jck[jd])
            if not uws:
                return 2
            # For each uw check whether there is some uw2 with wx(uw,uw2) and rrh(*, uw2)
            for uw in uws:
                if uw in wx_map:
                    for uw2 in wx_map[uw]:
                        # if some rrh pair exists with ijb == uw2
                        for (stk, ijb) in rrh_set:
                            if ijb == uw2:
                                return 0
            # has jck but no full chain
            return 1

        total_cost = 0

        # Process each goal fact individually and sum estimated costs
        for goal in self.goals:
            if goal in state:
                continue  # already satisfied

            parts = get_parts(goal)
            pred = parts[0]

            if pred == "ixsb7u9c":
                _, r6, pm = parts
                # If some r1frsf5x already exists for this pair, small base cost
                if (r6, pm) in r1_index:
                    base_cost = 1  # one ymbljzxy action likely
                    # try to find best candidate jdovh among those producing r1frsf5x
                    candidates = r1_index[(r6, pm)]
                    best_extra = 10
                    for jd in candidates:
                        extra = has_chain_for_jd(jd)
                        if extra < best_extra:
                            best_extra = extra
                    # If somehow no chain for these jd's, consider other jd candidates too
                    if best_extra == 10:
                        # fallback: try any jd that has jckjsxca (static or state)
                        best_extra = min((has_chain_for_jd(jd) for jd in all_jd_candidates), default=2)
                else:
                    # r1frsf5x missing: assume producing it needs ~2-3 actions (p8 + dlx (+setup))
                    base_cost = 3
                    # pick best jdovh candidate that would be used to produce r1frsf5x
                    # prefer jd with jckjsxca chain already present
                    best_extra = min((has_chain_for_jd(jd) for jd in all_jd_candidates), default=2)

                # Normalize the extra to an action estimate:
                if best_extra == 0:
                    extra_cost = 0
                elif best_extra == 1:
                    extra_cost = 1
                else:
                    extra_cost = 2

                total_cost += base_cost + extra_cost

            elif pred == "y01m0fhm":
                # For y01m0fhm(uw): sn0ysqqm produces it but needs rpzrvk8g and jck/wx/rrh support
                _, uw = parts
                # If rpzrvk8g already present for some jdovh and this uw, cost smaller
                rpz_ok = any(jd for (jd, u) in rpz_set if u == uw)
                base_cost = 1 if rpz_ok else 2
                # Check whether there exists some jd with chain for jd->uw (jckjsxca and wx/rrh wrt some uw2)
                # We accept any jd that has jckjsxca(jd, some_jehs5) and wx/jck chain mapping to uw (ijb)
                # Conservative check: test jd that has jckjsxca and whether a chain exists to some ijb that matches rrh
                # If there exists any jd with full chain, extra 0, if only jck present extra 1 else extra 2
                best_extra = 2
                for jd in all_jd_candidates:
                    val = has_chain_for_jd(jd)
                    if val < best_extra:
                        best_extra = val
                        if best_extra == 0:
                            break
                if best_extra == 0:
                    extra_cost = 0
                elif best_extra == 1:
                    extra_cost = 1
                else:
                    extra_cost = 1  # be slightly optimistic for y01m0fhm (creation of rpz/other small costs)
                total_cost += base_cost + extra_cost

            else:
                # Generic fallback for any other goal predicate:
                # If goal not present, assume a small cost (1) to create it (domain is STRIPS with mostly unary actions).
                total_cost += 1

        # Ensure the heuristic returns a non-zero positive integer when there are missing goals.
        # (We already returned 0 above if all goals satisfied.)
        return int(total_cost)