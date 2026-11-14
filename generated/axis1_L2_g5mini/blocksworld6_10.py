from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return list of tokens in a PDDL fact string, e.g. "(pred a b)" -> ["pred", "a", "b"]."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A lightweight domain-dependent heuristic for the secret_domain (dnbnd6kg).
    The heuristic estimates the number of actions required to achieve the goal
    facts by analyzing which core predicates are currently true and by using
    short, domain-informed templates of how the operators can establish those
    facts. It is not admissible but aims to be informative and fast to compute.

    Assumptions
    -----------
    - Facts are represented as strings in the form "(predicate arg1 arg2 ...)".
    - The number of "tokens" (bclhkf4r) in the world is conserved by the domain
      operations (tokens are moved/transformed between objects).
    - The main predicates the domain manipulates are:
        * zzar8qlk            (global boolean toggle)
        * bclhkf4r ?x        (object holds a token)
        * mdawkki4 ?x        (object in a transitional state)
        * chxwa5j4 ?x        (object has marker)
        * gaijkwef ?x ?y     (directed relation between objects)
    - Typical operator patterns observed:
        * mri4ci1h(mdawkki4 X) -> sets zzar8qlk, bclhkf4r X, chxwa5j4 X  (1 action)
        * fqixw4x9(zzar8qlk, bclhkf4r X, chxwa5j4 X) -> mdawkki4 X  (1 action)
        * inz39ca6(bclhkf4r A, zzar8qlk, gaijkwef A B) -> transfers token A->B,
          leaves mdawkki4 A (1 action)
        * j2i3gtal(bclhkf4r Y, mdawkki4 X) -> creates gaijkwef X Y and bclhkf4r X (1 action)
    - The heuristic uses these shortest, domain-typical sequences as building
      blocks for estimating costs. It does not search exhaustively but applies
      small pattern-based rules to estimate the minimal number of operations.

    Heuristic Initialization
    ------------------------
    - Store the task goal facts (self.goals).
    - No other static facts are required by this heuristic (the domain files
      we have supply no relevant static information).

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given state, the heuristic proceeds as follows:

    1. Quick goal-check:
       - If all goal facts are already true in the state, return 0.

    2. Parse the state into fast lookup structures:
       - zz_flag: whether "(zzar8qlk)" is present.
       - bcl_set: objects currently holding bclhkf4r.
       - md_set: objects currently holding mdawkki4.
       - chx_set: objects currently holding chxwa5j4.
       - ga_set: set of (src, dst) pairs for gaijkwef.
       - ga_rev: mapping dst -> set(src) to quickly find incoming edges.

    3. For each goal fact that is missing, apply a small pattern-based cost
       estimation depending on the predicate:
       - gaijkwef X Y:
         * If already present cost 0.
         * If mdawkki4 X and bclhkf4r Y present -> j2i3gtal creates it in 1 action.
         * If either mdawkki4 X or bclhkf4r Y is present -> estimate 2 actions.
         * Otherwise estimate 3 actions.
       - bclhkf4r X:
         * If present cost 0.
         * If mdawkki4 X present -> mri4ci1h produces bclhkf4r in 1 action.
         * If there exists some Y with bclhkf4r and gaijkwef Y->X and zz_flag True
           -> inz39ca6 can transfer in 1 action; if zz_flag False add 1 (conservative).
         * If some other bclhkf4r exists (token can be moved indirectly) -> estimate 2.
         * Otherwise estimate 3.
       - chxwa5j4 X:
         * If present cost 0.
         * If mdawkki4 X present -> mri4ci1h produces chxwa5j4 in 1 action.
         * If object has bclhkf4r -> we estimate 2 actions (flip to mdawkki4 and back to create chx).
         * Otherwise estimate 3.
       - mdawkki4 X:
         * If present cost 0.
         * If bclhkf4r X present -> many domain operators can produce mdawkki4 from a bcl
           in one action variant (we conservatively estimate 1).
         * Otherwise estimate 2.

    4. Sum the individual estimates for all missing goal facts and return the sum.
       The estimate is small, finite and biased toward states that already
       satisfy strong preconditions (md, bcl, chx, or appropriate ga links).

    Notes on design choices
    -----------------------
    - The heuristic is intentionally simple and fast: it inspects the local
      facts relevant to each goal and uses a few domain-specific rules that
      correspond to the shortest operators that can establish those facts.
    - It is not exhaustive: it does not perform a full reachability analysis
      (which would be more expensive) and therefore is not guaranteed to be
      admissible. It does, however, capture common, short transformations
      used by the domain operators which gives informed guidance in practice.
    """

    def __init__(self, task):
        # Store goal facts (frozenset of strings)
        self.goals = task.goals
        # Static facts are not required for this heuristic, but keep reference if needed.
        self.static = task.static

    def __call__(self, node):
        state = node.state

        # 1) Return zero if goals already satisfied
        if self.goals <= state:
            return 0

        # 2) Parse state into convenient structures
        zz_flag = "(zzar8qlk)" in state
        bcl_set = set()
        md_set = set()
        chx_set = set()
        ga_set = set()
        ga_rev = dict()  # dst -> set(src)

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "bclhkf4r" and len(parts) >= 2:
                bcl_set.add(parts[1])
            elif pred == "mdawkki4" and len(parts) >= 2:
                md_set.add(parts[1])
            elif pred == "chxwa5j4" and len(parts) >= 2:
                chx_set.add(parts[1])
            elif pred == "gaijkwef" and len(parts) >= 3:
                src, dst = parts[1], parts[2]
                ga_set.add((src, dst))
                ga_rev.setdefault(dst, set()).add(src)

        total_cost = 0

        # 3) For each missing goal, estimate cost using small rule set
        for goal in self.goals:
            if goal in state:
                continue  # already satisfied

            parts = get_parts(goal)
            if not parts:
                continue
            pred = parts[0]

            # gaijkwef X Y
            if pred == "gaijkwef" and len(parts) >= 3:
                src, dst = parts[1], parts[2]
                if (src, dst) in ga_set:
                    cost = 0
                else:
                    # Best case: md(src) and bcl(dst) -> j2i3gtal creates ga(src,dst) in 1
                    if src in md_set and dst in bcl_set:
                        cost = 1
                    # If either precondition exists, estimate 2
                    elif src in md_set or dst in bcl_set:
                        cost = 2
                    else:
                        # neither precondition present, need a sequence of operations
                        cost = 3
                total_cost += cost

            # bclhkf4r X
            elif pred == "bclhkf4r" and len(parts) >= 2:
                obj = parts[1]
                if obj in bcl_set:
                    cost = 0
                elif obj in md_set:
                    # mri4ci1h: mdawkki4 -> bclhkf4r (and sets zz and chx) in one action
                    cost = 1
                else:
                    # Check for a direct transferable token: some Y with bcl and a link Y->obj
                    transferers = ga_rev.get(obj, set())
                    found_transferable = any((y in bcl_set) for y in transferers)
                    if found_transferable:
                        # inz39ca6 requires zz_flag True. If it's False, add 1 conservatively.
                        cost = 1 if zz_flag else 2
                    elif bcl_set:
                        # tokens exist elsewhere, estimate 2 steps to move one here (short sequence)
                        cost = 2
                    else:
                        # No tokens anywhere -> costly (may be unsolvable); estimate conservatively
                        cost = 3
                total_cost += cost

            # chxwa5j4 X
            elif pred == "chxwa5j4" and len(parts) >= 2:
                obj = parts[1]
                if obj in chx_set:
                    cost = 0
                elif obj in md_set:
                    # mri4ci1h creates chx from md in 1 action
                    cost = 1
                elif obj in bcl_set:
                    # likely need to flip bcl->md->mri to create chx; estimate 2
                    cost = 2
                else:
                    cost = 3
                total_cost += cost

            # mdawkki4 X
            elif pred == "mdawkki4" and len(parts) >= 2:
                obj = parts[1]
                if obj in md_set:
                    cost = 0
                elif obj in bcl_set:
                    # some operators can produce md from bcl in short sequences (conservatively 1)
                    cost = 1
                else:
                    # no direct precondition present; estimate small cost to bring token/conditions here
                    cost = 2
                total_cost += cost

            else:
                # Unknown predicate in goal: fall back to 1 (should be finite)
                total_cost += 1

        # Ensure returned value is integer and finite
        if total_cost < 0:
            total_cost = 0
        return int(total_cost)