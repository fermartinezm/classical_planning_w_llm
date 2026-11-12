from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return list of tokens of a fact string '(pred a b)' -> ['pred','a','b']."""
    return fact[1:-1].split()


def match(fact, *pattern):
    """Return True if fact matches pattern (supports '*' wildcard)."""
    parts = get_parts(fact)
    return all(fnmatch(p, pat) for p, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A lightweight, domain-dependent heuristic for the secret_domain PDDL domain.
    The heuristic estimates the number of actions required to reach the goal
    by inspecting which goal predicates are not yet satisfied and estimating
    the minimal (approximate) number of operator applications needed to achieve
    them. It is intentionally not admissible: the estimate favors informativeness
    (to guide greedy best-first search) and computational efficiency.

    Assumptions
    -----------
    - The domain predicates are:
        - zzar8qlk (global 0-ary flag)
        - bclhkf4r ?x
        - mdawkki4 ?x
        - gaijkwef ?x ?y
        - chxwa5j4 ?x
    - Operators in the domain create/consume these predicates in a small set
      of recognizable patterns. For example:
        - j2i3gtal adds (gaijkwef A B) when (mdawkki4 A) and (bclhkf4r B) hold.
        - mri4ci1h creates (zzar8qlk), (bclhkf4r X) and (chxwa5j4 X) from (mdawkki4 X).
        - fqixw4x9 and inz39ca6 create/remove mdawkki4 and toggle zzar8qlk.
    - Exact minimal costs are difficult to compute cheaply; therefore the
      heuristic uses small integer approximations for the cost to establish
      individual facts and composes these costs for compound goal facts.
    - Static facts are not required for this heuristic (the example domain has
      none), but the constructor stores them for completeness.

    Heuristic Initialization
    ------------------------
    - Stores the task goals (set of PDDL facts).
    - Stores static facts (if any) for potential future extensions.
    - No expensive preprocessing is performed (keeps initialization cheap).

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    1. Parse the current state into efficient lookup sets:
       - has_zz: whether "(zzar8qlk)" is present.
       - has_bcl[x], has_md[x], has_chx[x] for object-level predicates.
       - has_ga[(a,b)] for gaijkwef facts.

    2. For each goal fact:
       - If it already holds in the state, contribute 0 to the heuristic.
       - Otherwise estimate a small integer cost based on the predicate:
         * Goal (bclhkf4r X)
           - 0 if present
           - 1 if mdawkki4 X present (mri4ci1h produces bclhkf4r X)
           - 2 otherwise (approximate: create mdawkki4 then convert)
         * Goal (chxwa5j4 X)
           - 0 if present
           - 1 if mdawkki4 X present (mri4ci1h produces chxwa5j4)
           - 2 otherwise
         * Goal (mdawkki4 X)
           - 0 if present
           - 1 if (bclhkf4r X and chxwa5j4 X and zzar8qlk) present (fqixw4x9 creates mdawkki4)
           - 1 if there exists a neighbor relation suggesting an operator may produce it
             (we keep a conservative 1 or 2)
           - default 2 otherwise
         * Goal (gaijkwef A B)
           - 0 if present
           - Otherwise, we model creation by j2i3gtal, which requires (mdawkki4 A)
             and (bclhkf4r B). The estimated cost is:
                 cost_md(A) + cost_bcl(B) + 1
             where cost_md/cost_bcl follow the rules above.
         * Goal (zzar8qlk)
           - 0 if present
           - 1 if any mdawkki4 X exists (mri4ci1h creates zzar8qlk)
           - 2 otherwise

    3. Sum the costs for all unsatisfied goal facts to produce the final heuristic
       value. This is a simple additive heuristic; it does not attempt to detect
       action reuse between goals (admissibility would require more care).
       The values chosen (0,1,2, sometimes +1) are small integers that capture
       readily available shortcuts in the domain (e.g., turning mdawkki4 into
       bclhkf4r or chxwa5j4 is cheap), and larger values reflect the need to do
       a small sequence of actions.

    Notes
    -----
    - The heuristic returns 0 exactly when all goals are satisfied.
    - It always returns a finite integer for solvable states because we bound
      estimated costs by small integers.
    - The heuristic is designed to be fast to compute and to provide useful
      guidance to greedy best-first search in this domain.
    """

    def __init__(self, task):
        self.goals = set(task.goals)
        self.static = set(task.static) if hasattr(task, "static") else set()

    def __call__(self, node):
        state = node.state  # frozenset of fact strings

        # Quick goal check: if all goals satisfied -> heuristic 0
        if self.goals <= state:
            return 0

        # Build fast lookup structures
        has_zz = "(zzar8qlk)" in state

        has_bcl = set()
        has_md = set()
        has_chx = set()
        has_ga = set()  # set of tuples (a,b)

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "bclhkf4r" and len(parts) >= 2:
                has_bcl.add(parts[1])
            elif pred == "mdawkki4" and len(parts) >= 2:
                has_md.add(parts[1])
            elif pred == "chxwa5j4" and len(parts) >= 2:
                has_chx.add(parts[1])
            elif pred == "gaijkwef" and len(parts) >= 3:
                has_ga.add((parts[1], parts[2]))

        # Helper cost estimators for atomic facts (small integer approximations).
        def cost_bcl(x):
            if x in has_bcl:
                return 0
            if x in has_md:
                return 1  # mri4ci1h creates bclhkf4r from mdawkki4
            # otherwise approximate cost to create md then convert
            return 2

        def cost_chx(x):
            if x in has_chx:
                return 0
            if x in has_md:
                return 1  # mri4ci1h creates chxwa5j4 from mdawkki4
            return 2

        def cost_md(x):
            if x in has_md:
                return 0
            # fqixw4x9 can produce mdawkki4 X if zzar8qlk, bclhkf4r X and chxwa5j4 X hold.
            if has_zz and x in has_bcl and x in has_chx:
                return 1
            # inz39ca6 can produce mdawkki4 X if bclhkf4r X and gaijkwef X Y and zzar8qlk
            # We conservatively check if any gaijkwef from X exists and bcl present -> 1 or 2
            for (a, b) in has_ga:
                if a == x and x in has_bcl and has_zz:
                    return 1
            # Otherwise estimate cost 2 to create a supporting mdawkki4
            return 2

        total = 0
        for g in self.goals:
            parts = get_parts(g)
            pred = parts[0]
            if pred == "bclhkf4r" and len(parts) >= 2:
                obj = parts[1]
                total += cost_bcl(obj)
            elif pred == "chxwa5j4" and len(parts) >= 2:
                obj = parts[1]
                total += cost_chx(obj)
            elif pred == "mdawkki4" and len(parts) >= 2:
                obj = parts[1]
                total += cost_md(obj)
            elif pred == "gaijkwef" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                if (a, b) in has_ga:
                    continue
                # j2i3gtal produces (gaijkwef a b) and requires mdawkki4 a and bclhkf4r b.
                # estimate cost as cost_md(a) + cost_bcl(b) + 1 (for j2i3gtal itself)
                total += cost_md(a) + cost_bcl(b) + 1
            elif pred == "zzar8qlk":
                if has_zz:
                    continue
                # mri4ci1h produces zzar8qlk when mdawkki4 x exists, so cost 1 if any md exists
                total += 1 if len(has_md) > 0 else 2
            else:
                # Unknown predicate: fall back to 1 (unknown but needed)
                total += 1

        # Ensure heuristic is at least 1 when goals not satisfied (and not zeroed above)
        return max(1, total)