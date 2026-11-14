from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate obj1 obj2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the 'rb7qr1fs' (secret_domain) domain.

    Summary
    -------
    This heuristic estimates the number of actions needed to achieve all conjunctive goal predicates.
    The domain involves complex relations between objects of different types, with goals focusing on certain predicates 
    involving objects of types `dbk5i2a5` and `chxau8fy` (visible in goals like `(ixsb7u9c ...)`) and `y01m0fhm` predicates over `uwq7ubjf` objects.
    It leverages the presence of these key predicates as indicators of progress and estimates how many such facts remain to reach all goals.

    Assumptions
    -----------
    - The goal conjunctively requires certain `(ixsb7u9c <dbk5i2a5> <chxau8fy>)` facts and `(y01m0fhm <uwq7ubjf>)` facts to be true.
    - Achieving each missing goal fact requires at least one action.
    - Facts like `(r1frsf5x <rb7qr1fs> <dbk5i2a5> <chxau8fy>)` represent necessary intermediate steps before `(ixsb7u9c ...)` can be true.
    - Similarly, predicates `(jckjsxca <rb7qr1fs> <uwq7ubjf>)` and `(w1yp6zl7 <rb7qr1fs> <uwq7ubjf>)` relate to enabling `(y01m0fhm)` predicates.
    - The heuristic is not guaranteed to be admissible but provides a meaningful, efficient estimate guiding the search.

    Heuristic Initialization
    ------------------------
    - Extract the goal facts into sets by predicate to identify which `(ixsb7u9c ...)` and `(y01m0fhm ...)` need to be achieved.
    - Extract static facts linking objects and static relations (like membership to various sets).
    - Track sets of objects by type based on static facts for filtering and reasoning, e.g., all `uwq7ubjf` with `(a8x3330w)`, all `rb7qr1fs` with `(bszmiquc)` or `(a298chgk)`.
    - Precompute relations from static facts that indicate possible object interactions, e.g., `(b2mwd6x4 <rb7qr1fs> <uwq7ubjf> <uwq7ubjf>)` edges for counting steps.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify how many goal `(ixsb7u9c ...)` facts are still missing in the current state.
       - Each missing `(ixsb7u9c ?r6gw2 ?pmq6p)` requires at least one action to produce.
       - Often an `(r1frsf5x ?jdovh ?r6gw2 ?pmq6p)` fact must exist before `(ixsb7u9c ...)` is made true, so missing these imply additional needed actions.
       - Approximate by counting missing `(ixsb7u9c ...)` plus missing supporting `(r1frsf5x ...)` facts related to the goal.

    2. For `(y01m0fhm ?lt0bb)` goal facts:
       - Count how many are still missing.
       - Achieving `(y01m0fhm ?lt0bb)` requires relevant `(w1yp6zl7 ?jdovh ?lt0bb)` and `(jckjsxca ?jdovh ?lt0bb)` intermediate facts.
       - Approximate by counting missing `(y01m0fhm)` plus the needed supporting facts `(w1yp6zl7 ...)` or `(jckjsxca ...)` if relevant.

    3. Additionally check goals of predicates like `(rpzrvk8g ?jdovh ?lt0bb)`, if any, and the requirement of `(jckjsxca ?jdovh ?lt0bb)` presence to support `(w1yp6zl7 ...)` or `(y01m0fhm ...)`.

    4. Estimate actions by summing counts of missing goal facts and their key dependencies as a proxy of the minimal number of steps needed.

    5. This heuristic focuses on counting missing key "product" predicates and factors in missing intermediate predicates that are blocking goal achievement.

    6. This simple counting approach is efficient to compute and effectively guides BFS to focus on achieving these core predicates.

    Note: The heuristic always returns 0 on goal states by definition of no missing goals.
    """

    def __init__(self, task):
        self.goals = task.goals  # Set of goal facts (strings).

        # Extract goal ixsb7u9c pairs: (ixsb7u9c <dbk5i2a5> <chxau8fy>)
        self.goal_ixsb7u9c = set()
        # Extract goal y01m0fhm singletons: (y01m0fhm <uwq7ubjf>)
        self.goal_y01m0fhm = set()

        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            pred = parts[0]
            if pred == "ixsb7u9c" and len(parts) == 3:
                self.goal_ixsb7u9c.add((parts[1], parts[2]))
            elif pred == "y01m0fhm" and len(parts) == 2:
                self.goal_y01m0fhm.add(parts[1])

        # Prepare static indices for efficiency:
        # Map rb7qr1fs objects having (bszmiquc ?jdovh) predicate:
        self.objects_bszmiquc = set()
        # Map rb7qr1fs objects having (a298chgk ?jdovh) predicate:
        self.objects_a298chgk = set()
        # Map (b2mwd6x4 ?jdovh ?uwq7ubjf ?uwq7ubjf) edges for reachability counting:
        self.b2mwd6x4_relations = set()

        for fact in task.static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "bszmiquc" and len(parts) == 2:
                self.objects_bszmiquc.add(parts[1])
            elif pred == "a298chgk" and len(parts) == 2:
                self.objects_a298chgk.add(parts[1])
            elif pred == "b2mwd6x4" and len(parts) == 4:
                self.b2mwd6x4_relations.add((parts[1], parts[2], parts[3]))

    def __call__(self, node):
        state = node.state

        # Quick check: if goal already reached, heuristic = 0.
        if self.goals <= state:
            return 0

        # Extract current sets of achieved facts from the state:
        # - Existing (ixsb7u9c ?dbk5i2a5 ?chxau8fy)
        current_ixsb7u9c = set()
        # - Existing (r1frsf5x ?rb7qr1fs ?dbk5i2a5 ?chxau8fy)
        current_r1frsf5x = set()
        # - Existing (y01m0fhm ?uwq7ubjf)
        current_y01m0fhm = set()
        # - Existing (w1yp6zl7 ?rb7qr1fs ?uwq7ubjf)
        current_w1yp6zl7 = set()
        # - Existing (jckjsxca ?rb7qr1fs ?uwq7ubjf)
        current_jckjsxca = set()
        # - Existing (rpzrvk8g ?rb7qr1fs ?uwq7ubjf)
        current_rpzrvk8g = set()

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "ixsb7u9c" and len(parts) == 3:
                current_ixsb7u9c.add((parts[1], parts[2]))
            elif pred == "r1frsf5x" and len(parts) == 4:
                current_r1frsf5x.add((parts[1], parts[2], parts[3]))
            elif pred == "y01m0fhm" and len(parts) == 2:
                current_y01m0fhm.add(parts[1])
            elif pred == "w1yp6zl7" and len(parts) == 3:
                current_w1yp6zl7.add((parts[1], parts[2]))
            elif pred == "jckjsxca" and len(parts) == 3:
                current_jckjsxca.add((parts[1], parts[2]))
            elif pred == "rpzrvk8g" and len(parts) == 3:
                current_rpzrvk8g.add((parts[1], parts[2]))

        # Count missing ixsb7u9c goal facts
        missing_ixsb7u9c = self.goal_ixsb7u9c - current_ixsb7u9c

        # For each missing ixsb7u9c fact, check if its corresponding r1frsf5x fact exists.
        # Missing r1frsf5x means more actions needed to produce ixsb7u9c.
        missing_r1frsf5x = set()
        for (r6gw2, pmq6p) in missing_ixsb7u9c:
            # The corresponding r1frsf5x must have the same dbk5i2a5 and chxau8fy, linked to some rb7qr1fs (unknown here).
            # We approximate by counting all missing r1frsf5x with the same ?r6gw2 and ?pmq6p in goals or state.
            # Because r1frsf5x facts link a rb7qr1fs node, but goal doesn't specify which.
            # We check if any current r1frsf5x has this pair.
            found = False
            for r in self.objects_bszmiquc:  # candidate rb7qr1fs from static facts with bszmiquc (available nodes)
                if (r, r6gw2, pmq6p) in current_r1frsf5x:
                    found = True
                    break
            if not found:
                missing_r1frsf5x.add((r6gw2, pmq6p))

        # Heuristic component 1: sum missing ixsb7u9c + missing corresponding r1frsf5x
        cost_ixsb7u9c = len(missing_ixsb7u9c) + len(missing_r1frsf5x)

        # Count missing y01m0fhm goal facts
        missing_y01m0fhm = self.goal_y01m0fhm - current_y01m0fhm

        # For each missing y01m0fhm(?lt0bb), consider also required jckjsxca and w1yp6zl7 facts to produce it:
        # (y01m0fhm ?lt0bb) <- possibly from (w1yp6zl7 ?jdovh ?lt0bb) with (jckjsxca ?jdovh ?lt0bb)
        cost_y01m0fhm = 0
        for lt0bb in missing_y01m0fhm:
            # Check if there exists a rb7qr1fs node rjdovh with (w1yp6zl7 rjdovh lt0bb) and (jckjsxca rjdovh lt0bb)
            found_support = False
            for r in self.objects_bszmiquc:
                if ((r, lt0bb) in current_w1yp6zl7) and ((r, lt0bb) in current_jckjsxca):
                    found_support = True
                    break
            if not found_support:
                # Count one for y01m0fhm and one extra if no support directly found (to approximate cost to create support)
                cost_y01m0fhm += 2
            else:
                # Only one needed for y01m0fhm itself
                cost_y01m0fhm += 1

        # Additionally, some goals may involve (rpzrvk8g ?jdovh ?lt0bb), which could be a prerequisite for w1yp6zl7
        # Count missing (rpzrvk8g) for needed (w1yp6zl7)
        missing_rpzrvk8g = set()
        for (jdovh, lt0bb) in current_w1yp6zl7:
            # If (rpzrvk8g jdovh lt0bb) missing, this might increase cost, so we roughly check if rpzrvk8g required
            if (jdovh, lt0bb) not in current_rpzrvk8g:
                missing_rpzrvk8g.add((jdovh, lt0bb))
        cost_rpzrvk8g = len(missing_rpzrvk8g)  # count added cost for missing rpzrvk8g facts

        # Sum all costs for heuristic estimate
        heuristic_value = cost_ixsb7u9c + cost_y01m0fhm + cost_rpzrvk8g

        # Ensure heuristic is never zero unless goal is reached
        if heuristic_value == 0:
            # But goal not reached, so minimal estimate of 1
            heuristic_value = 1

        return heuristic_value
