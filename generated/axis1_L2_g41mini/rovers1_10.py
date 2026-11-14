from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - fact: string with PDDL fact, e.g. "(predicate obj1 obj2)".
    - args: pattern components where "*" is a wildcard.
    Returns True if the pattern matches the fact.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain (rb7qr1fs).
    
    Summary:
    ----------
    This heuristic estimates the number of actions required to fulfill all conjunctive goal facts,
    primarily focusing on achieving the (ixsb7u9c ?r6gw2 ?pmq6p) facts present in the goal state.
    These facts represent a required relationship between ?r6gw2 (dbk5i2a5 type objects) and ?pmq6p (chxau8fy type objects).
    The heuristic counts how many goal (ixsb7u9c ...) facts are not yet achieved, as these need explicit action applications 
    (at least one action per missing such fact) to be made true.

    Assumptions:
    ----------
    - The main goal is a conjunction of (ixsb7u9c ?r6gw2 ?pmq6p) facts.
    - Achieving each such goal fact requires at least one action (ymbljzxy or dlx472ba).
    - Other goal conditions are derived or less costly; focusing on these captures the core work needed.
    - The heuristic is non-admissible but informative to guide search efficiently.
    - Static facts such as (rycjvfag ?eeuld ?pmq6p), and (cs7680d7 ?r6gw2 ?lt0bb) are used to relate objects to help check possible achievements.

    Heuristic Initialization:
    ----------
    - Extract all goal facts.
    - Identify goal (ixsb7u9c ?r6gw2 ?pmq6p) facts and store their argument pairs.
    - Extract static facts to build:
      * For each pmq6p (chxau8fy), which eeuld (tom0ncko) relate via (rycjvfag ?eeuld ?pmq6p).
      * For each r6gw2 (dbk5i2a5), which lt0bb (uwq7ubjf) relate via (cs7680d7 ?r6gw2 ?lt0bb).
      * For each jdovh (rb7qr1fs) and lt0bb (uwq7ubjf), whether (jckjsxca ?jdovh ?lt0bb) holds.
      * For pairs of jdovh and uwq7ubjf, the links via (b2mwd6x4 ?jdovh ?lt0bb ?jehs5) and (wx95011q ?lt0bb ?jeh3h).
      This information can help estimate partial progress or dependencies.

    Step-By-Step Thinking for Computing Heuristic:
    ----------
    1. For each goal (ixsb7u9c r6gw2 pmq6p):
       - Check if it is already achieved in the current state.
       - If achieved, no cost added.
       - If not achieved:
         a) Check if any corresponding supporting facts exist that indicate partial progress,
            like (r1frsf5x ?jdovh ?r6gw2 ?pmq6p) or (wfrx4exs ?eeuld ?jdovh) with proper relations.
         b) If partial progress is found, estimate smaller cost (e.g. 1).
         c) Otherwise, estimate cost as 2 (need to build support and then achieve).
    2. Also check for necessary supportive conditions in the state (like (y01m0fhm ?lt0bb)) that appear in the goal,
       and incorporate their absence as heuristic increments.
    3. Sum all increments to estimate the number of actions needed to reach the goal.
    4. Return 0 if goal already satisfied.

    This heuristic is cheap to compute, focuses strongly on missing key goal facts, and accounts for intermediate dependencies
    without fully performing complex relaxed reachability computations.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract all goal ixsb7u9c facts as tuples (r6gw2, pmq6p)
        self.goal_ixsb7u9c = set()
        self.goal_y01m0fhm = set()  # track goals of y01m0fhm predicates (uwq7ubjf objects)
        for g in self.goals:
            parts = get_parts(g)
            if parts[0] == "ixsb7u9c":
                # (ixsb7u9c r6gw2 pmq6p)
                self.goal_ixsb7u9c.add((parts[1], parts[2]))
            elif parts[0] == "y01m0fhm":
                # (y01m0fhm uwq7ubjf)
                self.goal_y01m0fhm.add(parts[1])

        # Build static indexes to help estimate heuristic

        # Map pmq6p (chxau8fy) -> set of eeuld (tom0ncko), from (rycjvfag ?eeuld ?pmq6p)
        self.pmq6p_to_eeuld = {}
        # Map r6gw2 (dbk5i2a5) -> set of lt0bb (uwq7ubjf) from (cs7680d7 ?r6gw2 ?lt0bb)
        self.r6gw2_to_lt0bb = {}
        # Map (jdovh, lt0bb) pairs that satisfy (jckjsxca ?jdovh ?lt0bb)
        self.jdovh_lt0bb_jckjsxca = set()
        # Map (jdovh, val) pairs where val is uwq7ubjf, from (rpzrvk8g ?jdovh ?val)
        self.jdovh_to_rpzrvk8g = {}
        # Map (jdovh, r6gw2, pmq6p) from static facts (r1frsf5x ?jdovh ?r6gw2 ?pmq6p)
        self.r1frsf5x_set = set()
        # Map (eeuld, jdovh) from (efxvv9ax ?eeuld ?jdovh)
        self.efxvv9ax_set = set()
        # Map (lt0bb, val) from (wx95011q ?lt0bb ?val)
        self.lt0bb_to_wx95011q = {}
        # Map uwq7ubjf that satisfy (a8x3330w ?lt0bb)
        self.a8x3330w_set = set()
        # Map (zmrjf, jdovh) from (z6yvwl7n ?zmrjf ?jdovh)
        self.z6yvwl7n_set = set()

        for fact in self.static:
            parts = get_parts(fact)
            pred = parts[0]

            if pred == "rycjvfag" and len(parts) == 3:
                eeuld, pmq6p = parts[1], parts[2]
                self.pmq6p_to_eeuld.setdefault(pmq6p, set()).add(eeuld)

            elif pred == "cs7680d7" and len(parts) == 3:
                r6gw2, lt0bb = parts[1], parts[2]
                self.r6gw2_to_lt0bb.setdefault(r6gw2, set()).add(lt0bb)

            elif pred == "jckjsxca" and len(parts) == 3:
                jdovh, lt0bb = parts[1], parts[2]
                self.jdovh_lt0bb_jckjsxca.add((jdovh, lt0bb))

            elif pred == "rpzrvk8g" and len(parts) == 3:
                jdovh, val = parts[1], parts[2]
                self.jdovh_to_rpzrvk8g.setdefault(jdovh, set()).add(val)

            elif pred == "r1frsf5x" and len(parts) == 4:
                jdovh, r6gw2, pmq6p = parts[1], parts[2], parts[3]
                self.r1frsf5x_set.add((jdovh, r6gw2, pmq6p))

            elif pred == "efxvv9ax" and len(parts) == 3:
                eeuld, jdovh = parts[1], parts[2]
                self.efxvv9ax_set.add((eeuld, jdovh))

            elif pred == "wx95011q" and len(parts) == 3:
                lt0bb, val = parts[1], parts[2]
                self.lt0bb_to_wx95011q.setdefault(lt0bb, set()).add(val)

            elif pred == "a8x3330w" and len(parts) == 2:
                self.a8x3330w_set.add(parts[1])

            elif pred == "z6yvwl7n" and len(parts) == 3:
                self.z6yvwl7n_set.add((parts[1], parts[2]))

    def __call__(self, node):
        state = node.state

        # If all goals are satisfied, heuristic is zero.
        if self.goals <= state:
            return 0

        # Helper inline functions for checking facts in current state
        def has_fact(p, *args):
            # Compose a fact string and check if it is present in state
            return f"({p} {' '.join(args)})" in state

        def facts_matching(predicate):
            # Yield facts with given predicate in current state
            for f in state:
                if f.startswith(f"({predicate} "):
                    yield f

        cost = 0

        # Count how many goal ixsb7u9c facts are not yet fulfilled.
        # For each missing (ixsb7u9c r6gw2 pmq6p), add cost based on partial progress.

        for (r6gw2, pmq6p) in self.goal_ixsb7u9c:
            if has_fact("ixsb7u9c", r6gw2, pmq6p):
                continue  # achieved, no cost

            # Check if any jdovh exists for which (r1frsf5x jdovh r6gw2 pmq6p) is true in state
            # or related partial facts that support achieving this goal
            partial_found = False

            # Search jdovh candidates linked to r6gw2 and pmq6p
            jdovh_candidates = set()
            # From static: all jdovh for which (r1frsf5x jdovh r6gw2 pmq6p) exists
            for (jdovh, rr6, ppm) in self.r1frsf5x_set:
                if rr6 == r6gw2 and ppm == pmq6p:
                    jdovh_candidates.add(jdovh)

            # Check if any of these have (r1frsf5x jdovh r6gw2 pmq6p) in current state
            for jdovh in jdovh_candidates:
                if has_fact("r1frsf5x", jdovh, r6gw2, pmq6p):
                    # further check if (wfrx4exs ?eeuld ?jdovh) and (efxvv9ax ?eeuld ?jdovh) hold,
                    # indicating partial progress from eeuld linked to jdovh.
                    # To do this, find eeuld related to pmq6p from static and check these facts in state.
                    eeuld_candidates = self.pmq6p_to_eeuld.get(pmq6p, set())
                    for eeuld in eeuld_candidates:
                        if has_fact("wfrx4exs", eeuld, jdovh) and has_fact("efxvv9ax", eeuld, jdovh):
                            partial_found = True
                            break
                if partial_found:
                    break

            if partial_found:
                cost += 1  # partial progress detected, small cost
            else:
                cost += 2  # no visible progress, higher cost

        # Add cost for any goal (y01m0fhm lt0bb) fact not yet achieved in state.
        for lt0bb in self.goal_y01m0fhm:
            if not has_fact("y01m0fhm", lt0bb):
                cost += 1

        # If cost is zero but goal not fully reached (edge case), return 1 to ensure progress
        if cost == 0:
            cost = 1

        return cost
