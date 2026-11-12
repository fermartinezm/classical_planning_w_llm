from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(ixsb7u9c jk58y1hv no6b5lgm)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain.

    Summary
    -------
    This heuristic estimates the number of actions still required to reach the goal by focusing on the goal facts that involve the predicate (ixsb7u9c ?dbk5i2a5 ?chxau8fy).
    For each such goal fact, it checks if it already holds in the current state.
    If not, the heuristic estimates how many actions are needed to establish it, considering the relevant facts and actions in the domain.
    Given the domain structure, achieving a single goal fact (ixsb7u9c r6gw2 pmq6p) requires:
      - Having a certain fact (r1frsf5x jdovh r6gw2 pmq6p) true for some ?jdovh,
      - That itself requires (wfrx4exs eeuld jdovh),
      - (efxvv9ax eeuld jdovh) and (bszmiquc jdovh) must hold,
      - So the heuristic estimates the minimal number of chains to satisfy all goals,
      - Counting missing intermediate facts to reach the goal.
    The heuristic value sums estimates over all missing goal facts.

    Assumptions
    -----------
    - Goal facts always refer to (ixsb7u9c ?dbk5i2a5 ?chxau8fy).
    - Each goal fact can be achieved independently (heuristic sums over them).
    - Relationships of predicates given by static facts and the effects of actions can be used to estimate prerequisite conditions.
    - The heuristic is not necessarily admissible but should be informative and efficient.

    Heuristic Initialization
    ------------------------
    - Extract all goal facts involving predicate "ixsb7u9c".
    - Extract static facts relevant to:
      * Links between (r1frsf5x ?jdovh ?r6gw2 ?pmq6p)
      * Which 'jdovh' satisfy (bszmiquc ?jdovh), (a298chgk ?jdovh), (dfhfijzr ?jdovh)
      * Which 'eeuld' and pairings (efxvv9ax ?eeuld ?jdovh) hold.
    - Store facts about possible uwq7ubjf objects from (wx95011q ?uwq1 ?uwq2), useful to check related pairings.
    - Store (jckjsxca ?jdovh ?uwq7ubjf) facts indicating what 'uwq7ubjf' each 'rb7qr1fs' currently "has".
    - Prepare structures to quickly check presence of prerequisites and plan for missing ones.

    Step-By-Step Thinking for Computing Heuristic
    ----------------------------------------------
    For each goal fact (ixsb7u9c r6gw2 pmq6p):
    1. Check if (ixsb7u9c r6gw2 pmq6p) is already true in the state:
        - If yes: no cost for this goal.
    2. Otherwise, identify candidate 'jdovh' objects for which:
       - (r1frsf5x jdovh r6gw2 pmq6p) holds (from static)
       - And jdovh satisfies (bszmiquc jdovh) (static)
    3. For each such 'jdovh', check whether:
       - (wfrx4exs eeuld jdovh) is true for some 'eeuld' with (efxvv9ax eeuld jdovh).
         These represent a required intermediate fact.
       - (jckjsxca jdovh uwq7ubjf) must hold for some uwq7ubjf (an "agent" or resource),
       - (cs7680d7 r6gw2 uwq7ubjf) must hold relating the dbk5i2a5 and uwq7ubjf.
    4. Compute minimum cost to establish these conditions, estimating missing facts as +1 per missing fact (presence or transition).
    5. Pick the minimal estimated cost over all candidate 'jdovh' for this goal fact.
    6. Sum these minimal costs for all goal facts.

    - The heuristic value is zero only if all goal facts are currently true in the state.
    - Always finite for solvable states because the domain is connected, and it counts missing prerequisites linearly.
    """

    def __init__(self, task):
        self.goals = {g for g in task.goals if g.startswith("(ixsb7u9c ")}
        self.static = task.static

        # Extract static facts into dictionaries indexed by relevant objects for quick lookup.

        # r1frsf5x jdovh r6gw2 pmq6p links
        self.r1frsf5x_by_r6gw2_pmqp = {}
        for fact in self.static:
            if match(fact, "r1frsf5x", "*", "*", "*"):
                parts = get_parts(fact)
                jdovh, r6gw2, pmq6p = parts[1], parts[2], parts[3]
                self.r1frsf5x_by_r6gw2_pmqp.setdefault((r6gw2, pmq6p), set()).add(jdovh)

        # bszmiquc jdovh - occurs for jdovh that are "active"
        self.bszmiquc_set = {get_parts(fact)[1] for fact in self.static if match(fact, "bszmiquc", "*")}

        # efxvv9ax eeuld jdovh
        self.efxvv9ax_set = set()
        for fact in self.static:
            if match(fact, "efxvv9ax", "*", "*"):
                parts = get_parts(fact)
                eeuld, jdovh = parts[1], parts[2]
                self.efxvv9ax_set.add((eeuld, jdovh))

        # jckjsxca jdovh uwq7ubjf
        self.jckjsxca_set = set()
        for fact in self.static:
            if match(fact, "jckjsxca", "*", "*"):
                parts = get_parts(fact)
                jdovh, uwq7ubjf = parts[1], parts[2]
                self.jckjsxca_set.add((jdovh, uwq7ubjf))

        # cs7680d7 r6gw2 uwq7ubjf
        self.cs7680d7_set = set()
        for fact in self.static:
            if match(fact, "cs7680d7", "*", "*"):
                parts = get_parts(fact)
                r6gw2, uwq7ubjf = parts[1], parts[2]
                self.cs7680d7_set.add((r6gw2, uwq7ubjf))

        # wfrx4exs eeuld jdovh are dynamic facts, so will be checked in the state.

        # Also record all uwq7ubjf objects from wx95011q predicate (static)
        self.wx95011q_set = set()
        for fact in self.static:
            if match(fact, "wx95011q", "*", "*"):
                parts = get_parts(fact)
                self.wx95011q_set.add((parts[1], parts[2]))

    def __call__(self, node):
        state = node.state

        # Fast membership check helpers for state facts
        def in_state(fact_str):
            return fact_str in state

        # Check if all goals are met: if yes, heuristic = 0
        if self.goals <= state:
            return 0

        heuristic_sum = 0

        for goal_fact in self.goals:
            # Goal fact format: (ixsb7u9c r6gw2 pmq6p)
            _, r6gw2, pmq6p = get_parts(goal_fact)

            # If goal fact already true, no cost
            if goal_fact in state:
                continue

            # Candidate jdovh for this (r6gw2, pmq6p) from static r1frsf5x facts
            candidates_jdovh = self.r1frsf5x_by_r6gw2_pmqp.get((r6gw2, pmq6p), set())
            candidates_jdovh = [jd for jd in candidates_jdovh if jd in self.bszmiquc_set]
            if not candidates_jdovh:
                # No candidate jdovh found, large heuristic estimate
                heuristic_sum += 10
                continue

            min_cost_for_goal = float('inf')

            for jdovh in candidates_jdovh:
                cost = 0
                # Check if (r1frsf5x jdovh r6gw2 pmq6p) is guaranteed from static (given jdovh is candidate)
                # So no cost to create this fact.

                # Check if (wfrx4exs eeuld jdovh) holds for some eeuld with (efxvv9ax eeuld jdovh)
                eeuld_candidates = [ee for (ee, jd) in self.efxvv9ax_set if jd == jdovh]

                # Among them check if corresponding (wfrx4exs eeuld jdovh) is true in state
                wfrx4exs_holds = False
                for eeuld in eeuld_candidates:
                    if in_state(f"(wfrx4exs {eeuld} {jdovh})"):
                        wfrx4exs_holds = True
                        chosen_eeuld = eeuld
                        break

                if not wfrx4exs_holds:
                    # To have (wfrx4exs eeuld jdovh) true requires action p8i30ct9 with preconditions:
                    # (jckjsxca jdovh lt0bb), (iaub97vi eeuld r6gw2),
                    # (cs7680d7 r6gw2 lt0bb), (bszmiquc jdovh), and (efxvv9ax eeuld jdovh).
                    # All but (jckjsxca jdovh lt0bb) and (iaub97vi eeuld r6gw2) assumed to be static or fulfilled.
                    # We'll estimate cost += 1 to apply p8i30ct9 to get this fact.

                    # Choose some lt0bb to satisfy precondition (jckjsxca jdovh lt0bb) and (cs7680d7 r6gw2 lt0bb)
                    # Find intersection of uwq7ubjf that both relate to jdovh and r6gw2
                    possible_lt0bb = {lt0 for (jd, lt0) in self.jckjsxca_set if jd == jdovh}
                    possible_lt0bb &= {lt0 for (r6, lt0) in self.cs7680d7_set if r6 == r6gw2}

                    if not possible_lt0bb:
                        # No suitable lt0bb, assign higher cost
                        cost += 3
                    else:
                        cost += 1  # One action to achieve (wfrx4exs ...)

                # Check if (jckjsxca jdovh lt0bb) already true for some lt0bb from above
                # If not in state, cost += 1 (one action to move it or create it, e.g. xvldpwsb)

                # Find lt0bb candidates satisfying (jckjsxca jdovh lt0bb) and (cs7680d7 r6gw2 lt0bb)
                lt0bb_candidates = {lt0 for (jd, lt0) in self.jckjsxca_set if jd == jdovh}
                lt0bb_candidates &= {lt0 for (r6, lt0) in self.cs7680d7_set if r6 == r6gw2}

                # Try to find one such fact in the current state
                jckjsxca_holds = False
                for lt0 in lt0bb_candidates:
                    if in_state(f"(jckjsxca {jdovh} {lt0})"):
                        jckjsxca_holds = True
                        break
                if not jckjsxca_holds:
                    cost += 1  # One action needed to establish that relation (e.g. xvldpwsb or xmjnictn)

                # Check (ixsb7u9c r6gw2 pmq6p) depends on having (r1frsf5x jdovh r6gw2 pmq6p) and (wfrx4exs eeuld jdovh)
                # We have counted cost to establish them if missing
                # Next, the actual goal fact likely comes from the effect of dlx472ba:
                # Preconditions: (efxvv9ax eeuld jdovh), (bszmiquc jdovh), (wfrx4exs eeuld jdovh),
                # (rycjvfag eeuld pmq6p), (jckjsxca jdovh jeh3h), (cs7680d7 r6gw2 jeh3h)
                # Effect: (r1frsf5x jdovh r6gw2 pmq6p), (ixsb7u9c r6gw2 pmq6p)
                #
                # Check if (rycjvfag eeuld pmq6p) holds, else +1 cost
                rycjvfag_holds = False
                if wfrx4exs_holds:
                    for fact in state:
                        if match(fact, "rycjvfag", chosen_eeuld, pmq6p):
                            rycjvfag_holds = True
                            break
                if not rycjvfag_holds:
                    cost += 1

                # Check if there exists uwq7ubjf jeh3h making (jckjsxca jdovh jeh3h) and (cs7680d7 r6gw2 jeh3h)
                # Both need to be true in state or cost +1
                possible_jeh3h = {x for (jd, x) in self.jckjsxca_set if jd == jdovh}
                possible_jeh3h &= {x for (r6, x) in self.cs7680d7_set if r6 == r6gw2}

                jeh3h_holds = False
                for jeh3h in possible_jeh3h:
                    if in_state(f"(jckjsxca {jdovh} {jeh3h})") and in_state(f"(cs7680d7 {r6gw2} {jeh3h})"):
                        jeh3h_holds = True
                        break
                if not jeh3h_holds:
                    cost += 1

                # Best candidate jdovh cost is minimum
                if cost < min_cost_for_goal:
                    min_cost_for_goal = cost

            # If no candidate with finite cost found, assign a fallback cost
            if min_cost_for_goal == float('inf'):
                min_cost_for_goal = 10

            heuristic_sum += min_cost_for_goal

        return heuristic_sum
