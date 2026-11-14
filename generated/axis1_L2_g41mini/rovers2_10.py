from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
    - `args`: The expected pattern(s) with optional wildcards '*'.
    - Returns `True` if the fact matches the pattern; otherwise `False`.
    """
    parts = get_parts(fact)
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain.

    Summary
    -------
    This heuristic estimates the minimal number of actions required to achieve the goal
    from a given state in the secret_domain. It focuses on tracking progress towards the goal
    predicates related to the key resources: the (ixsb7u9c ?r6gw2 ?pmq6p) facts which appear
    as goals in example instances. The heuristic counts how many such goal predicates are 
    already satisfied and estimates the necessary actions to complete the remaining ones.

    Assumptions
    -----------
    - The ultimate goal predicates are always of the form (ixsb7u9c ?r6gw2 ?pmq6p), i.e., these
      represent final achieved conditions linking objects of types dbk5i2a5 and chxau8fy.
    - Achieving a single (ixsb7u9c ...) fact requires a sequence of actions involving
      assigning or transporting resources (implied by related predicates r1frsf5x).
    - The domain static facts encode fixed relationships, such as the existence of b2mwd6x4 and wx95011q
      between objects, which influence applicability but do not directly count toward the goal.
    - The heuristic does not guarantee admissibility but aims for a close estimate that minimizes search nodes.
    - All goals must be achieved individually and correspond to distinct (ixsb7u9c ...) facts.

    Heuristic Initialization
    ------------------------
    - Extract from goal conditions the set of required (ixsb7u9c ?r6gw2 ?pmq6p) facts (object pairs).
    - Extract static facts describing:
      * which ?r6gw2 (dbk5i2a5) and ?pmq6p (chxau8fy) objects exist.
      * which ?jdovh (rb7qr1fs) and ?lt0bb (uwq7ubjf) objects have specific relation properties (like wx95011q and b2mwd6x4).
    - No full graph or cost computations due to complexity; rely on simple counts and proxy predicates that can 
      be evaluated quickly.
    - Use the static predicates bszmiquc and a298chgk to identify relevant ?jdovh objects (rb7qr1fs types) needed for progress.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a state:

    1. Check how many goal (ixsb7u9c ?r6gw2 ?pmq6p) facts are already true in the state.
       These represent completed goal subgoals.

    2. For all remaining goal facts not satisfied, estimate the minimal number of actions needed to achieve each.

       - Each (ixsb7u9c ?r6gw2 ?pmq6p) requires an (r1frsf5x ?jdovh ?r6gw2 ?pmq6p) fact to be true.
       - Without (r1frsf5x ?jdovh ?r6gw2 ?pmq6p), the heuristic assumes at least one action (ymbljzxy or dlx472ba) needed to create it.
       - To create (r1frsf5x ?jdovh ?r6gw2 ?pmq6p), the preconditions involve multiple predicates such as (wfrx4exs ?eeuld ?jdovh), (efxvv9ax ?eeuld ?jdovh), and connectivity predicates like (cs7680d7 ?r6gw2 ?lt0bb), (wx95011q ?jehs5 ?ijb5b), and (b2mwd6x4 ?jdovh ?jehs5 ?ijb5b).
       - Since a detailed path search is expensive, estimate that each missing (r1frsf5x ...) fact requires at least 2 actions on average:
         one to prepare the (wfrx4exs ...) predicate and one to apply dlx472ba, creating the (r1frsf5x ...) fact.

    3. Sum the minimal action counts for all unsatisfied goal facts.

    4. Additionally, check if secondary goal predicates such as (y01m0fhm ?lt0bb) appear in the goal and estimate 1 action if missing.

    5. Return the total estimated action count as the heuristic value.

    Note:
    - This heuristic intentionally errs on the optimistic side by assuming minimal action sequences per missing fact.
    - It is efficiently computable using simple set membership and counts of key predicates.
    """

    def __init__(self, task):
        """Initialize data structures for quick heuristic computation."""

        # Extract goal ixsb7u9c facts: tuples of (?r6gw2, ?pmq6p) representing goal pairs
        self.goal_ixsb7u9c = set()
        self.goal_y01m0fhm = set()  # set of ?lt0bb uwq7ubjf for goal predicate y01m0fhm
        for goal in task.goals:
            parts = get_parts(goal)
            if not parts:
                continue
            pred = parts[0]
            args = parts[1:]
            if pred == "ixsb7u9c" and len(args) == 2:
                self.goal_ixsb7u9c.add((args[0], args[1]))  # (?r6gw2, ?pmq6p)
            elif pred == "y01m0fhm" and len(args) == 1:
                self.goal_y01m0fhm.add(args[0])  # ?lt0bb

        # Store static facts for quick lookup of relevant predicates:
        self.static = task.static

        # Cache related predicates sets for quick membership tests
        self.efxvv9ax_pairs = set()
        self.wfrx4exs_pairs = set()
        self.r1frsf5x_triples = set()
        self.bszmiquc_set = set()
        self.a298chgk_set = set()
        self.y01m0fhm_set = set()
        self.jckjsxca_pairs = set()
        self.cs7680d7_pairs = set()
        self.wx95011q_pairs = set()
        self.b2mwd6x4_triples = set()
        # These sets will help estimate what is already established

        for fact in self.static:
            parts = get_parts(fact)
            p = parts[0]
            args = parts[1:]
            if p == "efxvv9ax" and len(args) == 2:
                self.efxvv9ax_pairs.add((args[0], args[1]))  # (eeuld, jdovh)
            elif p == "bszmiquc" and len(args) == 1:
                self.bszmiquc_set.add(args[0])  # jdovh
            elif p == "a298chgk" and len(args) == 1:
                self.a298chgk_set.add(args[0])  # jdovh
            elif p == "jckjsxca" and len(args) == 2:
                self.jckjsxca_pairs.add((args[0], args[1]))  # (jdovh, lt0bb)
            elif p == "cs7680d7" and len(args) == 2:
                self.cs7680d7_pairs.add((args[0], args[1]))  # (r6gw2, lt0bb)
            elif p == "wx95011q" and len(args) == 2:
                self.wx95011q_pairs.add((args[0], args[1]))  # (lt0bb, jeh3h)
            elif p == "b2mwd6x4" and len(args) == 3:
                self.b2mwd6x4_triples.add((args[0], args[1], args[2]))  # (jdovh, jehs5, ijb5b)

        # This task may have some facts in the static set that correspond to facts that hold in all states.
        # We won't need more than that for this heuristic.

    def __call__(self, node):
        """Estimate the minimal number of actions to achieve all goal predicates."""

        state = node.state

        # Quick goal check: if all goals achieved, heuristic is 0
        # Using the already stored goal predicates against current state.
        if all(
            f"(ixsb7u9c {r6gw2} {pmq6p})" in state for (r6gw2, pmq6p) in self.goal_ixsb7u9c
        ) and all(f"(y01m0fhm {lt0bb})" in state for lt0bb in self.goal_y01m0fhm):
            return 0

        # Count how many goal ixsb7u9c facts are currently true
        satisfied_ixsb7u9c = sum(
            1 for (r6gw2, pmq6p) in self.goal_ixsb7u9c if f"(ixsb7u9c {r6gw2} {pmq6p})" in state
        )
        unsatisfied_ixsb7u9c = len(self.goal_ixsb7u9c) - satisfied_ixsb7u9c

        # For each unsatisfied ixsb7u9c goal fact:
        # estimate at least 2 actions needed:
        #   one action to produce r1frsf5x fact (direct precondition)
        #   one action to produce ixsb7u9c fact given r1frsf5x fact
        # (e.g. dlx472ba produces r1frsf5x, and ymbljzxy produces ixsb7u9c)
        # If r1frsf5x is already true for that pair, then only 1 action needed.

        estimated_actions_for_ixsb7u9c = 0
        for (r6gw2, pmq6p) in self.goal_ixsb7u9c:
            ixsb_fact = f"(ixsb7u9c {r6gw2} {pmq6p})"
            r1frs_fact_match = None
            # Find matching r1frsf5x fact in state with matching ?r6gw2, ?pmq6p and any jdovh
            # Format: (r1frsf5x jdovh r6gw2 pmq6p)
            # Try to find any jdovh with that combination
            found_r1frsf5x = False
            for fact in state:
                if not fact.startswith("(r1frsf5x "):
                    continue
                parts = get_parts(fact)
                if len(parts) == 4:
                    _, jdovh, r_obj, p_obj = parts
                    if r_obj == r6gw2 and p_obj == pmq6p:
                        found_r1frsf5x = True
                        break

            if ixsb_fact in state:
                # Goal already achieved, no cost for this fact
                continue
            elif found_r1frsf5x:
                # One action needed to produce ixsb7u9c from r1frsf5x (ymbljzxy action)
                estimated_actions_for_ixsb7u9c += 1
            else:
                # Need to produce r1frsf5x first (1-2 actions), then ixsb7u9c (1 action)
                # We estimate conservatively 2 actions to produce r1frsf5x and 1 for ixsb7u9c, total 3
                estimated_actions_for_ixsb7u9c += 3

        # Additionally, count unsatisfied y01m0fhm facts (from goal)
        unsatisfied_y01m0fhm = 0
        for lt0bb in self.goal_y01m0fhm:
            if f"(y01m0fhm {lt0bb})" not in state:
                # One action (sn0ysqqm) suffices to produce this predicate
                unsatisfied_y01m0fhm += 1

        # Overall heuristic estimate is sum of estimated actions for unsatisfied goals
        heuristic_value = estimated_actions_for_ixsb7u9c + unsatisfied_y01m0fhm

        # Ensure heuristic is at least 1 if not zero for solvability
        return max(1, heuristic_value)
