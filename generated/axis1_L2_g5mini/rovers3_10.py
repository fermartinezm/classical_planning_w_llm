from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Return list of tokens inside a PDDL fact string '(pred arg1 arg2...)'."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A simple, efficient domain-dependent heuristic tailored to the secret_domain
    (rb7qr1fs). The majority of typical goals in this domain (observed in the
    instances) are of the form (ixsb7u9c <dbk5i2a5> <chxau8fy>) or (y01m0fhm <uwq7ubjf>).
    The heuristic focuses on estimating the number of actions required to
    establish these high-level goal facts by inspecting a small number of
    intermediate facts that are required by the domain operators that produce
    the goals. It trades admissibility for speed and informative estimates to
    reduce node expansions in greedy best-first search.

    Assumptions
    -----------
    - Typical goals are of predicates ixsb7u9c and y01m0fhm (but the heuristic
      supports any goal predicate, using a small default estimate).
    - Many of the static relations in this domain (e.g. wx95011q, rrhxwbyt,
      cs7680d7, rycjvfag, iaub97vi, bszmiquc) are present in the task.static
      and do not need to be produced by planning steps. The heuristic uses
      static facts together with the dynamic state to cheaply check whether
      useful prerequisites are already available.
    - The domain's operator graph indicates that ixsb7u9c is produced by an
      operator (ymbljzxy) that requires a r1frsf5x fact; r1frsf5x in turn is
      typically produced by dlx472ba (which itself often needs wfrx4exs,
      produced by p8i30ct9). Based on that, the heuristic assigns smaller
      costs when some of these intermediate facts are already present.

    Heuristic Initialization
    ------------------------
    The constructor extracts:
    - self.goals : the set of goal fact strings.
    - self.static : the set of static fact strings from the task.
    - A small index (dictionary) over static facts by predicate name for quick
      membership checks (self.static_by_pred).
    This precomputation makes the per-node heuristic computation cheap.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given search node the heuristic does the following:
    1. If all goals are already satisfied in the node's state, return 0.
    2. Otherwise, for each individual goal fact that is currently false:
       - Parse its predicate name and arguments.
       - Use predicate-specific, cheap rules to estimate how many actions
         are still needed to make that goal true. The heuristic is not
         fully precise; instead, it leverages the structure of the domain's
         operators and common intermediate facts to provide an informative,
         quickly computable estimate:
         * ixsb7u9c <r6> <pmq>:
           - If a matching r1frsf5x <jdovh> <r6> <pmq> is already true in the
             current state (or static), then we estimate a small remaining
             cost (2): one action to run ymbljzxy plus a small allowance for
             any missing supplementary preconditions.
           - Otherwise, we estimate a larger cost (3): assume we need the
             sequence p8i30ct9 -> dlx472ba -> ymbljzxy (wfrx4exs -> r1frsf5x ->
             ixsb7u9c) in the worst common case.
         * y01m0fhm <lt0bb>:
           - If there is a rpzrvk8g <jdovh> <lt0bb> present already, the cost
             is set to 1 (sn0ysqqm can likely produce y01m0fhm).
           - Otherwise we estimate cost 2 (one step to produce rpzrvk8g,
             another to produce y01m0fhm).
         * Other predicates (default):
           - A conservative small cost of 2 is returned per missing fact. This
             keeps the heuristic finite and inexpensive to evaluate for any
             goal predicate not explicitly modeled above.
    3. Sum the per-goal estimates to produce the final heuristic value.
    Notes:
    - The heuristic uses both the node state and task.static when checking for
      existing intermediate facts (some facts in this domain are static).
    - The heuristic is intentionally coarse but informative: it distinguishes
      between "goal already close" (intermediate fact present) and
      "goal requires synthesis of intermediates".
    """

    def __init__(self, task):
        # Goal facts are stored as provided by the planner (strings).
        self.goals = set(task.goals)
        # Static facts that hold in all states.
        self.static = set(task.static) if task.static is not None else set()

        # Build a small index of static facts by predicate for O(1)-like checks.
        self.static_by_pred = {}
        for fact in self.static:
            pred = get_parts(fact)[0]
            self.static_by_pred.setdefault(pred, set()).add(fact)

    def __call__(self, node):
        # Quick check: if all goals are in the state, heuristic = 0.
        state = set(node.state)
        if self.goals <= state:
            return 0

        # Treat available facts for cheap precondition checks as union of state+static.
        available = state | self.static

        # Index dynamic available facts by predicate to speed up lookups.
        available_by_pred = {}
        for fact in available:
            parts = get_parts(fact)
            pred = parts[0]
            available_by_pred.setdefault(pred, []).append(parts[1:])

        h_value = 0

        for goal in self.goals:
            if goal in state:
                continue  # achieved

            parts = get_parts(goal)
            pred = parts[0]
            args = parts[1:]

            # Cost estimation specialized per predicate
            if pred == "ixsb7u9c":
                # goal: (ixsb7u9c <r6> <pmq>)
                # Check if matching r1frsf5x exists (r1frsf5x ?jdovh ?r6 ?pmq)
                r6, pmq = args
                r1_exists = False
                for parts_r1 in available_by_pred.get("r1frsf5x", []):
                    # parts_r1 = [jdovh, r6gw2, pmq6p]
                    if len(parts_r1) >= 3 and parts_r1[1] == r6 and parts_r1[2] == pmq:
                        r1_exists = True
                        break
                if r1_exists:
                    # If r1frsf5x already present, assume ~2 actions remain (ymbljzxy +
                    # possible small fixes)
                    h_value += 2
                else:
                    # Otherwise assume we need to produce r1frsf5x and then ixsb7u9c:
                    # p8i30ct9 (wfrx4exs) -> dlx472ba (r1frsf5x) -> ymbljzxy (ixsb7u9c)
                    h_value += 3

            elif pred == "y01m0fhm":
                # goal: (y01m0fhm <lt0bb>)
                lt0bb = args[0]
                rpz_exists = False
                for parts_rpz in available_by_pred.get("rpzrvk8g", []):
                    # rpzrvk8g ?jdovh ?lt0bb
                    if len(parts_rpz) >= 2 and parts_rpz[1] == lt0bb:
                        rpz_exists = True
                        break
                if rpz_exists:
                    h_value += 1
                else:
                    # Need to produce rpzrvk8g and then y01m0fhm
                    h_value += 2

            else:
                # Default conservative estimate for other goal predicates.
                h_value += 2

        # Ensure finite non-negative integer heuristic.
        if h_value < 0:
            h_value = 0
        return int(h_value)