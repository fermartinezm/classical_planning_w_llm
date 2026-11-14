from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing surrounding parentheses
    and splitting on whitespace.

    Example:
      "(gaijkwef a b)" -> ["gaijkwef", "a", "b"]
      "(zzar8qlk)"     -> ["zzar8qlk"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (dnbnd6kg).
    It estimates the number of actions required to satisfy all goal facts by
    reasoning about three primary fact types in the domain:
      - bclhkf4r <obj>    (a unary predicate, think of it as a "token at object")
      - mdawkki4 <obj>    (a unary predicate representing a temporary/working state)
      - chxwa5j4 <obj>    (a unary predicate, related to mdawkki4 transitions)
      - gaijkwef <a> <b>  (a binary relation often produced by combining a mdawkki4 on `a`
                          and a bclhkf4r on `b` via an action)
      - zzar8qlk          (a 0-ary global flag that affects which operators are applicable)
    The heuristic is deliberately non-admissible: it uses a small, cheap model of how many
    operator applications are typically needed to create missing facts and sums estimates
    for individual missing goal facts.

    Assumptions
    -----------
    - Goals are conjunctions of ground facts and are independent for the purpose of this
      heuristic (we sum per-goal estimates). This ignores possible synergies but remains
      effective for guiding greedy best-first search.
    - Creating mdawkki4 on an object typically requires at least one action (e.g. fqixw4x9 or inz39ca6),
      and possibly one extra action to set up the required global flag or preconditions. We model
      creation costs with small integer constants (1-2) to remain conservative but informative.
    - Creating bclhkf4r for an object can be achieved directly from a mdawkki4 for that object
      (via mri4ci1h) or indirectly via other actions; we check for mdawkki4 first and otherwise
      assign a modest cost.
    - Creating a gaijkwef (binary goal) typically requires one action (j2i3gtal) when its two
      preconditions (mdawkki4 on the source and bclhkf4r on the target) are already present.
      If those preconditions are missing we add small additional costs to create them.

    Heuristic Initialization
    ------------------------
    - The constructor stores the set of goal facts (task.goals) for quick membership checks.
    - Static facts from the task are stored (task.static) but the domain contains no meaningful
      static relations for this heuristic, so they are merely retained for completeness.
    - Goals are parsed into a list of (predicate, args...) tuples for efficient per-goal handling.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    Given a planner node (with node.state being a frozenset of PDDL fact strings), the heuristic:
    1. If all goals are already satisfied (all goal facts are in the current state) return 0.
    2. Parse the state's facts to extract:
       - which objects have bclhkf4r, mdawkki4, chxwa5j4
       - which ordered pairs (a,b) are present for gaijkwef
       - whether the global atom (zzar8qlk) is currently true
    3. For each goal fact not present in the state, estimate a small integer cost:
       - Goal type bclhkf4r X:
           * If mdawkki4 X exists -> cost 1 (mri4ci1h can produce bclhkf4r from mdawkki4)
           * Else -> cost 2 (create mdawkki4 then finalize)
       - Goal type chxwa5j4 X:
           * If mdawkki4 X exists -> cost 1 (mri4ci1h produces chxwa5j4)
           * Else -> cost 2
       - Goal type gaijkwef A B:
           * If (A,B) already present -> cost 0
           * Else estimate: base cost 1 to apply the operator that creates the relation (j2i3gtal)
             plus extra costs for missing preconditions:
               - if mdawkki4 A is missing, add 2 (to create a mdawkki4 for A)
               - if bclhkf4r B is missing, add 2 (to create bclhkf4r on B)
             (This models that creating these preconditions often requires one or two actions.)
       - Other predicate forms (if any) are given a conservative fixed cost of 2 if missing.
    4. Sum the per-goal costs and return the integer result.

    Rationale
    ---------
    - The heuristic is cheap to compute: it performs simple set membership tests and a small
      amount of arithmetic. It intentionally overestimates modestly to encourage focused
      search towards satisfying missing preconditions rather than exploring unrelated actions.
    - It is 0 exactly when all goal facts are satisfied.
    - All estimates are finite integers, so the heuristic yields finite values for solvable states.
    """

    def __init__(self, task):
        # Goals are stored as a frozenset of goal fact strings.
        self.goals = task.goals
        # retain static facts (not used in depth for this domain, but stored for completeness)
        self.static = task.static

        # Parse goals once into (pred, args...) tuples for faster handling in __call__.
        parsed_goals = []
        for g in self.goals:
            parts = get_parts(g)
            if parts:
                parsed_goals.append(tuple(parts))
        self.parsed_goals = parsed_goals

    def __call__(self, node):
        state = node.state  # frozenset of fact strings

        # If goals already satisfied, heuristic is 0.
        if self.goals <= state:
            return 0

        # Extract available facts by predicate for quick membership tests.
        have_bcl = set()     # objects with (bclhkf4r X)
        have_mdawk = set()   # objects with (mdawkki4 X)
        have_chx = set()     # objects with (chxwa5j4 X)
        have_gaij = set()    # pairs (A,B) with (gaijkwef A B)
        has_zzar = "(zzar8qlk)" in state

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "bclhkf4r" and len(parts) >= 2:
                have_bcl.add(parts[1])
            elif pred == "mdawkki4" and len(parts) >= 2:
                have_mdawk.add(parts[1])
            elif pred == "chxwa5j4" and len(parts) >= 2:
                have_chx.add(parts[1])
            elif pred == "gaijkwef" and len(parts) >= 3:
                have_gaij.add((parts[1], parts[2]))
            # other predicates (including zzar8qlk) handled above / ignored

        total_cost = 0

        # For each goal fact, estimate the cost to achieve it from the current state.
        for goal in self.parsed_goals:
            pred = goal[0]
            args = goal[1:]

            # If the exact goal fact already present, no cost.
            goal_fact_str = "(" + " ".join(goal) + ")"
            if goal_fact_str in state:
                continue

            # Heuristic estimates per predicate type:
            if pred == "bclhkf4r" and len(args) == 1:
                obj = args[0]
                # If mdawkki4 already present on the object, one action (mri4ci1h) suffices.
                if obj in have_mdawk:
                    cost = 1
                else:
                    # otherwise assume we need to create mdawkki4 then finalize it.
                    cost = 2
                total_cost += cost

            elif pred == "chxwa5j4" and len(args) == 1:
                obj = args[0]
                # chxwa5j4 is produced alongside bclhkf4r by mri4ci1h from mdawkki4.
                if obj in have_mdawk:
                    cost = 1
                else:
                    cost = 2
                total_cost += cost

            elif pred == "gaijkwef" and len(args) == 2:
                a, b = args
                # If already present we've skipped earlier; otherwise estimate:
                # Base cost 1 for the operator that creates the relation (j2i3gtal)
                # plus extra cost for missing preconditions:
                #  - mdawkki4 on a (source)
                #  - bclhkf4r on b (target)
                cost = 1
                if a not in have_mdawk:
                    # creating mdawkki4 typically requires 1-2 actions; use 2
                    cost += 2
                if b not in have_bcl:
                    # creating bclhkf4r typically requires 1-2 actions; use 2
                    cost += 2
                total_cost += cost

            else:
                # Conservative fallback for unexpected predicates: assign small finite cost.
                total_cost += 2

        # Ensure heuristic produces an integer and is finite.
        return int(total_cost)