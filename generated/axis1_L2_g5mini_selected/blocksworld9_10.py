from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Remove surrounding parentheses and split a fact into its predicate and arguments.
    Example: "(gaijkwef a b)" -> ["gaijkwef", "a", "b"].
    """
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A lightweight, domain-dependent heuristic tailored to the secret_domain (dnbnd6kg).
    The heuristic estimates the number of actions required to achieve all goal facts
    by reasoning about how each goal predicate can be produced by the domain operators.
    It is not admissible (it may be optimistic) but aims to be informative and fast to compute.

    Assumptions
    -----------
    - Facts are strings in the form "(predicate arg1 arg2 ...)".
    - The goal is expressed as a set of facts; the heuristic returns 0 iff all goal facts
      are present in the current state.
    - We do not fully simulate all operator interactions or build a full relaxed planning
      graph. Instead we use compact, predicate-specific cost estimates based on which
      operators can produce which predicates in the domain.
    - Static facts are provided by the task but, for this domain, they typically are empty.
      We still record them in the constructor for potential future use.

    Heuristic Initialization
    ------------------------
    - Store the list of goal facts (as strings) from the task.
    - Keep static facts from the task as a set (even if empty).
    - Parse goal facts into (predicate, args) tuples for quicker per-goal handling.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For each goal fact g we compute a small estimated cost to make g true if it is currently false.
    The design uses knowledge of the domain operators:
      - gaijkwef X Y is created only by action j2i3gtal which has preconditions:
          mdawkki4 X  and  bclhkf4r Y
        Therefore:
          * If gaijkwef X Y already true -> cost 0
          * If both mdawkki4 X and bclhkf4r Y are present -> j2i3gtal can be applied -> cost 1
          * If one of the two preconditions is missing -> assume each missing precondition costs about 1 extra action -> cost = 1 + (#missing_preconds)
          (This yields cost 2 if one missing, cost 3 if both missing.)
      - bclhkf4r X and chxwa5j4 X are both produced (for a single X) by the action mri4ci1h when mdawkki4 X is true:
          mdawkki4 X  -> (mri4ci1h) -> bclhkf4r X  and chxwa5j4 X (and zzar8qlk)
        Therefore:
          * If goal already true -> cost 0
          * Else if mdawkki4 X is present -> cost 1
          * Else assume mdawkki4 can be produced with a small number of actions (optimistic) -> cost 2
      - mdawkki4 X can be created by a couple of domain actions (inz39ca6 or fqixw4x9), both of which often require the global zzar8qlk or other facts. To remain efficient and optimistic:
          * If mdawkki4 X is present -> cost 0
          * Else -> cost 2 (optimistic fixed cost)
      - Any other predicate (unexpected ones) is handled conservatively:
          * present -> 0, else -> 1
    Finally, the heuristic value for the state is the sum of the per-goal estimated costs.
    This sum is a cheap, informed estimate: it respects that some goals require more work
    (e.g., creating a gaijkwef relation usually involves ensuring two supporting facts),
    while remaining inexpensive to compute.

    Notes on properties:
    - The heuristic returns 0 exactly when the goal set is a subset of the state.
    - The heuristic always returns a finite integer.
    - It is intentionally heuristic (non-admissible) to improve guidance for greedy search.
    """

    def __init__(self, task):
        # Store goal facts and static facts (parsed).
        self.goals = set(task.goals)
        self.static = set(task.static) if task.static is not None else set()

        # Pre-parse goals into (predicate, args) tuples for faster access.
        parsed_goals = []
        for goal in self.goals:
            parts = get_parts(goal)
            if parts:
                parsed_goals.append((parts[0], parts[1:]))
        self.parsed_goals = parsed_goals

    def __call__(self, node):
        state = node.state  # frozenset of fact strings

        # If all goals already satisfied, heuristic is zero.
        if self.goals <= state:
            return 0

        # Helper to quickly check presence of fact by predicate and args.
        def fact_in_state(pred, *args):
            fact = "(" + " ".join((pred,)+args) + ")"
            return fact in state

        total_cost = 0

        for pred, args in self.parsed_goals:
            # Recreate the exact goal fact string
            goal_fact = "(" + " ".join((pred,)+tuple(args)) + ")"
            if goal_fact in state:
                continue  # cost 0 for this goal

            # Heuristic estimates per predicate type (domain-specific)
            if pred == "gaijkwef":
                # Needs mdawkki4 arg1 and bclhkf4r arg2 before j2i3gtal can create the relation.
                a, b = args[0], args[1]
                need_md = not fact_in_state("mdawkki4", a)
                need_bcl = not fact_in_state("bclhkf4r", b)
                # Base cost 1 to apply j2i3gtal plus extra for missing preconditions.
                cost = 1 + (1 if need_md else 0) + (1 if need_bcl else 0)
                total_cost += cost

            elif pred == "bclhkf4r":
                x = args[0]
                # Prefer mri4ci1h if mdawkki4 x exists (1 action). Otherwise assume small effort (2).
                if fact_in_state("mdawkki4", x):
                    total_cost += 1
                else:
                    total_cost += 2

            elif pred == "chxwa5j4":
                x = args[0]
                # chxwa5j4 x is produced by mri4ci1h from mdawkki4 x.
                if fact_in_state("mdawkki4", x):
                    total_cost += 1
                else:
                    total_cost += 2

            elif pred == "mdawkki4":
                x = args[0]
                # mdawkki4 might require a couple of actions to produce; optimistic estimate 2.
                total_cost += 2

            else:
                # Generic fallback: if missing, assume one action.
                total_cost += 1

        return total_cost