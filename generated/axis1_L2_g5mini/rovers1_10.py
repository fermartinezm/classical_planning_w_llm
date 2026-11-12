from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """
    Split a PDDL fact string into its components, stripping the surrounding
    parentheses. Example: "(at obj loc)" -> ["at", "obj", "loc"].
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-aware relaxed-plan (additive) heuristic specialized for the
    provided secret_domain. It computes an estimate of the number of
    actions required to satisfy all goal facts from a given state by
    building a relaxed cost graph where delete effects are ignored and
    operator costs are cumulative (h_add).

    Assumptions
    -----------
    - The planner represents facts as strings of the form "(pred arg1 arg2 ...)".
    - Operators available in the Task object contain preconditions and add effects
      that are also string facts in the same representation.
    - Static facts (task.static) hold in every state and are available at cost 0.
    - Goals are independent in the relaxed sense and their costs can be summed
      (the heuristic uses the additive relaxation, which is admissible in the
      relaxed problem but may over/under-estimate in the real problem).
    - If some goal is unreachable in the relaxed graph the heuristic returns
      a large finite value to keep the search progressing (the value is chosen
      to be large but finite).

    Heuristic Initialization
    ------------------------
    The constructor stores:
    - self.goals : the set of goal facts (strings).
    - self.operators : the list of operators from the task (each operator has
      preconditions and add_effects as frozensets of fact strings).
    - self.static : the set of static facts (strings) that are always true.
    These are used by the relaxed planner computation in __call__.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    1. Fast goal check:
       - If all goals are already satisfied in the state's fact set, the heuristic returns 0.

    2. Initialization of fact costs:
       - All facts that hold in the current state or are static facts are assigned cost 0.
       - All other facts are initially considered to have infinite cost.

    3. Relaxed propagation (additive, h_add):
       - Repeatedly consider every operator:
         * If every precondition of the operator has a finite cost,
           compute the operator cost as 1 + sum(cost(precondition) for precondition).
         * For every add-effect of that operator, relax the cost of the add-effect
           to be the minimum of its current cost and the operator cost.
       - Repeat propagation until no costs change or a safe iteration bound is reached.
       - This effectively constructs a relaxed reachability graph from the current
         facts to any other fact, measuring the additive cost to achieve each fact.

    4. Aggregation of goal costs:
       - Sum the computed costs for all goal facts. This is the additive heuristic value.
       - If any goal remains with infinite cost after relaxation, return a large finite
         penalty (1_000_000) to indicate that the goal appears unreachable in the
         relaxed graph while keeping the heuristic finite.

    5. Return value:
       - The resulting value is an estimate of the number of actions required to
         reach the goal from the current state in the original problem. It is
         not guaranteed to be admissible for the real problem, but it is
         informative and efficiently computable, which helps guide greedy
         best-first search.

    Notes on efficiency and domain-dependence
    ----------------------------------------
    - This heuristic is domain-aware because it uses the domain's operators and
      the task-specific static facts. It handles the particular goal predicates
      of the secret_domain without requiring predicate-specific hard-coding.
    - The algorithm ignores delete effects (relaxed planning), making it
      efficiently computable and typically informative for guiding search in
      structured domains like the secret_domain.
    """

    def __init__(self, task):
        # Goals the planner must achieve (set/frozenset of fact strings)
        self.goals = set(task.goals)
        # All operators (instances of Operator) for the domain
        self.operators = list(task.operators)
        # Static facts that hold in every state
        self.static = set(task.static)

    def __call__(self, node):
        """
        Compute the additive relaxed-plan heuristic (h_add) for the given node.
        Returns an integer heuristic estimate (0 for goal states).
        """
        state = set(node.state)

        # Quick check: if the goal is already satisfied, return 0.
        if self.goals <= state:
            return 0

        # Initialize costs: facts present in state or static facts cost 0.
        INF = float("inf")
        costs = {}

        # Seed costs with 0 for known facts
        for fact in state | self.static:
            costs[fact] = 0

        # For all other facts in the domain (we may only discover them),
        # we'll initialize on demand when encountered in operator effects.

        changed = True
        # A safe upper bound iteration count: operators * 10 (practical tradeoff)
        max_iters = max(1, len(self.operators) * 10)
        iters = 0

        while changed and iters < max_iters:
            changed = False
            iters += 1

            for op in self.operators:
                # Determine whether all preconditions have a finite cost
                precondition_costs = []
                preconds_known = True
                for p in op.preconditions:
                    c = costs.get(p, INF)
                    if c == INF:
                        preconds_known = False
                        break
                    precondition_costs.append(c)

                if not preconds_known:
                    continue

                # Cost to apply this operator in the relaxed problem:
                op_cost = 1 + sum(precondition_costs)

                # Relax costs for add effects
                for eff in op.add_effects:
                    old = costs.get(eff, INF)
                    if op_cost < old:
                        costs[eff] = op_cost
                        changed = True

        # Sum the costs of goal facts
        total = 0.0
        for g in self.goals:
            c = costs.get(g, INF)
            if c == INF:
                # If a goal is unreachable in the relaxed graph, return a large but finite cost.
                return 1_000_000
            total += c

        # Return integer heuristic value
        return int(total)