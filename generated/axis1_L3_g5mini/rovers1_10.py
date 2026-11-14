from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Utility: remove surrounding parentheses and split a PDDL fact into parts.
    Example: "(at_kitchen_bread content-portion1 bread-portion1)" ->
             ["at_kitchen_bread", "content-portion1", "bread-portion1"]
    (This function is provided to satisfy the requirement that fact parsing
    ignores surrounding brackets and to help any debugging.)
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A relaxed-additive (h_add) style heuristic specialized for the
    secret_domain ("content-portion") problems. The heuristic estimates the
    number of primitive actions required to satisfy all goal facts from the
    current state under a relaxation that ignores delete-effects. It returns
    0 for goal states and a finite positive integer estimate otherwise.

    Assumptions
    -----------
    - We assume the planning task operators provided by the planner are
      accessible via task.operators and that preconditions and add_effects
      are expressed as fact strings identical to those in states and static
      facts (including parentheses).
    - Static facts (task.static) are available in every state and count as
      already achieved with zero cost.
    - The delete effects of operators are ignored (relaxed planning),
      which makes the computation efficient and generally informative for
      greedy best-first search guidance (though not admissible).

    Heuristic Initialization
    ------------------------
    - The constructor stores:
      * self.goals: the set of goal fact strings.
      * self.operators: the set/list of Operator objects for the domain.
      * self.static: the set of static facts that hold in every state.
    - No heavy preprocessing is performed because the operator set and
      static facts are already compact for this domain.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the current state already satisfies all goals, return 0.
    2. Initialize a cost dictionary `costs` that records the minimal estimated
       cost (under the relaxed model) to achieve each fact:
       - Every fact that is currently true in the state or is static is
         assigned cost 0 (already achieved).
    3. Iteratively relax-apply operators:
       - For each operator whose every precondition already has an assigned
         cost, compute the operator cost as 1 + sum(cost(pre) for pre in preconds).
       - For every add-effect of that operator, update the effect's cost
         to the minimum cost observed so far (i.e., classical h_add update).
       - Continue until no more cost improvements occur or all goals have
         assigned costs.
    4. If every goal has an assigned cost, return the sum of the costs of
       the individual goal facts (this is the h_add estimate).
    5. If some goal remains unreachable in the relaxed computation, return
       a large finite value to indicate that the goal appears unreachable
       from the current state under the relaxed model.
    6. This heuristic is efficient (iterative fixed-point over operators)
       and tends to provide informative guidance for greedy best-first
       search in this domain because it accounts for conjunctive
       preconditions and chains of producer actions.

    Notes on Practical Behavior
    ---------------------------
    - The heuristic is admissible only in special cases; it is intended for
      guiding greedy best-first search (as requested) rather than optimal
      search.
    - Returning a single large finite number for seemingly unreachable goals
      is preferable to returning infinity, so planners using this heuristic
      can still expand states but deprioritize them.
    """

    def __init__(self, task):
        # Goals to be achieved (set of PDDL facts as strings)
        self.goals = set(task.goals)
        # Operators (instances of Operator, with .preconditions and .add_effects)
        self.operators = list(task.operators)
        # Static facts that hold in every state
        self.static = set(task.static)

    def __call__(self, node):
        state = set(node.state)

        # 2) Heuristic is 0 for goal states
        if self.goals <= state:
            return 0

        # Initialize costs: facts in the current state or static facts cost 0.
        costs = {}
        for f in state | self.static:
            costs[f] = 0

        # Iteratively propagate relaxed-add costs.
        changed = True
        # To avoid infinite loops in pathological domains, limit iterations:
        max_iterations = max(50, len(self.operators) * 5)
        iterations = 0

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            for op in self.operators:
                # Check whether all preconditions have an assigned cost
                preconds = op.preconditions
                # If no preconditions, operator is always applicable in relaxation.
                if not preconds:
                    op_cost = 1
                    preconds_known = True
                else:
                    preconds_known = all(p in costs for p in preconds)
                    if preconds_known:
                        # Additive relaxation: operator cost is 1 + sum(costs[pre])
                        op_cost = 1 + sum(costs[p] for p in preconds)
                if not preconds_known:
                    continue
                # Update costs for add effects
                for eff in op.add_effects:
                    prev = costs.get(eff)
                    if prev is None or op_cost < prev:
                        costs[eff] = op_cost
                        changed = True
            # Early stop if all goals have costs
            if all(g in costs for g in self.goals):
                break

        # If after propagation some goals have an assigned cost, sum them.
        if all(g in costs for g in self.goals):
            total = sum(costs[g] for g in self.goals)
            # Ensure an integer return
            return int(total)

        # Otherwise, some goals seemed unreachable under the relaxation.
        # Return a large finite cost (domain-size aware).
        # We choose a bound derived from operator count plus known costs.
        fallback = 100000
        # But if we have some costs, pick a more informed fallback:
        if costs:
            max_known = max(costs.values())
            fallback = int(max_known + len(self.operators) + 10)
        return fallback