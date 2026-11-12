from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Split a PDDL fact string into its parts, ignoring the surrounding parentheses.

    Example:
      "(at loc1 loc2)" -> ["at", "loc1", "loc2"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the 'transport' (secret_domain) domain.
    This heuristic computes a relaxed-plan-based estimate (h_add) of the number
    of actions required to reach the goal from a given state. The relaxed plan
    ignores negative effects (delete lists) of operators, which yields an
    efficiently computable, informative estimate for greedy best-first search.

    Assumptions
    -----------
    - Facts in states and operator preconditions/effects are represented as
      strings of the form "(predicate arg1 arg2 ...)".
    - The task object supplied to the constructor exposes:
        - task.goals : set of goal fact strings,
        - task.static : set of static facts (facts true in every state),
        - task.operators : iterable of Operator objects whose preconditions,
          add_effects and del_effects are sets of fact strings.
    - The domain has a single global 'capacity' fact in some states; the
      heuristic treats all facts uniformly (no special-case semantics), but
      the relaxed planning procedure implicitly reasons about the need to
      establish such facts via operators.
    - It is acceptable for the heuristic to be non-admissible (we prioritize
      guidance quality over admissibility).

    Heuristic Initialization
    ------------------------
    In the constructor we:
    - store the goal facts (self.goals),
    - store static facts in a readily available set (self.static_facts),
    - keep a reference to the task operators (self.operators).
    Static facts are kept although not used specially by the relaxed planner;
    they can accelerate the initial fact cost assignment because static facts
    can be considered always available (cost 0).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    The heuristic is the "sum of goal costs" extracted from a relaxed planning
    graph / cost-propagation (often called h_add). The steps for a given node
    are:

    1. Goal check:
       - If all goal facts are already true in the node.state, return 0.

    2. Initialization of fact costs:
       - For every fact present in the current state, set its cost to 0.
       - For every static fact in the task.static, also ensure cost 0.
       - All other facts are initialized with infinite cost.

    3. Operator application in the relaxed sense:
       - Iteratively consider each operator. If every precondition of the
         operator has a finite cost, the operator becomes usable in the
         relaxed planning sense.
       - The cost of applying that operator is computed as:
             op_cost = 1 + sum(cost(pre) for pre in preconditions)
         (one action plus the aggregated costs to achieve its preconditions).
       - For every add-effect produced by the operator, update the cost of
         that fact to the minimum between its current cost and op_cost.
       - Note: delete-effects are ignored (this is the delete-relaxation).

    4. Fixed-point / termination:
       - Repeat step 3 until no fact cost improves in an iteration or until
         all goal facts have finite costs.
       - If all goals have finite costs, compute the heuristic as the sum of
         the individual goal costs (h_add).
       - If some goals remain unreachable (infinite cost) after reaching a
         fixed point, return a large finite value to indicate difficulty
         (this keeps the heuristic finite for solvable states and gives a
         consistent ordering for search).

    5. Return:
       - The heuristic returns an integer (or float) cost estimate; in
         practice we return the sum of costs of the goal facts if reachable,
         or a large sentinel value (e.g., 10**6) if unreachable in the
         relaxed graph.

    Rationale and benefits
    ----------------------
    - The relaxed (additive) heuristic captures necessary action dependencies
      in the domain (e.g., to get a particular 'at' fact an operator that
      produces it must be applied, which itself may require other facts).
    - Because the heuristic ignores deletes, it is monotone and fast to
      compute; it typically provides good guidance for greedy best-first
      search in small-to-medium sized instances of this domain.
    """

    def __init__(self, task):
        # Goals that must be achieved (set of fact strings)
        self.goals = set(task.goals)
        # Static facts (true in every state); kept for immediate zero-cost facts
        self.static_facts = set(task.static) if hasattr(task, "static") else set()
        # Operators of the task (instances of Operator)
        self.operators = list(task.operators)

    def __call__(self, node):
        """
        Compute and return the h_add relaxed-plan heuristic value for the node.

        Returns 0 if the node is a goal state. Otherwise computes cost to reach
        each goal fact in the delete-relaxed planning graph and returns the
        sum of goal costs (or a large number if unreachable in the relaxed graph).
        """
        state = set(node.state)

        # 1) Goal check: if already satisfied -> heuristic 0
        if self.goals <= state:
            return 0

        # 2) Initialize cost map for facts: facts in state and static facts cost 0
        INF = float("inf")
        fact_cost = {}  # fact -> best known cost
        # Facts true in current state have zero cost
        for fact in state:
            fact_cost[fact] = 0.0
        # Static facts are always available with cost 0
        for fact in self.static_facts:
            fact_cost[fact] = 0.0

        # Precompute operator preconditions and add effects lists for speed
        ops_pre = [op.preconditions for op in self.operators]
        ops_add = [op.add_effects for op in self.operators]

        # 3) Cost propagation loop (relaxed, i.e., ignoring deletes)
        updated = True
        # We also stop early if all goals have finite costs.
        while updated:
            updated = False

            # If all goals already have finite cost, we can stop
            all_goals_finite = all(fact in fact_cost and fact_cost[fact] < INF for fact in self.goals)
            if all_goals_finite:
                break

            # For each operator, check whether all preconditions have known costs.
            for preconds, add_effects in zip(ops_pre, ops_add):
                # Check if all preconditions currently have finite costs
                pre_costs = []
                applicable_in_relax = True
                for pre in preconds:
                    c = fact_cost.get(pre, INF)
                    if c == INF:
                        applicable_in_relax = False
                        break
                    pre_costs.append(c)

                if not applicable_in_relax:
                    continue

                # Operator is usable in the relaxed sense; operator cost is
                # 1 (for the operator itself) plus sum of precondition costs.
                op_cost = 1.0 + sum(pre_costs)

                # Propagate costs to add effects
                for eff in add_effects:
                    old = fact_cost.get(eff, INF)
                    if op_cost < old:
                        fact_cost[eff] = op_cost
                        updated = True

        # 4) Compute heuristic: sum of goal fact costs (h_add)
        total = 0.0
        unreachable = False
        for g in self.goals:
            c = fact_cost.get(g, INF)
            if c == INF:
                unreachable = True
                break
            total += c

        if unreachable:
            # Return a large finite value if goals unreachable in relaxed graph.
            # This keeps the heuristic finite but indicates very high difficulty.
            return 10 ** 6

        # Return integer-like value when possible
        return total