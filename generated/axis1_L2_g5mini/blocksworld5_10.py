from heuristics.heuristic_base import Heuristic
from math import inf


def get_parts(fact):
    """
    Extract components of a PDDL fact string by removing surrounding parentheses
    and splitting on whitespace.

    Example:
        "(gaijkwef a b)" -> ["gaijkwef", "a", "b"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic tailored to the secret_domain (dnbnd6kg).
    It uses a delete-relaxed (additive) reachability estimate (h_add) over facts:
    - Compute minimal relaxed cost to make each fact true (ignoring delete effects).
    - Sum the relaxed costs of all goal facts to produce the heuristic value.

    Assumptions
    -----------
    - The planner represents facts as strings like "(pred obj1 obj2)".
    - Operators (task.operators) provide .preconditions and .add_effects as sets
      of those fact-strings (the Task/Operator representation from the planner).
    - Static facts (task.static) are true in every state and cost 0.
    - We ignore negative/delete effects to compute a fast, optimistic estimate.
    - Goals are conjunctive; we sum the costs for each goal fact (h_add).
      This heuristic is not guaranteed admissible but is informative and cheap.

    Heuristic Initialization
    ------------------------
    - Store the set of goal facts (task.goals).
    - Store the set of static facts (task.static) and the full operator list.
    - No other domain-specific static preprocessing is required because
      the domain is small and operators already capture the necessary structure.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If all goal facts are already true in the given state, return 0.
    2. Build the initial relaxed-cost dictionary:
       - Every fact that is true in the current state or in task.static has cost 0.
       - All other facts are given cost = +inf (unreached).
    3. Iteratively relax operators:
       - For each operator, if every precondition has a finite relaxed cost,
         compute op_cost = 1 + sum(cost(pre) for pre in preconditions).
         (One action step for the operator plus the cost to achieve its preconditions.)
       - For every add-effect of that operator, if op_cost is smaller than the
         current known cost of the fact, update the fact's cost to op_cost.
       - Repeat until no costs change in a full pass.
    4. The heuristic value is the sum of the relaxed costs for each goal fact.
       - If any goal fact remains at infinite cost, we return a large finite number
         (a cap) to indicate the state appears unsolvable in the relaxed model.
    5. Return the integer heuristic value.
    """

    def __init__(self, task):
        """
        Initialize heuristic with task-level information.

        Stores:
         - self.goals: set of goal fact strings.
         - self.static: set of static fact strings.
         - self.operators: list of Operator objects from the task.
        """
        self.goals = set(task.goals)
        self.static = set(task.static)
        self.operators = list(task.operators)

    def __call__(self, node):
        """
        Compute the h_add (additive relaxed) heuristic for the node's state.

        Returns:
            An integer heuristic estimate (lower is better). 0 if goals already satisfied.
        """
        state = set(node.state)

        # If goals already satisfied, heuristic is zero.
        if self.goals <= state:
            return 0

        # Initialize fact costs: 0 for facts in current state or static facts, inf otherwise.
        fact_cost = {}
        # Collect universe of facts that appear in operators/add effects/goals as potential facts.
        # Initialize costs lazily: known facts -> 0, others will be created with inf when needed.
        for f in state:
            fact_cost[f] = 0.0
        for f in self.static:
            fact_cost[f] = 0.0
        for g in self.goals:
            if g not in fact_cost:
                fact_cost[g] = inf

        # Also ensure every add_effect in operators is present with initial inf if not already set.
        for op in self.operators:
            for eff in op.add_effects:
                if eff not in fact_cost:
                    fact_cost[eff] = inf
            # Also ensure preconditions exist in fact_cost mapping
            for pre in op.preconditions:
                if pre not in fact_cost:
                    fact_cost[pre] = inf

        # Relaxation loop: propagate minimal relaxed costs until fixpoint.
        changed = True
        # To avoid infinite loops on weird domains, cap iterations by product of facts * ops (safe upper bound).
        max_iters = max(1000, len(fact_cost) * max(1, len(self.operators)))
        iters = 0
        while changed and iters < max_iters:
            iters += 1
            changed = False
            for op in self.operators:
                # Sum costs of preconditions: if any pre unreachable, skip.
                pre_cost_sum = 0.0
                unreachable = False
                for pre in op.preconditions:
                    c = fact_cost.get(pre, inf)
                    if c == inf:
                        unreachable = True
                        break
                    pre_cost_sum += c
                if unreachable:
                    continue
                op_cost = pre_cost_sum + 1.0  # 1 for applying this operator
                # Relax add effects
                for eff in op.add_effects:
                    if op_cost < fact_cost.get(eff, inf):
                        fact_cost[eff] = op_cost
                        changed = True

        # Sum costs for goals (h_add)
        total_cost = 0.0
        for g in self.goals:
            c = fact_cost.get(g, inf)
            if c == inf:
                # If some goal not reachable in the relaxed model, return a large finite penalty.
                # This keeps the heuristic finite while indicating difficulty.
                return 1000
            total_cost += c

        # Return integer heuristic (planner systems typically expect numeric values).
        # Round up to bias towards solvability when costs fractional due to sums.
        return int(total_cost + 0.9999)