from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return list of components of a fact string by removing surrounding parentheses and splitting."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A delete-relaxed, additive domain-dependent heuristic tailored to the structure
    of the secret_domain (rb7qr1fs). The heuristic estimates the number of
    operator applications required to satisfy all goal facts by solving a
    relaxed planning problem in which delete effects are ignored. It is
    deliberately optimistic (not necessarily admissible in the real problem)
    and designed to be fast while providing informative gradients for GBFS.

    Assumptions
    -----------
    - Facts are strings of the form "(predicate arg1 arg2 ...)".
    - Operator preconditions and add-effects use the same fact-string format.
    - Static facts (task.static) hold in every state; they are available at zero
      cost in the relaxed problem.
    - Each operator application has uniform base cost 1; the cost to achieve a
      fact via an operator is 1 + (additive costs of its preconditions).
    - The heuristic ignores delete-effects (standard relaxed/additive heuristic).
    - If a goal fact cannot be reached in the relaxed problem, a large finite
      penalty is returned instead of infinity.

    Heuristic Initialization
    ------------------------
    During initialization the heuristic extracts and stores:
    - self.goals: the set of goal fact strings to be achieved.
    - self.operators: the set/list of Operator objects from the task (each
      operator exposes .preconditions and .add_effects as frozensets of fact
      strings).
    - self.static: the set of static facts (task.static) converted to a set for
      O(1) membership tests.
    - self.static_by_pred: a mapping from predicate name to the set of static
      facts having that predicate (useful if future domain-specific
      specialization is needed).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the given node's state already satisfies all goal facts, return 0.
    2. Build the relaxed-cost estimate for every fact reachable in the relaxed
       problem:
       - Initialize a dictionary `fact_costs` mapping fact -> cost.
       - All facts currently true in the given state and all static facts are
         assigned cost 0.
       - All other facts are implicitly at infinite cost at the start.
    3. Propagate costs forward using a relaxed (no deletes) fixed-point:
       - Repeatedly consider each operator. If all its preconditions have
         finite cost, the cost of achieving any of its add-effects via this
         operator is operator_cost = 1 + sum(cost(precondition) for each
         precondition). If operator_cost improves the recorded cost of an
         add-effect, update that fact's cost.
       - Iterate until no fact cost improves during a full pass over operators
         (a monotone fixed-point).
    4. Combine goal costs to produce a heuristic value:
       - For each goal fact that is not already true in the current state,
         obtain its relaxed cost from `fact_costs`. The additive heuristic
         sums these costs (this is the h_add heuristic).
       - If any goal fact remains unreachable (cost infinite), return a large
         finite penalty (1000000) so the planner treats the state as poor but
         does not crash on infinite values.
    5. Return the computed sum as the heuristic estimate. This value is finite
       for any goal that is reachable in the relaxed model.

    Rationale / Domain Considerations
    ---------------------------------
    The domain's meaningful goal predicates in the provided instances are
    primarily ixsb7u9c(...) and y01m0fhm(...). Both are introduced by specific
    actions in the domain; the relaxed-additive propagation naturally identifies
    prerequisite facts and composes their costs. Although this heuristic does
    not exploit bespoke shortcuts for this particular domain, the additive
    relaxed approach captures chains of dependencies (e.g., producing r1frsf5x
    then using it to produce ixsb7u9c) and is efficient to compute. If needed,
    further domain-specific pruning (e.g., specializing for these predicates)
    can be added later without changing the relaxed-cost core.
    """

    def __init__(self, task):
        # Goals required in the problem (set of fact strings).
        self.goals = set(task.goals)

        # Operators from the task (Operator objects with .preconditions/.add_effects).
        # Convert to a list for deterministic iteration order.
        self.operators = list(task.operators)

        # Static facts that hold in every state.
        self.static = set(task.static)

        # Build a mapping from predicate -> set of static fact strings for quick access.
        self.static_by_pred = {}
        for fact in self.static:
            parts = get_parts(fact)
            pred = parts[0]
            self.static_by_pred.setdefault(pred, set()).add(fact)

    def __call__(self, node):
        """
        Compute the delete-relaxed additive heuristic (h_add) for the given node.

        Parameters
        ----------
        node : object
            A search node object that provides `node.state`, a frozenset of fact
            strings representing the current state.

        Returns
        -------
        float or int
            Estimated cost (number of operator applications) to reach the goals
            from node.state in the delete-relaxed model. Returns 0 if goals are
            already satisfied. Returns a large finite number (1e6) if one or
            more goals are unreachable in the relaxed problem.
        """
        state = set(node.state)  # Convert to mutable set for membership checks.

        # If goals already satisfied, heuristic is zero.
        if self.goals <= state:
            return 0

        INF = float("inf")
        # Initialize fact costs: facts true in the current state and static facts have cost 0.
        fact_costs = {}

        for f in state:
            fact_costs[f] = 0.0
        for f in self.static:
            fact_costs[f] = 0.0

        # Repeatedly relax operator add-effects until fixed point (Bellman-Ford style).
        improved = True
        # To prevent pathological long loops in degenerate domains, we cap iterations
        # to a reasonable number (operators * 50). In practice, propagation converges quickly.
        max_iters = max(1000, len(self.operators) * 50)
        iters = 0

        while improved and iters < max_iters:
            improved = False
            iters += 1
            for op in self.operators:
                # For this operator, compute sum of costs of its preconditions.
                total_pre_cost = 0.0
                pre_unreachable = False
                for pre in op.preconditions:
                    c = fact_costs.get(pre, INF)
                    if c == INF:
                        pre_unreachable = True
                        break
                    total_pre_cost += c

                if pre_unreachable:
                    # This operator is not applicable in the relaxed sense yet.
                    continue

                op_cost = 1.0 + total_pre_cost  # cost to achieve operator's add-effects via this op.

                # Update costs for each add-effect if we found a cheaper way.
                for add in op.add_effects:
                    old = fact_costs.get(add, INF)
                    if op_cost < old:
                        fact_costs[add] = op_cost
                        improved = True

        # Sum costs for unsatisfied goals (additive heuristic).
        total = 0.0
        for g in self.goals:
            if g in state:
                continue
            cost_g = fact_costs.get(g, INF)
            if cost_g == INF:
                # Unreachable in the relaxed problem: return a large finite penalty.
                return 1000000
            total += cost_g

        # Ensure returned value is an integer when possible (operators are unit-cost, sums are integral
        # if no fractional pre-costs remain). Return as int for consistency with other heuristics.
        try:
            # If total is very near an integer, round it to the nearest int.
            if abs(total - round(total)) < 1e-9:
                return int(round(total))
        except Exception:
            pass

        return total