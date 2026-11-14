from math import inf
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """
    Split a PDDL fact string into its components, ignoring the surrounding parentheses.

    Example:
        "(at loc1 loc2)" -> ["at", "loc1", "loc2"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent, delete-relaxed additive heuristic (h_add) specialized for the
    'secret_domain' (transport) domain. It estimates the number of actions required
    to reach the goal by solving a relaxed version of the planning problem that
    ignores negative effects (the delete-relaxation). The heuristic computes the
    minimal additive cost for each fact in the relaxed problem and sums the costs
    of all goal facts to obtain the heuristic value.

    Assumptions
    -----------
    - The planner represents facts as strings like "(predicate arg1 arg2)".
    - Static facts (task.static) are true in every state and are treated as already
      achieved (cost 0).
    - The delete-relaxation assumption is used: effects that remove facts are ignored.
      This makes the heuristic inadmissible in general but informative and efficient.
    - All operators have uniform action cost 1 (an action contributes 1 plus the
      cost of satisfying its preconditions in the relaxed model).
    - If a goal fact cannot be achieved in the relaxed model, the heuristic returns
      a large finite value to keep the returned value finite for search code. For
      solvable tasks, the relaxed model typically finds finite costs for reachable goals.

    Heuristic Initialization
    ------------------------
    - Store the task's goal facts (self.goals).
    - Store the set of all operator instances (self.operators).
    - Store the set of all known facts in the task (self.all_facts) to bound the
      relaxed cost propagation.
    - Store static facts (self.static) and treat them as initially true (cost 0).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Base costs:
       - Facts that are true in the current state or present as static facts are
         assigned cost 0.
       - All other facts are initialized with infinite cost.

    2. Operator costs in the delete-relaxed model:
       - For each operator, its relaxed cost to make its add-effects true is:
         op_cost = 1 + sum(cost(pre) for pre in op.preconditions)
         (i.e., one action plus the cost to achieve all preconditions).
       - Because deletes are ignored, an operator can be applied regardless of whether
         it also deletes facts required elsewhere in the real problem.

    3. Fixed-point propagation:
       - Repeatedly iterate over all operators and try to improve the cost of each
         fact in their add-effects using the operator cost computed above.
       - Continue until no fact cost can be improved (a least fixed point is reached)
         or until a safe iteration bound is passed.

    4. Goal aggregation:
       - The heuristic value is the sum of the relaxed costs of each individual
         goal fact (the additive heuristic h_add).
       - If any goal fact remains unreachable (cost == inf) in the relaxed model,
         return a large finite value (len(all_facts) * 1000) to keep the heuristic
         finite while effectively marking the state as costly.

    Notes on Efficiency and Guidance
    --------------------------------
    - The delete-relaxed additive heuristic strikes a balance between informativeness
      and computational cost. It is typically much more informative than a simple
      goal-count heuristic and remains efficiently computable because all computations
      operate over the finite set of facts and operators from the task.
    - This heuristic is domain-aware only in the sense that it uses the domain's
      operators and facts; it does not hard-code special-cases of particular
      predicates. That makes it robust to variations of the transport instances
      while still using domain dynamics to guide search.
    """

    def __init__(self, task):
        """
        Initialize the heuristic using the given Task object.

        Parameters:
        - task: an instance of Task containing facts, initial state, goals,
                operators, and static facts.
        """
        # Goal facts (a set of fact strings)
        self.goals = task.goals

        # All operators (set of Operator instances)
        self.operators = task.operators

        # All facts that are syntactically valid in the task (set of strings)
        self.all_facts = set(task.facts)

        # Static facts (true in every state)
        self.static = set(task.static)

    def __call__(self, node):
        """
        Compute the additive delete-relaxed heuristic (h_add) for the given node.

        Returns:
        - An integer heuristic value estimating the number of actions to reach the goal.
        """
        state = set(node.state)  # facts true in the current state

        # If goals already reached exactly return 0
        if self.goals <= state:
            return 0

        # Initialize costs: 0 for facts in current state or static, INF otherwise
        cost = {fact: (0 if (fact in state or fact in self.static) else inf) for fact in self.all_facts}

        # Relaxed propagation (fixed-point computation)
        # Bound iterations to avoid pathological loops: at most |facts| * 2 iterations
        max_iters = max(1, len(self.all_facts) * 2)
        for _ in range(max_iters):
            updated = False
            for op in self.operators:
                # Sum cost of preconditions
                pre_sum = 0
                impossible = False
                for pre in op.preconditions:
                    c = cost.get(pre, inf)
                    if c == inf:
                        impossible = True
                        # In h_add we still treat preconditions with INF as making op_cost INF
                        break
                    pre_sum += c
                if impossible:
                    continue  # this operator cannot yet contribute finite cost

                op_cost = 1 + pre_sum
                # Try to update add effects
                for eff in op.add_effects:
                    # Only consider effects that are legal facts in the task
                    if eff not in cost:
                        # If the operator can produce a fact not in all_facts, add it for completeness
                        cost[eff] = inf
                    if op_cost < cost[eff]:
                        cost[eff] = op_cost
                        updated = True
            if not updated:
                break

        # Aggregate goal costs (additive heuristic)
        total = 0
        for g in self.goals:
            c = cost.get(g, inf)
            if c == inf:
                # unreachable in relaxed model -> return a large finite penalty
                return len(self.all_facts) * 1000
            total += c

        # Return integer heuristic
        return int(total)