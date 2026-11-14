from math import inf
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing surrounding parentheses
    and splitting on whitespace.

    Example:
      "(at obj1 loc1)" -> ["at", "obj1", "loc1"]

    This function intentionally strips the leading '(' and trailing ')'.
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent relaxed-cost heuristic for the secret_domain (aixg4uih).
    The goal facts in this domain are of the form (qxafq0yb ?r) for z65nyce9 objects.
    This heuristic computes an h_add-style relaxed cost estimate: it treats facts
    that are true in the current state or are static as having cost 0 and then
    iteratively propagates operator costs (1 + sum(costs of preconditions))
    to their add-effects until a fixed point is reached. The heuristic value is
    the sum of the relaxed costs of all goal facts.

    Assumptions
    -----------
    - The Task object passed to the constructor contains grounded Operator
      instances (i.e., operators with concrete fact strings in their
      preconditions and add_effects). This is consistent with the planner's
      operator representation provided.
    - Static facts (task.static) are true in every state and thus have cost 0.
    - Costs of individual operators are uniform (each operator application
      contributes cost 1). We use an additive relaxation (h_add), summing
      the precondition costs to estimate operator cost.
    - If any goal fact is unreachable in the relaxed graph (cost = infinity),
      the heuristic returns a large finite value to keep the value finite.

    Heuristic Initialization
    ------------------------
    In the constructor we extract and store:
    - self.goals : the set of goal fact strings (task.goals).
    - self.static : the set of static facts (task.static), treated as always true.
    - self.operators : the set/list of grounded Operator objects (task.operators).
    The constructor does minimal precomputation because the main propagation
    uses the current state's facts in combination with static facts.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a search node with node.state (a set/frozenset of fact strings):

    1. Build the initial cost mapping:
       - For every fact f in (state U static), set cost[f] = 0.
       - For any other fact encountered in operators' add effects or preconditions,
         initialize cost[f] = infinity.

    2. Iteratively propagate costs using all grounded operators:
       - For each operator op:
         * If every precondition p in op.preconditions has a finite cost,
           compute op_cost = 1 + sum(cost[p] for p in op.preconditions).
         * For every fact a in op.add_effects, if op_cost < cost[a], set cost[a] = op_cost.
       - Repeat this propagation until no cost changes (fixed point).

       Rationale:
       - This is the additive relaxed planning heuristic (h_add), which provides a
         monotone, polynomial-time computable estimate that often guides greedy
         search well even though it is not admissible.

    3. Compute the heuristic value:
       - For each goal fact g in self.goals, read cost[g].
       - If all goal costs are finite, return the integer sum of these costs.
       - If any goal cost is infinite (unreachable in the relaxed graph),
         return a large finite value (for example 10**6) so that the heuristic
         remains finite while indicating the goal is (relaxationally) unreachable.

    Notes on Correctness and Practicality
    -------------------------------------
    - The heuristic returns 0 iff all goal facts are already present in the state.
    - Because it uses grounded operators from the task, no expensive grounding
      is performed here; the propagation iterates over existing operator instances.
    - The additive relaxation tends to be informative for domains where goals
      need multiple supporting facts that themselves require actions (as is the
      case in secret_domain).
    """

    def __init__(self, task):
        # Goals that must be achieved (set of fact strings)
        self.goals = set(task.goals)
        # Static facts that hold in every state (set of fact strings)
        self.static = set(task.static)
        # Operators (expected to be grounded Operator instances)
        # Keep as list for efficient iteration
        self.operators = list(task.operators)

        # Collect the universe of all facts that might appear in preconditions or effects.
        # This set is helpful to initialize cost table; we build it once here for efficiency.
        fact_universe = set()
        for op in self.operators:
            fact_universe.update(op.preconditions)
            fact_universe.update(op.add_effects)
            fact_universe.update(op.del_effects)
        # Also include known static and goal facts
        fact_universe.update(self.static)
        fact_universe.update(self.goals)
        self._fact_universe = fact_universe

    def __call__(self, node):
        """
        Compute the h_add relaxed-cost heuristic for the given search node.

        Parameters
        ----------
        node : A search node object with node.state being a frozenset of fact strings.

        Returns
        -------
        int : Estimated number of actions to reach the goal (sum of relaxed costs of goal facts).
              If any goal is unreachable in the relaxed graph, returns a large finite number.
        """
        state = set(node.state)  # convert to mutable set for membership tests

        # Initialize costs: 0 for facts in state or static, infinity otherwise
        cost = {fact: inf for fact in self._fact_universe}
        true_facts = state | self.static
        for f in true_facts:
            if f in cost:
                cost[f] = 0
            else:
                # If there is a fact in the current state or static not seen in the
                # precomputed universe, add it with cost 0 (robustness).
                cost[f] = 0
                self._fact_universe.add(f)

        # Fixed-point propagation of operator costs (h_add)
        changed = True
        while changed:
            changed = False
            for op in self.operators:
                # Check whether all preconditions have finite cost
                preconditions = op.preconditions
                pre_costs = []
                applicable_in_relaxation = True
                for p in preconditions:
                    c = cost.get(p, inf)
                    if c is inf:
                        applicable_in_relaxation = False
                        break
                    pre_costs.append(c)
                if not applicable_in_relaxation:
                    continue
                # Cost to apply this operator in the additive relaxation:
                op_cost = 1 + sum(pre_costs)
                # Relax: update costs of add effects
                for a in op.add_effects:
                    prev = cost.get(a, inf)
                    if op_cost < prev:
                        cost[a] = op_cost
                        changed = True

        # Sum the costs of goal facts
        total = 0
        for g in self.goals:
            c = cost.get(g, inf)
            if c is inf:
                # Unreachable in the relaxed graph -> return a large finite heuristic
                return 10 ** 6
            total += c

        # Return integer heuristic value
        # The additive relaxation can produce fractional-like sums when precondition costs
        # are integers here, so result is already integral; ensure int type.
        return int(total)