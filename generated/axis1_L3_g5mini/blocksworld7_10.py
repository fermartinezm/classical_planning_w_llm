from heuristics.heuristic_base import Heuristic
import math

def get_parts(fact):
    """Return the components of a fact string '(pred a b ...)' as a list ['pred','a','b',...]."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A delete-relaxed, domain-dependent heuristic for the `transport` (secret_domain) domain.
    This heuristic estimates the number of actions required to reach the goal by computing an
    additive relaxed-plan cost (h_add). It treats actions as if they only add facts and never
    delete them, and computes the minimal summed cost to achieve each goal fact. The heuristic
    is intended to be efficiently computable and to guide greedy best-first search (it need
    not be admissible).

    Assumptions
    -----------
    - The task provides fully grounded operators (no parameterized operators).
    - Facts are represented as strings in the form '(predicate arg1 arg2)'.
    - Static facts (task.static) are available in every state and are considered initial facts.
    - Costs of individual actions are uniform (1 per operator application).
    - Resource constraints (like a single `capacity` fact) are modeled as ordinary facts in the
      relaxed problem: an operator that adds `capacity` will make it available for other operators
      in the relaxed model (i.e., deletes are ignored).

    Heuristic Initialization
    ------------------------
    In the constructor we extract and store:
    - the goal facts (task.goals),
    - the static facts (task.static),
    - the complete operator set for the task (task.operators).
    These are used to build the relaxed planning cost propagation.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Goal check:
       - If all goal facts already hold in the current state, return 0.

    2. Prepare the relaxed-cost problem:
       - Initialize a cost map for facts. Facts true in the current state (including static facts)
         receive cost 0.
       - All other facts that appear in operator preconditions or add effects or in the goal
         are initialized with infinite cost.

    3. Cost propagation (h_add computation):
       - Repeatedly consider every operator. If every precondition of the operator has a finite
         cost (i.e., is achievable in the relaxed model), then this operator is applicable in
         the relaxed sense.
       - The relaxed cost of applying this operator is:
            cost(op) = 1 + sum(cost(p) for p in preconditions)
         (one action to apply the operator plus the costs to achieve all its preconditions).
       - For each fact added by the operator, if cost(op) is less than the current recorded cost
         for that fact, update the fact's cost to cost(op).
       - Iterate until no more improvements are possible (fixpoint).

    4. Extract heuristic value:
       - The heuristic is the sum of the costs of all individual goal facts (additive heuristic).
       - If any goal has infinite cost in the relaxed model (unreachable under delete-relaxation),
         return a large finite integer to indicate a high cost (we avoid returning infinite so the
         search algorithm can still compare nodes).

    Notes on properties
    -------------------
    - Heuristic is 0 exactly in states that satisfy all goal facts (explicit check at start).
    - For solvable problems (and in practice for many solvable states), the delete-relaxed
      propagation yields finite costs for goal facts; when it doesn't, we return a large finite
      number to remain robust.
    - The heuristic is efficient: it performs a polynomial number of iterations over grounded
      operators and facts, and uses only simple arithmetic operations.
    """

    def __init__(self, task):
        # Goals as a set of fact strings
        self.goals = set(task.goals)
        # Static facts (true in every state)
        self.static = set(task.static)
        # Ground operators for the task (Operator objects with preconditions and add_effects as sets)
        self.operators = list(task.operators)

        # Precompute the universe of facts that are relevant: all preconditions, add effects, and goals.
        facts = set(self.static) | set(self.goals)
        for op in self.operators:
            facts |= set(op.preconditions)
            facts |= set(op.add_effects)
        self.relevant_facts = facts

    def __call__(self, node):
        """
        Compute the h_add relaxed-plan heuristic value for the given search node.

        Parameters
        ----------
        node: search node object with attribute `state` (a frozenset of fact strings)

        Returns
        -------
        int: estimated number of actions to reach the goal (finite integer; 0 if already at goal)
        """
        state = set(node.state)
        # If goal satisfied, heuristic is zero.
        if self.goals <= state:
            return 0

        # Initialize costs: facts true in current state or static facts cost 0, others infinite.
        cost = {fact: math.inf for fact in self.relevant_facts}
        initial_facts = state | self.static
        for f in initial_facts:
            if f in cost:
                cost[f] = 0

        # Iteratively relax operators (h_add propagation)
        changed = True
        # To be safe, cap iterations to avoid pathological infinite loops (shouldn't happen).
        iterations = 0
        max_iterations = 10000
        while changed and iterations < max_iterations:
            iterations += 1
            changed = False
            for op in self.operators:
                # Check if all preconditions are known and achievable (finite cost)
                preconds = op.preconditions
                precond_costs = []
                unreachable = False
                for p in preconds:
                    # If a precondition is not in our relevant facts, treat it as unreachable unless it's in the current state/static
                    if p not in cost:
                        # If p is present in the current state or static, its cost is 0; otherwise unreachable in relaxed model
                        if p in initial_facts:
                            precond_costs.append(0)
                        else:
                            unreachable = True
                            break
                    else:
                        c = cost[p]
                        if math.isinf(c):
                            unreachable = True
                            break
                        precond_costs.append(c)
                if unreachable:
                    continue
                # relaxed operator application cost = 1 + sum(costs of preconditions)
                op_cost = 1 + sum(precond_costs)
                # For every added fact, possibly improve its cost
                for f in op.add_effects:
                    # Only consider facts that are in our relevant set (we precomputed)
                    if f in cost and op_cost < cost[f]:
                        cost[f] = op_cost
                        changed = True

        # Sum the costs of goal facts (additive heuristic)
        total = 0.0
        for g in self.goals:
            c = cost.get(g, math.inf)
            if math.isinf(c):
                # If a goal is unreachable in relaxed model, return a large finite value.
                # Use sum of numbers of relevant facts as a heuristic upper bound.
                return 9999
            total += c

        # Return integer estimate
        return int(math.ceil(total))