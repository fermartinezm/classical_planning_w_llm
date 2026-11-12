from heuristics.heuristic_base import Heuristic
import math

def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing surrounding parentheses
    and splitting on whitespace.
    Example: "(at_kitchen_bread content-portion1 bread-portion2)" ->
             ["at_kitchen_bread", "content-portion1", "bread-portion2"]
    """
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A relaxed-additive (h_add) style domain-dependent heuristic tailored to the
    secret_domain ("content-portion") task. It computes for each goal fact an
    estimate of the minimal number of actions required to make that fact true
    in a relaxed planning sense (ignoring delete effects). The heuristic value
    returned for a state is the sum of the estimated costs of the individual
    goal facts (i.e., assumes conjunction of goals), which is appropriate for
    guiding greedy best-first search in this domain.

    Assumptions
    -----------
    - All operators have unit action cost (each operator application is counted
      as cost 1). This is consistent with the STRIPS-style actions in the
      domain files.
    - The heuristic ignores delete effects (relaxed planning graph), which
      typically underestimates true costs but gives informative guidance that
      is cheaply computable.
    - The planning task (task) provides concrete operator instances where
      preconditions, add effects and delete effects are represented as strings
      in the same format as state facts. The task also exposes goal facts and
      static facts.
    - If a goal fact is already present in the state, its cost is 0.

    Heuristic Initialization
    ------------------------
    The constructor stores:
    - self.goals: the set of goal facts (strings) that must be achieved.
    - self.operators: the set/list of operator instances available in the task.
    - self.static: the static facts from the task (not strictly required for
      the relaxed propagation, but preserved for possible domain-dependent
      extensions).
    Additionally, helper utilities to parse facts are provided (get_parts).

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given search node (state) the heuristic proceeds as follows:

    1. Immediate goal check:
       - If all goal facts are already true in the current state, return 0.

    2. Relaxed reachability propagation (h_add style):
       - Initialize a dictionary `fact_cost` mapping facts that are true in the
         current state to cost 0.
       - Repeatedly consider every operator in the task. For an operator,
         if every precondition of the operator has a known finite cost in
         `fact_cost`, compute the operator cost as:
             op_cost = 1 + sum(cost(precondition) for each precondition)
         (This corresponds to the additive assumption: satisfying multiple
         preconditions requires summing their costs, then adding the action
         itself.)
       - For every fact in the operator's add effects, set:
             fact_cost[fact] = min(previous_cost, op_cost)
         This performs a relaxation where we record the cheapest known way
         (in the relaxed model) to make each fact true.
       - Continue iterating until no new facts get a cheaper cost or until
         all goal facts have assigned costs.

    3. Combine goal costs:
       - The heuristic value is the sum of the costs assigned to each goal
         fact (conjunctive goals). If any goal remains unreachable (no cost
         assigned after fixed-point propagation), the heuristic returns a
         large finite value (math.inf is not used; instead a large integer)
         to indicate (for search guidance) that the state is far from the
         goal under relaxation.

    4. Robustness and termination:
       - The relaxed propagation is monotonic (costs only decrease). The loop
         terminates when no improvements are possible. Complexity is bounded
         by the number of operators and facts.
    """

    def __init__(self, task):
        # Store goals and operators for use during heuristic computation.
        self.goals = set(task.goals)
        # Operators are instances of Operator with .preconditions and .add_effects
        self.operators = list(task.operators)
        # Preserve static facts in case domain-dependent tuning is required later.
        self.static = set(task.static)

    def __call__(self, node):
        """
        Compute an h_add-style heuristic estimate for `node.state`.

        Returns
        -------
        int
            Estimated number of actions to reach a goal from node.state.
            Returns 0 if the state already satisfies all goals. Returns a
            large finite integer when one or more goals are unreachable in
            the relaxed model.
        """
        state = node.state

        # If already a goal state, heuristic is 0.
        if self.goals <= state:
            return 0

        # Initialize costs: facts true in the state have cost 0.
        fact_cost = {fact: 0 for fact in state}

        # We will attempt to improve costs by repeatedly applying relaxed operators.
        changed = True
        # Safety bound: limit iterations to avoid pathological infinite loops.
        # In practice the fixed-point will be reached in a small number of passes.
        max_iterations = max(50, len(self.operators) * 5)
        iters = 0

        while changed and iters < max_iterations:
            changed = False
            iters += 1

            # Early exit if all goals are already assigned costs.
            if all(goal in fact_cost for goal in self.goals):
                break

            for op in self.operators:
                # All preconditions must have assigned costs in the relaxed model.
                # op.preconditions is a frozenset of fact strings.
                preconds = op.preconditions
                # Skip operators with preconditions we can't (yet) achieve.
                if not preconds:
                    op_pre_cost = 0
                else:
                    pre_costs = []
                    applicable_relaxed = True
                    for p in preconds:
                        if p in fact_cost:
                            pre_costs.append(fact_cost[p])
                        else:
                            applicable_relaxed = False
                            break
                    if not applicable_relaxed:
                        continue
                    op_pre_cost = sum(pre_costs)

                # Operator cost = 1 (this action) + cost to reach all preconditions.
                op_cost = 1 + op_pre_cost

                # For each add-effect, try to improve its fact_cost.
                for eff in op.add_effects:
                    prev = fact_cost.get(eff, math.inf)
                    if op_cost < prev:
                        fact_cost[eff] = op_cost
                        changed = True

        # After fixed-point propagation compute the heuristic as sum of goal costs.
        total = 0
        unreachable = False
        for g in self.goals:
            if g in fact_cost:
                total += fact_cost[g]
            else:
                unreachable = True
                break

        if unreachable:
            # Return a large finite estimate to indicate distant/unreachable under relaxation.
            return 10**6

        # Ensure the heuristic is an integer value (number of actions).
        return int(total)