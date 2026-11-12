from heuristics.heuristic_base import Heuristic

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-aware heuristic for the `content-portion` (secret_domain) planning tasks.
    The heuristic estimates the number of actions to reach the goals by computing a
    relaxed cost (additive relaxed-plan cost / h_add) over the STRIPS operators that
    are part of the task. The relaxed cost ignores delete effects and treats each
    operator application as costing 1 plus the summed cost of its preconditions.
    The heuristic returns the sum of minimal relaxed costs for all goal facts.

    Assumptions
    -----------
    - Operators in task.operators are instances of the provided Operator class,
      exposing .preconditions and .add_effects as frozensets of fact-strings.
    - Facts in states, goal set, operator preconditions and effects use the same
      string representation (e.g. "(predicate arg1 arg2)").
    - The heuristic is not required to be admissible; it is optimized for guidance
      and typically produces informative, non-admissible estimates.
    - If a goal is already true in the current state, its cost contribution is 0.
    - If a goal is unreachable in the relaxed model (no operator sequence adds it),
      a large finite penalty is returned so the heuristic value remains finite.

    Heuristic Initialization
    ------------------------
    The constructor extracts:
    - self.goals: the set of goal fact-strings from the Task.
    - self.static: the set of static facts from the Task (kept for potential
      domain-specific inspections; not required by the relaxed-cost computation).
    - self.operators: the set/list of operators available in the Task.
    These are stored for repeated use during heuristic evaluations.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    1. If the state's facts already satisfy all goals, return 0 immediately.
    2. Initialize a cost map `fact_cost` with cost 0 for every fact that is true
       in the given state (these are the facts available at zero cost).
    3. Repeatedly attempt to "apply" operators in the relaxed sense:
       - An operator is considered applicable in the relaxed model when all its
         preconditions already have a finite cost in `fact_cost`.
       - If applicable, the operator application cost is:
            op_cost = 1 + sum(cost(pre) for pre in operator.preconditions)
         (1 represents the cost of applying the operator itself).
       - For each add-effect fact produced by the operator, if op_cost is
         lower than the current known cost for that fact (or the fact is unknown),
         update the cost of that fact to op_cost.
    4. Iterate step 3 until no cost improves in an iteration (a fixed point).
    5. The heuristic value is the sum of the computed minimal costs for each goal
       fact. If any goal fact was not assigned a cost during the relaxed expansion,
       it is treated as unreachable in the relaxed model and assigned a large
       penalty value (configurable; chosen here as 10_000) to keep the heuristic
       finite.
    6. Return the final summed cost (integer). This is the additive relaxed-plan
       estimate (h_add), which tends to be informative and efficiently computable.

    Notes on Design Choices
    -----------------------
    - Using an additive relaxed-cost computation (h_add) trades admissibility for
      more informative guidance compared to simple goal counting. It is domain-
      independent in its algorithm, but uses the domain operators directly, which
      makes it effective for the secret_domain operators and goals.
    - The chosen large penalty for unreachable goals is intentionally finite so
      the heuristic never returns an infinite value; planners may interpret a large
      heuristic as undesirable but still handle such states consistently.
    """

    def __init__(self, task):
        """Extract and store goal facts, operators and static facts from the task."""
        # Goals: a frozenset of fact-strings that must be true in the goal.
        self.goals = set(task.goals)
        # Operators: list for iteration (each operator has .preconditions and .add_effects).
        self.operators = list(task.operators)
        # Static facts retained for possible domain-specific heuristics or future use.
        self.static = set(task.static)

        # A finite large penalty for facts unreachable in the relaxed model.
        self._unreachable_penalty = 10_000

    def __call__(self, node):
        """
        Compute the additive relaxed-plan heuristic (h_add) for the given search node.

        Parameters
        ----------
        node : search node object
            Expected to expose node.state, a frozenset of fact-strings.

        Returns
        -------
        int
            Estimated number of actions required to reach the goal from the given state.
        """
        state = set(node.state)

        # Quick check: if all goals already hold in the current state, return 0.
        if self.goals <= state:
            return 0

        # Initialize costs: facts present in the state have cost 0.
        fact_cost = {fact: 0 for fact in state}

        # Repeatedly relax operators to propagate costs to their add-effects.
        changed = True
        # To avoid pathological infinite loops, cap iterations to a reasonable upper bound.
        # A safe upper bound is number of facts times number of operators; but keep it generous.
        max_iterations = max(50, len(fact_cost) + len(self.operators) + 5)
        iterations = 0

        while changed and iterations < max_iterations:
            iterations += 1
            changed = False

            for op in self.operators:
                # All preconditions must have a known cost in the relaxed model to apply op.
                # If there are no preconditions (preconditions set empty), op is always applicable.
                preconds = op.preconditions
                # Check if every precond has an assigned cost.
                missing_pre = False
                pre_cost_sum = 0
                for p in preconds:
                    if p in fact_cost:
                        pre_cost_sum += fact_cost[p]
                    else:
                        missing_pre = True
                        break

                if missing_pre:
                    continue  # can't apply this operator yet

                # cost to apply operator in relaxed model
                op_cost = 1 + pre_cost_sum

                # For each add-effect, possibly improve its cost.
                for eff in op.add_effects:
                    current = fact_cost.get(eff)
                    if current is None or op_cost < current:
                        fact_cost[eff] = op_cost
                        changed = True

        # Sum the costs of all goal facts, applying a penalty for any unreachable goals.
        total_cost = 0
        for g in self.goals:
            if g in fact_cost:
                total_cost += fact_cost[g]
            else:
                # unreachable in relaxed model: assign large finite penalty
                total_cost += self._unreachable_penalty

        return int(total_cost)