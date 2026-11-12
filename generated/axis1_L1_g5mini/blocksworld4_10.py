from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact string by removing surrounding
    parentheses and splitting on whitespace.

    Example:
      "(s0899d5u b1 b2)" -> ["s0899d5u", "b1", "b2"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    An additive (delete-relaxed) domain-dependent heuristic tailored for the
    secret_domain. The heuristic builds a relaxed planning graph from the
    current state: facts true in the state (and static facts) have cost 0;
    each operator whose preconditions all have assigned costs contributes its
    add-effects with cost = sum(costs(preconditions)) + 1. The heuristic
    returns the sum of the relaxed costs of the goal facts (the classical
    h_add heuristic). This yields an informative, efficiently computable
    estimate of the number of actions required to reach the goal under the
    delete-relaxation.

    Assumptions
    -----------
    - The planner represents facts as strings with parentheses, e.g.
      "(predicate arg1 arg2)". The helper `get_parts` removes the
      parentheses and splits the fact into components.
    - Static facts (task.static) are true in every state and are initialized
      with cost 0.
    - The heuristic ignores delete effects (delete-relaxation). It therefore
      may be inadmissible but is effective at guiding greedy best-first
      search.
    - If a goal fact is unreachable in the relaxed graph, the heuristic
      returns a large finite value (treated as "very costly").

    Heuristic Initialization
    ------------------------
    - Store:
      * self.operators: list of Operator objects from the task.
      * self.goals: set of goal fact strings.
      * self.static_facts: set of static fact strings.
      * self.objects: set of all object names observed in the task's facts,
        initial state, goals and operators (extracted by parsing facts and
        ignoring the predicate names). This demonstrates that object
        extraction ignores parentheses as required.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    1. If the state's facts already satisfy all goal facts, return 0.
    2. Initialize a cost map `costs`:
       - For each fact in the current state and for each static fact, set
         costs[fact] = 0 (they are available immediately).
    3. Iteratively expand operators in a relaxed fashion:
       - For each operator whose every precondition p has an assigned cost
         in `costs`, compute the operator application cost as
         op_cost = sum(costs[p] for p in preconditions) + 1.
       - For every add-effect a of this operator, if a is not yet in
         `costs` or op_cost < costs[a], set costs[a] = op_cost.
       - Repeat until no cost improves in an iteration.
    4. Compute the heuristic value as the sum of costs for all goal facts.
       - If any goal fact is not present in `costs` after relaxation, the
         goal is unreachable in the relaxed problem; return a large finite
         value (1_000_000) to indicate high cost.
    5. Return the integer sum of goal costs.
    """

    def __init__(self, task):
        """Extract static information and operators from the task."""
        # Operators and goals for the relaxed expansion.
        self.operators = list(task.operators)
        self.goals = set(task.goals)
        self.static_facts = set(task.static)

        # Collect objects appearing in facts (ignore the predicate name).
        objects = set()
        sources = []
        # gather facts/strings from initial, goals, static, and operator pre/add/del
        if hasattr(task, "initial_state") and task.initial_state:
            sources.extend(list(task.initial_state))
        if task.goals:
            sources.extend(list(task.goals))
        if task.static:
            sources.extend(list(task.static))
        for op in self.operators:
            sources.extend(list(op.preconditions))
            sources.extend(list(op.add_effects))
            sources.extend(list(op.del_effects))

        for fact in sources:
            # facts are strings like "(pred a b)"; get_parts handles brackets
            try:
                parts = get_parts(fact)
            except Exception:
                continue
            # skip predicate name (parts[0]); collect the rest as objects
            for token in parts[1:]:
                objects.add(token)
        self.objects = objects

    def __call__(self, node):
        """
        Compute the additive relaxed-planning heuristic (h_add) from the given
        node.state.

        Returns:
          int: estimated cost (sum of relaxed costs of goal facts). Returns 0
               if the node state already satisfies the goal. If some goal is
               unreachable in the relaxed graph, returns a large finite value.
        """
        state = node.state

        # If goals already satisfied, return zero as required.
        if self.goals <= state:
            return 0

        # Initialize costs: facts in current state and static facts cost 0.
        costs = {}
        for fact in state:
            costs[fact] = 0
        for fact in self.static_facts:
            costs[fact] = 0

        # Iteratively relax operators: propagate costs to add effects.
        improved = True
        while improved:
            improved = False
            for op in self.operators:
                # Check whether all preconditions have assigned costs.
                pre_costs = []
                all_known = True
                for p in op.preconditions:
                    if p in costs:
                        pre_costs.append(costs[p])
                    else:
                        all_known = False
                        break
                if not all_known:
                    continue
                # Cost to "apply" this operator in the relaxed graph
                op_cost = sum(pre_costs) + 1
                # Update add effects costs
                for add in op.add_effects:
                    prev = costs.get(add)
                    if prev is None or op_cost < prev:
                        costs[add] = op_cost
                        improved = True

        # Sum costs of goal facts; if any goal is unreachable, return large value.
        total = 0
        LARGE = 1_000_000
        for g in self.goals:
            c = costs.get(g)
            if c is None:
                return LARGE
            total += c

        # Return integer heuristic value.
        return int(total)