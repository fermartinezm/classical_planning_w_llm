from heuristics.heuristic_base import Heuristic
import math


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing surrounding parentheses
    and splitting the content.

    Example: "(at obj1 loc2)" -> ["at", "obj1", "loc2"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the secret_domain (rb7qr1fs). The heuristic
    computes a relaxed (delete-relaxed) cost estimate for achieving all goal
    facts from the current state using an h_add style propagation. Costs are
    propagated from currently true facts (cost 0) through operators (each
    operator accrues cost equal to 1 plus the sum of costs of its preconditions)
    to their add effects. The heuristic value is the sum of the estimated costs
    of the individual goal facts.

    Assumptions
    -----------
    - Facts are represented as strings with surrounding parentheses, e.g.
      "(predicate arg1 arg2)".
    - The planning task provided contains Operator instances (task.operators)
      whose preconditions and add_effects are sets of such fact strings.
    - Static facts (task.static) are facts that are true in every state and
      therefore treated as initially available facts with cost 0.
    - The heuristic is not required to be admissible; it is tuned to provide
      informative estimates quickly by using the h_add relaxation.
    - If some goal fact cannot be reached under the delete-relaxed model, the
      heuristic returns a large finite value (rather than infinity) to remain
      usable by greedy best-first search.

    Heuristic Initialization
    ------------------------
    - Store the task goals and the list of operators.
    - Build a complete set of known facts (from operators' preconditions and
      add-effects, the provided static facts, and the goal facts).
    - Index static facts by predicate for quick lookup (this is helpful when
      debugging or extending the heuristic and satisfies the requirement to
      extract static information).

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    1. Early exit: if all goals are already satisfied in the given state,
       return 0.

    2. Build the universe of facts to reason about:
       - Include all facts that appear in operator preconditions, operator
         add-effects, the static facts, and the goals.

    3. Initialize cost estimates:
       - Facts true in the current state or present in static facts have cost 0.
       - All other facts start with cost = +infinity.

    4. Relaxed cost propagation (h_add style):
       - Repeatedly inspect every operator op in the task:
         * If every precondition p of op has a finite cost, compute the
           operator-cost as 1 + sum(cost[p] for p in preconditions).
         * For each add-effect a of op, if operator-cost < cost[a], update
           cost[a] := operator-cost.
       - Continue the propagation until a full pass over all operators
         produces no improvements (a fixed point is reached).

    5. Compute heuristic value:
       - Sum the final costs for each goal fact.
       - If any goal fact remains unreachable under the relaxed model
         (cost == infinity), return a large finite surrogate value so that
         the heuristic remains usable by greedy best-first search. The value
         is chosen proportional to the number of known facts to stay finite
         but large relative to realistic plan lengths.

    6. Return the computed integer heuristic value.

    Rationale
    ---------
    - The h_add propagation is an inexpensive, informative domain-independent
      relaxation that leverages the actual operators from the domain file.
    - This approach naturally captures multi-step dependencies among facts
      (via operator preconditions) and is robust across instances of the same
      domain.
    """

    def __init__(self, task):
        """
        Extract and store useful task information and static facts.

        Parameters
        ----------
        task : Task
            The planning Task object containing operators, goals, static facts,
            etc.
        """
        # Store the task and goals
        self.task = task
        self.goals = set(task.goals)

        # Operators as a list for fast iteration
        self.operators = list(task.operators)

        # Static facts: store both the frozenset and an index by predicate name
        self.static = set(task.static)
        self.static_by_pred = {}
        for fact in self.static:
            parts = get_parts(fact)
            pred = parts[0]
            self.static_by_pred.setdefault(pred, set()).add(fact)

        # Build a universe of facts that can appear (preconds, adds, static, goals)
        universe = set(self.static)
        for op in self.operators:
            universe.update(op.preconditions)
            universe.update(op.add_effects)
            universe.update(op.del_effects)
        universe.update(self.goals)
        self.all_facts = universe

    def __call__(self, node):
        """
        Compute the relaxed (h_add-like) heuristic value for the node's state.

        Parameters
        ----------
        node : SearchNode
            The node object containing the attribute 'state' (a frozenset of
            fact strings) describing the current world state.

        Returns
        -------
        int
            Estimated cost (number of actions) to reach all goals from the
            current state under the delete-relaxed model. Returns 0 if the
            state's facts already satisfy all goals. If some goal is
            unreachable in the relaxed model, a large finite integer is
            returned.
        """
        state = set(node.state)

        # If already a goal state, heuristic is zero
        if self.goals <= state:
            return 0

        INF = math.inf

        # Initialize costs: facts true in current state or static facts have cost 0
        cost = {fact: INF for fact in self.all_facts}
        for f in state:
            if f in cost:
                cost[f] = 0
        for f in self.static:
            if f in cost:
                cost[f] = 0

        # Propagate costs until fixpoint
        improved = True
        # Limit iterations conservatively to avoid pathological loops: at most
        # number of facts times number of operators (practically enough)
        max_iters = len(self.all_facts) * max(1, len(self.operators))
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Iterate over operators to attempt to lower cost of their add effects
            for op in self.operators:
                # Compute sum of costs of preconditions; if any precondition is
                # unreachable, op is currently not applicable in the relaxed sense.
                pre_sum = 0.0
                pre_unreachable = False
                for p in op.preconditions:
                    c = cost.get(p, INF)
                    if c == INF:
                        pre_unreachable = True
                        break
                    pre_sum += c
                if pre_unreachable:
                    continue
                op_cost = pre_sum + 1.0  # One action to apply op after achieving preconditions
                # Update costs of add effects
                for a in op.add_effects:
                    # Only consider facts known in the universe
                    if a not in cost:
                        continue
                    if op_cost < cost[a] - 1e-9:
                        cost[a] = op_cost
                        improved = True

        # Sum costs for goal facts
        total = 0.0
        unreachable = False
        for g in self.goals:
            c = cost.get(g, INF)
            if c == INF:
                unreachable = True
            else:
                total += c

        if unreachable:
            # Return a large finite value for unreachable-in-relaxed-model goals.
            # Use a value proportional to the number of known facts to keep it finite.
            fallback = max(1000, len(self.all_facts) * 10)
            # Also add the partial sum of reachable goals so heuristic reflects progress.
            return int(fallback + total)
        else:
            # Return integer heuristic; h_add can be fractional because of summation,
            # but for guiding search an integer is sufficient.
            return int(math.ceil(total))