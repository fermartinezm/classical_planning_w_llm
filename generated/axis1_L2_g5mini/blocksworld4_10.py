from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract components of a PDDL fact string by removing surrounding parentheses
    and splitting on whitespace.

    Example:
      "(gaijkwef a b)" -> ["gaijkwef", "a", "b"]

    This helper intentionally strips the leading '(' and trailing ')' as required.
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the secret_domain (dnbnd6kg). The heuristic
    computes an additive relaxed-plan estimate (h_add) based on the delete-relaxation:
    it assumes operator delete effects do not remove facts. This yields a fast,
    informative estimate of the number of actions required to reach the goal from
    a given state, and is effective to guide greedy best-first search even though
    it is not admissible.

    Assumptions
    -----------
    - Facts are represented as strings of the form "(predicate arg1 arg2 ...)".
    - Operators are provided as Task.Operator objects with `preconditions`,
      `add_effects` and `del_effects` (all sets of fact strings).
    - The task's `static` facts are available but not modified during planning;
      they can be used to speed up the relaxation propagation if needed.
    - The heuristic ignores delete effects (delete-relaxation) and uses the
      additive cost aggregation: an operator's cost is 1 and the cost to make
      all of its preconditions true is the sum of their relaxed costs.

    Heuristic Initialization
    ------------------------
    In the constructor we store:
    - self.goals : the set of goal facts to be achieved.
    - self.operators : list of operator instances for fast iteration.
    - self.static : the set of static facts (kept for possible extensions).
    - self.fact_to_adding_ops : a mapping from a fact string to the operators
      that can add that fact (used to accelerate propagation).

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    1. Immediate goal check:
       - If every goal fact is already present in the current state, return 0.
       - This ensures the heuristic is 0 exactly for goal states.

    2. Initialize relaxed costs:
       - Every fact that is true in the current state has relaxed cost 0.
       - Static facts (task.static) are also given cost 0.

    3. Relaxed forward propagation (additive / h_add):
       - Repeatedly scan operators. For an operator whose every precondition
         already has a relaxed cost, the operator becomes applicable in the
         relaxed sense. The operator's cost is 1 plus the sum of costs of its
         preconditions (additive aggregation).
       - For each add-effect of that operator, update the fact's relaxed cost
         to the minimum seen so far (since multiple operators might achieve
         the same fact with different relaxed costs).
       - Continue until no costs change or until all goals have costs.

    4. Extract heuristic value:
       - If all goal facts have relaxed costs, return the sum of their costs.
         (This is the classical h_add estimate.)
       - If one or more goals are unreachable in the relaxed model (which can
         happen if no operator can produce them), return a large finite value
         (we use sum of missed goals multiplied by a heuristic penalty). This
         keeps the heuristic finite for solvable states and produces large
         values for obviously unreachable goals.

    Rationale
    ---------
    - The additive delete-relaxed heuristic captures cumulative resource
      requirements (precondition conjunctions) better than a simple count of
      unsatisfied goals and is still fast to compute for the small operators
      sets typical in this domain.
    - Ignoring delete effects generally yields optimistic estimates, but the
      heuristic remains informative to guide greedy best-first search and is
      cheap to compute.
    """

    def __init__(self, task):
        """
        Extract goals, operators and static facts from the task and build
        auxiliary data structures for efficient computation.
        """
        # Goals (set of fact strings)
        self.goals = set(task.goals)
        # Operators (list of Operator instances)
        self.operators = list(task.operators)
        # Static facts (set) - stored for completeness / potential use
        self.static = set(task.static)

        # Map each fact to the operators that can add it (accelerates updates)
        self.fact_to_adding_ops = {}
        for op in self.operators:
            for eff in op.add_effects:
                self.fact_to_adding_ops.setdefault(eff, []).append(op)

    def __call__(self, node):
        """
        Compute the additive relaxed-plan heuristic (h_add) for the node.state.

        Returns an integer cost estimate. If the node's state already satisfies
        all goals, returns 0.
        """
        state = set(node.state)  # ensure mutability; facts are strings like "(pred a b)"

        # If already goal, return 0 (heuristic is zero only for goal states)
        if self.goals <= state:
            return 0

        # Fact costs in the relaxed model: facts true in state/static have cost 0
        fact_costs = {f: 0 for f in (state | self.static)}

        # We will propagate costs until fixpoint or until all goals have costs.
        changed = True
        # To avoid infinite loops, also bound iterations by number of facts * ops
        # but typically propagation converges quickly.
        while changed:
            changed = False

            # Early exit if all goals have known costs
            if all(g in fact_costs for g in self.goals):
                break

            # Scan every operator and try to relax its add effects
            for op in self.operators:
                # If any precondition is not yet reachable in the relaxed model, skip
                if not op.preconditions <= fact_costs.keys():
                    continue

                # Operator cost under additive aggregation = 1 + sum(costs of preconditions)
                op_cost = 1 + sum(fact_costs[p] for p in op.preconditions)

                # For each add effect, see if we can improve its cost
                for eff in op.add_effects:
                    old_cost = fact_costs.get(eff)
                    if old_cost is None or op_cost < old_cost:
                        fact_costs[eff] = op_cost
                        changed = True

            # If no change, loop will exit.

        # If not all goals are reachable in the relaxed model, return a large but finite estimate.
        missing_goals = [g for g in self.goals if g not in fact_costs]
        if missing_goals:
            # Finite penalty: count of missing goals times an upper bound estimate.
            # Use the number of operators as a proxy for a per-goal cost.
            penalty_per_goal = max(1, len(self.operators))
            return sum(fact_costs.get(g, penalty_per_goal) for g in self.goals)

        # h_add: sum of costs of individual goal facts
        total = sum(fact_costs[g] for g in self.goals)
        # Return integer estimate (costs are integers by construction)
        return int(total)