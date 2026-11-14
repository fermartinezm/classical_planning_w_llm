from heuristics.heuristic_base import Heuristic
import math

def get_parts(fact):
    """Return the space-separated parts of a fact string, stripping surrounding parentheses."""
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent relaxed-plan (additive) heuristic specialized to the
    "secret_domain" planning domain. The heuristic computes a delete-relaxed
    shortest-cost estimate (h_add) for achieving all goal facts from the
    current state. It treats each operator application as costing 1 and
    accumulates costs of preconditions additively to estimate the cost of
    producing new facts. The heuristic returns 0 for goal states.

    Assumptions
    -----------
    - The planner's internal representation uses full PDDL fact strings,
      e.g., "(predicate arg1 arg2)". get_parts() above removes the parentheses
      and splits on spaces.
    - Operators are available through task.operators and follow the Operator
      interface given in Task (preconditions, add_effects, del_effects).
    - Static facts in task.static are always true and included in initial
      reachable fact set.
    - Each operator application has unit action cost (1). This is a modeling
      choice that keeps the heuristic efficient and informative.
    - The domain's interplay of sensors, cameras and rovers can create
      conditional orderings; by using the delete-relaxation we obtain a fast
      lower-complexity estimate that is well-suited to greedy best-first
      search guidance (not necessarily admissible for optimal search).

    Heuristic Initialization
    ------------------------
    In the constructor we extract:
    - self.goals: the set of required goal facts (task.goals).
    - self.static: the immutable facts (task.static) that are true in every
      state and do not need to be re-derived.
    - self.operators: the list (set) of Operator instances for the domain.
    These are stored to avoid repeated extraction on each heuristic call.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    1. If the current state already satisfies all goals, return 0 immediately.
    2. Build the initial reachable fact set as the union of:
         - current state facts (node.state)
         - static facts (self.static)
       For these initial facts, assign distance 0 (they cost nothing to achieve).
    3. Use a delete-relaxed fixed-point propagation (h_add style):
       - Repeatedly consider every operator whose preconditions all have
         known finite distances.
       - The relaxed cost to apply an operator is 1 (for the action) plus the
         sum of the costs of its preconditions (additive aggregation).
       - For each add-effect fact of that operator, if the computed operator
         cost is smaller than the currently recorded distance for that fact,
         update the fact's distance to the operator cost.
       - Continue until no distance can be improved.
    4. Compute the heuristic value as the sum of distances of all goal facts.
       - If any goal fact remains unreachable (distance is infinite), return
         a large finite number (indicating the goals are not reachable under
         the delete-relaxed model from this state). This keeps the heuristic
         finite but large for unreachable configurations.
    5. Return the integer heuristic estimate. This estimate is intended to be
       informative for greedy best-first search, reducing node expansions by
       preferring states with small additive relaxed costs toward the goal.
    """

    def __init__(self, task):
        """
        Initialize the heuristic with the planning task information.

        Parameters
        ----------
        - task: a Task instance (see code-file-task). It provides:
          - task.goals: set of goal fact strings
          - task.static: set of static fact strings
          - task.operators: iterable of Operator instances
        """
        self.goals = set(task.goals)
        self.static = set(task.static)
        # Operators are stored as given (Operator objects with preconditions and add_effects)
        self.operators = list(task.operators)

    def __call__(self, node):
        """
        Compute the additive relaxed-plan heuristic (h_add) for the given node.

        Parameters
        ----------
        - node: a search node with attribute `state` (a frozenset of fact strings).

        Returns
        -------
        - An integer heuristic estimate of the number of actions required to
          reach the goals from node.state under the delete-relaxation.
        """
        state = set(node.state)

        # 1) If goals already satisfied, return 0
        if self.goals <= state:
            return 0

        # Initialize distances: facts in current state or static have cost 0.
        distances = {}
        for fact in state:
            distances[fact] = 0
        for fact in self.static:
            # static facts are always reachable at zero cost
            if fact not in distances:
                distances[fact] = 0

        # For all other facts unknown yet, they implicitly have infinite distance.
        INF = math.inf

        # Iteratively improve distances using delete-relaxed operator applications.
        improved = True
        while improved:
            improved = False
            # Consider each operator
            for op in self.operators:
                # If operator has no preconditions, its application cost is 1.
                if not op.preconditions:
                    op_cost = 1
                    # Apply to add effects
                    for eff in op.add_effects:
                        prev = distances.get(eff, INF)
                        if op_cost < prev:
                            distances[eff] = op_cost
                            improved = True
                    continue

                # Check if all preconditions are reachable (finite distance)
                pre_costs = []
                all_reachable = True
                for pre in op.preconditions:
                    d = distances.get(pre, INF)
                    if d == INF:
                        all_reachable = False
                        break
                    pre_costs.append(d)
                if not all_reachable:
                    continue

                # h_add: cost to achieve all preconditions plus cost of this action
                op_cost = 1 + sum(pre_costs)

                # Relax effects: if we found a cheaper way to create effect, update
                for eff in op.add_effects:
                    prev = distances.get(eff, INF)
                    if op_cost < prev:
                        distances[eff] = op_cost
                        improved = True

        # Sum distances of all goal facts (additive heuristic).
        total = 0.0
        for g in self.goals:
            d = distances.get(g, INF)
            if d == INF:
                # unreachable under relaxed model => return a large finite value
                return 10**6
            total += d

        # Ensure integer return
        return int(total)