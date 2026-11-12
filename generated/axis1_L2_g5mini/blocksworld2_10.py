from math import inf
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing surrounding parentheses
    and splitting on whitespace.

    Example:
      "(gaijkwef a b)" -> ["gaijkwef", "a", "b"]

    This helper ensures we ignore the surrounding brackets when decomposing facts.
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent relaxed (additive) heuristic for the secret_domain
    (dnbnd6kg). The heuristic estimates the number of actions required to
    satisfy the remaining goal facts by computing minimal additive costs to
    achieve each fact in a relaxed problem where delete effects are ignored.
    Costs are computed from the given state's set of true facts and the
    domain's operators (as provided via the Task object).

    Assumptions
    -----------
    - The planner represents facts as strings like "(predicate arg1 arg2)".
      get_parts(...) is available to split such facts while ignoring brackets.
    - Operators provided in the Task are already fully instantiated (their
      preconditions and add_effects are concrete fact strings).
    - We ignore delete effects (relaxed planning graph). This makes the
      heuristic inadmissible but usually informative and cheap to compute.
    - If a goal fact is unreachable in the relaxed problem we return a finite
      large estimate (penalizing unreachable goals) instead of infinity.

    Heuristic Initialization
    ------------------------
    During initialization we extract:
    - self.goals: the set of goal fact strings that must be true.
    - self.operators: the list of operator instances available in the task.
    - self.static: the set of static facts (kept for potential extensions).
    No heavy preprocessing is necessary because operators are already
    instantiated and small in number.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    1. If the current state already satisfies all goals, return 0.

    2. Initialize a dictionary `costs` mapping facts to their currently known
       minimal cost (in number of actions) to achieve them in the relaxed
       problem. All facts present in the current state have cost 0.

    3. Repeatedly relax-apply operators to improve costs:
       - For each operator, compute the cost to apply it as:
           op_cost = 1 + sum(cost(pre) for pre in op.preconditions)
         If some precondition is currently unknown (no cost assigned),
         treat its cost as +infinity for this operator (i.e., the operator
         cannot yet improve any add effect).
       - For each fact in op.add_effects, if op_cost is smaller than the
         current known cost for that fact, update cost[fact] = op_cost.
       - Iterate this process until no costs improve in a full pass.

       This procedure computes the classical additive (h_add) relaxed
       heuristic: each fact's cost is the minimal additive cost to achieve it
       in the delete-free relaxation.

    4. Compute the heuristic value:
       - If every goal fact has a finite cost, return the sum of costs for
         the goal facts (additive heuristic).
       - If some goal facts are unreachable in the relaxation, return a
         finite penalized estimate: sum(costs of reachable goals) +
         penalty_per_missing * number_of_missing_goals. The penalty keeps
         the heuristic finite while strongly preferring states where goals
         are reachable in the relaxed sense.

    Notes on design choices
    -----------------------
    - The additive (sum) heuristic guides greedy best-first search by
      preferring states that reduce the total estimated remaining action
      count. It is more informed than naive unsatisfied-goals counting while
      remaining efficiently computable.
    - The relaxation ignores deletes; in this domain (with toggling facts
      like zzar8qlk and transfers between objects) this is a reasonable
      trade-off: we obtain useful distance estimates while avoiding expensive
      exact reasoning about cyclic toggles.
    """

    def __init__(self, task):
        """
        Initialize heuristic with task-specific information.

        Parameters
        ----------
        task : Task
            The planning task instance (provides goals, operators, static facts).
        """
        # goal facts (set of strings)
        self.goals = set(task.goals)
        # list of Operator objects (each has preconditions and add_effects)
        self.operators = list(task.operators)
        # static facts (kept for future extensions; not required by current heuristic)
        self.static = set(task.static)

    def __call__(self, node):
        """
        Compute the additive relaxed heuristic (h_add) for the node's state.

        Returns
        -------
        int
            Estimated number of actions to reach the goal from node.state.
            Returns 0 if the state already satisfies all goals.
            Returns a finite penalized estimate if some goals are unreachable
            in the relaxed problem.
        """
        state = set(node.state)

        # If goal already satisfied, heuristic is zero.
        if self.goals <= state:
            return 0

        # Initialize costs: facts true in the current state cost 0.
        costs = {fact: 0 for fact in state}

        # Iteratively improve costs using relaxed operator application.
        improved = True
        # To avoid pathological long loops in degenerate cases, cap iterations.
        max_iterations = 1000
        iteration = 0

        while improved and iteration < max_iterations:
            iteration += 1
            improved = False

            for op in self.operators:
                # Sum cost of preconditions; if some precondition has no cost yet
                # we consider op currently inapplicable in the relaxed sense.
                preconds = op.preconditions
                # If no preconditions: cost to apply is 1.
                if not preconds:
                    op_cost = 1
                else:
                    total_pre_cost = 0
                    unreachable = False
                    for p in preconds:
                        if p in costs:
                            total_pre_cost += costs[p]
                        else:
                            unreachable = True
                            break
                    if unreachable:
                        # Operator cannot be applied yet in the relaxation.
                        continue
                    op_cost = 1 + total_pre_cost

                # Try to improve costs of add effects.
                for add in op.add_effects:
                    prev_cost = costs.get(add, inf)
                    if op_cost < prev_cost:
                        costs[add] = op_cost
                        improved = True

            # Small micro-optimization: if all goals have known finite cost, we
            # can stop early (no need to continue refining other facts).
            if all(g in costs for g in self.goals):
                break

        # After propagation, compute heuristic value for goals.
        missing_goals = [g for g in self.goals if g not in costs]
        reachable_goal_cost = sum(costs[g] for g in self.goals if g in costs)

        if not missing_goals:
            # All goals reachable in relaxed problem: return additive cost.
            # Round to integer (costs are integers by construction).
            return int(reachable_goal_cost)
        else:
            # Some goals unreachable in relaxation: penalize heavily but return
            # a finite estimate. The penalty should be large enough to prefer
            # states that make progress toward all goals.
            penalty_per_missing = 10
            penalty = penalty_per_missing * len(missing_goals)
            return int(reachable_goal_cost + penalty)