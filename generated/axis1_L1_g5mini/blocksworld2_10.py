from heuristics.heuristic_base import Heuristic
import math

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A relaxed (delete-relaxed) additive cost heuristic tailored for the
    secret_domain (z1spq3ra). The heuristic computes, for the given state,
    a fast estimate of the number of actions required to satisfy all
    goal facts by performing a relaxed fixpoint where operator delete
    effects are ignored. The returned value is the sum of the minimal
    relaxed costs of the individual goal facts (the h_add heuristic).

    Assumptions
    -----------
    - Operators in the planning task are fully instantiated (no variables).
    - Every operator has unit action cost (each operator contributes cost 1).
    - Delete effects are ignored to produce a fast, monotone underestimate
      of dependencies between facts. This makes the heuristic non-admissible
      but effective for greedy best-first search guidance in practice.
    - The task's static facts are available (even if empty); they are stored
      for potential domain-specific extensions.

    Heuristic Initialization
    ------------------------
    In the constructor we:
    - Store the set of goal facts (task.goals).
    - Store the set of static facts (task.static) for potential use.
    - Store the list of operators (task.operators) so they can be consulted
      while computing the relaxed reachability / additive costs.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    1. Goal-check:
       - If all goal facts are already true in the current state, return 0.

    2. Initialization of facts:
       - All facts that are true in the current state are assigned cost 0
         (they already hold).
       - Facts not present are considered to have infinite cost until proven
         reachable via operators.

    3. Operator relaxation and propagation:
       - Repeatedly scan all operators. For each operator whose preconditions
         all have finite costs, compute the operator application cost as:
             op_cost = 1 + sum(cost(pre) for pre in op.preconditions)
         (we count the unit cost of the operator plus the additive cost of
         achieving its preconditions).
       - For every add-effect fact of that operator, if op_cost is lower
         than the currently recorded cost for that fact, update it.
       - Continue iterations until a full pass produces no cost updates
         (fixpoint reached). This is the standard h_add propagation
         under delete-relaxation.

    4. Combining goal costs:
       - The heuristic value is the sum of the individual costs of each
         goal fact (h_add sums costs for conjunctive goals).
       - If any goal fact remains unreachable in the relaxed graph (infinite
         cost), the heuristic returns a large finite penalty (to keep the
         search numeric and avoid infinities) — here we use a large constant.

    Notes on correctness and practicality
    ------------------------------------
    - The heuristic is 0 exactly when the goal is satisfied (check performed).
    - It always returns a finite number for states where goals are reachable
      in the relaxed graph; for cases where the relaxed graph does not
      reach some goal fact, a large finite constant is returned to indicate
      high cost.
    - This implementation is efficient: each iteration scans all operators
      and updates only when strictly improving costs. For typical problem
      sizes in this domain the number of iterations until fixpoint is small.
    """

    def __init__(self, task):
        """
        Extract goals, static facts and operators from the task.

        Parameters
        ----------
        task : Task
            The planning task object which contains goals, static facts
            and instantiated operators.
        """
        # Goal facts: set / frozenset of strings like "(yfm3frbn b1)"
        self.goals = set(task.goals)
        # Store static facts (may be empty for this domain)
        self.static = set(task.static)
        # Operators: the Task uses Operator instances with concrete fact strings
        self.operators = list(task.operators)
        # A large finite cost to represent unreachable goals under the relaxation
        self._UNREACHABLE_COST = 10**6

    def __call__(self, node):
        """
        Compute the heuristic value for the given search node.

        Parameters
        ----------
        node : SearchNode
            The node whose state attribute is a frozenset of fact strings.

        Returns
        -------
        int
            Estimated number of actions to reach a goal state (lower is better).
        """
        state = node.state

        # Quick goal test: if all goals already hold, heuristic is zero.
        if self.goals <= state:
            return 0

        # Initialize costs: facts true in state cost 0; others are infinite.
        fact_cost = {fact: 0 for fact in state}

        # For facts not present, we will lazily add them to fact_cost when discovered.
        INF = float("inf")

        # Iteratively propagate costs using delete-relaxed additive rule.
        improved = True
        while improved:
            improved = False
            # For each operator, check if all preconditions have known finite cost.
            for op in self.operators:
                # If an operator has no preconditions, it is immediately applicable
                # with op_cost = 1.
                preconds = op.preconditions
                # Determine precondition total cost; if any precond unreachable -> skip.
                total_pre_cost = 0
                pre_unreachable = False
                for pre in preconds:
                    cost = fact_cost.get(pre, INF)
                    if cost == INF:
                        pre_unreachable = True
                        break
                    total_pre_cost += cost
                if pre_unreachable:
                    continue
                # Cost of applying this operator (unit action cost 1).
                op_cost = 1 + total_pre_cost

                # For each add-effect, see if we can improve its cost.
                for add in op.add_effects:
                    old_cost = fact_cost.get(add, INF)
                    if op_cost < old_cost:
                        fact_cost[add] = op_cost
                        improved = True

        # Sum the additive costs of all goal facts.
        total = 0
        for g in self.goals:
            c = fact_cost.get(g, INF)
            if c == INF:
                # Some goal unreachable in relaxed graph; return a large finite penalty.
                return self._UNREACHABLE_COST
            total += c

        # Return integer heuristic value (rounded up).
        # The heuristic may be non-admissible because deletes were ignored
        return int(math.ceil(total))