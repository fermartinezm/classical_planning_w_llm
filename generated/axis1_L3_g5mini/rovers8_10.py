from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Return the token list of a PDDL fact string, ignoring surrounding parentheses."""
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the secret_domain ("content-portion") domain.
    The heuristic estimates the number of actions still required to satisfy the
    goal facts by performing a shallow, goal-directed backward search (a small
   -depth relaxed-plan style estimate). It reasons about which operators can
    add a missing goal fact, then estimates the cost to achieve that operator's
    preconditions. The method ignores delete effects (i.e., it is a relaxed
    estimate) and uses a small recursion depth and conservative fallback costs
    to keep the estimate efficiently computable.

    Assumptions
    -----------
    - The task's operators are fully grounded (their preconditions/add-effects
      are concrete fact strings) as provided to the planner.
    - Static facts provided in task.static are always true in any reachable
      state and can be used to satisfy preconditions at zero cost.
    - The domain actions required to establish a goal fact are not arbitrarily
      deep; a shallow backward search (default depth 4) provides a useful
      approximation.
    - Costs are unit per operator; the heuristic returns an integer estimate
      equal to the (approximate) number of operators required.

    Heuristic Initialization
    ------------------------
    The constructor extracts:
    - self.goals: set of goal fact strings that must hold.
    - self.static: set of facts that are true in every state (task.static).
    - self.operators: list of Operator instances (task.operators).
    - self.effect_to_ops: a mapping from a fact string (an add-effect) to the
      list of operators that add that fact (for fast identification of
      candidate actions when reasoning backward from goals).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node/state:
    1. If all goals are already satisfied in the state, return 0.
    2. Otherwise, for each goal fact g that is not true in the state:
       a. Compute an estimated cost to achieve g using a recursive backward
          search with limited depth (default depth is 4). The cost of
          achieving g is:
             - 0 if g is already true in the current state or is a static fact.
             - min_{op in OpsAdding(g)} [ 1 + sum(cost(pre) for pre in op.preconditions) ]
               where cost(pre) is computed recursively.
          If no operator can (within the search depth) achieve a precondition,
          that operator is ignored. If none of the operators can be used, a
          conservative fallback cost is returned.
       b. Memoize intermediate fact costs to avoid repeated work.
    3. Sum the per-goal estimated costs to produce the final heuristic value.
       (This sum corresponds to a simple relaxed-plan aggregation and will
       typically over-estimate interactions and concurrency; it is acceptable
       because the greedy best-first search does not require admissibility.)
    4. Return the integer sum as the heuristic.

    Notes on parameters chosen:
    - recursion_depth: A small depth (4) balances informativeness and speed.
    - fallback_cost and max_cost: Conservative finite costs used when the
      backward search cannot find a supporting operator within the allowed
      depth. They ensure the heuristic is always finite for solvable states.
    """

    def __init__(self, task):
        """Extract goals, static facts, and build effect -> operators index."""
        self.goals = task.goals
        self.static = set(task.static)
        # Operators assumed to be grounded Operator instances
        self.operators = list(task.operators)

        # Map each effect fact to the list of operators that add it.
        self.effect_to_ops = {}
        for op in self.operators:
            for eff in op.add_effects:
                self.effect_to_ops.setdefault(eff, []).append(op)

        # Parameters controlling recursion depth and fallback costs.
        self.recursion_depth = 4
        self.fallback_cost = 4   # conservative cost when we cannot prove a smaller cost
        self.max_cost = 1000     # safety cap to keep heuristic finite

    def __call__(self, node):
        """
        Compute the heuristic estimate for the given search node.

        Returns an integer estimate of the number of actions required to
        satisfy all goal facts from the current node.state.
        """
        state = node.state

        # Quick check: 0 for goal states (requirement from checklist).
        if self.goals <= state:
            return 0

        memo = {}  # memoization: fact -> estimated cost

        def fact_cost(fact, depth):
            """
            Estimate the minimum number of operators required to make 'fact' true.

            - If fact already true in state or fact is static -> cost 0.
            - If memoized -> return cached value.
            - If depth == 0 -> return fallback cost (conservative).
            - Otherwise consider every operator that adds 'fact' and take the
              minimum over 1 + sum(cost(pre) for pre in op.preconditions).
            - If no operator applicable (or all lead to unsatisfiable preconditions
              within depth), return fallback cost.
            """
            # Already true in current state or static facts -> zero cost.
            if fact in state or fact in self.static:
                return 0

            if fact in memo:
                return memo[fact]

            if depth <= 0:
                memo[fact] = self.fallback_cost
                return memo[fact]

            candidate_ops = self.effect_to_ops.get(fact, [])
            best = None

            for op in candidate_ops:
                # Sum the estimated costs for op's preconditions (relaxed).
                total = 1  # cost for applying op itself
                skip_op = False
                for pre in op.preconditions:
                    # If pre is already true (state or static), cost 0.
                    if pre in state or pre in self.static:
                        continue
                    # Otherwise compute recursively.
                    c = fact_cost(pre, depth - 1)
                    # If cost is extremely large treat as unsatisfiable within depth.
                    if c >= self.max_cost:
                        skip_op = True
                        break
                    total += c
                    # Early abort if total already worse than best found.
                    if best is not None and total >= best:
                        skip_op = True
                        break
                if not skip_op:
                    if best is None or total < best:
                        best = total

            if best is None:
                # No operator found within allowed depth -> conservative fallback.
                memo[fact] = self.fallback_cost
            else:
                memo[fact] = min(best, self.max_cost)

            return memo[fact]

        # Sum costs for each unsatisfied goal fact.
        total_cost = 0
        for goal_fact in self.goals:
            if goal_fact in state:
                continue
            c = fact_cost(goal_fact, self.recursion_depth)
            # Guard to keep finite and reasonably scaled
            if c is None:
                c = self.fallback_cost
            total_cost += min(c, self.max_cost)

        # Ensure returned value is an int and finite.
        if total_cost >= self.max_cost:
            total_cost = int(self.max_cost - 1)
        return int(total_cost)