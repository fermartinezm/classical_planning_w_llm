from functools import lru_cache
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Remove surrounding parentheses and split a fact into its parts."""
    return fact[1:-1].strip().split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the "content-portion" (secret_domain) planning
    task. The heuristic estimates the number of actions required to satisfy the
    remaining (unsatisfied) goal facts by performing a shallow backward-chaining
    analysis over the ground operators in the task.

    Assumptions
    -----------
    - Operators provided to the planner are fully ground (their preconditions
      and effects are concrete facts as strings).
    - Static facts (task.static) are always true and do not need to be achieved.
    - The planner represents facts as strings with surrounding parentheses,
      e.g. "(predicate arg1 arg2)". The helper get_parts removes parentheses.
    - The heuristic is not required to be admissible. It aims to be informative
      and cheaply computable, hence it performs a limited-depth backward search.
    - If a fact is not producible by any operator (and is not already true or
      static) the heuristic returns a small finite estimate rather than +inf.

    Heuristic Initialization
    ------------------------
    In the constructor the heuristic extracts and indexes:
    - All goal facts that must be achieved (self.goals).
    - The set of static facts (self.static) that are true in every state.
    - A mapping from each fact string to the list of operators that add that
      exact fact (self.fact_adders). This supports quick lookup of which
      operators can achieve a specific fact during the backward-chaining pass.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node (with concrete state facts) the heuristic proceeds as follows:

    1. If the node's state already satisfies all goals, return 0.

    2. For each goal fact that is not yet true in the state:
       - Use a recursive, depth-limited backward search to estimate the minimal
         number of operator applications required to make that fact true.
       - The search works on ground facts and operators:
         a) If the fact is already true in the current state or is static, cost = 0.
         b) Otherwise, consider each operator that has this exact fact in its
            add-effects. For such an operator, estimate the cost as:
               cost(op) = 1 + sum(cost(precondition) for each precondition)
            where cost(precondition) is computed recursively.
         c) The estimated cost for the fact is the minimum cost(op) across all
            operators that can produce the fact.
       - The recursion is depth-limited (MAX_DEPTH) to keep the heuristic
         efficient; if the depth limit is reached, a conservative finite
         fallback cost (FALLBACK_COST) is returned.
       - Additionally, cycles are detected and handled by returning the
         fallback cost for already visited facts to avoid infinite loops.

    3. The heuristic value for the state is the sum of the estimated costs for
       all unsatisfied goal facts. (This may overestimate when goals share
       subgoals, but is useful to guide greedy best-first search.)

    Implementation notes
    --------------------
    - The backward search operates on ground facts (strings) and uses memoization
      to cache computed costs for facts within a single heuristic call.
    - The approach is deliberately shallow and conservative to remain fast while
      capturing domain structure: it takes into account which operators can
      directly achieve a goal and how many of their preconditions are already
      satisfied, as well as one or two layers of producers for missing
      preconditions.
    """

    def __init__(self, task):
        """
        Initialize indices and static information from the task.

        Parameters
        ----------
        task : heuristics.task.Task-like object
            The planning task provides .goals (set of goal fact strings),
            .static (set of static fact strings), and .operators (iterable of
            Operator objects with .add_effects and .preconditions as sets
            of fact strings).
        """
        self.goals = task.goals
        self.static = set(task.static)

        # Build mapping: fact_string -> list of operators that add that fact.
        fact_adders = {}
        for op in task.operators:
            for add in op.add_effects:
                fact_adders.setdefault(add, []).append(op)
        self.fact_adders = fact_adders

        # Parameters controlling recursion and fallback costs
        self.MAX_DEPTH = 3
        self.FALLBACK_COST = 4  # small finite cost when a fact is hard/non-producible

    def __call__(self, node):
        """
        Compute heuristic estimate for the given search node.

        Parameters
        ----------
        node : search.Node-like object
            Node with .state attribute, a set/frozenset of fact strings.

        Returns
        -------
        int
            Non-negative integer estimate of remaining actions to reach the goals.
        """
        state = set(node.state)

        # Quick check: if goals already satisfied, heuristic is 0.
        if self.goals <= state:
            return 0

        # Memoization cache for fact costs during this heuristic computation.
        memo = {}

        def estimate_fact(fact, depth, visited):
            """
            Estimate cost (number of operator applications) to make 'fact' true.

            - If fact is already true in state or is static -> cost 0.
            - If depth exceeds MAX_DEPTH -> FALLBACK_COST.
            - If no operator adds this exact fact -> FALLBACK_COST.
            - Otherwise consider all operators that add the fact and compute:
                cost(op) = 1 + sum(estimate_cost(pre) for pre in op.preconditions)
              using minimal cost across operators.

            Uses memoization and visited set to prevent cycles.
            """
            if fact in state or fact in self.static:
                return 0

            if fact in memo:
                return memo[fact]

            if depth >= self.MAX_DEPTH:
                memo[fact] = self.FALLBACK_COST
                return memo[fact]

            if fact in visited:
                # Cycle detected -> fallback finite cost
                memo[fact] = self.FALLBACK_COST
                return memo[fact]

            adders = self.fact_adders.get(fact, [])
            if not adders:
                # No operator produces this fact -> fallback cost
                memo[fact] = self.FALLBACK_COST
                return memo[fact]

            best_cost = None
            new_visited = visited | {fact}
            for op in adders:
                # Sum costs for preconditions that are not presently true/static.
                pre_cost = 0
                for pre in op.preconditions:
                    if pre in state or pre in self.static:
                        continue
                    c = estimate_fact(pre, depth + 1, new_visited)
                    pre_cost += c
                    # Early pruning if current pre_cost already worse than best_cost
                    if best_cost is not None and pre_cost + 1 >= best_cost:
                        break
                else:
                    op_cost = 1 + pre_cost  # 1 for applying this operator
                    if best_cost is None or op_cost < best_cost:
                        best_cost = op_cost

            if best_cost is None:
                # Nothing gave a finite cost within depth -> fallback
                memo[fact] = self.FALLBACK_COST
            else:
                memo[fact] = best_cost
            return memo[fact]

        total = 0
        for goal in self.goals:
            if goal in state:
                continue
            total += estimate_fact(goal, 0, frozenset())

        # Ensure heuristic is integer
        return int(total)