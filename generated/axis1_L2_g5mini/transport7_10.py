from heuristics.heuristic_base import Heuristic
from collections import deque
import math

def get_parts(fact):
    """Return the components of a PDDL fact string by stripping surrounding
    parentheses and splitting on whitespace.

    Example:
      "(at obj1 loc2)" -> ["at", "obj1", "loc2"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    This is a domain-dependent relaxed-cost heuristic for the secret_domain
    (mtfkspuf) domain. It estimates the number of actions required to satisfy
    the remaining goal facts by performing a relaxed reachability analysis:
    operators are treated as if they only add facts (delete effects are
    ignored). Costs are accumulated additively (h_add style): the cost of an
    operator is 1 plus the sum of the costs of its preconditions. The cost of
    a fact is the minimal such cost discovered in the relaxed graph. The
    heuristic value for a state is the sum of the costs of all goal facts that
    are currently not true.

    Assumptions
    -----------
    - The planner represents facts exactly as strings like "(pred a b)".
    - The operators available on the Task are instances of the planner's
      Operator class and expose `preconditions` and `add_effects` as sets of
      such fact strings.
    - Static facts (task.static) hold in all states and are available for
      satisfying preconditions at zero cost.
    - Using an additive relaxed-cost estimate (h_add) provides a good,
      efficiently computable, and non-admissible heuristic for guiding greedy
      best-first search in this domain.

    Heuristic Initialization
    ------------------------
    In the constructor we:
    - Remember the goal facts (task.goals).
    - Store a reference to the operators (task.operators).
    - Store the set of static facts (task.static).
    The constructor does not attempt expensive preprocessing beyond storing
    these sets because operator and fact counts are moderate in the provided
    instances and the per-state relaxed-cost computation below is kept
    efficient.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the current state already satisfies all goal facts, return 0.
    2. Build an initial cost map for facts:
       - Facts that are true in the current state or are static facts receive
         cost 0 (already achieved).
       - All other facts start with cost = +infinity.
    3. Use a relaxation loop (like a Dijkstra / Bellman-Ford over facts and
       operators) to compute minimal additive costs for facts under the
       assumption that operators only add facts:
       - For each operator whose preconditions all have finite costs,
         compute the operator cost:
            op_cost = 1 + sum(cost(precondition) for precondition in preconds)
         and relax all its add_effect facts:
            cost(effect) = min(cost(effect), op_cost)
       - Repeat until no cost improves or all goal facts have finite costs.
    4. The heuristic value is the sum of costs of goal facts that are not
       currently true in the state. If some goal fact remains unreachable in
       the relaxation (infinite cost), return a large finite value (we use
       len(all_facts) + number_of_operators as a conservative bound) to keep
       the heuristic finite while signaling that the state is far from a
       relaxed-solution.
    5. This heuristic runs in time O(iterations * num_ops * avg_preconds),
       and in practice converges quickly on the provided instances.
    """

    def __init__(self, task):
        # Goals are stored as a frozenset of fact strings.
        self.goals = task.goals
        # Keep operators and static facts for the relaxed-cost propagation.
        self.operators = list(task.operators)
        self.static = set(task.static)
        # For a reasonable finite cap when some goal is unreachable in the
        # relaxation, compute a conservative upper bound.
        # We will use number of unique facts in the task if available, else fallback.
        try:
            self._finite_cap = len(task.facts) + len(self.operators)
        except Exception:
            self._finite_cap = 1000

    def __call__(self, node):
        """
        Compute the heuristic value for the given search node.

        Parameters
        ----------
        node : search node object
            The node contains a `.state` attribute which is a frozenset of
            fact strings currently true in the world.

        Returns
        -------
        int
            Estimated number of actions to reach a goal state (non-admissible).
        """
        state = node.state

        # 1) If all goals already satisfied -> heuristic 0.
        if self.goals <= state:
            return 0

        # 2) Initialize fact costs: facts in current state or static -> 0, others -> inf.
        fact_cost = {}
        INF = float("inf")

        # Start with all known facts (from state and static) at cost 0.
        for f in state:
            fact_cost[f] = 0
        for f in self.static:
            fact_cost.setdefault(f, 0)

        # Collect universe of facts to bound the algorithm if needed.
        # We will gather also all add effects and preconditions to reason about unreachable facts.
        all_facts = set(fact_cost.keys())
        for op in self.operators:
            all_facts.update(op.add_effects)
            all_facts.update(op.preconditions)
        # Ensure every fact has an entry in fact_cost (default INF).
        for f in all_facts:
            fact_cost.setdefault(f, INF)

        # 3) Relaxation loop: repeatedly try to apply operators under the
        # additive cost model until no improvement or until all goal facts
        # have finite cost.
        changed = True
        # To avoid endless loops in degenerate cases, cap the maximum number
        # of iterations by number of facts + operators (conservative).
        max_iters = max(50, len(all_facts) + len(self.operators))
        iters = 0
        while changed and iters < max_iters:
            iters += 1
            changed = False
            for op in self.operators:
                # If any precondition is not known at all, skip until it may be discovered.
                # Here we require finite cost for all preconditions to compute operator cost.
                preconds = op.preconditions
                # If there are no preconditions, op_cost = 1.
                skip = False
                sum_pre = 0
                for p in preconds:
                    c = fact_cost.get(p, INF)
                    if c == INF:
                        skip = True
                        break
                    sum_pre += c
                if skip:
                    continue
                op_cost = 1 + sum_pre
                # Relax add effects
                for eff in op.add_effects:
                    if op_cost < fact_cost.get(eff, INF):
                        fact_cost[eff] = op_cost
                        changed = True
            # Early termination if all goals have finite cost
            all_goals_finite = True
            for g in self.goals:
                if fact_cost.get(g, INF) == INF and g not in state:
                    all_goals_finite = False
                    break
            if all_goals_finite:
                break

        # 4) Sum the costs for goal facts that are not yet true.
        total = 0
        unreachable = False
        for g in self.goals:
            if g in state:
                continue
            c = fact_cost.get(g, INF)
            if c == INF:
                unreachable = True
            else:
                total += int(math.ceil(c))

        if unreachable:
            # If any goal was unreachable in the relaxed analysis, return a large
            # finite value rather than infinity so the search algorithm can still
            # compare nodes. Use conservative cap computed in constructor.
            return self._finite_cap + total

        return total