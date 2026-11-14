from math import inf
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """
    Remove surrounding parentheses from a PDDL fact string and split into tokens.
    Example: "(at obj1 loc1)" -> ["at", "obj1", "loc1"]
    This function intentionally strips the first and last character (the parentheses)
    so callers don't have to remember to do that.
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A relaxed-planning (additive-relaxation) based heuristic tailored for the
    secret_domain ("content-portion") planning tasks. The heuristic computes
    an approximate number of actions still required to satisfy all goal facts
    by building a cost estimate for each reachable fact under the relaxed
    semantics (i.e., ignoring delete effects). The final heuristic value is
    the sum of the independent costs of the unmet goal facts (the h_add
    estimate). This provides a relatively informed, domain-sensitive value
    while remaining efficient.

    Assumptions
    -----------
    - All operators in task.operators are grounded (their preconditions and
      effects are concrete facts).
    - Static facts in task.static are always true and available at zero cost.
    - Operator application cost is 1 (every operator counts as one action).
    - Delete effects are ignored (relaxed planning graph). This makes the
      heuristic inadmissible but fast and informative for greedy best-first search.

    Heuristic Initialization
    ------------------------
    In the constructor we store:
    - self.goals: the set of goal facts that must hold.
    - self.static: the set of static facts that are true in every state.
    - self.operators: the list of operators for the task (used to relaxively
      derive new facts and their estimated costs).
    We also expose small helpers (get_parts) to parse facts when needed.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the given state already satisfies all goals, return 0.
    2. Initialize a cost mapping for facts:
       - Facts that are true in the current state have cost 0.
       - Facts that are static (task.static) also have cost 0 (they are always available).
       - All other facts start with cost = +infinity.
    3. Relaxed propagation:
       - Repeatedly consider each operator. If all of its preconditions have a
         finite cost, then the operator is achievable in the relaxed model.
       - The relaxed cost to achieve the operator's add effects is:
             op_cost = 1 + sum(cost(precondition) for each precondition)
         (This is the classical additive relaxation; it accounts for the cost of
         establishing all preconditions and the cost of applying the operator).
       - For every add effect of that operator, if op_cost is smaller than the
         current recorded cost for that fact, update the fact's cost.
       - Continue this propagation until no further cost improvements occur.
    4. Heuristic value:
       - For each goal fact that is not already true in the current state,
         take its relaxed cost from the cost mapping and sum these costs.
       - If one (or more) goal facts remain unreachable under the relaxation
         (cost = +infinity), the heuristic returns a large finite value
         (indicating the state is unlikely to be useful). For solvable tasks
         with sufficient grounding, reachable goals will acquire finite costs.
    5. Return the integer heuristic value (sum of relaxed costs for unsatisfied goals).
    """

    def __init__(self, task):
        """Store goals, static facts, and the operators for the relaxation."""
        # The goal facts to be satisfied (a frozenset of strings).
        self.goals = task.goals

        # Static facts are treated as always available at zero cost.
        self.static = set(task.static)

        # Operators (Operator instances) for relaxed propagation.
        # We copy to a list for deterministic iteration.
        self.operators = list(task.operators)

    def __call__(self, node):
        """
        Compute the additive relaxed (h_add) heuristic for the given search node.

        Returns an integer estimate of the number of actions to reach the goal
        from the node's state. Returns 0 iff the node state already satisfies
        all goals.
        """
        state = node.state

        # 1) If the node already satisfies all goals, heuristic is 0.
        if self.goals <= state:
            return 0

        # 2) Initialize fact costs.
        fact_cost = {}  # mapping fact string -> numeric relaxed cost
        # Facts true in the current (concrete) state are available for free.
        for f in state:
            fact_cost[f] = 0
        # Static facts are also available for free (even if not in the state set).
        for f in self.static:
            fact_cost[f] = 0

        # 3) Relaxed propagation (iterative fixpoint).
        changed = True
        while changed:
            changed = False
            for op in self.operators:
                # Determine if all preconditions have finite cost.
                preconds = op.preconditions
                precond_costs = []
                all_reachable = True
                for p in preconds:
                    c = fact_cost.get(p, inf)
                    if c == inf:
                        all_reachable = False
                        break
                    precond_costs.append(c)
                if not all_reachable:
                    continue

                # If reachable, compute operator relaxed cost using additive model.
                op_cost = 1 + sum(precond_costs)  # 1 for applying the operator

                # Relax add effects: set their cost to min(existing_cost, op_cost).
                for e in op.add_effects:
                    old = fact_cost.get(e, inf)
                    if op_cost < old:
                        fact_cost[e] = op_cost
                        changed = True

        # 4) Sum costs for goal facts not yet true in current state.
        total = 0
        for g in self.goals:
            if g in state:
                continue
            c = fact_cost.get(g, inf)
            if c == inf:
                # Unreachable under relaxation: return a large but finite cost.
                # This keeps the heuristic finite while signaling very poor states.
                return 10**6
            total += c

        # Return integer estimate.
        return int(total)