from heuristics.heuristic_base import Heuristic
from typing import List, Dict, Set


def get_parts(fact: str) -> List[str]:
    """
    Extract components of a PDDL fact by removing the surrounding parentheses
    and splitting on whitespace.

    Example:
      "(s0899d5u b1 b2)" -> ["s0899d5u", "b1", "b2"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the secret_domain (z1spq3ra).
    The heuristic computes an additive relaxed-plan estimate (h_add): it computes
    the minimum cost (in number of actions, assuming each action costs 1)
    to achieve each atomic goal when delete-effects are ignored, and returns
    the sum of these individual costs. This provides a quick, informative,
    non-admissible estimate that accounts for conjunctive goal structure and
    chained operator preconditions.

    Assumptions
    -----------
    - Each operator has unit cost.
    - Delete effects are ignored (relaxed planning graph), allowing efficient
      propagation of cheapest-achievement costs for facts.
    - Facts are represented as strings in the form "(predicate arg1 arg2 ...)".
    - The operator instances available through task.operators correctly list
      preconditions and add_effects using the same string representation.
    - Static facts (task.static) are available but not strictly required for
      the additive propagation (they are treated as facts true with cost 0).

    Heuristic Initialization
    ------------------------
    - Store the task goals (self.goals).
    - Store the operator set (self.operators).
    - Record static facts (task.static) to seed the relaxed cost propagation
      as facts with zero cost (they hold in every state).
    - No domain-specific hand-coded rules are required because the additive
      relaxed-planning propagation uses the operators defined for the domain.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node (with a concrete state), the heuristic proceeds
    as follows:

    1. Fast goal-check:
       - If all goal facts are already present in the current state, return 0.

    2. Initialization of fact costs:
       - For every fact true in the current state (node.state) or in the
         static facts, set its relaxed-achievement cost to 0.
       - For all other facts initialize cost to +inf.

    3. Repeated propagation (fixed-point):
       - Repeatedly consider every operator:
         - If all of the operator's preconditions have finite costs,
           compute the operator's precondition-cost as the sum of those costs.
         - Each add-effect of that operator can then be achieved for
           cost (precondition-cost + 1) (1 for applying the operator).
         - If this new cost is better (smaller) than the recorded cost of
           that add-effect, update it.
       - Continue until no fact cost improves in an iteration.

    4. Construct heuristic value:
       - For each goal fact, look up its computed cost.
       - The h_add heuristic returns the sum of the individual goal costs.
         This captures the idea that conjunctive goals require achieving
         multiple facts (and may double-count shared sub-achievements).
       - If any goal remains unreachable in the relaxed sense (cost = +inf),
         the heuristic returns a finite large estimate instead of infinity
         (we cap unreachable contributions with a fixed large penalty). This
         ensures the heuristic always returns a finite integer, which is
         useful for greedy best-first search implementations.

    Rationale and benefits
    ----------------------
    - The additive relaxed-cost heuristic balances fidelity and efficiency:
      it reasons about action chaining (via operator preconditions)
      while avoiding the combinatorial cost of a full planner.
    - It is domain-generic but produces informative estimates for the
      secret_domain because it uses the actual operators (including those
      that create/transfer yfm3frbn, jjtnwk3y, and s0899d5u facts).
    """

    def __init__(self, task):
        # Goals are the set of facts that must be true in a goal state.
        self.goals: Set[str] = set(task.goals)
        # Operators (instances of Operator) from the task; each has preconditions and add_effects as strings.
        self.operators = list(task.operators)
        # Static facts true in every state: treat them as cost-0 seeds.
        self.static_facts: Set[str] = set(task.static) if task.static is not None else set()

    def __call__(self, node):
        """
        Compute the additive relaxed-plan heuristic (h_add) value for the node's state.

        Returns an integer estimate (sum of per-goal relaxed costs). If a goal
        is unreachable in the relaxed problem, a finite large penalty is used
        instead to ensure the heuristic always returns a finite integer.
        """
        state: Set[str] = set(node.state)

        # 1) If all goals satisfied, heuristic is zero.
        if self.goals <= state:
            return 0

        INF = float("inf")

        # 2) Initialize costs: facts present in the current state or static facts have cost 0.
        fact_costs: Dict[str, float] = {}
        # Seed costs for facts known to be true initially.
        for fact in state | self.static_facts:
            fact_costs[fact] = 0.0

        # To speed up iterations, collect all candidate facts (from operators' add_effects and goals).
        candidate_facts = set(self.goals)
        for op in self.operators:
            candidate_facts.update(op.add_effects)
            candidate_facts.update(op.preconditions)
        # Ensure every candidate fact has an entry in fact_costs (initialized to INF where unknown).
        for f in candidate_facts:
            if f not in fact_costs:
                fact_costs[f] = INF

        # 3) Relaxed propagation (similar to h_add).
        changed = True
        # To avoid iterating forever in degenerate cases, cap iterations by number of facts + operators.
        max_iters = max(10, len(candidate_facts) + len(self.operators) + 5)
        iters = 0
        while changed and iters < max_iters:
            changed = False
            iters += 1
            # For each operator, if all preconditions have finite cost, it can contribute to its add effects.
            for op in self.operators:
                # Compute sum of precondition costs; if a precondition is unknown (not in fact_costs), treat as INF.
                pre_cost = 0.0
                unreachable = False
                for pre in op.preconditions:
                    c = fact_costs.get(pre, INF)
                    if c == INF:
                        unreachable = True
                        break
                    pre_cost += c
                if unreachable:
                    continue
                # Applying the operator costs 1 in the relaxed metric.
                op_cost = pre_cost + 1.0
                # Relax each add-effect.
                for add in op.add_effects:
                    if op_cost < fact_costs.get(add, INF):
                        fact_costs[add] = op_cost
                        changed = True

        # 4) Sum up goal costs. If a goal is unreachable in the relaxed sense, use a finite penalty.
        total_cost = 0.0
        unreachable_penalty = 10.0  # finite penalty per unreachable goal (keeps heuristic finite)
        for g in self.goals:
            c = fact_costs.get(g, INF)
            if c == INF:
                total_cost += unreachable_penalty
            else:
                total_cost += c

        # Return integer heuristic value (rounded up).
        # Ensure at least 1 if there's any unmet goal (heuristic shouldn't return 0 except goal states).
        h_value = int(total_cost + 0.999999)
        return max(1, h_value) if not (self.goals <= state) else 0