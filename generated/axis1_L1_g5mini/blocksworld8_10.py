from heuristics.heuristic_base import Heuristic
import math

def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing the surrounding parentheses
    and splitting on whitespace.
    Example: "(s0899d5u b1 b2)" -> ["s0899d5u", "b1", "b2"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    This heuristic estimates the number of actions required to reach the goal
    from a given state in the 'secret_domain' (domain id: z1spq3ra).
    It computes a relaxed (delete-relaxed) cost for each fact using a simple
    add-cost relaxation: facts true in the current state have cost 0; for other
    facts the heuristic computes the minimal cost at which they become true
    in a relaxed planning graph where operators only add facts (delete effects
    are ignored for cost propagation). The heuristic value returned is the
    sum of the relaxed costs of the goal facts (the so-called h_add metric).
    This provides an efficiently computable, informative estimate tailored to
    the domain's STRIPS operators while remaining inexpensive.

    Assumptions
    -----------
    - Every operator has uniform action cost 1.
    - The planner represents facts as strings '(predicate arg1 arg2 ...)'.
    - Static facts, if any, are available via task.static (they are treated as
      always true and are given cost 0).
    - We ignore delete effects (relaxed planning). This makes the heuristic
      inadmissible in general but more informative than a trivial goal-count.
    - If some goal facts are not reachable in the relaxed graph, the heuristic
      returns a finite but large value (penalized) so search can continue.

    Heuristic Initialization
    ------------------------
    The constructor extracts:
    - The set of goal facts (task.goals).
    - The set/list of operators available in the task (task.operators).
    - Static facts (task.static) and marks them as cost 0 in the initial
      relaxed fact-cost mapping.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Early exit:
       - If the provided state already satisfies all goal facts (task.goal_reached),
         return 0.

    2. Initialization:
       - Build a mapping `fact_cost` that assigns a numeric cost to facts.
       - Facts present in the current state or in static facts are initialized
         with cost 0.

    3. Relaxed cost propagation (add-cost / h_add style):
       - Repeatedly iterate over all operators. For each operator whose
         preconditions all have finite costs (i.e., are reachable in the relaxed
         sense), compute the operator cost as:
             op_cost = 1 + sum(cost(pre) for pre in op.preconditions)
         - For every fact in the operator's add effects, set:
             fact_cost[add_fact] = min(fact_cost.get(add_fact, +inf), op_cost)
       - Continue until a fixed point is reached (no cost improves).

    4. Extract heuristic value:
       - For each goal fact, read its relaxed cost from `fact_cost`.
       - If a goal fact is unreachable in the relaxed graph, treat it as
         unreachable_count and assign a finite penalty per unreachable goal.
       - The heuristic value returned is:
             sum(costs of reachable goal facts) + penalty_for_unreachable_goals

    5. Notes on domain behavior and why this helps:
       - The domain's operators move and toggle predicates (yfm3frbn, jjtnwk3y,
         a4ha3ujl, s0899d5u) and rely on a global resource (ug241uf7).
       - The relaxed add-cost computation implicitly accounts for chains of
         operator applications (for example, creating jjtnwk3y then using it to
         produce s0899d5u) because operator preconditions and their costs are
         summed in op_cost. Although delete effects (which cause interactions
         and resource contention in the real domain) are ignored, the metric
         captures reasonably how many operator applications are required to
         establish goal facts and thus effectively guides greedy best-first
         search in practice.
    """

    def __init__(self, task):
        # Store goals and operator references for use in the heuristic computation.
        self.goals = set(task.goals)
        self.operators = list(task.operators)
        # static facts are treated as always true (cost 0)
        self.static = set(task.static) if hasattr(task, "static") else set()

    def __call__(self, node):
        state = node.state

        # If already a goal state, heuristic is zero.
        if self.goals <= state:
            return 0

        # Initialize relaxed fact costs:
        # facts in the current (concrete) state or static facts have cost 0.
        fact_cost = {}
        for fact in state:
            fact_cost[fact] = 0
        for fact in self.static:
            fact_cost[fact] = 0

        INF = float("inf")

        # Fixed-point iteration to compute relaxed costs (h_add).
        improved = True
        # To keep the loop bounded in degenerate cases, cap iterations to a
        # reasonably large number proportional to number of facts and operators.
        max_iters = max(50, 5 * (len(fact_cost) + len(self.operators) + len(self.goals)))
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for op in self.operators:
                # If an operator has no preconditions, its preconditions cost is 0.
                preconds = op.preconditions
                # Check that all preconditions are reachable (have finite cost).
                pre_costs = []
                all_reachable = True
                for p in preconds:
                    c = fact_cost.get(p, INF)
                    if c == INF:
                        all_reachable = False
                        break
                    pre_costs.append(c)
                if not all_reachable:
                    continue
                # Cost of using this operator in the relaxed plan:
                op_cost = 1 + sum(pre_costs)
                # Try to improve cost estimates for add effects.
                for add in op.add_effects:
                    old = fact_cost.get(add, INF)
                    if op_cost < old:
                        fact_cost[add] = op_cost
                        improved = True

        # Sum the costs of individual goal facts (h_add).
        total_cost = 0.0
        unreachable = 0
        for g in self.goals:
            c = fact_cost.get(g, INF)
            if c == INF:
                unreachable += 1
            else:
                total_cost += c

        # If some goal facts are unreachable in the relaxed graph, add a penalty.
        # This keeps the heuristic finite while strongly preferring states that
        # make goals reachable in the relaxed sense.
        if unreachable > 0:
            # Penalty per unreachable goal; chosen moderately large to guide search
            # away from states that do not even reach goals in the relaxed graph.
            penalty_per_unreachable = 20
            total_cost += unreachable * penalty_per_unreachable

        # Return an integer heuristic estimate.
        # Ensure heuristic is at least 1 for non-goal states (except when sum is 0
        # due to some peculiarity), to avoid returning 0 erroneously.
        hval = int(math.ceil(total_cost))
        if hval == 0 and not (self.goals <= state):
            hval = 1
        return hval