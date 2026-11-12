from heuristics.heuristic_base import Heuristic
import math

def get_parts(fact):
    """
    Extract components of a PDDL fact string by removing surrounding parentheses
    and splitting on whitespace.

    Example:
      "(at_kitchen_bread content-portion1 bread-portion2)" ->
      ["at_kitchen_bread", "content-portion1", "bread-portion2"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A relaxed (additive) planning-graph based heuristic for the `content-portion`
    (secret_domain) PDDL domain. The heuristic computes a cost estimate for
    reaching all goal facts by repeatedly relaxing operators (ignoring delete
    effects) and computing the minimum additive cost to make each fact true.
    The returned heuristic value is the sum of the relaxed-costs of all goal
    facts (the classical h_add heuristic). This is not necessarily admissible,
    but is effective at guiding greedy best-first search in this domain.

    Assumptions
    -----------
    - Operators are treated in a relaxed manner: delete effects are ignored.
    - Facts that are present in the current state or in the static facts have
      cost 0.
    - Each operator has unit cost (we count an operator application as cost 1)
      plus the additive costs of its preconditions.
    - The heuristic ignores negative interactions between goals (shared
      preconditions will be counted multiple times because we sum goal costs).
    - If a goal is unreachable under the relaxation then its cost is considered
      'infinite' and a large finite value is returned by the heuristic.

    Heuristic Initialization
    ------------------------
    The constructor extracts and stores:
    - self.goals: the set of goal facts (strings).
    - self.static: the set of static facts (strings) that are true in every
      state (these facts are given zero cost in the relaxation).
    - self.operators: the list of Operator objects available in the task. The
      operator objects already store their preconditions and add-effects as
      strings, so they are used directly in the relaxation algorithm.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node (which exposes node.state as a frozenset of fact
    strings), the heuristic performs the following steps:

    1. Initial Costs:
       - Create a mapping `costs` from fact -> numeric cost.
       - Facts that are present in the current state or in the static facts are
         assigned cost 0.
       - All other facts start with cost INF (a large finite number).

    2. Relaxed Operator Propagation (fixed-point iteration):
       - Repeatedly iterate over all operators in the task.
       - For each operator, check whether all its preconditions have finite
         costs in the current `costs` map. If they do, compute the operator's
         relaxed cost:
             op_cost = 1 + sum(cost(pre) for pre in preconditions)
         (1 for applying the operator plus additive cost of preconditions).
       - For each add-effect fact of that operator, if the current stored cost
         for the effect is greater than op_cost, update it to op_cost.
       - Continue iterating until no cost improves in a full pass (reach a
         fixed point). This computes the minimal additive (h_add) cost of
         making each fact true under the relaxation.

    3. Combine Goal Costs:
       - Sum the computed costs of all goal facts. This sum is the heuristic
         value returned.
       - If any goal still has cost INF (unreachable under relaxed rules),
         a large finite value is returned (math.inf is avoided so the search
         machinery can still compare numeric heuristic values). This large
         value indicates the goals are effectively unreachable under the
         current relaxation.

    4. Properties:
       - If all goals are already true in the node.state, their costs are all
         zero and the heuristic returns 0.
       - For solvable states the relaxation will usually assign finite costs to
         goal facts, producing a finite heuristic value.
       - The heuristic is relatively cheap: fixed-point iteration over the set
         of operators is usually efficient because operator preconditions and
         effects are small compared to the number of facts.

    Implementation notes
    --------------------
    - The algorithm ignores delete effects (relaxation), so it is fast and
      generally informative but not admissible.
    - The additive formulation (h_add) is chosen because it captures the
      cost of satisfying each goal in a compositional way and frequently
      performs well as a greedy search guide in domains with structured
      operator dependencies such as this domain.
    """

    def __init__(self, task):
        # goal facts are stored as strings (e.g., "(no_gluten_bread-2 content-portion2 tray1)")
        self.goals = task.goals
        # static facts: facts true in every state (strings)
        self.static = task.static
        # Operators: instances of the Operator class (with preconditions/add_effects/del_effects)
        self.operators = list(task.operators)

    def __call__(self, node):
        """
        Compute the h_add relaxed heuristic value for the given node.

        Parameters
        ----------
        node : search node object
            The planner's search node containing node.state (a frozenset of
            fact strings).

        Returns
        -------
        float
            The heuristic estimate (sum of relaxed costs of goal facts). If a
            goal is unreachable in the relaxation a large finite value is
            returned.
        """
        state = node.state
        INF = 10 ** 6  # large finite cost for unreachable facts under the relaxation

        # Initialize costs: facts present in the current state or in static facts -> 0
        costs = {}
        # Seed known facts
        for fact in state:
            costs[fact] = 0.0
        for fact in self.static:
            # If a static fact wasn't in the state, it still should be treated as cost 0
            if costs.get(fact, INF) > 0.0:
                costs[fact] = 0.0

        # Initialize all other facts implicitly to INF when looked up via costs.get(..., INF).

        # Fixed-point iteration: propagate operator costs to their add-effects.
        changed = True
        while changed:
            changed = False
            for op in self.operators:
                # Sum costs of preconditions. If any precondition is still INF, skip.
                pre_sum = 0.0
                all_pre_finite = True
                for pre in op.preconditions:
                    c = costs.get(pre, INF)
                    if c >= INF:
                        all_pre_finite = False
                        break
                    pre_sum += c
                if not all_pre_finite:
                    continue
                op_cost = 1.0 + pre_sum
                # Relax each add-effect
                for eff in op.add_effects:
                    old = costs.get(eff, INF)
                    if op_cost < old:
                        costs[eff] = op_cost
                        changed = True

        # Sum the costs of all goal facts
        total = 0.0
        for g in self.goals:
            c = costs.get(g, INF)
            if c >= INF:
                # goal unreachable under relaxation -> return large finite value
                return float(INF)
            total += c

        return total