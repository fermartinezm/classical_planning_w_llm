from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the components of a PDDL fact string by removing surrounding
    parentheses and splitting on whitespace.
    Example: "(at obj loc)" -> ["at", "obj", "loc"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent, efficiently computable relaxed-plan heuristic for the
    secret_domain. This heuristic computes an additive relaxed-plan cost (h_add)
    for the current state: facts that hold in the current state or are static
    have cost 0; each operator contributes cost 1 plus the sum of costs of its
    preconditions; the cost of a fact is the minimum cost found among all
    operators that add it. The heuristic value returned is the sum of the
    (relaxed) costs of all goal facts that are not yet true in the current
    state.

    Assumptions
    -----------
    - The planner represents facts as strings with surrounding parentheses,
      e.g., "(predicate arg1 arg2)". get_parts() removes parentheses.
    - task.operators contains Operator instances with preconditions/add/del
      represented as strings consistent with the state's fact strings.
    - Static facts are provided in task.static and remain true in all states.
    - We ignore delete effects (relaxed planning graph), so the heuristic is
      not necessarily admissible but is informative and cheap to compute.
    - If a goal fact cannot be reached in the relaxed graph, we assign a
      finite large penalty to keep the heuristic finite (so search can proceed).

    Heuristic Initialization
    ------------------------
    In the constructor we:
    - Store the set of goal facts (task.goals).
    - Store static facts (task.static) as a frozenset for quick membership tests.
    - Cache the list of operators from task.operators to iterate efficiently.
      Operators are expected to provide preconditions and add_effects as sets
      of fact strings.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    1. If the node's state already satisfies all goals, return 0.
    2. Initialize a mapping `fact_cost` from fact -> relaxed cost:
       - All facts present in the current state or in static facts get cost 0.
    3. Repeatedly try to "apply" operators in the relaxed sense:
       - If all preconditions of an operator have an assigned cost, then the
         operator can (relaxed) produce its add effects with cost:
             op_cost = 1 + sum(cost(pre) for pre in preconditions)
       - For every add-effect fact, update its cost to the minimum of its
         current cost and op_cost.
       - Iterate until no cost changes in a pass (fixed point).
    4. Compute the heuristic as the sum of costs for every goal fact that is
       not true in the current state. If a goal fact did not receive any cost
       in the relaxed propagation (unreachable in the relaxed graph), add a
       finite large penalty (UNREACHABLE_COST) for that fact to keep the
       heuristic finite.
    5. Return the integer heuristic value. This gives an informative estimate
       of remaining operator count while being efficient to compute.

    Notes on design choices
    -----------------------
    - We use the additive relaxed plan (h_add) because it is cheap and often
      effective for guiding greedy best-first search in domains where goals
      can be decomposed across independent "facts" (which fits the structure
      of the provided domain with multiple goal facts of different predicates).
    - A finite penalty for unreachable goals prevents infinite heuristic
      values and allows the search to continue; in practice, solvable states
      will assign finite operator-based costs to all reachable goals.
    """

    UNREACHABLE_COST = 1000  # finite penalty for goals unreachable in relaxed graph

    def __init__(self, task):
        """Extract static facts, goal facts and cache operators for quick use."""
        # Goals are a set of fact strings.
        self.goals = task.goals
        # Static facts that hold in every state.
        self.static = frozenset(task.static)
        # Cache operators (Operator instances).
        self.operators = list(task.operators)

    def __call__(self, node):
        """Compute the h_add-style relaxed-plan heuristic for the given node."""
        state = node.state  # frozenset of fact strings

        # If already a goal state, heuristic is 0.
        if self.goals <= state:
            return 0

        # fact_cost: mapping from fact string -> minimal relaxed cost found.
        fact_cost = {}

        # Initialize costs of facts that are true now or static (cost 0).
        for f in state:
            fact_cost[f] = 0
        for f in self.static:
            if f not in fact_cost:
                fact_cost[f] = 0

        # Iteratively propagate relaxed costs using operators (ignore deletes).
        changed = True
        # To limit runtime in pathological cases, also cap iterations to a safe bound:
        max_iterations = len(self.operators) * 50 + 10
        iterations = 0

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            # For each operator compute relaxed applicability/cost.
            for op in self.operators:
                # Check if all preconditions have known costs.
                preconds = op.preconditions
                missing_pre = False
                pre_sum = 0
                for p in preconds:
                    if p in fact_cost:
                        pre_sum += fact_cost[p]
                    else:
                        missing_pre = True
                        break
                if missing_pre:
                    continue  # cannot fire this operator in the relaxed model yet

                op_cost = pre_sum + 1  # apply operator costs one action

                # Update add effects
                for add in op.add_effects:
                    old_cost = fact_cost.get(add)
                    if old_cost is None or op_cost < old_cost:
                        fact_cost[add] = op_cost
                        changed = True

            # Early exit optimization: if all goals have costs, we can stop.
            all_goals_have_cost = True
            for g in self.goals:
                if g not in fact_cost:
                    all_goals_have_cost = False
                    break
            if all_goals_have_cost:
                break

        # Sum up costs for goals not yet true in the current state.
        total = 0
        for g in self.goals:
            if g in state:
                continue  # already achieved, cost 0
            if g in fact_cost:
                total += fact_cost[g]
            else:
                # Unreachable in relaxed graph -> assign finite penalty.
                total += self.UNREACHABLE_COST

        # Return integer heuristic.
        return int(total)