from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing surrounding parentheses
    and splitting on whitespace.

    Example:
      "(at loc1 loc2)" -> ["at", "loc1", "loc2"]

    This utility ensures we ignore the surrounding brackets when parsing facts.
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the 'spanner' (secret_domain) domain.
    The heuristic uses a relaxed planning graph (the classical h_add relaxation).
    It computes the cost to achieve every fact in a delete-relaxed manner (i.e.,
    operators only add facts and never delete them) and returns the sum of the
    additive costs of the goal facts (h_add). This provides an informative,
    efficiently computable estimate of the number of actions needed to reach
    the goal under the delete-relaxation.

    Assumptions
    -----------
    - All facts and operator preconditions / effects are represented as strings
      exactly as in the planner (e.g., "(tightened nut1 locatable1)").
    - Static facts (task.static) are true in every state and have zero cost.
    - Actions each have unit cost (we count each operator application as cost 1).
    - The heuristic ignores delete effects (delete relaxation), which makes it
      inadmissible but typically informative for greedy best-first search.
    - If a goal fact cannot be derived in the relaxed graph, the heuristic
      returns a large finite value as a penalty.

    Heuristic Initialization
    ------------------------
    In the constructor we store:
    - self.goals: the set of goal facts (strings) the task requires.
    - self.static: the static facts of the task (facts true in every state).
    - self.operators: the set/list of operator objects (with preconditions and
      add_effects) provided by the task.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the current state already satisfies all goals, return 0.
    2. Initialize a mapping `cost[fact] = 0` for every fact that holds in the
       current state as well as for every static fact (they require no cost).
    3. Iteratively relax operators:
       - For each operator, if all its preconditions already have finite
         costs (i.e., they are present in `cost`), then the relaxed cost to
         apply this operator is 1 + sum(cost[p] for p in preconditions).
       - For each fact in the operator's add_effects, record the minimal
         discovered relaxed cost (min over operators that add the fact).
    4. Repeat step 3 until no cost improves OR until all goal facts have
       assigned costs.
    5. The heuristic value is the sum of the costs of the goal facts (h_add).
       - If any goal is unreachable in the relaxed graph (no cost assigned),
         return a large finite penalty (e.g., 1000).
    6. This estimate is informative because it accounts for required
       preconditions and how achieving one fact can enable many operators,
       but it is optimistic because operator deletions are ignored.
    """

    def __init__(self, task):
        """
        Initialize data structures needed by the heuristic.
        """
        self.goals = set(task.goals)
        self.static = set(task.static)
        # Keep operators as list for iteration. Each operator provides:
        # operator.preconditions, operator.add_effects (sets of fact strings)
        self.operators = list(task.operators)

    def __call__(self, node):
        """
        Compute the relaxed h_add heuristic value for the node's state.

        Returns an integer estimate of the number of actions required to
        reach the goal in the delete-relaxed problem. Returns 0 if the node's
        state is a goal state. If one or more goals are unreachable in the
        relaxed graph, returns a large finite penalty value.
        """
        state = node.state

        # Quick goal check: if all goals already true, h = 0.
        if self.goals <= state:
            return 0

        # cost_map maps fact string -> minimal relaxed cost to achieve it.
        cost_map = {}

        # Initialize cost 0 for facts present in the current state and for static facts.
        for fact in state:
            cost_map[fact] = 0
        for fact in self.static:
            cost_map[fact] = 0

        # Iteratively relax operators and propagate costs.
        changed = True
        # Bound iterations to avoid pathological infinite loops; typically the
        # process converges quickly. Use a safe upper bound: number of facts * 2.
        max_iterations = max(4, len(cost_map) + 2 * len(self.operators))
        iterations = 0

        while changed and iterations < max_iterations:
            iterations += 1
            changed = False

            # For each operator, check if all preconditions have known costs.
            for op in self.operators:
                # All preconditions must be reachable in the relaxed graph.
                preconds = op.preconditions
                # If operator has no preconditions, pre_cost_sum is 0.
                pre_known = True
                pre_cost_sum = 0
                for p in preconds:
                    if p in cost_map:
                        pre_cost_sum += cost_map[p]
                    else:
                        pre_known = False
                        break

                if not pre_known:
                    continue

                # Relaxed cost to apply this operator: 1 (for the operator) plus
                # the sum of costs to establish its preconditions (h_add).
                op_cost = 1 + pre_cost_sum

                # Update cost for each add effect if this op gives a better cost.
                for eff in op.add_effects:
                    old = cost_map.get(eff)
                    if old is None or op_cost < old:
                        cost_map[eff] = op_cost
                        changed = True

            # Early exit if all goals have been assigned a cost.
            if all(goal in cost_map for goal in self.goals):
                break

        # If some goals are unreachable in relaxed graph, return a large finite penalty.
        unreachable_goals = [g for g in self.goals if g not in cost_map]
        if unreachable_goals:
            # A large but finite heuristic value. Should be bigger than typical
            # solution lengths for the domain so the search avoids such states.
            return 1000

        # Sum the additive costs for all goal facts (h_add).
        total = sum(cost_map[g] for g in self.goals)
        # Ensure integer return.
        return int(total)