from heuristics.heuristic_base import Heuristic
from math import inf


def get_parts(fact):
    """Return the space-separated parts of a fact string, without surrounding parentheses."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-aware, efficiently computable heuristic based on the additive
    relaxed-planning heuristic (h_add). It estimates the number of actions
    required to reach the goal from a given state by computing minimal
    costs to achieve individual facts when delete effects are ignored.
    This approach captures useful interactions in the secret_domain: many
    operators transfer the "bclhkf4r" token, create/remove "gaijkwef" links,
    and toggle the global flag "zzar8qlk". By reasoning in the delete-relaxed
    model we obtain fast, informative estimates for guiding a greedy
    best-first search.

    Assumptions
    -----------
    - The planner provides instantiated Operator objects in task.operators whose
      preconditions and add effects are represented as exact fact strings
      (including parentheses), matching the format of the state representation.
    - Static facts are available in task.static and remain true in every state.
    - Goals are a set of fact strings (task.goals).
    - Ignoring delete effects gives a heuristic that is not necessarily
      admissible but is informative and cheap to compute for guiding greedy search.

    Heuristic Initialization
    ------------------------
    The constructor stores:
    - self.goals         : set of goal fact strings extracted from the task
    - self.static       : static facts (set of strings)
    - self.operators    : list of instantiated operators for the task
    - self.all_facts    : all known fact names in the task (for bounds and checks)

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the current state already satisfies all goals, return 0.
    2. Build an initial cost mapping:
       - cost(f) = 0 for all facts f that are true in the current state or are static.
       - cost(f) = +inf for other facts (will be discovered/updated).
    3. Perform a fixed-point computation (value-iteration) of minimal costs
       under the delete-relaxation:
       - For each operator, compute an operator cost:
           cost(op) = 1 + sum(cost(p) for p in op.preconditions)
         (an op contributes cost 1 plus the cost to achieve its preconditions).
       - For every fact in op.add_effects, if cost(op) < cost(fact), update cost(fact).
       - Iterate these updates until no cost improves.
       This yields the additive/delete-relaxed cost (h_add) for every reachable fact.
    4. The heuristic value is the sum of the per-goal costs:
           h = sum(cost(goal) for goal in self.goals)
       If any goal remains unreachable in the relaxed model (cost = +inf),
       return a large finite estimate (bounded by the number of operators
       and facts) rather than +inf so the greedy search keeps exploring.
    5. Return the computed (integer) cost estimate.
    """

    def __init__(self, task):
        """Extract goals, static facts, and operators from the task."""
        # Goals are a set of facts that must hold in the final state.
        self.goals = set(task.goals)
        # Static facts that hold in every state.
        self.static = set(task.static)
        # Operators are instances of Operator with preconditions and add_effects as fact strings.
        self.operators = list(task.operators)
        # Store a reference to the task for potential bounds.
        self._task = task
        # All known facts (may be useful for bounds); fall back to empty set if not provided.
        try:
            self.all_facts = set(task.facts)
        except Exception:
            self.all_facts = set()

    def __call__(self, node):
        """
        Compute the additive relaxed heuristic (h_add) for the given node.

        Returns an integer estimate of the number of actions to reach the goals.
        """
        state = set(node.state)  # current true facts (strings)
        # Goal check: heuristic is zero exactly for states that satisfy all goals.
        if self.goals <= state:
            return 0

        # Initialize costs: 0 for facts currently true (or static), +inf otherwise.
        cost = {}
        init_facts = state | self.static
        for f in init_facts:
            cost[f] = 0

        # Ensure that every goal has an entry (initialize to +inf if not present).
        for g in self.goals:
            if g not in cost:
                cost[g] = inf

        # Iteratively relax operators: cost(add) = min(cost(add), 1 + sum(cost(pre)))
        changed = True
        # To avoid infinite loops and to be efficient, we cap iterations by a reasonable bound.
        # A safe upper bound is number of operators times number of distinct facts + 5.
        bound = max(1000, len(self.operators) * max(1, len(self.all_facts)))
        iterations = 0

        while changed and iterations < bound:
            iterations += 1
            changed = False
            for op in self.operators:
                # Compute cost to make op applicable in relaxed model:
                # cost(op) = 1 + sum(cost(pre) for pre in op.preconditions)
                op_preconds = op.preconditions
                # If any precondition currently has infinite cost, we can't yet compute a finite cost.
                pre_cost = 0
                skip = False
                for p in op_preconds:
                    p_cost = cost.get(p, inf)
                    if p_cost == inf:
                        skip = True
                        break
                    pre_cost += p_cost
                if skip:
                    continue
                op_cost = 1 + pre_cost
                # Update costs of add effects if op gives a cheaper way.
                for a in op.add_effects:
                    prev = cost.get(a, inf)
                    if op_cost < prev:
                        cost[a] = op_cost
                        changed = True

        # Sum the costs of goal facts. If some goals remain unreachable in the relaxed model,
        # return a large but finite estimate to keep search proceeding.
        total = 0
        unreachable = False
        for g in self.goals:
            c = cost.get(g, inf)
            if c == inf:
                unreachable = True
            else:
                total += c

        if unreachable:
            # Provide a finite upper bound: conservative estimate based on operators and goals.
            # This keeps the heuristic finite for solvable states; for complex unreachable
            # cases we still return a large value to deprioritize the branch.
            upper_bound = max(1, len(self.operators)) * (len(self.goals) + 1)
            # Add number of facts as additional slack if available.
            upper_bound += len(self.all_facts)
            return upper_bound

        # The h_add values computed can be non-integers if costs had non-integer parts,
        # but here all op costs are integers. Return integer.
        return int(total)