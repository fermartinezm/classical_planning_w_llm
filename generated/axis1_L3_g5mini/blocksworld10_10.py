from heuristics.heuristic_base import Heuristic
import math

INF = math.inf

def get_parts(fact):
    """Return the list of tokens inside a PDDL fact string, e.g. '(at a b)' -> ['at', 'a', 'b']."""
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the "transport" (secret_domain) domain.
    The heuristic is an additive relaxed-plan heuristic (h_add) computed on the
    current state using the task's operators. It treats operator effects as
    only adding facts (i.e., ignores delete-effects) and sums the estimated
    cost of achieving each goal fact. This provides a reasonably informed,
    efficiently computable estimate that guides greedy best-first search well
    in practice for STRIPS-like domains such as this one.

    Assumptions
    -----------
    - The input `task` provides:
        * `task.operators`: a set of Operator objects with `preconditions`
          and `add_effects` stored as strings (e.g., '(at a b)').
        * `task.goals`: a set of goal fact strings.
        * `task.facts`: a set containing the universe of fact strings possible.
        * `task.static`: static facts (not used heavily here but stored).
    - Facts are represented exactly as strings with surrounding parentheses.
    - The heuristic does not account for delete effects and therefore is not
      guaranteed to be admissible. It is, however, finite for reachable goals
      and tends to guide search effectively.

    Heuristic Initialization
    ------------------------
    In the constructor we extract and store:
    - self.goals: the set of goal fact strings.
    - self.operators: the list of operators (as provided).
    - self.static: the set of static facts (task.static) for potential use.
    - self.all_facts: the universe of possible facts (task.facts), used to
      initialize distance estimates.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the current state already satisfies all goals, return 0.
    2. Initialize a cost-table `costs` for every possible fact:
         - cost = 0 for facts true in the current state.
         - cost = INF for all other facts.
    3. Iteratively relax operator applications in a fashion similar to
       the h_add (additive) heuristic:
         - For each operator, if all of its preconditions have finite cost,
           compute the operator cost as sum(cost(preconditions)) + 1 (1 for
           the operator application).
         - For every fact in the operator's add effects, if the operator cost
           improves the current known cost for that fact, update it.
         - Repeat until a fixed point is reached (no improvements).
    4. Compute the heuristic value as the sum of the costs of the individual
       goal facts. This is the "additive" aggregation: h_add(goal1) + ...
    5. If any goal has infinite cost after relaxation (meaning it was not
       reached in the relaxed reachability analysis), return a finite but
       large heuristic value to avoid returning INF. This keeps the planner's
       search working even if a goal appears unreachable under the relaxed
       assumptions (it will be deprioritized).
    6. Return the final numeric estimate (integer).

    Notes on Design Choices
    -----------------------
    - The additive aggregation typically gives stronger guidance than h_max
      while remaining cheap to compute.
    - Ignoring delete effects can underestimate actual needed actions, but
      that is acceptable here because admissibility is not required and the
      heuristic's primary role is to reduce node expansions.
    """

    def __init__(self, task):
        # Store references to useful task components.
        self.goals = set(task.goals)
        self.operators = list(task.operators)
        self.static = set(task.static)
        # The set of all facts potentially relevant (used for cost dictionary keys).
        # If task.facts is not exhaustive in some planner variants, the algorithm
        # will still work because unknown facts will not be updated and remain INF.
        self.all_facts = set(task.facts)

    def __call__(self, node):
        """
        Compute the h_add heuristic for the given search node.

        Parameters
        ----------
        node : search node object exposing `state` (a frozenset of fact strings)

        Returns
        -------
        int
            The heuristic estimate (non-negative integer). Returns 0 for goal
            states, finite large values for states where relaxed analysis
            cannot reach all goals.
        """
        state = node.state

        # 1) If already a goal state, heuristic is 0.
        if self.goals <= state:
            return 0

        # 2) Initialize costs: 0 for facts true in the state, INF otherwise.
        # Use the known universe of facts; also ensure that any goal or any
        # fact appearing in operators is present as a key so it can be updated.
        costs = {}

        # Start with all known facts from task.facts
        for fact in self.all_facts:
            costs[fact] = 0 if fact in state else INF

        # Ensure that any fact mentioned in operators (preconditions/adds)
        # and goals are keys in the costs dictionary (for safety).
        for op in self.operators:
            for f in op.preconditions | op.add_effects:
                if f not in costs:
                    costs[f] = INF
        for g in self.goals:
            if g not in costs:
                costs[g] = INF

        # 3) Iterative relaxation (h_add style).
        changed = True
        # We cap iterations to avoid pathological infinite loops (shouldn't happen).
        max_iterations = len(costs) * 4 + 50
        iterations = 0
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            # For every operator, if all preconditions have finite cost,
            # compute op_cost = sum(pre_costs) + 1 and update add effect costs.
            for op in self.operators:
                # Sum costs of all preconditions; if any precondition is INF,
                # the operator is currently not applicable in the relaxed sense.
                pre_sum = 0.0
                skip = False
                for p in op.preconditions:
                    c = costs.get(p, INF)
                    if c == INF:
                        skip = True
                        break
                    pre_sum += c
                if skip:
                    continue
                op_cost = pre_sum + 1.0
                # Update cost of each add-effect if we can improve it.
                for eff in op.add_effects:
                    if op_cost < costs.get(eff, INF):
                        costs[eff] = op_cost
                        changed = True

        # 4) Aggregate goal costs (additive).
        total_cost = 0.0
        missing_goals = 0
        for g in self.goals:
            c = costs.get(g, INF)
            if c == INF:
                missing_goals += 1
            else:
                total_cost += c

        # 5) If some goals unreachable in relaxed analysis, return a large finite cost.
        if missing_goals > 0:
            # The constant penalty is chosen to be larger than any reasonable plan
            # length for the domain; it keeps the heuristic finite while strongly
            # discouraging states from which goals appear unreachable.
            penalty_per_missing = 1000
            total_cost += missing_goals * penalty_per_missing

        # Return as integer (heuristic values for search are integers in this planner).
        # We round up to encourage progress.
        return int(math.ceil(total_cost))