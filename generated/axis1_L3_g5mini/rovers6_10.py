from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Remove surrounding parentheses and split a PDDL fact string into parts.
    Example: "(at obj1 obj2)" -> ["at", "obj1", "obj2"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the `content-portion` (secret_domain) planning domain.
    The heuristic provides a fast, informative estimate of the number of actions
    required to reach the goal from the current state. It is intended to guide
    greedy best-first search (so it need not be admissible), but it should be
    informative enough to reduce node expansions.

    Assumptions
    -----------
    - The Task passed to the constructor has fully grounded operators (each
      Operator.preconditions and Operator.add_effects contain concrete fact strings).
    - Goals are conjunctive facts (a set of ground facts).
    - Static facts in task.static are true in every reachable state and can be
      treated as already satisfied without extra cost.
    - Each ground goal fact is produced by one or more grounded operators;
      when multiple operators can add a goal, this heuristic picks the cheapest
      operator according to a simple cost model described below.

    Heuristic Initialization
    ------------------------
    In the constructor we:
    - Remember the set of goal facts (self.goals).
    - Store the set of static facts (self.static).
    - Build a mapping from each ground fact string to the list of grounded
      operators that add this exact fact (self.fact_adders). This allows fast
      lookup of which operators can achieve a particular ground goal fact.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given search node / state:
    1. If all goals are already true in the state's fact set, return 0.
    2. Otherwise, for each goal fact g that is not yet true in the state:
       a. Look up the set of grounded operators that add exactly the fact g
          (using self.fact_adders). If no operator adds g, treat g as difficult
          (assign a capped large cost).
       b. For each operator op that can add g, compute a simple estimated
          cost to make op applicable:
            - Base cost: 1 (to apply op and thereby add g).
            - Extra cost: count of op preconditions that are currently not
              satisfied in the state and are not static facts.
              Each unsatisfied precondition is treated as one "extra" action.
              (This is a lightweight relaxation: each missing precondition is
              assumed to be achievable in one step. This is not always true,
              but it is a fast, informative estimate.)
            - Total candidate cost for op = 1 + number_of_unsatisfied_preconds.
       c. Take the minimum candidate cost among the operators that add g. If
          there are none, use a fixed penalty cost for that goal.
       d. Cap the per-goal cost to a reasonable maximum (to avoid extreme
          values from long precondition chains and to remain efficient).
    3. The heuristic value is the sum of the per-goal costs for all unsatisfied
       goals. Summing assumes independence of goals; it is simple and cheap.
       Note: this may overcount when a single action can achieve multiple
       goals, but that is acceptable for greedy guidance.
    4. Return the integer heuristic value.

    Notes on design choices
    -----------------------
    - The heuristic leverages grounded operator preconditions and static facts
      for a quick estimate: missing preconditions imply at least one extra
      action each.
    - Costs are intentionally simple (1 + #missing_preconditions) to remain
      very fast. A small per-goal cap prevents unbounded costs and keeps the
      heuristic finite even when full precondition chains exist.
    - This estimator is not guaranteed to be admissible but tends to be
      informative: it distinguishes easy goals (immediately achievable) from
      goals that require additional setup actions.
    """

    def __init__(self, task):
        """
        Initialize heuristic with task-specific information.

        Parameters
        ----------
        task : Task
            The planning task object (see Task in codebase). It must have
            attributes `.goals`, `.static`, and `.operators`.
        """
        # Goals that must be true in final state (set of ground fact strings).
        self.goals = task.goals

        # Static facts (true in every state).
        self.static = set(task.static)

        # Build mapping: ground fact string -> list of operators that add that fact.
        # This supports fast lookup of which operators can achieve a particular goal fact.
        self.fact_adders = {}  # dict: fact_string -> [Operator, ...]
        for op in task.operators:
            for add in op.add_effects:
                self.fact_adders.setdefault(add, []).append(op)

        # Tunable parameters:
        # maximum cost we attribute to a single goal to keep heuristic finite and stable
        self.max_cost_per_goal = 10
        # penalty for goals that no grounded operator produces (should be rare in well-formed tasks)
        self.unproducible_goal_penalty = 8

    def __call__(self, node):
        """
        Compute heuristic estimate for `node`.

        Parameters
        ----------
        node : SearchNode
            Node object with attribute `state` (a frozenset of fact strings).

        Returns
        -------
        int
            Estimated number of actions required to reach goal from node.state.
        """
        state = node.state

        # If all goals already satisfied, heuristic is 0 (exact).
        if self.goals <= state:
            return 0

        total_cost = 0

        # For each goal not yet satisfied, estimate cost to achieve it.
        for goal_fact in self.goals:
            if goal_fact in state:
                continue  # already satisfied

            # If the goal is static but not in state (shouldn't occur), we give high cost.
            if goal_fact in self.static:
                total_cost += min(self.unproducible_goal_penalty, self.max_cost_per_goal)
                continue

            # Operators that can add this exact ground fact.
            adders = self.fact_adders.get(goal_fact, [])

            if not adders:
                # No grounded operator adds this exact goal fact in the grounded task:
                # give a penalty but keep it finite.
                total_cost += min(self.unproducible_goal_penalty, self.max_cost_per_goal)
                continue

            # Compute minimal candidate cost among adders.
            best_cost = None
            for op in adders:
                # Count preconditions that are not currently satisfied and not static.
                # Those unsatisfied preconditions are treated as one extra action each.
                # (This is a lightweight relaxation.)
                unsatisfied = 0
                for pre in op.preconditions:
                    if pre in state or pre in self.static:
                        continue
                    unsatisfied += 1

                candidate_cost = 1 + unsatisfied  # 1 for applying this op
                if best_cost is None or candidate_cost < best_cost:
                    best_cost = candidate_cost

            # best_cost should always be set because adders non-empty
            if best_cost is None:
                best_cost = self.unproducible_goal_penalty

            # Cap per-goal cost
            best_cost = min(best_cost, self.max_cost_per_goal)
            total_cost += best_cost

        # Return an integer heuristic value
        return int(total_cost)