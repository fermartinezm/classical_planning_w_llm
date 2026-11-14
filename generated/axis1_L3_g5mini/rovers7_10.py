from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return components of a PDDL fact string by removing surrounding parentheses and splitting."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the 'content-portion' (secret_domain) PDDL domain.
    The heuristic estimates the number of actions needed to satisfy the remaining goal facts.
    It is not guaranteed to be admissible, but is designed to be informative and cheap to compute
    so it can guide a greedy best-first search efficiently.

    Assumptions
    -----------
    - The planning Task object passed to the constructor contains:
        - task.goals: a set of goal facts (strings).
        - task.static: a set of static facts (strings) that hold in every state.
        - task.operators: a set of Operator objects with .preconditions, .add_effects and .del_effects.
    - A single operator application typically costs '1' in the search; we approximate missing
      prerequisites with small integer costs (1 if plausibly achievable by one action, 2 otherwise).
    - Many goals in this domain are of the form "no_gluten_bread-2 ..." or "no_gluten_sandwich-2 ...".
      Those can require multi-step sequences (put_on_tray, make_sandwich, move_tray-2, ...).
      The heuristic captures some of that dependency information by inspecting operator add effects
      and preconditions, but it intentionally remains shallow to remain efficient.

    Heuristic Initialization
    ------------------------
    - self.goals: set of goal facts to achieve.
    - self.static: set of static facts (task.static) used to treat some preconditions as already
      'free' (cost 0) because they hold in all states.
    - self.operators: list of operator objects from the task.
    - self.adders: a mapping from a fact string to the list of operators that can add that fact.
      This allows the heuristic to reason about how a missing fact could be achieved and what
      preconditions are required for that achievement.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given state (node.state, a frozenset of fact strings), the heuristic computes an estimate
    as follows:

    1. If the state already satisfies all goal facts, return 0.

    2. For each goal fact g that is not true in the current state:
       a. Find all operators that have g in their add effects (these are candidate "producers").
       b. For each candidate operator op:
          i. Compute the set of its preconditions that are not currently satisfied in the state,
             and that are not static (these are the truly missing prerequisites).
         ii. For each missing precondition m, estimate how "hard" it is to obtain:
             - If there exists at least one operator that can add m (i.e., m is in self.adders),
               assume it can be achieved by a single additional action (cost 1).
             - Otherwise, assume it is harder and give it a higher small constant cost (cost 2).
        iii. The total estimated cost for using op to achieve g is:
             cost(op -> g) = 1 (to apply op itself) + sum(estimated costs for each missing precondition).
       c. Use the minimum cost among all candidate operators as the estimated cost to achieve g.
          If no operator produces g, use a small fallback (3) to keep the heuristic finite but
          penalize states where the goal seems currently unreachable via a direct producer.
    3. The heuristic value is the sum of these per-goal estimates. This treats goals as independent
       (a relaxation), which makes computation efficient while still capturing some causal structure.

    Notes on design choices
    -----------------------
    - The heuristic is intentionally shallow (only looks one step ahead plus a 1-step estimate for
      missing preconditions) to keep evaluations fast. This usually suffices for domains with many
      independent or weakly-coupled goals such as this one.
    - Using different weights for missing preconditions (1 vs. 2) helps distinguish facts that are
      plausibly obtainable in one action from those that look harder, reducing the number of
      expanded nodes compared to a naive goal-count heuristic.
    - The heuristic always returns small integers and is 0 exactly on states that satisfy all goals.
    """

    def __init__(self, task):
        """Extract goals, static facts and build mapping from facts to operators that add them."""
        self.goals = set(task.goals)
        self.static = set(task.static)
        # List of operators from the task (Operator objects).
        self.operators = list(task.operators)

        # Map: fact_string -> [operator, operator, ...] that add the fact.
        self.adders = {}
        for op in self.operators:
            for eff in op.add_effects:
                self.adders.setdefault(eff, []).append(op)

    def __call__(self, node):
        """Compute heuristic estimate for node.state (a frozenset of fact strings)."""
        state = set(node.state)

        # 1) If already a goal state, return 0.
        if self.goals <= state:
            return 0

        total_estimate = 0

        # For each unsatisfied goal, compute a small approximate cost to achieve it.
        for goal in self.goals:
            if goal in state:
                continue

            best_cost = None
            candidate_ops = self.adders.get(goal, [])

            # Evaluate each candidate operator that can add this goal.
            for op in candidate_ops:
                # Determine missing preconditions ignoring those that are static (cost 0).
                missing = [p for p in op.preconditions if p not in state and p not in self.static]

                # Estimate cost to obtain the missing preconditions.
                # If a missing precondition can itself be produced by some operator,
                # assume cost 1; otherwise assume cost 2 (harder).
                missing_cost = 0
                for m in missing:
                    if m in self.adders:
                        missing_cost += 1
                    else:
                        missing_cost += 2

                op_cost = 1 + missing_cost  # 1 for executing op itself.
                if best_cost is None or op_cost < best_cost:
                    best_cost = op_cost

            # If no operator can directly add the goal, apply a conservative fallback.
            # Use a small finite penalty so heuristic stays finite for solvable states.
            if best_cost is None:
                best_cost = 3

            total_estimate += best_cost

        # Ensure we return a finite integer.
        return int(total_estimate)