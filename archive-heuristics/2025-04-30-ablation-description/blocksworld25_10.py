# Assuming heuristics.heuristic_base provides a base class named Heuristic
from heuristics.heuristic_base import Heuristic

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the blocksworld domain.

    # Summary
    This heuristic estimates the cost to reach the goal by counting the number of goal conditions
    that are not currently satisfied in the state.

    # Assumptions:
    - The heuristic counts any goal fact (on, on-table, clear, arm-empty) that is not true.
    - This is a simple relaxation and does not consider the dependencies between goals
      or the cost of clearing blocks that are in the way. It is equivalent to the
      number of unsatisfied goal literals.

    # Heuristic Initialization
    - The set of goal facts is extracted from the task definition.

    # Step-By-Step Thinking for Computing Heuristic
    1. Get the set of goal facts from the task object (`self.goals`).
    2. Get the current state, which is a frozenset of true facts, from the node object (`node.state`).
    3. Initialize a counter variable, `unsatisfied_goals_count`, to 0.
    4. Iterate through each fact (`goal_fact`) in the set of goal facts (`self.goals`).
    5. For each `goal_fact`, check if it is present in the current state (`state`).
    6. If `goal_fact` is *not* found in the `state`, increment `unsatisfied_goals_count` by 1.
    7. After checking all goal facts, the final value of `unsatisfied_goals_count` is the heuristic estimate.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        # The set of facts that must hold in goal states.
        self.goals = task.goals
        # Blocksworld domain has no static facts, so task.static is not needed.
        # static_facts = task.static

    def __call__(self, node):
        """Estimate the minimum cost to reach the goal state."""
        state = node.state

        # Count the number of goal facts that are not in the current state.
        unsatisfied_goals_count = 0
        for goal_fact in self.goals:
            if goal_fact not in state:
                unsatisfied_goals_count += 1

        return unsatisfied_goals_count
