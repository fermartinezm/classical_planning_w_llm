from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to reach the goal state by counting the number of goal predicates that are not satisfied in the current state.
    For each unsatisfied goal predicate, it adds a cost of 1 to the heuristic estimate.
    This heuristic is a simple set-level heuristic that provides a basic estimate of the distance to the goal.

    # Assumptions:
    - Each unsatisfied goal predicate requires at least one action to be satisfied.
    - The heuristic does not consider the dependencies between goal predicates or the specific actions required to achieve them.
    - It assumes a unit cost for each unsatisfied goal predicate.

    # Heuristic Initialization
    - The heuristic initializes by storing the goal predicates from the task definition.
    - No static facts are used in this heuristic.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Iterate through each goal predicate in the task's goal description.
    3. For each goal predicate, check if it is present in the current state.
    4. If a goal predicate is NOT present in the current state, increment the heuristic value by 1.
    5. After checking all goal predicates, the accumulated heuristic value is returned as the estimated cost to reach the goal state.

    This heuristic is efficiently computable as it only requires checking the presence of goal predicates in the current state.
    It is domain-dependent as it is specifically designed for the Blocksworld domain and its predicates.
    While simple, it can guide a greedy best-first search by prioritizing states that satisfy more goal predicates.
    """

    def __init__(self, task):
        """Initialize the heuristic by storing the goal predicates."""
        self.goals = task.goals
        static_facts = task.static # Static facts are not used in this heuristic

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        heuristic_value = 0
        for goal in self.goals:
            if goal not in state:
                heuristic_value += 1
        return heuristic_value
