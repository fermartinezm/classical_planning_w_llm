from heuristics.heuristic_base import Heuristic

class secret_domainHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain.

    Summary:
    This heuristic estimates the number of actions required to achieve the goal
    by counting the number of goal conditions that are not yet satisfied in the current state.
    Each unsatisfied goal condition indicates that at least one action is needed.

    Assumptions:
    - Each unsatisfied goal condition will require at least one action to be satisfied.
    - The static facts (if any) are always true and do not need to be checked explicitly.
    - The goal conditions are independent, meaning achieving one does not affect others.

    Heuristic Initialization:
    - The goal conditions are extracted from the task during initialization.

    Step-By-Step Thinking for Computing Heuristic:
    1. For each goal condition in the task's goal set, check if it is present in the current state.
    2. Count the number of goal conditions that are not satisfied.
    3. The total count is the heuristic value, representing the estimated number of actions needed.
    """

    def __init__(self, task):
        """Initialize the heuristic by storing the goal conditions."""
        self.goals = task.goals

    def __call__(self, node):
        """Compute the heuristic value for the given node's state."""
        state = node.state
        # Count the number of goal conditions not satisfied in the current state.
        return sum(1 for goal in self.goals if goal not in state)
