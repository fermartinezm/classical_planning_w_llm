from heuristics.heuristic_base import Heuristic

class secret_domainHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain.

    Summary:
    This heuristic estimates the number of actions required to reach the goal state by counting the number of unsatisfied goal conditions. Specifically, it checks for missing 'on' relations (x272hpoe), base conditions (dhjk1eyy), and clear conditions (xk87au5b) in the current state compared to the goal state.

    Assumptions:
    - The goal conditions are provided as a set of facts that must be satisfied.
    - The state is represented as a set of strings, each corresponding to a fact.

    Heuristic Initialization:
    - Extracts the goal conditions from the task and stores them for quick lookup.

    Step-By-Step Thinking for Computing Heuristic:
    1. For the current state, iterate over each goal condition.
    2. For each goal condition, check if it is present in the current state.
    3. Count the number of goal conditions not satisfied in the current state.
    4. Return this count as the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic with the goal conditions from the task."""
        self.goal_conditions = task.goals

    def __call__(self, node):
        """Compute the heuristic value for the current node's state."""
        state = node.state
        # Count the number of goal conditions not satisfied in the current state
        return sum(1 for goal in self.goal_conditions if goal not in state)
