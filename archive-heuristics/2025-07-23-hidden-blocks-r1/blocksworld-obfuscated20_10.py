from heuristics.heuristic_base import Heuristic

class secret_domainHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain.

    Summary:
    This heuristic estimates the number of actions required to reach the goal state
    by counting the number of unsatisfied goal conditions in the current state.

    Assumptions:
    - Each goal condition is independent, and achieving one condition does not affect others.
    - The cost of achieving each goal condition is at least 1 action, making the heuristic an underestimate in many cases.
    - The heuristic does not account for interactions between actions or the need for specific sequences of actions.

    Heuristic Initialization:
    - The goal conditions are stored during initialization for quick access during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
    1. For the current state, compare the set of goal conditions against the state's predicates.
    2. Count the number of goal conditions that are not present in the current state.
    3. Return this count as the heuristic value, representing the estimated number of actions needed to achieve all goals.
    """

    def __init__(self, task):
        """
        Initialize the heuristic with the task's goal conditions.
        """
        self.goals = task.goals

    def __call__(self, node):
        """
        Compute the heuristic value for the given node's state.
        
        :param node: The node representing the current state in the search.
        :return: The number of unsatisfied goal conditions.
        """
        state = node.state
        # Count the number of goals not satisfied in the current state
        unsatisfied = len(self.goals - state)
        return unsatisfied
