from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to reach the goal state
    by counting the number of goal predicates that are not satisfied in the
    current state. For each unsatisfied goal predicate, it adds a cost of 1 to the
    heuristic value. This heuristic is admissible in relaxed planning but not necessarily
    admissible in classical planning due to ignoring preconditions and negative effects.
    However, for greedy best-first search, admissibility is not required, and this
    heuristic aims to provide a reasonable estimate to guide the search efficiently.

    # Assumptions:
    - The problem is solvable in the Blocksworld domain using the defined actions.
    - The goal state is specified as a conjunction of predicates.

    # Heuristic Initialization
    - The heuristic initializes by storing the goal predicates from the task definition.
    - No static facts are used in this heuristic.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Iterate through each goal predicate in the task's goal definition.
    3. For each goal predicate, check if it is present in the current state.
    4. If a goal predicate is NOT present in the current state, increment the heuristic value by 1.
    5. After checking all goal predicates, the accumulated heuristic value is returned as the estimate
       of the distance to the goal.

    This heuristic essentially counts the number of "problems" that still need to be solved to reach the goal.
    Each unsatisfied goal predicate is considered as one unit of "work" required.
    It is a simple and fast heuristic to compute, suitable for guiding greedy best-first search
    in the Blocksworld domain.
    """

    def __init__(self, task):
        """Initialize the heuristic by storing the goal predicates."""
        self.goals = task.goals

    def __call__(self, node):
        """
        Estimate the number of actions to reach the goal state from the current state.
        This is done by counting the number of goal predicates not satisfied in the current state.
        """
        state = node.state
        heuristic_value = 0
        for goal in self.goals:
            if goal not in state:
                heuristic_value += 1
        return heuristic_value
