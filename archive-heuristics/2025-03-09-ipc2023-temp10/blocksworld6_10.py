from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class blocksworld6Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goal state
    by counting the number of blocks that are not in their goal positions,
    plus the number of blocks that have a block on top of them that shouldn't be there,
    plus the number of blocks that are clear but shouldn't be.

    # Assumptions:
    - Each misplaced block requires at least one unstack and one stack operation.
    - Blocks that are clear but shouldn't be need to be stacked.
    - Blocks with incorrect blocks on top need to be unstacked.

    # Heuristic Initialization
    - The heuristic initializes by extracting the goal state from the task.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize a counter for the heuristic value.
    2. Iterate through the goal state facts.
    3. For each 'on' fact in the goal, check if it exists in the current state. If not, increment the counter.
    4. For each 'clear' fact in the goal, check if it exists in the current state. If not, increment the counter.
    5. Iterate through the current state facts.
    6. For each 'on' fact in the current state, check if it is present in the goal state. If not, increment the counter.
    7. For each 'clear' fact in the current state, check if it is present in the goal state. If not, increment the counter.
    8. Return the final heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        heuristic_value = 0

        # Check if the goal is already reached
        if self.goals <= state:
            return 0

        # Count misplaced blocks based on 'on' and 'clear' predicates in the goal state
        for goal_fact in self.goals:
            if goal_fact.startswith('(on'):
                if goal_fact not in state:
                    heuristic_value += 1
            elif goal_fact.startswith('(clear'):
                if goal_fact not in state:
                    heuristic_value += 1

        # Count misplaced blocks based on 'on' and 'clear' predicates in the current state
        for state_fact in state:
            if state_fact.startswith('(on'):
                if state_fact not in self.goals:
                    heuristic_value += 1
            elif state_fact.startswith('(clear'):
                if state_fact not in self.goals:
                    heuristic_value += 1

        return heuristic_value
