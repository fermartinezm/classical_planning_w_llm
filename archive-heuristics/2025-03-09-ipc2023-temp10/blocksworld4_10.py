from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class blocksworld4Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goal state
    by counting the number of blocks that are not in their goal positions. It considers
    both blocks that are not on the correct block and blocks that are not clear when
    they should be.

    # Assumptions
    - Each block needs to be moved at least once if it's not in the correct position.
    - Stacking or unstacking a block requires at least one action.
    - The heuristic is not admissible.

    # Heuristic Initialization
    - Extract the goal state facts.
    - Identify the 'on' and 'clear' relationships required in the goal.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize a counter for the heuristic value.
    2. Extract the current 'on' and 'clear' relationships from the current state.
    3. For each 'on' fact in the goal state, check if it exists in the current state.
       If not, increment the counter.
    4. For each 'clear' fact in the goal state, check if it exists in the current state.
       If not, increment the counter.
    5. Return the final count as the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        goal_on = set()
        goal_clear = set()

        # Extract goal 'on' and 'clear' facts
        for goal in self.goals:
            if goal.startswith('(on'):
                goal_on.add(goal)
            elif goal.startswith('(clear'):
                goal_clear.add(goal)

        # Count the number of misplaced blocks
        misplaced_blocks = 0
        for on_fact in goal_on:
            if on_fact not in state:
                misplaced_blocks += 1

        for clear_fact in goal_clear:
            if clear_fact not in state:
                misplaced_blocks += 1

        return misplaced_blocks
