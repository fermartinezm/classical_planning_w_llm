from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class blocksworld10Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goal state in the Blocksworld domain.
    It considers the number of blocks that are not in their goal positions, the number of blocks that have
    blocks on top of them that should not be there, and the number of blocks that are clear but should not be.

    # Assumptions
    - The heuristic assumes that each misplaced block requires at least one 'unstack' and one 'stack' action.
    - It also assumes that each block that needs to be clear requires at least one 'unstack' action.
    - The arm-empty condition is not explicitly considered, as it is implicitly handled by the 'unstack' and 'stack' actions.

    # Heuristic Initialization
    - The heuristic initializes by extracting the goal state from the task definition.
    - It identifies the 'on', 'on-table', and 'clear' predicates in the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the goal state information: Identify the 'on', 'on-table', and 'clear' relationships that must hold in the goal.
    2. Identify misplaced blocks: Compare the current state with the goal state to find blocks that are not in their goal positions.
       - Count the number of blocks that are on the wrong block or table.
    3. Identify blocks with incorrect support: Count the number of blocks that have other blocks on top of them in the current state,
       but should not have any blocks on top of them in the goal state.
    4. Identify blocks that are incorrectly clear: Count the number of blocks that are clear in the current state but should have
       another block on top of them in the goal state.
    5. Sum the misplaced blocks, blocks with incorrect support, and incorrectly clear blocks to estimate the number of actions required.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.static = task.static

        self.goal_on = set()
        self.goal_on_table = set()
        self.goal_clear = set()

        for goal in self.goals:
            if goal.startswith('(on '):
                self.goal_on.add(goal)
            elif goal.startswith('(on-table '):
                self.goal_on_table.add(goal)
            elif goal.startswith('(clear '):
                self.goal_clear.add(goal)

    def __call__(self, node):
        """Estimate the cost to reach the goal state from the current state."""
        state = node.state
        misplaced_blocks = 0
        incorrect_support = 0
        incorrectly_clear = 0

        # Check 'on' relationships
        for goal_fact in self.goal_on:
            if goal_fact not in state:
                misplaced_blocks += 1

        # Check 'on-table' relationships
        for goal_fact in self.goal_on_table:
            if goal_fact not in state:
                misplaced_blocks += 1

        # Check blocks with incorrect support
        for fact in state:
            if fact.startswith('(on '):
                if fact not in self.goal_on:
                    incorrect_support += 1

        # Check blocks that are incorrectly clear
        for fact in state:
            if fact.startswith('(clear '):
                block = fact[7:-1]
                goal_clear_blocks = {g[7:-1] for g in self.goal_clear}
                if block not in goal_clear_blocks:
                    incorrectly_clear += 1

        return misplaced_blocks + incorrect_support + incorrectly_clear
