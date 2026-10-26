from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class blocksworld3Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goal state in the Blocksworld domain.
    It considers the number of blocks that are not in their goal positions, the number of blocks that have blocks
    on top of them that should not be there, and the number of blocks that are clear but should not be.

    # Assumptions
    - Each block needs to be moved at least once if it is not in its goal position.
    - A block needs to be unstacked if it has the wrong block on top of it.
    - A block needs to be stacked if it is clear but should have a block on top of it.
    - The arm-empty and holding predicates are ignored, as they are implicitly handled by the other actions.

    # Heuristic Initialization
    - Extract the goal state from the task.
    - Store the goal 'on' relationships in a dictionary for easy access.
    - Store the goal 'clear' blocks in a set.
    - Store the blocks that should be on the table in a set.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize a counter for the heuristic value.
    2. Iterate through the current state and compare it to the goal state.
    3. For each 'on' predicate in the goal, check if it exists in the current state. If not, increment the counter.
    4. For each 'clear' predicate in the goal, check if it exists in the current state. If not, increment the counter.
    5. For each 'on' predicate in the current state, check if it is in the goal state. If not, increment the counter.
    6. For each 'clear' predicate in the current state, check if it should be clear in the goal state. If not, increment the counter.
    7. If the current state is the goal state, return 0.
    8. Return the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.static = task.static

        self.goal_on = {}
        self.goal_clear = set()
        self.goal_on_table = set()

        for goal in self.goals:
            if goal.startswith('(on '):
                parts = goal[1:-1].split()
                block1 = parts[1]
                block2 = parts[2]
                self.goal_on[block1] = block2
            elif goal.startswith('(clear '):
                parts = goal[1:-1].split()
                block = parts[1]
                self.goal_clear.add(block)
            elif goal.startswith('(on-table '):
                parts = goal[1:-1].split()
                block = parts[1]
                self.goal_on_table.add(block)

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        heuristic = 0

        if node.state == self.goals:
            return 0

        # Check if 'on' relationships are correct
        for block1, block2 in self.goal_on.items():
            if f'(on {block1} {block2})' not in state:
                heuristic += 1

        # Check if 'clear' relationships are correct
        for block in self.goal_clear:
            if f'(clear {block})' not in state:
                heuristic += 1

        # Check if 'on-table' relationships are correct
        for block in self.goal_on_table:
            if f'(on-table {block})' not in state:
                heuristic += 1

        # Check for blocks that are incorrectly stacked
        for fact in state:
            if fact.startswith('(on '):
                parts = fact[1:-1].split()
                block1 = parts[1]
                block2 = parts[2]
                if block1 in self.goal_on and self.goal_on[block1] != block2:
                    heuristic += 1
                elif block1 not in self.goal_on:
                    heuristic += 1

            elif fact.startswith('(clear '):
                parts = fact[1:-1].split()
                block = parts[1]
                is_goal_clear = block in self.goal_clear
                is_goal_on = False
                for b1, b2 in self.goal_on.items():
                    if b2 == block:
                        is_goal_on = True
                        break
                if not is_goal_clear and is_goal_on:
                    heuristic += 1

        return heuristic
