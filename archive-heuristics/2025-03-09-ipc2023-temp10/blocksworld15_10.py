from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class blocksworld15Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goal state in the Blocksworld domain.
    It considers the number of blocks that are not in their goal positions, and the number of blocks that have blocks
    on top of them that should not be there. It also considers the blocks that are on the table when they should not be.

    # Assumptions
    - Each block needs to be moved at most once to its correct position.
    - Moving a block requires picking it up, potentially unstacking it, and then stacking it in the correct place.
    - The arm can only hold one block at a time.

    # Heuristic Initialization
    - Extract the goal `on` and `on-table` relations.
    - Identify all blocks involved in the goal.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Initialize the heuristic value to 0.
    2.  Extract the goal state information to determine the desired `on` and `on-table` relationships.
    3.  Iterate through the current state and compare it with the goal state.
    4.  For each block that is not in its goal position (either `on` another block or `on-table`), increment the heuristic value.
        - This includes blocks that are on the table but should be stacked.
        - This also includes blocks that are stacked incorrectly.
    5.  For each block that has another block on top of it that should not be there, increment the heuristic value.
    6.  Return the final heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static
        self.goal_on = {}
        self.goal_on_table = set()
        self.blocks = set()

        for goal in self.goals:
            if goal.startswith('(on '):
                parts = goal[1:-1].split()
                block1 = parts[1]
                block2 = parts[2]
                self.goal_on[block1] = block2
                self.blocks.add(block1)
                self.blocks.add(block2)
            elif goal.startswith('(on-table '):
                parts = goal[1:-1].split()
                block = parts[1]
                self.goal_on_table.add(block)
                self.blocks.add(block)
            elif goal.startswith('(clear '):
                parts = goal[1:-1].split()
                block = parts[1]
                self.blocks.add(block)

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        h = 0

        # Check if the goal is reached
        if node.state >= self.goals:
            return 0

        # Blocks that are not in the correct position
        for block1, block2 in self.goal_on.items():
            correct_relation = f'(on {block1} {block2})'
            if correct_relation not in state:
                h += 1

        for block in self.goal_on_table:
            correct_relation = f'(on-table {block})'
            if correct_relation not in state:
                h += 1

        # Blocks that have blocks on top of them that should not be there
        for fact in state:
            if fact.startswith('(on '):
                parts = fact[1:-1].split()
                block1 = parts[1]
                block2 = parts[2]
                if block1 in self.goal_on and self.goal_on[block1] != block2:
                    h += 1
                elif block1 not in self.goal_on and block1 in self.blocks:
                    h += 1

        # Blocks that are on the table but should not be
        for fact in state:
            if fact.startswith('(on-table '):
                parts = fact[1:-1].split()
                block = parts[1]
                if block not in self.goal_on_table and block in self.blocks:
                    h += 1

        return h
