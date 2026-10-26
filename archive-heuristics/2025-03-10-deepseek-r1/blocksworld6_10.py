from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class blocksworld6Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to arrange blocks into their goal positions. It considers:
    - Each misplaced goal block (not in correct support chain) adds 2 actions.
    - Each block on top of a goal block that shouldn't be there adds 2 actions.

    # Assumptions:
    - The goal is a conjunction of 'on' and 'on-table' predicates.
    - Blocks not mentioned in the goal can be in any position.
    - Moving a block requires two actions (pickup and putdown/stack).

    # Heuristic Initialization
    - Extract goal structure to determine required supports for each goal block.
    - Collect all blocks mentioned in the goal.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each goal block, check if it's in the correct position by verifying its entire support chain.
    2. Add 2 for each misplaced goal block.
    3. For each block on top of a goal block, check if it's supposed to be there. If not, add 2.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goal_on = {}  # Maps each goal block to its required support (None for on-table)
        self.goal_blocks = set()

        # Parse goal predicates
        for goal in task.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'on':
                a, b = parts[1], parts[2]
                self.goal_on[a] = b
                self.goal_blocks.update([a, b])
            elif parts[0] == 'on-table':
                a = parts[1]
                self.goal_on[a] = None
                self.goal_blocks.add(a)

    def __call__(self, node):
        state = node.state

        # Build current_on and current_on_table
        current_on = {}
        current_on_table = set()
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'on':
                a, b = parts[1], parts[2]
                current_on[a] = b
            elif parts[0] == 'on-table':
                a = parts[1]
                current_on_table.add(a)

        h = 0

        # Check each goal block's support chain
        for block in self.goal_blocks:
            current_block = block
            correct = True
            while True:
                if current_block not in self.goal_on:
                    # Should be on-table
                    if current_block not in current_on_table or current_block in current_on:
                        correct = False
                    break
                else:
                    required_support = self.goal_on[current_block]
                    actual_support = current_on.get(current_block)
                    if actual_support != required_support:
                        correct = False
                        break
                    current_block = required_support
            if not correct:
                h += 2

        # Check for blocks on top of goal blocks that shouldn't be there
        for upper_block, lower_block in current_on.items():
            if lower_block in self.goal_blocks:
                if upper_block in self.goal_on:
                    if self.goal_on[upper_block] != lower_block:
                        h += 2
                else:
                    # upper_block should be on-table but is on a goal block
                    h += 2

        return h
