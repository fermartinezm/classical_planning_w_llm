from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class blocksworld20Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goal by considering:
    - Blocks that are not in their goal positions, requiring movement.
    - Blocks that are on top of goal blocks and need to be moved to clear the goal blocks.

    # Assumptions:
    - Each block can have at most one block on top of it (valid state).
    - Moving a block requires unstacking all blocks above it first.
    - The cost to move a block is 2 actions per block above it plus 2 actions for the block itself.

    # Heuristic Initialization
    - Extract the goal conditions to determine the required positions for each block.
    - Identify all blocks mentioned in the goal (goal blocks).

    # Step-By-Step Thinking for Computing Heuristic
    1. For each goal block:
        a. If it's not in the correct position, add 2 actions per block above it plus 2 for itself.
        b. If its goal support is another block not in the correct position, add their movement cost.
    2. For each non-goal block:
        a. If it's on top of any goal block, add 2 actions per block above it plus 2 for itself.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goal_supports = {}  # Maps each goal block to its goal support (block or 'table')
        for goal in task.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'on':
                block, support = parts[1], parts[2]
                self.goal_supports[block] = support
            elif parts[0] == 'on-table':
                block = parts[1]
                self.goal_supports[block] = 'table'
        self.goal_blocks = set(self.goal_supports.keys())

    def __call__(self, node):
        state = node.state

        # Build current_on and current_above
        current_on = {}
        on_facts = [fact for fact in state if fact.startswith('(on ') or fact.startswith('(on-table')]
        for fact in on_facts:
            parts = fact[1:-1].split()
            if parts[0] == 'on':
                block, support = parts[1], parts[2]
                current_on[block] = support
            elif parts[0] == 'on-table':
                block = parts[1]
                current_on[block] = 'table'

        # Compute current_above for each block
        current_above = {}
        for block in current_on:
            count = 0
            current_block = block
            while True:
                # Find blocks directly on top of current_block
                next_blocks = [b for b, s in current_on.items() if s == current_block]
                if not next_blocks:
                    break
                count += len(next_blocks)
                current_block = next_blocks[0]  # Only one block can be on top
            current_above[block] = count

        heuristic_value = 0

        # Process each block in the current state
        for block in current_on:
            if block in self.goal_blocks:
                # Check if the block is in the correct position
                current_support = current_on[block]
                goal_support = self.goal_supports[block]
                if current_support != goal_support:
                    # Add cost for this block
                    heuristic_value += 2 * (current_above.get(block, 0) + 1)
                    # Check if the goal support is a block and needs to be moved
                    if goal_support != 'table' and goal_support in self.goal_supports:
                        # Check if the goal support is in the correct position
                        if current_on.get(goal_support, 'table') != self.goal_supports.get(goal_support, 'table'):
                            heuristic_value += 2 * (current_above.get(goal_support, 0) + 1)
            else:
                # Check if this block is on top of any goal block
                current_support = current_on.get(block, 'table')
                found_goal_block = False
                while current_support != 'table':
                    if current_support in self.goal_blocks:
                        found_goal_block = True
                        break
                    current_support = current_on.get(current_support, 'table')
                if found_goal_block:
                    # Add cost for this block
                    heuristic_value += 2 * (current_above.get(block, 0) + 1)

        return heuristic_value
