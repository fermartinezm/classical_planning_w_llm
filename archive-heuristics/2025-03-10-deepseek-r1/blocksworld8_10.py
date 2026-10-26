from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class blocksworld8Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goal by considering blocks that are misplaced in the goal structure and blocks that are on top of goal blocks but not part of the goal. For each such block, the heuristic adds 2 actions per block in its current stack (including itself).

    # Assumptions
    - The goal is a conjunction of 'on' and 'on-table' predicates.
    - Blocks not mentioned in the goal can be in any position but may need to be moved if they are on top of goal blocks.
    - Each block in the goal must be in the correct position, and all supporting blocks must also be correct.

    # Heuristic Initialization
    - Extract 'on' and 'on-table' predicates from the goal to build a dictionary of goal_parents for each block in the goal.
    - Static facts are not used in this heuristic.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each block in the current state:
        a. If the block is part of the goal structure and is not correctly placed (considering the entire chain down to the table), add 2 actions for each block in its current stack (including itself).
        b. If the block is not part of the goal structure but is on top of a goal block, add 2 actions for each block in its current stack (including itself).
    2. Sum all the computed actions to get the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goal_parents = {}
        for goal in task.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'on':
                self.goal_parents[parts[1]] = parts[2]
            elif parts[0] == 'on-table':
                self.goal_parents[parts[1]] = 'table'

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal."""
        state = node.state
        current_parents = {}
        current_children = {}

        # Parse current state to build parent and child relationships
        for fact in state:
            if fact.startswith('(on '):
                parts = fact[1:-1].split()
                if parts[0] == 'on' and len(parts) == 3:
                    block, parent = parts[1], parts[2]
                    current_parents[block] = parent
                    current_children[parent] = block  # Track only one child per parent
            elif fact.startswith('(on-table '):
                parts = fact[1:-1].split()
                if parts[0] == 'on-table' and len(parts) == 2:
                    block = parts[1]
                    current_parents[block] = 'table'
                    # Note: current_children for 'table' may be overwritten, but handled in heuristic logic

        total = 0

        for block in current_parents:
            # Check if the block is part of the goal structure
            if block in self.goal_parents:
                # Check if the block is correctly placed
                if not self._is_correct(block, current_parents):
                    count = self._count_blocks_above(block, current_children)
                    total += 2 * (count + 1)
            else:
                # Check if the block is on top of a goal block
                current_parent = current_parents[block]
                if current_parent in self.goal_parents:
                    count = self._count_blocks_above(block, current_children)
                    total += 2 * (count + 1)

        return total

    def _is_correct(self, block, current_parents):
        """Check if a block is correctly placed according to the goal structure."""
        current_block = block
        while True:
            if current_block not in self.goal_parents:
                return True
            current_parent = current_parents.get(current_block, 'table')
            goal_parent = self.goal_parents.get(current_block, 'table')
            if current_parent != goal_parent:
                return False
            if goal_parent == 'table':
                return True
            current_block = goal_parent

    def _count_blocks_above(self, block, current_children):
        """Count the number of blocks stacked on top of the given block."""
        count = 0
        current = current_children.get(block, None)
        while current is not None:
            count += 1
            current = current_children.get(current, None)
        return count
