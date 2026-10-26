from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class blocksworld1Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to move all blocks to their goal positions. Each block not in its correct position contributes 2 to the heuristic value, accounting for the pickup and putdown/stack actions required.

    # Assumptions:
    - Each block move requires two actions (pickup and stack/putdown).
    - Blocks not mentioned in the goal can be in any position and do not affect the heuristic.
    - Held blocks are considered misplaced and contribute 2 to the heuristic.

    # Heuristic Initialization
    - Extract the goal conditions to determine the correct position (on or on-table) for each block mentioned in the goal.
    - Identify all blocks mentioned in the goal to focus the heuristic calculation.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all blocks mentioned in the goal's on and on-table predicates.
    2. For each such block:
        a. If the block is currently being held, add 2 (needs to be put down or stacked).
        b. If the block is not in its goal position (on the correct block or table), add 2.
    3. Sum these values to get the total heuristic estimate.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and relevant blocks."""
        self.goal_on = {}
        self.goal_table = set()
        self.goal_blocks = set()

        # Extract goal conditions and relevant blocks
        for goal in task.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'on':
                block = parts[1]
                under = parts[2]
                self.goal_on[block] = under
                self.goal_blocks.update([block, under])
            elif parts[0] == 'on-table':
                block = parts[1]
                self.goal_table.add(block)
                self.goal_blocks.add(block)

        # Remove 'table' from goal_blocks if present
        self.goal_blocks.discard('table')

    def __call__(self, node):
        """Estimate the number of actions needed to achieve the goal state."""
        state = node.state
        current_on = {}
        current_holding = set()

        # Parse current state
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'on':
                current_on[parts[1]] = parts[2]
            elif parts[0] == 'on-table':
                current_on[parts[1]] = 'table'
            elif parts[0] == 'holding':
                current_holding.add(parts[1])

        heuristic_value = 0

        for block in self.goal_blocks:
            if block in current_holding:
                # Block is being held, needs to be placed
                heuristic_value += 2
            else:
                current_parent = current_on.get(block, 'table')
                # Determine goal parent
                if block in self.goal_table:
                    goal_parent = 'table'
                else:
                    goal_parent = self.goal_on.get(block, None)

                if current_parent != goal_parent:
                    heuristic_value += 2

        return heuristic_value
