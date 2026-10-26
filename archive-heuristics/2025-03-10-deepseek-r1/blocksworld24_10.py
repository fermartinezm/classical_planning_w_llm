from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

class Blocksworld24Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal by considering the number of misplaced blocks and the blocks that need to be moved to access them. Each misplaced block contributes a cost based on the number of blocks above it plus two actions (pickup and stack).

    # Assumptions
    - All blocks mentioned in the goal must be in specific positions (on or on-table).
    - Blocks not mentioned in the goal are ignored.
    - Moving a block requires two actions (pickup and stack/putdown) plus one action per block above it that needs to be moved first.

    # Heuristic Initialization
    - Extract the goal conditions to build a map of each block's goal parent (the block it should be on or None for on-table).
    - Static facts are not used in this heuristic.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Extract Goal Structure**: For each block in the goal, determine its required parent (the block it should be on or table).
    2. **Current State Analysis**: For each block in the current state, determine its current parent (block or table).
    3. **Check Correct Placement**: A block is correctly placed if it is on its goal parent and the goal parent is correctly placed (recursively).
    4. **Count Misplaced Blocks**: For each block not correctly placed, count the number of blocks stacked above it in the current state.
    5. **Calculate Heuristic**: Sum (number of blocks above + 2) for all misplaced blocks. This accounts for moving the blocks above (each requiring an unstack) and the misplaced block itself (pickup and stack).
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goal_parent_map = {}
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'on':
                block, parent = parts[1], parts[2]
                self.goal_parent_map[block] = parent
            elif parts[0] == 'on-table':
                block = parts[1]
                self.goal_parent_map[block] = None

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        current_parent_map = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                block, parent = parts[1], parts[2]
                current_parent_map[block] = parent
            elif parts[0] == 'on-table':
                block = parts[1]
                current_parent_map[block] = None

        memo = {}

        def is_correctly_placed(block):
            if block in memo:
                return memo[block]
            if block not in self.goal_parent_map:
                memo[block] = True
                return True
            current_parent = current_parent_map.get(block, None)
            goal_parent = self.goal_parent_map.get(block, None)
            if current_parent != goal_parent:
                memo[block] = False
                return False
            if goal_parent is None:
                memo[block] = True
                return True
            result = is_correctly_placed(goal_parent)
            memo[block] = result
            return result

        misplaced_blocks = [block for block in self.goal_parent_map if not is_correctly_placed(block)]

        def count_blocks_above(block):
            count = 0
            current_block = block
            while True:
                found = False
                for upper_block, parent in current_parent_map.items():
                    if parent == current_block:
                        count += 1
                        current_block = upper_block
                        found = True
                        break
                if not found:
                    break
            return count

        return sum(count_blocks_above(block) + 2 for block in misplaced_blocks)
