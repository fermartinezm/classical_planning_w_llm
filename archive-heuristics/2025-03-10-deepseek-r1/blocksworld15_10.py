from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict

class blocksworld15Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goal by considering:
    1. Blocks that are not in their correct position (on or on-table) as per the goal, requiring 2 actions each.
    2. Blocks that are on top of blocks required to be clear in the goal, requiring 2 actions per such block.

    # Assumptions
    - Moving a block requires at least two actions (pickup and putdown/stack).
    - Clear goals are achieved by moving all blocks on top of the specified block.
    - Blocks not mentioned in the goal can be in any state.

    # Heuristic Initialization
    - Extract goal conditions for 'on', 'on-table', and 'clear' predicates.
    - Build a map of each block's goal parent (another block or table) and clear requirements.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each block in the goal's 'on' and 'on-table' conditions:
        a. Check if the block is currently on the correct parent.
        b. Recursively verify that the parent block is also correctly positioned.
        c. If not, add 2 to the heuristic.
    2. For each block in the goal's 'clear' conditions:
        a. Count the number of blocks currently on top of it.
        b. Add 2 for each such block.
    """

    def __init__(self, task):
        """Extract goal information and static facts."""
        self.goal_parent_map = {}  # Maps each block to its goal parent (block or 'table')
        self.clear_goals = set()   # Blocks that must be clear in the goal

        # Extract 'on', 'on-table', and 'clear' goals
        for goal in task.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'on':
                self.goal_parent_map[parts[1]] = parts[2]
            elif parts[0] == 'on-table':
                self.goal_parent_map[parts[1]] = 'table'
            elif parts[0] == 'clear':
                self.clear_goals.add(parts[1])

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        current_parent_map = {}
        current_blocks_above = defaultdict(list)  # Maps each block to the blocks on top of it
        holding_blocks = set()

        # Parse current state to find parent relationships and blocks being held
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'on':
                current_parent_map[parts[1]] = parts[2]
                current_blocks_above[parts[2]].append(parts[1])
            elif parts[0] == 'on-table':
                current_parent_map[parts[1]] = 'table'
            elif parts[0] == 'holding':
                holding_blocks.add(parts[1])

        # Set parent to 'arm' for held blocks
        for block in holding_blocks:
            current_parent_map[block] = 'arm'

        # Check recursively if a block is correctly placed
        cache = {}
        def is_correct(block):
            if block in cache:
                return cache[block]
            # Check if the block is part of the goal
            if block not in self.goal_parent_map:
                # If not part of the goal, consider it correct
                cache[block] = True
                return True
            goal_parent = self.goal_parent_map[block]
            current_parent = current_parent_map.get(block, 'table')
            if current_parent != goal_parent:
                cache[block] = False
                return False
            if goal_parent == 'table':
                cache[block] = True
                return True
            # Recursively check the goal parent
            parent_correct = is_correct(goal_parent)
            cache[block] = parent_correct
            return parent_correct

        # Calculate heuristic for 'on' and 'on-table' goals
        h = 0
        for block in self.goal_parent_map:
            if not is_correct(block):
                h += 2

        # Calculate heuristic for 'clear' goals
        for block in self.clear_goals:
            # Count blocks on top of this block
            num_above = len(current_blocks_above.get(block, []))
            h += 2 * num_above

        return h
