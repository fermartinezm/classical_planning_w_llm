from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class blocksworld25Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state by considering:
    1. Blocks that are not in their correct position (either on the wrong block or on the table).
    2. The number of blocks stacked above each misplaced block, which need to be moved first.

    # Assumptions
    - Each block move requires unstacking blocks above it first.
    - Moving a block and each block above it requires 2 actions per block (unstack and putdown or stack).

    # Heuristic Initialization
    - Extracts goal conditions to determine the correct positions of each block.
    - Constructs a mapping of each block to its goal support (either another block or the table).

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is a goal state; return 0 if true.
    2. Parse the current state to determine the current support (what each block is on).
    3. Build stacks to determine how many blocks are above each block in the current state.
    4. For each block not in its correct position, add 2 actions per block above it plus 2 actions for the block itself.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract goal on and on-table information
        self.goal_on = {}
        self.goal_on_table = set()
        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'on':
                self.goal_on[parts[1]] = parts[2]
            elif parts[0] == 'on-table':
                self.goal_on_table.add(parts[1])

        # Build a map from block to its goal support
        self.goal_support = {}
        for block in self.goal_on:
            self.goal_support[block] = self.goal_on[block]
        for block in self.goal_on_table:
            self.goal_support[block] = 'table'

    def __call__(self, node):
        state = node.state

        # Check if current state is a goal state
        if self.goals.issubset(state):
            return 0

        current_on = {}
        on_table = set()

        # Parse current state
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'on':
                current_on[parts[1]] = parts[2]
            elif parts[0] == 'on-table':
                on_table.add(parts[1])

        # Build current_support: block -> 'table' or supporting block
        current_support = {}
        for block in current_on:
            current_support[block] = current_on[block]
        for block in on_table:
            current_support[block] = 'table'

        # Collect all blocks from current state and goal
        all_blocks = set(current_support.keys()).union(on_table).union(self.goal_support.keys())

        # Build stacks to compute current_above for each block
        visited = set()
        stacks = []
        for block in all_blocks:
            if block not in visited and block in current_support:
                stack = []
                current = block
                while True:
                    if current in visited:
                        break
                    visited.add(current)
                    stack.append(current)
                    if current not in current_support or current_support[current] == 'table':
                        break
                    current = current_support[current]
                stacks.append(stack)

        block_to_stack = {}
        for stack in stacks:
            for i, blk in enumerate(stack):
                block_to_stack[blk] = (stack, i)

        heuristic_value = 0

        # Check each block's current support against goal support
        for block in all_blocks:
            current_sup = current_support.get(block, 'table')
            goal_sup = self.goal_support.get(block, 'table')

            if current_sup != goal_sup:
                # Compute current_above
                if block in block_to_stack:
                    stack, index = block_to_stack[block]
                    current_above = len(stack) - index - 1
                else:
                    current_above = 0
                heuristic_value += 2 * (current_above + 1)

        return heuristic_value
