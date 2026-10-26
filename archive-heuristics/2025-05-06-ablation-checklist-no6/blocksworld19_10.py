# Assuming heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    Estimates the number of actions needed to reach the goal state.
    The heuristic counts:
    1. Blocks that are not on their correct goal base (block or table).
    2. Blocks that are currently on top of another block, where the block on top is *not* the one that should be there in the goal.
    3. The arm is holding a block when the goal requires it to be empty.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and identifying all blocks.
        """
        self.goals = task.goals

        # Parse goal facts to build goal structure
        self.goal_on_map = {} # Maps block -> block_below in goal
        self.goal_table_bases = set() # Set of blocks that should be on the table in goal
        self.all_blocks = set() # Collect all block names

        for goal in self.goals:
            parts = self._get_parts(goal)
            if parts and parts[0] == 'on':
                block_on_top, block_below = parts[1], parts[2]
                self.goal_on_map[block_on_top] = block_below
                self.all_blocks.add(block_on_top)
                self.all_blocks.add(block_below)
            elif parts and parts[0] == 'on-table':
                block = parts[1]
                self.goal_table_bases.add(block)
                self.all_blocks.add(block)
            elif parts and len(parts) > 1 and parts[1] != 'arm-empty': # Collect blocks from other goal types if any
                 self.all_blocks.add(parts[1])
            elif parts and parts[0] == 'arm-empty':
                 pass # arm-empty is a state property, not a block

        # Collect blocks from initial state as well, in case some blocks are not in the goal
        for fact in task.initial_state:
             parts = self._get_parts(fact)
             if parts and len(parts) > 1 and parts[1] != 'arm-empty':
                 self.all_blocks.add(parts[1])
             if parts and len(parts) > 2:
                 self.all_blocks.add(parts[2])

    def _get_parts(self, fact):
        """Extract the components of a PDDL fact."""
        # Remove parentheses and split by spaces
        content = fact[1:-1].strip()
        if not content: # Handle empty fact like '()' though unlikely
            return []
        return content.split()

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # If the state is a goal state, the heuristic is 0.
        if self.goals <= state:
            return 0

        h = 0

        # Parse current state facts
        current_on_map = {} # Maps block -> block_below in current state
        current_table_blocks = set() # Set of blocks currently on the table
        current_holding = None # The block currently held, or None
        blocks_with_something_on_top = set() # Set of blocks that have something on top

        for fact in state:
            parts = self._get_parts(fact)
            if parts and parts[0] == 'on':
                block_on_top, block_below = parts[1], parts[2]
                current_on_map[block_on_top] = block_below
                blocks_with_something_on_top.add(block_below)
            elif parts and parts[0] == 'on-table':
                block = parts[1]
                current_table_blocks.add(block)
            elif parts and parts[0] == 'holding':
                current_holding = parts[1]

        # Part 1: Blocks not on correct base.
        # Iterate over all blocks identified during initialization
        for block in self.all_blocks:
            goal_support = self.goal_on_map.get(block, 'table' if block in self.goal_table_bases else None)

            # Find current support for the block
            current_support = current_on_map.get(block)
            if current_support is None: # Not on another block
                if block in current_table_blocks:
                    current_support = 'table'
                elif current_holding == block:
                    current_support = 'arm'
                # else: block is not on/on-table/held? Assume valid states.

            # If the block has a defined goal support and is not on that support
            if goal_support is not None and current_support != goal_support:
                h += 1 # Block is not on its correct base

        # Part 2: Blocks that have something on top (and shouldn't).
        # Iterate over blocks that currently have something on top
        for block_below in blocks_with_something_on_top:
            # Find the block Z currently on block_below
            current_block_on_top = None
            for fact in state:
                parts = self._get_parts(fact)
                if parts and parts[0] == 'on' and parts[2] == block_below:
                    current_block_on_top = parts[1]
                    break # Assuming only one block can be on top

            # Find the block X that should be on block_below in the goal (if any).
            goal_block_on_top = None
            for block_on_top, block_below_goal in self.goal_on_map.items():
                if block_below_goal == block_below:
                    goal_block_on_top = block_on_top
                    break # Assuming only one block can be on top in the goal

            # If there is a block currently on block_below, and it's not the correct one according to the goal
            if current_block_on_top is not None and current_block_on_top != goal_block_on_top:
                 h += 1 # This block on top is blocking.

        # Part 3: Arm not empty when it should be.
        # Check if (arm-empty) is a goal and is not true in the current state
        if "(arm-empty)" in self.goals and "(arm-empty)" not in state:
            h += 1

        return h
