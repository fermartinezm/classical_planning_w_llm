from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Assumes fact is a string like "(predicate arg1 arg2)"
    if fact.startswith('(') and fact.endswith(')'):
        content = fact[1:-1].strip()
        if not content: # Handle empty parentheses like '()'
            return []
        return content.split()
    return [] # Should not happen with valid PDDL facts


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are 'out of place'
    relative to the goal configuration. A block is out of place if it is
    not on the correct block or table below it as specified in the goal,
    OR if it is on the correct block/table below it but has the wrong block
    (or any block, if it should be clear) currently stacked on top of it.

    # Assumptions
    - The goal specifies the desired position (on another block or on the table)
      for all blocks involved in the required goal stacks.
    - Blocks not mentioned in goal 'on' or 'on-table' predicates are not part
      of the strictly defined goal structure (their position doesn't directly
      contribute to the heuristic value, although they might block goal blocks).
    - Standard Blocksworld goals are assumed (no `(holding X)` goals, `(clear X)`
      usually means X is the top of a goal stack).

    # Heuristic Initialization
    - Parses the goal facts to build two mappings:
        - `goal_pos`: Maps each block in a goal stack to the block or 'table' it should be directly on.
        - `goal_block_above`: Maps each block or 'table' that is *below* another block in a goal stack to the block that should be directly on top of it.
    - Identifies the set of all blocks (`goal_blocks`) that are part of the goal configuration (mentioned in 'on' or 'on-table' goals).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state facts to determine the current position of each block.
       This creates a `current_pos` map where keys are blocks and values are the block
       they are currently on, 'table' if on the table, or 'arm' if being held.
    2. Initialize the heuristic value `h` to 0.
    3. Iterate through each block (`block`) that is part of the goal configuration (`self.goal_blocks`):
       a. Determine the block that should be directly below the current block in the goal stack (`target_below`). This is 'table' if the block should be on the table, or the block name it should be on. This is obtained from `self.goal_pos.get(block)`. It is None if the block is the top of a goal stack (i.e., not a key in `self.goal_pos`).
       b. Determine the block that is currently directly below the current block (`current_below`). This is 'table' if it's on the table, the block name it's on, or 'arm' if being held. This is obtained from `current_pos.get(block)`. It is None if the block is not currently on anything, on the table, or held (e.g., invalid state or not in problem).
       c. Check if the block is in the correct position relative to the block below it as defined in the goal (`is_in_final_relative_pos`). This is true if `target_below` is 'table' and `current_below` is 'table', OR if `target_below` is a block name and `current_below` is that same block name.
       d. If the block's position relative to the block below it is defined in the goal (i.e., `block` is a key in `self.goal_pos`) AND it is NOT in the correct position relative to the block below it, increment `h` by 1. This counts blocks that need to be moved because they are on the wrong base.
       e. Find the block that is currently directly on top of the current block (`block_on_top`).
       f. Find the block that should be directly on top of the current block in the goal stack (`block_should_be_on_top = self.goal_block_above.get(block)`). This is None if nothing should be on this block in the goal (i.e., it's the top of a goal stack).
       g. If there is a block currently on top (`block_on_top` is not None) AND it is NOT the block that should be on top (`block_on_top != block_should_be_on_top`), increment `h` by 1. This counts blocks that are correctly placed below but are blocked from above by the wrong block, OR blocks that should be clear but have something on them.
    4. Return the final value of `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration.
        """
        self.goals = task.goals

        # Map block -> block_below_it_in_goal_stack or 'table'
        self.goal_pos = {}
        # Set of all blocks involved in goal 'on' or 'on-table' predicates
        self.goal_blocks = set()
        # Map block_below -> block_above_it_in_goal_stack (for quick lookup)
        self.goal_block_above = {}

        # Parse goal facts
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'on':
                block_on_top, block_below = parts[1], parts[2]
                self.goal_pos[block_on_top] = block_below
                self.goal_blocks.add(block_on_top)
                self.goal_blocks.add(block_below)
                self.goal_block_above[block_below] = block_on_top
            elif predicate == 'on-table':
                block = parts[1]
                self.goal_pos[block] = 'table'
                self.goal_blocks.add(block)
            # 'clear' goals are implicitly handled by goal_block_above not having an entry for top blocks

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse current state facts
        current_pos = {} # block -> block_below or 'table' or 'arm'

        # Build current_pos map
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'on':
                current_pos[parts[1]] = parts[2]
            elif predicate == 'on-table':
                current_pos[parts[1]] = 'table'
            elif predicate == 'holding':
                current_pos[parts[1]] = 'arm' # Indicate it's held

        h = 0

        # Iterate through each block that is part of the goal configuration
        for block in self.goal_blocks:
            target_below = self.goal_pos.get(block)
            current_below = current_pos.get(block) # Can be None if block is not on anything/table/held

            # --- Part 1: Check position relative to below ---
            # This applies only to blocks whose position relative to below is defined in the goal
            # (i.e., blocks that are not the top of a goal stack).
            if block in self.goal_pos:
                is_in_final_relative_pos = False
                if target_below == 'table':
                    is_in_final_relative_pos = (current_below == 'table')
                elif target_below is not None: # Should always be true if block in self.goal_pos
                    is_in_final_relative_pos = (current_below == target_below)

                if not is_in_final_relative_pos:
                     h += 1

            # --- Part 2: Check block on top ---
            # This applies to all goal blocks.
            # Find the block currently on top of `block`.
            block_on_top = None
            # Iterate through current_pos values to find who is on top of `block`
            for other_block, pos in current_pos.items():
                if pos == block:
                    block_on_top = other_block
                    break # Assuming only one block can be directly on another

            # Find the block that should be on top of `block` in the goal.
            block_should_be_on_top = self.goal_block_above.get(block) # None if `block` is top of goal stack

            # If there is a block currently on top, AND it's not the one that should be there...
            if block_on_top is not None and block_on_top != block_should_be_on_top:
                 h += 1 # The block on top is wrong/missing.


        return h
