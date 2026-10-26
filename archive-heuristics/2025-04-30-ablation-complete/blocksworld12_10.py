# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Note: The Heuristic base class is assumed to be provided externally.
# For standalone testing, you might need a dummy definition like:
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#     def __call__(self, node):
#         raise NotImplementedError

from fnmatch import fnmatch # Import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    # Split by space, handling multiple spaces
    return fact[1:-1].split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goal
    configuration of blocks. It identifies blocks that are part of a "wrong"
    stack. A stack is considered wrong if its base block is not on the correct
    block/table according to the goal, or if the stack is on a block that
    should be clear in the goal state but isn't. The heuristic counts the total
    number of unique blocks within these "wrong" stacks and multiplies by 2,
    estimating 2 actions (pickup/unstack + putdown/stack) per block that needs
    to be moved.

    # Assumptions
    - Action costs are uniform (implicitly 1 for each PDDL action). The heuristic
      uses a cost of 2 per block that is part of a misplaced or obstructing stack
      as a rough estimate (pickup/unstack + putdown/stack).
    - The heuristic focuses on achieving the correct stack structure and clear
      conditions specified in the goal.

    # Heuristic Initialization
    - Parses the goal facts to determine the desired block configuration:
      - `goal_below`: A dictionary mapping a block to the block it should be
        directly on top of, or 'table'.
      - `goal_clear_blocks`: A set of blocks that should have nothing on top
        in the goal state.
    - Blocksworld has no static facts relevant to the heuristic calculation.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Parse the current state to determine the current block configuration:
        - `current_below`: A dictionary mapping a block to the block it is
          currently directly on top of, or 'table'. Blocks that are held
          or not mentioned in 'on'/'on-table' facts are not keys in this map.
        - `current_above`: A dictionary mapping a block to the block currently
          directly on top of it.
        - `all_blocks_in_state`: A set of all blocks present in the state (those appearing
          in 'on' or 'on-table' facts, or being held).
    2.  Collect all unique blocks mentioned in either the goal or the state (`all_relevant_blocks`).
    3.  Define a helper function `get_stack_above_list(block, current_above_map)`
        that returns a list of blocks directly or indirectly stacked on top of
        the given `block` using the `current_above_map`.
    4.  Identify the set of blocks that are "wrongly based": `wrongly_based_blocks`.
        A block `B` is wrongly based if it is part of the goal configuration
        (i.e., `goal_below[B]` is defined) and its current position below
        (`current_below.get(B)`) is different from its target position below
        (`goal_below.get(B)`).
    5.  Identify the set of blocks that are "obstructed clear": `obstructed_clear_blocks`.
        A block `B` is obstructed clear if it is in the goal_clear_blocks set
        AND it currently has something on top of it (checked using `current_above.get(B)`).
    6.  Initialize an empty set `blocks_to_move`.
    7.  For each block in `wrongly_based_blocks`: Add the block itself and all
        blocks in the stack above it (obtained using `get_stack_above_list`)
        to the `blocks_to_move` set.
    8.  For each block in `obstructed_clear_blocks`: If this block is *not*
        already in `wrongly_based_blocks` (to avoid double-counting stacks),
        add all blocks in the stack above it (obtained using `get_stack_above_list`)
        to the `blocks_to_move` set.
    9.  The heuristic value is 2 times the total number of unique blocks in the
        `blocks_to_move` set.

    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration.
        """
        # Assuming task object has 'goals' attribute which is a frozenset of goal fact strings
        self.goals = task.goals

        # Parse goal facts to build goal_below and goal_clear_blocks
        self.goal_below = {}
        self.goal_clear_blocks = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]
            if predicate == 'on':
                if len(parts) == 3:
                    block_on_top, block_below = parts[1], parts[2]
                    self.goal_below[block_on_top] = block_below
            elif predicate == 'on-table':
                 if len(parts) == 2:
                    block = parts[1]
                    self.goal_below[block] = 'table'
            elif predicate == 'clear':
                 if len(parts) == 2:
                    block = parts[1]
                    self.goal_clear_blocks.add(block)

        # Blocksworld has no relevant static facts for this heuristic.
        # self.static = task.static # Access static facts if needed

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state # Assuming state is a frozenset of fact strings

        # Parse current state to build current_below, current_above, and all_blocks_in_state
        current_below = {}
        current_above = {}
        all_blocks_in_state = set()
        # current_holding_block = None # Not strictly needed for this heuristic logic

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]
            if predicate == 'on':
                if len(parts) == 3:
                    block_on_top, block_below = parts[1], parts[2]
                    current_below[block_on_top] = block_below
                    current_above[block_below] = block_on_top # Mapping block_below -> block_on_top
                    all_blocks_in_state.add(block_on_top)
                    all_blocks_in_state.add(block_below)
            elif predicate == 'on-table':
                if len(parts) == 2:
                    block = parts[1]
                    current_below[block] = 'table'
                    all_blocks_in_state.add(block)
            elif predicate == 'holding':
                 if len(parts) == 2:
                    # current_holding_block = parts[1] # Store if needed
                    all_blocks_in_state.add(parts[1])
            # 'clear' and 'arm-empty' are not needed for structure mapping

        # Collect all unique blocks mentioned in either the goal or the state
        all_relevant_blocks = set(self.goal_below.keys()) | self.goal_clear_blocks | all_blocks_in_state

        # Helper function to get list of blocks in the stack above a given block
        # Uses the current_above map (block_below -> block_on_top)
        def get_stack_above_list(block, current_above_map):
            stack = []
            current = block
            # Find the block directly on top of 'current'
            block_on_top = current_above_map.get(current)
            while block_on_top is not None:
                stack.append(block_on_top)
                current = block_on_top # Move up the stack
                block_on_top = current_above_map.get(current) # Find next block on top
            return stack

        # Identify blocks that are wrongly based
        # A block B is wrongly based if it's in the goal_below mapping
        # AND its current_below position is different from its goal_below position.
        wrongly_based_blocks = {
            block for block in all_relevant_blocks
            if self.goal_below.get(block) is not None and current_below.get(block) != self.goal_below.get(block)
        }

        # Identify blocks that should be clear in the goal but aren't
        # A block B is obstructed clear if it's in the goal_clear_blocks set
        # AND it currently has something on top of it.
        obstructed_clear_blocks = {
             block for block in all_relevant_blocks
             if block in self.goal_clear_blocks and current_above.get(block) is not None
        }

        # Collect all blocks that are part of a "wrong" stack
        # A block is in a "wrong" stack if it is wrongly based OR
        # it is on top of a wrongly based block OR
        # it is on top of a block that should be clear but isn't.
        blocks_to_move = set()

        # Add wrongly based blocks and everything above them
        for block in wrongly_based_blocks:
            blocks_to_move.add(block)
            blocks_to_move.update(get_stack_above_list(block, current_above))

        # Add blocks on top of obstructed_clear_blocks, but only if the base block
        # of this stack wasn't already covered by being wrongly_based.
        for block in obstructed_clear_blocks:
             if block not in wrongly_based_blocks:
                 blocks_to_move.update(get_stack_above_list(block, current_above))

        # The heuristic is 2 * the number of unique blocks that need to be moved.
        total_cost = 2 * len(blocks_to_move)

        return total_cost
