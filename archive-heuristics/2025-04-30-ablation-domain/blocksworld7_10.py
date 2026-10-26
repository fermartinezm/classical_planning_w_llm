from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state.
    It counts the cost of moving blocks that are in the wrong position relative
    to the block below them (or table), and the cost of clearing blocks that
    are on top of blocks that need to be moved or should be clear in the goal.

    # Assumptions
    - Actions have a cost of 1.
    - Moving a block from one position to another typically requires 2 actions:
      pick/unstack and put/stack.
    - Clearing a block (moving something off its top) requires moving the block
      on top, which also costs 2 actions (pick/unstack + put/stack).
    - A held block needs 1 action to be placed.
    - 'table' and 'arm' are special locations/states, not blocks.
    - The heuristic is non-admissible.

    # Heuristic Initialization
    - Stores the full set of goal facts.
    - Collects all block names present in the initial state or goal facts.
    - Extracts the goal configuration (`on` and `on-table` facts) to build
      maps representing the desired block positions relative to the block below
      (`goal_below_map`) and the block above (`goal_above_map`).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the current state is the goal state. If yes, return 0.
    2. Parse the current state to build maps representing the current block
       positions relative to the block below (`current_below_map`), the block
       above (`current_above_map`), and identify the currently held block
       (`current_holding_block`).
    3. Initialize heuristic value `h = 0`.
    4. Iterate through all blocks identified in the problem instance:
       a. Determine the block's current position relative to the block below
          (or table/arm) and its goal position relative to the block below
          (or table).
       b. Count the number of blocks currently stacked directly or indirectly
          on top of the current block.
       c. If the block has a specified goal position (is part of a goal stack)
          AND its current position relative to the block below is different
          from its goal position:
          - Add 1 to `h` if the block is currently held (cost to place it).
          - Add 2 to `h` if the block is on the table or another block (cost to pick/unstack and place).
          - If the block is not held, add 2 for each block currently stacked on top (cost to clear it).
       d. Else, if the block has a specified goal position AND its current
          position relative to the block below is correct, BUT it should be
          clear in the goal state and is not clear in the current state:
          - Add 2 for each block currently stacked on top (cost to clear it).
    5. If there is a block currently being held, AND it does NOT have a specified
       goal position (meaning it's an auxiliary block or misplaced block not
       explicitly in goal stacks), add 1 to `h` (cost to put it down somewhere).
    6. Return `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configurations and
        identifying all blocks.
        """
        self.goals = task.goals # Store the full set of goal facts

        # Collect all block names from initial state and goal
        self.all_blocks = set()
        # Add objects from initial state
        for fact in task.initial_state:
             parts = get_parts(fact)
             # Assuming object names are the arguments to predicates like on, on-table, clear, holding
             if parts[0] in ['on', 'on-table', 'clear', 'holding']:
                 for part in parts[1:]: # Skip predicate name
                     if part not in ['table', 'arm']: # Exclude keywords
                         self.all_blocks.add(part)
             # Also add objects from goal facts
        for goal in self.goals:
             parts = get_parts(goal)
             # Assuming object names are the arguments to predicates like on, on-table, clear
             if parts[0] in ['on', 'on-table', 'clear']:
                 for part in parts[1:]: # Skip predicate name
                     if part not in ['table', 'arm']: # Exclude keywords
                          self.all_blocks.add(part)

        # Build goal maps: block -> block_below and block_below -> block_on_top
        self.goal_below_map = {}
        self.goal_above_map = {} # Maps block_below -> block_on_top (assuming unique block on top in goal)

        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == 'on':
                block_on_top, block_below = parts[1], parts[2]
                self.goal_below_map[block_on_top] = block_below
                self.goal_above_map[block_below] = block_on_top
            elif predicate == 'on-table':
                block_on_table = parts[1]
                self.goal_below_map[block_on_table] = 'table'
            # 'clear' facts in goal are implicitly handled by checking goal_above_map == 'clear'

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # If the state is the goal state, the heuristic is 0.
        if self.goals <= state:
            return 0

        # Parse current state
        current_on_map = {} # Maps block_on_top -> block_below
        current_on_table_set = set()
        current_holding_block = None
        # current_clear_set = set() # Not strictly needed for this heuristic logic

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'on':
                current_on_map[parts[1]] = parts[2]
            elif predicate == 'on-table':
                current_on_table_set.add(parts[1])
            elif predicate == 'holding':
                current_holding_block = parts[1]
            # elif predicate == 'clear':
            #      current_clear_set.add(parts[1])
            # 'arm-empty' is implicit if current_holding_block is None

        # Build current maps: block -> block_below and block_below -> block_on_top
        current_below_map = {}
        current_above_map = {} # Maps block_below -> block_on_top (assuming unique block on top in current state)

        for block in self.all_blocks:
            if block == current_holding_block:
                current_below_map[block] = 'arm'
            elif block in current_on_table_set:
                current_below_map[block] = 'table'
            elif block in current_on_map:
                 # Find the block B such that block is on B
                 current_below_map[block] = current_on_map[block]
            # else: block is not in state in a known location (shouldn't happen in valid states)
            # current_below_map will not have the block, get() will return None.

        # Build current_above_map from current_on_map
        for block_on_top, block_below in current_on_map.items():
             current_above_map[block_below] = block_on_top # Assumes only one block on top in current state

        h = 0

        # Iterate through all blocks to find misplaced ones or ones needing clearing
        for block in self.all_blocks:
            current_pos = current_below_map.get(block)
            goal_pos = self.goal_below_map.get(block)

            # Calculate blocks on top of the current block in current state
            blocks_on_top_count = 0
            current = current_above_map.get(block)
            while current and current != 'clear':
                blocks_on_top_count += 1
                current = current_above_map.get(current)

            # Condition 1: Block is in the wrong place relative to the block below it,
            # AND it's a block whose final position is specified in the goal.
            if goal_pos is not None and current_pos != goal_pos:
                # Cost to move this block:
                # If held: 1 action (put/stack)
                # If on table/block: 2 actions (pick/unstack + put/stack)
                if current_pos == 'arm':
                    h += 1
                else:
                    h += 2
                    # Cost to clear this block (if not held): 2 actions per block on top
                    h += blocks_on_top_count * 2

            # Condition 2: Block is in the correct immediate position (relative to below),
            # AND its final position is specified in the goal,
            # BUT it should be clear in the goal state and is not clear in the current state.
            elif goal_pos is not None and current_pos == goal_pos:
                goal_block_above = self.goal_above_map.get(block, 'clear')
                # Check if it should be clear in goal but isn't currently clear
                if goal_block_above == 'clear' and blocks_on_top_count > 0:
                    # Cost to clear this block: 2 actions per block on top
                    h += blocks_on_top_count * 2

        # Add cost for the held block *only if* it was not a block with a specified goal position.
        # If a block with a goal position is held, its cost was added in Condition 1 (as 1).
        # If a block *without* a goal position is held, it still needs to be put down (1 action).
        if current_holding_block is not None and self.goal_below_map.get(current_holding_block) is None:
             h += 1

        return h
