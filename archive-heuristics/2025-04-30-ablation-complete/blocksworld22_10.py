from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    by summing costs associated with blocks that are not in their correct
    goal position or are blocking other blocks from reaching their goal
    position. It considers the cost of moving a block and the cost of
    clearing blocks stacked on top of it.

    # Assumptions
    - Standard Blocksworld actions (pickup, putdown, stack, unstack).
    - The cost of moving a block from its current position to its goal
      position (if not held) is estimated as 2 actions (e.g., unstack/pickup + stack/putdown).
    - The cost of clearing a block is estimated as 1 action for each block
      currently stacked directly or indirectly on top of it.
    - If a block is currently held, it needs 1 action to be placed.
    - The heuristic is non-admissible and designed for greedy best-first search.

    # Heuristic Initialization
    The heuristic pre-processes the goal state to identify:
    - `goal_on_map`: A dictionary mapping a block to the block it should be
      stacked directly on top of in the goal state (e.g., {'b1': 'b2'} for (on b1 b2)).
    - `goal_on_table_set`: A set of blocks that should be on the table
      in the goal state.
    - `goal_clear_set`: A set of blocks that should be clear (have nothing
      on top) in the goal state.
    - `goal_blocks`: A set of all blocks mentioned in the goal configuration.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is computed as follows:
    1. Parse the current state to determine the position of each block
       (on another block, on the table, or held) and build an inverse
       stack map (`current_above_map`) to quickly find blocks stacked above.
       Also identify which blocks are currently clear and if the arm is holding a block.
    2. Initialize the total heuristic cost `h` to 0.
    3. Iterate through each block that is part of the goal configuration (`goal_blocks`).
    4. For the current block `B`:
       a. Determine its desired goal position (on block Y or on table) based on `goal_on_map` and `goal_on_table_set`.
       b. Determine its current position (on block Z, on table, or held) based on the parsed state.
       c. Count the number of blocks currently stacked directly or indirectly on top of `B` using the `current_above_map`.
       d. If `B` is currently held:
          - Add 1 to `h` (estimated cost to place the block).
       e. If `B` is not held, and its current position is different from its goal position (relative to the block below or table):
          - Add 2 to `h` (estimated cost to move the block: unstack/pickup + stack/putdown).
          - Add the count of blocks currently stacked above `B` to `h` (estimated cost to clear `B`).
       f. If `B` is not held, its current position is the same as its goal position (relative to the block below or table), but the goal requires `B` to be clear (`B` in `goal_clear_set`) and it is not clear (`B` not in `current_clear_set`):
          - Add the count of blocks currently stacked above `B` to `h` (estimated cost to clear `B`).
    5. Return the total computed cost `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration.

        Args:
            task: The planning task object containing initial state, goals, etc.
        """
        self.goals = task.goals

        # Parse goal state to get goal configuration
        self.goal_on_map = {}
        self.goal_on_table_set = set()
        self.goal_clear_set = set()
        self.goal_blocks = set() # Keep track of all blocks mentioned in goal config

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'on':
                block, block_below = parts[1], parts[2]
                self.goal_on_map[block] = block_below
                self.goal_blocks.add(block)
                self.goal_blocks.add(block_below)
            elif predicate == 'on-table':
                block = parts[1]
                self.goal_on_table_set.add(block)
                self.goal_blocks.add(block)
            elif predicate == 'clear':
                block = parts[1]
                self.goal_clear_set.add(block)
                self.goal_blocks.add(block) # Clear goals might mention blocks not in on/on-table goals

        # Static facts are not used in this heuristic.
        # static_facts = task.static

    def _count_blocks_above(self, block, current_above_map):
        """Counts blocks stacked directly or indirectly on top of the given block."""
        count = 0
        temp_block = block
        # Iterate upwards from the block using the inverse map
        while temp_block in current_above_map:
            temp_block = current_above_map[temp_block]
            count += 1
        return count

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer representing the estimated cost to reach the goal.
        """
        state = node.state

        # Parse current state
        current_on_map = {}
        current_on_table_set = set()
        current_clear_set = set()
        current_holding = None
        current_above_map = {} # Inverse map: maps block_below -> block_above

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'on':
                block, block_below = parts[1], parts[2]
                current_on_map[block] = block_below
                current_above_map[block_below] = block # Build inverse map
            elif predicate == 'on-table':
                block = parts[1]
                current_on_table_set.add(block)
            elif predicate == 'clear':
                block = parts[1]
                current_clear_set.add(block)
            elif predicate == 'holding':
                block = parts[1]
                current_holding = block
            # Ignore arm-empty

        h = 0
        # Iterate through all blocks that are part of the goal configuration
        for block in self.goal_blocks:
            # Determine goal position
            goal_pos = None
            if block in self.goal_on_map:
                goal_pos = self.goal_on_map[block]
            elif block in self.goal_on_table_set:
                goal_pos = 'table' # Use a special string 'table'
            # If block is only in goal_clear_set, goal_pos remains None

            # Determine current position
            current_pos = None
            if current_holding == block:
                current_pos = 'holding' # Use a special string 'holding'
            elif block in current_on_map:
                current_pos = current_on_map[block]
            elif block in current_on_table_set:
                current_pos = 'table' # Use a special string 'table'
            # If a block is in goal_blocks but not in current state (impossible in STRIPS)
            # or if it's in state but not in goal_blocks (we ignore these)
            # the logic handles it by only iterating over goal_blocks.

            blocks_above_B = self._count_blocks_above(block, current_above_map)

            if current_pos == 'holding':
                # Block is held. It needs to be placed.
                h += 1
                # Blocks above it is 0 when held.
            elif goal_pos is not None and goal_pos != current_pos:
                # Block is in a stack/table but wrong place relative to below.
                h += 2 # Cost to move B (unstack/pickup + stack/putdown).
                h += blocks_above_B # Cost to clear B.
            elif block in self.goal_clear_set and block not in current_clear_set:
                # Block is in correct place (relative to below), but not clear (and should be).
                # This case only applies if goal_pos == current_pos (implicitly handled by elif structure)
                h += blocks_above_B # Cost to clear B.

        return h
