from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    This heuristic estimates the number of actions required to reach the goal state
    by considering three components:
    1. The number of blocks that are not part of a correctly built goal stack
       segment from the table up.
    2. The number of blocks that are obstructing the goal configuration by being
       on top of other blocks where they shouldn't be according to the goal state.
    3. A penalty for holding a block when the goal state has not yet been reached,
       encouraging the agent to utilize or put down the held block.

    The heuristic is not admissible but aims to be informative for greedy best-first search.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration and identifying all blocks.

        Args:
            task: The planning task object containing initial state, goals, etc.
        """
        super().__init__(task) # Call base class constructor

        # Map goal support: block -> block_below or 'table'
        # Example: {'b1': 'b2', 'b2': 'table'} for goal (on b1 b2), (on-table b2)
        self.goal_below = {}
        # Map goal stack above: block_below -> block_above or None (if clear)
        # Example: {'b2': 'b1', 'table': 'b2'} for goal (on b1 b2), (on-table b2)
        self.goal_above = {}
        # Set of blocks that should be on the table in the goal
        # Example: {'b2'} for goal (on b1 b2), (on-table b2)
        self.goal_on_table = set()
        # Set of all blocks involved in the problem (from initial or goal)
        self.all_blocks = set()

        # Parse goal facts to build goal configuration maps and collect blocks
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts[0] == 'on':
                block, below = parts[1], parts[2]
                self.goal_below[block] = below
                self.goal_above[below] = block
                self.all_blocks.add(block)
                self.all_blocks.add(below)
            elif parts[0] == 'on-table':
                block = parts[1]
                self.goal_below[block] = 'table'
                self.goal_above[block] = None # Block on table is clear above in goal
                self.goal_on_table.add(block)
                self.all_blocks.add(block)
            elif parts[0] == 'clear':
                 # Explicit clear goal. Add block to all_blocks.
                 self.all_blocks.add(parts[1])

        # Add blocks from initial state to ensure all blocks are considered
        for fact in task.initial_state:
            parts = get_parts(fact)
            # Assuming predicates like 'on', 'on-table', 'holding', 'clear'
            # have objects as parameters.
            if len(parts) > 1:
                 self.all_blocks.update(parts[1:])

        # Remove non-object terms that might have been added during parsing
        self.all_blocks.discard('table')
        self.all_blocks.discard('arm-empty')
        self.all_blocks.discard('clear')
        self.all_blocks.discard('holding')
        self.all_blocks.discard('on')
        self.all_blocks.discard('on-table')


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer estimate of the remaining cost to reach the goal.
        """
        state = node.state

        # If the state is the goal state, the heuristic is 0.
        if self.task.goal_reached(state):
             return 0

        # Build current configuration maps from the state facts
        current_below = {} # block -> block_below or 'table'
        current_above = {} # block_below -> block_above
        current_on_table = set()
        current_holding = None
        # current_clear = set() # Explicit clear facts are not directly used in this heuristic logic

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                block, below = parts[1], parts[2]
                current_below[block] = below
                current_above[below] = block # Store who is on top
            elif parts[0] == 'on-table':
                block = parts[1]
                current_below[block] = 'table'
                current_on_table.add(block)
                # current_above[block] remains None if nothing is on it
            elif parts[0] == 'holding':
                current_holding = parts[1]
            # elif parts[0] == 'clear':
            #     current_clear.add(parts[1]) # Not used

        # Memoization dictionary for the recursive is_correctly_stacked function
        self._correctly_stacked_cache = {}

        def is_correctly_stacked(block):
            """
            Recursively checks if a block is part of a correctly built goal stack
            segment from the table up in the current state.

            Args:
                block: The block to check.

            Returns:
                True if the block is correctly stacked according to the goal, False otherwise.
            """
            # Check cache first
            if block in self._correctly_stacked_cache:
                return self._correctly_stacked_cache[block]

            result = False # Default result is False

            # Determine the block's current support (block below, table, or arm)
            current_below_val = current_below.get(block)
            if current_holding == block:
                 current_below_val = 'arm' # Special case: block is held by the arm

            # Case 1: The block should be on the table in the goal state
            if block in self.goal_on_table:
                if current_below_val == 'table':
                    result = True # It is currently on the table, matching the goal

            # Case 2: The block should be on another block in the goal state
            elif block in self.goal_below:
                goal_below_val = self.goal_below[block]
                # Check if the block is currently on its correct goal support
                if current_below_val == goal_below_val:
                    # If it's on the correct support, check if that support is correctly stacked.
                    if goal_below_val == 'table':
                         # The support is the table, which is always considered correctly stacked as a base.
                         result = True
                    else:
                         # Recursively check the block below in the goal stack.
                         result = is_correctly_stacked(goal_below_val)

            # Note: Blocks not explicitly mentioned in 'on' or 'on-table' goal facts
            # are not considered by this function and do not contribute to h1.

            # Store the result in the cache before returning
            self._correctly_stacked_cache[block] = result
            return result

        # Heuristic Component 1: Count blocks not part of a correctly built goal stack segment
        # Iterate over blocks that have a specific goal support defined ('on' or 'on-table')
        h1 = 0
        blocks_with_goal_pos = set(self.goal_below.keys()) | self.goal_on_table
        for block in blocks_with_goal_pos:
             if not is_correctly_stacked(block):
                 h1 += 1

        # Heuristic Component 2: Count blocks that are obstructing the goal configuration
        # Count blocks that are currently on top of another block, but are NOT
        # supposed to be directly on that block in the goal state.
        h2 = 0
        # Iterate through blocks that have something on top in the current state (keys in current_above).
        for block_below, block_on_top in current_above.items():
             # Check if block_on_top is supposed to be directly on block_below in the goal state.
             goal_block_on_top = self.goal_above.get(block_below)
             if goal_block_on_top != block_on_top:
                 # block_on_top is currently on block_below, but this is not the goal configuration
                 # for block_below (either block_below should be clear, or have a different block on top).
                 # This block_on_top is obstructing the goal stack structure below it.
                 h2 += 1

        # Heuristic Component 3: Penalty for holding a block when not in goal
        # If the arm is holding a block and the goal has not been reached,
        # it likely requires at least one action (putdown or stack) involving the held block.
        h3 = 0
        # We already checked if goal is reached at the beginning.
        if current_holding is not None:
             h3 = 1

        # The total heuristic value is the sum of the components.
        h = h1 + h2 + h3

        return h
