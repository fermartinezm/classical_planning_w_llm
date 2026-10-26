from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Robustly handle potential malformed facts
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    Estimates the number of blocks that are either in the wrong position
    or are in the correct position but have the wrong block on top.

    Heuristic value is the sum of:
    1. For each block B that is part of the goal stack structure (mentioned in goal on/on-table):
       If B's current support (block below it, table, or arm) is different from its goal support: +1.
    2. For each block B that is part of the goal stack structure:
       If B's current support is the same as its goal support, BUT the block currently on top of B
       is different from the block that should be on top of B according to the goal (or if B should be clear).
       This condition is only checked if the block B is not currently held.
       If the block currently on top is different from the desired one: +1.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration.
        """
        # The base class constructor might initialize self.goals and self.static
        # depending on its implementation. The example heuristics don't explicitly
        # call super().__init__, but they do access task.goals and task.static.
        # Let's explicitly get them from task.
        self.goals = task.goals
        # self.static = task.static # Blocksworld has no static facts relevant here

        # Parse goal predicates to understand the desired stack structure.
        # goal_support: maps block -> block it should be on (or 'table')
        # goal_above: maps block -> block that should be directly on top of *this* block (None if block should be clear)
        self.goal_support = {}
        self.goal_above = {}
        self.goal_blocks = set() # Blocks mentioned in goal on/on-table predicates

        # Build goal_support and identify goal_blocks
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip malformed goals

            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block, support = parts[1], parts[2]
                self.goal_support[block] = support
                self.goal_blocks.add(block)
                self.goal_blocks.add(support)
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                self.goal_support[block] = 'table'
                self.goal_blocks.add(block)

        # Build goal_above based on goal_support
        # Iterate through all blocks that are supports in the goal
        for block_below in self.goal_blocks:
             # Find the block that should be on top of block_below in the goal
             block_above = None
             for goal in self.goals:
                 parts = get_parts(goal)
                 if parts[0] == 'on' and len(parts) == 3 and parts[2] == block_below:
                     block_above = parts[1]
                     break # Assuming only one block can be on another in the goal

             # If a block was found that should be on top, record it
             if block_above is not None:
                 self.goal_above[block_below] = block_above
             # If no block was found, it means block_below should be clear in the goal.
             # We don't need to add an entry for None, get() will return None by default.


        # Note: This heuristic only considers blocks explicitly mentioned in goal 'on' or 'on-table' predicates.
        # Blocks that are not in the goal configuration but are in the state
        # are only accounted for if they are blocking a goal block.


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Parse current state to understand the current stack structure.
        # current_support: maps block -> block it is currently on (or 'table')
        # current_above: maps block -> block that is currently directly on top of it
        current_support = {}
        current_above = {}
        is_holding = None # Block currently held, or None

        # First pass to build initial support/above maps and find held block
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block, support = parts[1], parts[2]
                current_support[block] = support
                current_above[support] = block
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                current_support[block] = 'table'
            elif predicate == 'holding' and len(parts) == 2:
                is_holding = parts[1]
            # 'clear' and 'arm-empty' are not needed for building support/above maps

        h = 0

        # Iterate through blocks that are part of the goal configuration
        for block in self.goal_blocks:
            desired_support = self.goal_support.get(block)

            # Only consider blocks that are part of the goal stack structure (have a desired support)
            if desired_support is not None:
                current_sup = current_support.get(block)

                # Determine the block's current support (on, on-table, or held)
                if is_holding == block:
                    current_sup = 'arm'
                elif current_sup is None:
                    # If not held and not found in 'on' or 'on-table' facts,
                    # assume it's on the table. This handles blocks that might be
                    # in the initial state but not explicitly on/on-table in the state representation
                    # (though PDDL states usually list these).
                    current_sup = 'table'


                # Condition 1: Block is in the wrong place (support is different)
                if current_sup != desired_support:
                    h += 1
                else: # Block is in the right place (support is the same)
                    # Condition 2: Block is in the right place but wrongly covered
                    # This condition only applies if the block is *not* held.
                    if is_holding != block:
                        current_block_above = current_above.get(block)
                        desired_block_above = self.goal_above.get(block) # None if 'block' should be clear in goal

                        # Check if the block currently on top is different from the desired one
                        if current_block_above != desired_block_above:
                             h += 1

        return h
