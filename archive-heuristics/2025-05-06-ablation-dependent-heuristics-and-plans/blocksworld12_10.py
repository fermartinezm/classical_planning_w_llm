from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact string."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not in their
    correct position relative to the block they should be stacked on (or the table),
    considering the goal stack structure. A block is considered "not correctly
    stacked" if it is not on its designated goal support (block or table), or
    if its designated goal support is itself not correctly stacked. Blocks held
    by the arm are also considered not correctly stacked. This heuristic focuses
    on achieving the correct relative positions of blocks as defined by the
    `(on ?x ?y)` and `(on-table ?x)` goal predicates.

    # Assumptions
    - The goal defines a specific stack configuration for all blocks relevant to the goal.
    - The goal predicates involving blocks are primarily `(on ?x ?y)` and `(on-table ?x)`.
    - All blocks mentioned in the goal configuration are present in the problem.
    - Standard Blocksworld problems ensure a consistent and achievable goal structure.

    # Heuristic Initialization
    - Parses the goal predicates (`on` and `on-table`) to build the target stack
      configuration (`goal_config`), mapping each block to the block it should be
      on, or the special string 'table'.
    - Identifies the set of all blocks that are part of this goal configuration.
      These are the blocks whose positions matter for this heuristic.

    # Step-By-Step Thinking for Computing Heuristic
    1. For a given state, determine the current position of each block:
       - On another block (`on ?x ?y`).
       - On the table (`on-table ?x`).
       - Held by the arm (`holding ?x`).
       Store this in `current_config`. This is done for all blocks present in the state facts that define position.
    2. Initialize a memoization dictionary (`memo`) within the `__call__` method
       to store the result of the `is_correctly_stacked` check for each block.
       This prevents redundant computation in the recursive calls.
    3. Define a recursive helper function `is_correctly_stacked(block)`:
       - If the status for `block` is already in the `memo`, return the stored value.
       - Determine the block's current support (`current_under`) from `current_config`.
       - If the block is not found in `current_config` (which shouldn't happen in a valid state where every block is located somewhere), treat it as not correctly stacked. Store False in `memo` and return False.
       - If the block is currently held by the arm (`current_under == 'arm'`), it is
         not in its final stacked position, so it's not correctly stacked. Store False
         in `memo` and return False.
       - Retrieve the block's goal support (`goal_under`) from `goal_config`. This block
         is guaranteed to be in `goal_config` because we iterate over `self.all_blocks_in_goal_config`.
       - If `goal_under` is the special string 'table': The block is correctly stacked
         if and only if its `current_under` is also 'table'. Store the result in `memo`
         and return it.
       - If `goal_under` is a block (i.e., not 'table'): The block is correctly stacked
         if and only if its `current_under` is the same block (`goal_under`) AND the
         block `goal_under` is itself correctly stacked (recursive call: `is_correctly_stacked(goal_under)`).
         Store the result in `memo` and return it.
    4. Initialize a counter `not_correctly_stacked_count` to 0.
    5. Iterate through each `block` in the set `self.all_blocks_in_goal_config`
       (the blocks whose goal positions are specified).
    6. For each `block`, call `is_correctly_stacked(block)`. If it returns False,
       increment `not_correctly_stacked_count`.
    7. The heuristic value is the final `not_correctly_stacked_count`.

    This heuristic is 0 if and only if all blocks relevant to the goal configuration
    are in their correct positions relative to their goal support, forming the
    desired stacks, and none of these blocks are in the arm. For standard Blocksworld
    problems, this state is equivalent to the goal state (assuming `clear` goals
    are for stack tops and `arm-empty` is a goal).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration and blocks.
        """
        self.goals = task.goals

        # Build goal_config: map block -> goal_under_block or 'table'
        self.goal_config = {}
        # Identify all blocks that are part of the goal configuration
        self.all_blocks_in_goal_config = set()

        # Process goals to build goal_config and identify relevant blocks
        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == "on":
                block, under_block = parts[1], parts[2]
                self.goal_config[block] = under_block
                self.all_blocks_in_goal_config.add(block)
                self.all_blocks_in_goal_config.add(under_block)
            elif predicate == "on-table":
                block = parts[1]
                self.goal_config[block] = 'table'
                self.all_blocks_in_goal_config.add(block)
            # 'clear' and 'arm-empty' goals are not used for the structural config

        # Now self.all_blocks_in_goal_config contains exactly the blocks whose
        # goal positions are specified by on/on-table predicates.
        # The heuristic will count how many of *these* blocks are not correctly stacked.

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Build current_config: map block -> current_under_block or 'table' or 'arm'
        current_config = {}
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "on":
                block, under_block = parts[1], parts[2]
                current_config[block] = under_block
            elif predicate == "on-table":
                block = parts[1]
                current_config[block] = 'table'
            elif predicate == "holding":
                block = parts[1]
                current_config[block] = 'arm' # Special value for arm
            # 'clear' and 'arm-empty' are not needed for current_config

        # Memoization dictionary for is_correctly_stacked
        memo = {}

        def is_correctly_stacked(block):
            """
            Recursive helper to check if a block is in its correct goal stack position.
            Only called for blocks within self.all_blocks_in_goal_config.
            """
            if block in memo:
                return memo[block]

            current_under = current_config.get(block)

            # If block is not in current_config, it's not on anything or table. Invalid state.
            # Treat as not correctly stacked. This case indicates a problem with the state representation.
            if current_under is None:
                 memo[block] = False
                 return False

            # Case 1: Block is in the arm. It's not correctly stacked.
            if current_under == 'arm':
                 memo[block] = False
                 return False

            # goal_under must exist for blocks in self.all_blocks_in_goal_config
            goal_under = self.goal_config[block]

            # Case 2: Block has a goal position on the table.
            if goal_under == 'table':
                result = (current_under == 'table')
                memo[block] = result
                return result

            # Case 3: Block has a goal position on another block (goal_under is a block).
            # It's correctly stacked if it's currently on the goal_under AND goal_under is correctly stacked.
            result = (current_under == goal_under) and is_correctly_stacked(goal_under)
            memo[block] = result
            return result

        # Count blocks that are not correctly stacked according to the goal config
        not_correctly_stacked_count = 0
        for block in self.all_blocks_in_goal_config:
            if not is_correctly_stacked(block):
                not_correctly_stacked_count += 1

        return not_correctly_stacked_count
