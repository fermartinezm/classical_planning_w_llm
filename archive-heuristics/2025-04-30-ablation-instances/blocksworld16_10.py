# Need to import Heuristic base class
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not in their correct
    final position relative to the block below them in the goal configuration.
    A block is considered correctly placed if it is on the correct block (or table)
    according to the goal, AND the block it is on is also correctly placed.

    # Assumptions
    - The goal specifies the desired position for a subset of blocks using
      `(on ?x ?y)` and `(on-table ?x)` predicates.
    - Blocks not mentioned in these goal predicates are not considered by this
      heuristic (they don't need a specific final position relative to another block).
    - The heuristic assumes a unique goal position for each block it considers
      (i.e., a block appears as the first argument in at most one `(on ?x ?y)`
      or one `(on-table ?x)` goal fact). This is standard for Blocksworld goals.

    # Heuristic Initialization
    - Extracts the goal configuration from `task.goals`, creating a dictionary
      (`goal_config`) mapping each block `B` with a specified goal position to
      its target (`Y` if goal is `(on B Y)`, or 'table' if goal is `(on-table B)`).

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the goal facts (`task.goals`) to create a dictionary (`goal_config`)
       mapping each block `B` with a specified goal position to its target
       (`Y` if goal is `(on B Y)`, or 'table' if goal is `(on-table B)`).
       Only `(on ?x ?y)` and `(on-table ?x)` goal facts are processed.
    2. For a given state, parse the current facts (`node.state`) to create a
       dictionary (`current_config`) mapping each block `B` to what it is
       currently on (`X` if state has `(on B X)`, 'table' if `(on-table B)`,
       or 'arm' if `(holding B)`).
    3. Define a recursive helper function `is_correctly_placed(block)` that checks
       if a block is in its correct goal position relative to the block below it:
       - Use memoization (`memo` dictionary) to store results and avoid redundant
         calculations for the same block during the recursive calls.
       - If the block is not found as a key in `goal_config`, it means its specific
         position is not defined by the goal facts processed. For this heuristic,
         such blocks are treated as "correctly placed" relative to the goal structure
         being tracked, and they do not contribute to the misplaced count.
       - If the block is currently held (`current_config.get(block) == 'arm'`), it's
         not in a stable final position, so it's not correctly placed.
       - Retrieve the block's goal target (`goal_config[block]`) and current target
         (`current_config.get(block)`). The `.get()` method handles cases where a
         block from `goal_config` might not be found in the `current_config` keys
         (e.g., due to an unexpected state representation), returning `None`.
       - If the `current_target` is `None` (block not found in `on`, `on-table`,
         or `holding` facts), it's not correctly placed.
       - If the `current_target` matches the `goal_target`:
         - If the `goal_target` is 'table', the block is correctly placed.
         - If the `goal_target` is another block `Y`, the block is correctly placed
           *only if* `Y` is also correctly placed (this is the recursive step:
           call `is_correctly_placed(Y)`).
       - If the `current_target` does not match the `goal_target`, the block is not
         correctly placed.
       - Store the result in `memo` before returning.
    4. Initialize a counter `misplaced_goal_blocks_count` to 0.
    5. Iterate through all blocks that are keys in the `goal_config` dictionary.
       These are the blocks whose positions are explicitly specified in the goal.
    6. For each such block, call the `is_correctly_placed(block)` helper function.
    7. If `is_correctly_placed(block)` returns `False`, increment the
       `misplaced_goal_blocks_count`.
    8. The final heuristic value is the total `misplaced_goal_blocks_count`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal configuration."""
        self.goals = task.goals
        # Static facts are empty for blocksworld, not needed.
        # static_facts = task.static

        # Build the goal configuration map: block -> target_block_or_table
        # This maps a block to the object (another block or 'table') it should be on.
        self.goal_config = {}
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            # Ensure the fact has enough parts to be an 'on' or 'on-table' predicate
            if len(parts) >= 2:
                predicate = parts[0]
                if predicate == 'on' and len(parts) == 3:
                    block, target = parts[1], parts[2]
                    self.goal_config[block] = target
                elif predicate == 'on-table' and len(parts) == 2:
                    block = parts[1]
                    self.goal_config[block] = 'table'
            # Ignore other goal predicates like (clear ?x) or (arm-empty) for this heuristic
            # and ignore malformed facts.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Build the current configuration map: block -> current_target_block_or_table_or_arm
        # This maps a block to the object (another block, 'table', or 'arm') it is currently on.
        current_config = {}
        # We could also track what's on top of each block or which blocks are clear,
        # but this specific heuristic only needs the 'below' relationship.

        for fact in state:
            parts = get_parts(fact)
            # Ensure the fact has enough parts to be a relevant predicate
            if len(parts) >= 2:
                predicate = parts[0]
                if predicate == 'on' and len(parts) == 3:
                    block, below = parts[1], parts[2]
                    current_config[block] = below
                elif predicate == 'on-table' and len(parts) == 2:
                    block = parts[1]
                    current_config[block] = 'table'
                elif predicate == 'holding' and len(parts) == 2:
                    block = parts[1]
                    current_config[block] = 'arm'
            # Ignore other state facts like (clear ?x) or (arm-empty) for current_config
            # and ignore malformed facts.

        # Memoization dictionary for the recursive correctness check
        memo = {}

        def is_correctly_placed(block):
            """
            Checks if a block is in its correct goal position relative to the
            block below it, according to the goal configuration.
            Uses memoization.
            """
            # Return memoized result if available
            if block in memo:
                return memo[block]

            # If block has no goal position specified in self.goal_config,
            # it doesn't contribute to the count of misplaced *goal* blocks.
            # Treat as correctly placed for the purpose of this specific count.
            if block not in self.goal_config:
                 memo[block] = True
                 return True

            goal_target = self.goal_config[block]
            current_target = current_config.get(block) # Use .get() to handle blocks not found in state facts

            # If the block is currently held, it's not in its final position.
            if current_target == 'arm':
                memo[block] = False
                return False

            # If the block is not found in current_config (e.g., not on anything, not on table, not held),
            # it's in an unexpected state or irrelevant. Treat as not correctly placed for safety.
            # This case might indicate an issue with state representation if a block from goal_config isn't found.
            if current_target is None:
                 memo[block] = False
                 return False

            # Check if the block is on the correct target
            if current_target == goal_target:
                # If the target is 'table', the block is correctly placed.
                if goal_target == 'table':
                    memo[block] = True
                    return True
                # If the target is another block, check if that block is correctly placed.
                else:
                    # Recursive call to check the block below it in the goal stack
                    # Ensure the target block is also in goal_config or treat it appropriately
                    # If goal_target is not in goal_config, it means the goal specifies
                    # block is on goal_target, but goal_target itself doesn't have a
                    # specified position. This is unusual but possible. We treat such
                    # a goal_target as "correctly placed" relative to the goal structure
                    # we are tracking. The recursive call handles this via the first check.
                    target_block_is_correct = is_correctly_placed(goal_target)
                    memo[block] = target_block_is_correct
                    return target_block_is_correct
            else:
                # The block is on the wrong target.
                memo[block] = False
                return False

        # Calculate the heuristic value: count blocks in goal_config that are not correctly placed
        misplaced_goal_blocks_count = 0
        # Iterate only through blocks that have a specified goal position
        for block in self.goal_config:
            if not is_correctly_placed(block):
                misplaced_goal_blocks_count += 1

        return misplaced_goal_blocks_count
