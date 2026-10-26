# Need to ensure the Heuristic base class is available.
# Assuming it's in a module named 'heuristics.heuristic_base'
# If running standalone, a mock class is needed.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Mock Heuristic class for testing purposes if the actual module is not available
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static

        def __call__(self, node):
            raise NotImplementedError

from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not in their
    correct final position relative to the table, considering the required
    stack structure defined by the goal state. A block is considered
    correctly placed relative to the table if it is on the correct block
    according to the goal, and that block is also correctly placed relative
    to the table, recursively, until the base block is correctly on the table.

    # Assumptions
    - The goal state defines a specific configuration of blocks in stacks
      or on the table using `(on ?x ?y)` and `(on-table ?x)` predicates.
    - All blocks involved in goal `on` or `on-table` predicates must reach
      their specified position relative to the table.
    - The heuristic counts how many such blocks are *not* in their correct
      position relative to the table.
    - The heuristic ignores `(clear ?x)` and `(arm-empty)` goals.

    # Heuristic Initialization
    - The heuristic parses the goal conditions to build a map (`goal_below`)
      that specifies, for each block that is part of a goal stack or goal
      table position, which block (or the table) should be directly below it
      in the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to determine the immediate support for each block.
       Create a map (`current_below`) where `current_below[block]` is the block
       directly below it, 'table' if it's on the table, or 'holding' if it's
       in the arm.
    2. Initialize a memoization dictionary (`memo`) to store the result of
       `is_correctly_placed` checks for blocks to avoid redundant computations
       in the recursive calls.
    3. Define a recursive helper function `is_correctly_placed(block, current_below, goal_below, memo)`:
       - If the result for `block` is already in `memo`, return it.
       - If `block` is 'table', it is the base and is always correctly placed relative to itself. Store `True` in `memo` and return `True`.
       - If the block is not expected to be in a specific position relative to the table
         according to the goal (`block not in goal_below`), it cannot be the correct
         base for a block that *is* in the goal structure. Therefore, it is not
         correctly placed *as a base for a goal block*. Store `False` in `memo` and return `False`.
       - Get the block that should be below `block` in the goal (`goal_pos = goal_below[block]`).
       - Get the block that is currently below `block` (`current_pos = current_below.get(block)`).
         If `block` is not found in `current_below` (e.g., floating block), it's not correctly placed. Store `False` in `memo` and return `False`.
       - If `current_pos == 'holding'`, it's not correctly placed relative to the table. Store `False` in `memo` and return `False`.
       - Check base case: block should be on the table
       - If `goal_pos == 'table'`:
         - The block is correctly placed relative to the table if and only if it is currently on the table (`current_pos == 'table'`). Store the result in `memo` and return it.
       - Check recursive case: block should be on another block
       - If `goal_pos` is a block `y`:
         - The block is correctly placed relative to the table if and only if it is currently on block `y` (`current_pos == y`) AND block `y` is also correctly placed relative to the table (`is_correctly_placed(y, current_below, goal_below, memo)`). Store the result in `memo` and return it.
    4. Initialize the heuristic value `h = 0`.
    5. Iterate through each block `b` that is present in the `goal_below` map (i.e., each block that has a defined goal position in a stack or on the table).
    6. For each such block `b`, call `is_correctly_placed(b, current_below, self.goal_below, memo)`.
    7. If the function returns `False` (meaning the block is not correctly placed relative to the table), increment the heuristic value `h`.
    8. Return the final heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the goal stack structure.
        """
        # The set of facts that must hold in goal states.
        self.goals = task.goals
        # Static facts are not needed for this heuristic in blocksworld.
        self.static = task.static # Keep it as per base class

        # Build the goal stack structure: block -> block_below_in_goal (or 'table')
        self.goal_below = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: # Skip malformed facts
                continue
            predicate = parts[0]
            if predicate == "on":
                if len(parts) == 3:
                    block, block_below = parts[1], parts[2]
                    self.goal_below[block] = block_below
            elif predicate == "on-table":
                 if len(parts) == 2:
                    block = parts[1]
                    self.goal_below[block] = 'table'
            # Ignore 'clear' and 'arm-empty' goals for this heuristic

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        Estimates the number of blocks not correctly placed relative to the table.
        """
        state = node.state

        # Build the current stack structure: block -> block_below_currently (or 'table' or 'holding')
        current_below = {}
        # We don't strictly need all_blocks_in_state for this heuristic logic,
        # as we iterate over blocks in goal_below.
        # all_blocks_in_state = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: # Skip malformed facts
                continue
            predicate = parts[0]
            if predicate == "on":
                if len(parts) == 3:
                    block, block_below = parts[1], parts[2]
                    current_below[block] = block_below
                    # all_blocks_in_state.add(block)
                    # all_blocks_in_state.add(block_below)
            elif predicate == "on-table":
                if len(parts) == 2:
                    block = parts[1]
                    current_below[block] = 'table'
                    # all_blocks_in_state.add(block)
            elif predicate == "holding":
                 if len(parts) == 2:
                    block = parts[1]
                    current_below[block] = 'holding'
                    # all_blocks_in_state.add(block)
            # We don't need 'clear' or 'arm-empty' for the stack structure

        # Memoization dictionary for is_correctly_placed
        memo = {}

        def is_correctly_placed(block):
            """
            Recursive helper to check if a block is in its correct goal position
            relative to the table in the current state.
            """
            # Check memoization cache
            if block in memo:
                return memo[block]

            # Base case: 'table' is always correctly placed relative to itself
            if block == 'table':
                 memo[block] = True
                 return True

            # If the block is not in the goal structure, it cannot be the correct
            # base for a block that *is* in the goal structure. Therefore, it is not
            # correctly placed *as a base for a goal block*.
            if block not in self.goal_below:
                 memo[block] = False
                 return False

            goal_pos = self.goal_below[block]
            current_pos = current_below.get(block) # Use .get() in case block is not in current_below (e.g., floating)

            # If block is not found in current_below (e.g., floating block not on table/other block/held), it's not correctly placed.
            if current_pos is None:
                 memo[block] = False
                 return False

            # If block is being held, it's not correctly placed relative to the table.
            if current_pos == 'holding':
                memo[block] = False
                return False

            # Check base case: block should be on the table
            if goal_pos == 'table':
                result = (current_pos == 'table')
                memo[block] = result
                return result

            # Check recursive case: block should be on another block
            # It's correctly placed if it's on the correct block AND that block is correctly placed
            # Ensure current_pos is a block (not 'table' when goal_pos is a block)
            if current_pos != 'table':
                 result = (current_pos == goal_pos) and is_correctly_placed(current_pos)
            else: # current_pos is 'table' but goal_pos is a block
                 result = False # Cannot be correctly placed if it's on the table but should be on a block

            memo[block] = result
            return result

        # Calculate heuristic: count blocks in goal_below that are not correctly placed
        h = 0
        # Iterate only over blocks that are part of the goal stack definition
        for block in self.goal_below:
            # Call the recursive check. The check handles cases where the block
            # might not be in current_below or is being held.
            if not is_correctly_placed(block):
                h += 1

        return h
