from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected fact format, maybe log a warning or raise error
        # print(f"Warning: Invalid fact format encountered: {fact}")
        return [] # Return empty list for invalid format
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the difficulty of reaching the goal state by counting:
    1. The number of blocks that are part of a goal stack but are not in their
       correct position relative to the block (or table) they should be on,
       considering the goal stack structure recursively.
    2. The number of blocks that are required to be clear in the goal state
       but currently have a block on top of them.
    3. A penalty if the arm is currently holding a block.

    # Assumptions
    - The goal state defines specific stacks of blocks using `on` and `on-table` predicates.
    - `clear` goals typically apply to the top block of a goal stack.
    - The goal predicates (`on`, `on-table`) define complete goal stacks down to the table for all blocks involved in those stacks.

    # Heuristic Initialization
    - Parses the goal predicates (`on`, `on-table`, `clear`) to build:
      - A map of the required support for each block (`block -> block_below` or `block -> 'table'`). This map includes all blocks that are the first argument of an `on` goal or the argument of an `on-table` goal.
      - A set of blocks that must be clear in the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state predicates (`on`, `on-table`, `holding`) to build:
       - A map of the current support for each block (`block -> block_below`, `block -> 'table', or `block -> 'holding'`).
       - A map of which blocks are currently on top of other blocks (`block_below -> set of blocks_above`). This is used to check the 'blocked clear' condition.
       - Identify if the arm is holding a block.
    2. Calculate the 'incorrectly positioned' count:
       - Use a recursive helper function with memoization (`_is_correct`) to determine if a block is
         in its correct goal position *and* the block below it (in the goal stack)
         is also correctly positioned, all the way down to the table. The recursion
         follows the required support chain defined by `self.goal_support`.
       - Iterate through all blocks that are defined as being 'on' something or 'on-table'
         in the goal state (i.e., the keys in the `self.goal_support` map).
       - Count how many of these blocks are *not* recursively correctly positioned according
         to the `_is_correct` function.
    3. Calculate the 'blocked clear' count:
       - Iterate through all blocks that are required to be clear in the goal state
         (from `self.goal_clear_blocks`).
       - For each such block, check if there is any block currently on top of it
         in the current state (using the `current_block_above` map).
       - Count how many blocks required to be clear are currently blocked.
    4. Calculate the 'arm busy' penalty:
       - Add 1 to the heuristic if the arm is currently holding a block.
    5. The total heuristic value is the sum of the 'incorrectly positioned' count,
       the 'blocked clear' count, and the 'arm busy' penalty.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal support relationships and
        blocks that need to be clear in the goal state.
        """
        super().__init__(task)

        # Map block to its required support in the goal state ('table' or another block)
        self.goal_support = {}
        # Set of blocks that must be clear in the goal state
        self.goal_clear_blocks = set()

        # Collect goal information
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: # Skip invalid facts
                continue
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block_above, block_below = parts[1], parts[2]
                self.goal_support[block_above] = block_below
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                self.goal_support[block] = 'table'
            elif predicate == 'clear' and len(parts) == 2:
                block = parts[1]
                self.goal_clear_blocks.add(block)
            # Ignore other goal predicates like arm-empty if they exist


    def _is_correct(self, block, current_support, goal_support, memo):
        """
        Recursive helper to check if a block is in its correct goal position
        relative to the stack below it, based on goal_support.
        Assumes 'block' is a key in goal_support or is 'table'.
        """
        # Base case: The table is always correctly positioned
        if block == 'table':
            return True

        # Check memoization table
        if block in memo:
            return memo[block]

        # Get the required support for this block from the goal
        # We assume block is a key in goal_support based on how this function is called
        required_support = goal_support[block]
        actual_support = current_support.get(block)

        # If actual_support is None, the block's position is unknown/invalid in state
        # (not on, on-table, or holding). Treat as incorrect.
        if actual_support is None:
            memo[block] = False
            return False

        # Check if the immediate support is correct
        if actual_support != required_support:
            memo[block] = False
            return False

        # If immediate support is correct, recursively check the support below
        # This recursive call will eventually hit 'table' or a block whose
        # correctness is already memoized.
        result = self._is_correct(required_support, current_support, goal_support, memo)
        memo[block] = result
        return result


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Build current support map and block above map
        current_support = {}
        current_block_above = {} # support -> set of blocks on top
        is_holding = None

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block_above, block_below = parts[1], parts[2]
                current_support[block_above] = block_below
                if block_below not in current_block_above:
                    current_block_above[block_below] = set()
                current_block_above[block_below].add(block_above)
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                current_support[block] = 'table'
                if 'table' not in current_block_above:
                    current_block_above['table'] = set()
                current_block_above['table'].add(block)
            elif predicate == 'holding' and len(parts) == 2:
                block = parts[1]
                current_support[block] = 'holding'
                is_holding = block # Track which block is held
            # Ignore other state predicates like arm-empty, clear

        # --- Calculate Incorrectly Positioned Count (Recursive Correctness) ---
        # Memoization table for _is_correct. Initialize with base case.
        memo = {'table': True}
        incorrectly_positioned_count = 0

        # Check correctness for all blocks that have a defined goal position ('on' or 'on-table' goal)
        # These are the blocks that are keys in self.goal_support.
        for block in self.goal_support.keys():
             if not self._is_correct(block, current_support, self.goal_support, memo):
                 incorrectly_positioned_count += 1

        # --- Calculate Blocked Clear Count ---
        blocked_clear_count = 0
        for block_to_be_clear in self.goal_clear_blocks:
            # Check if anything is on top of this block in the current state
            # A block is blocked if it is a key in current_block_above and the set is not empty.
            if block_to_be_clear in current_block_above and current_block_above[block_to_be_clear]:
                 blocked_clear_count += 1

        # --- Calculate Arm Busy Penalty ---
        # Add 1 if the arm is holding a block. This penalizes states where the arm
        # is occupied, as the block needs to be placed before other actions like
        # picking up another block can occur.
        arm_busy_penalty = 0
        if is_holding is not None:
             arm_busy_penalty = 1

        # --- Combine and Return ---
        # The total heuristic is the sum of structural mismatches, blocks that
        # should be clear but aren't, and a penalty for a busy arm.
        total_cost = incorrectly_positioned_count + blocked_clear_count + arm_busy_penalty

        # Ensure heuristic is 0 for goal states.
        # In a goal state:
        # - All blocks in goal_support keys will be recursively correct -> incorrectly_positioned_count = 0.
        # - All blocks in goal_clear_blocks will be clear (nothing on them) -> blocked_clear_count = 0.
        # - Arm will be empty (assuming arm-empty is part of goal or initial state and maintained) -> arm_busy_penalty = 0.
        # So, heuristic is 0 for goal states.

        return total_cost
