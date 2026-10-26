from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace and inner structure if needed,
    # but for simple blocksworld facts, split() is sufficient after stripping parens.
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by counting blocks that are not in their correct goal position within their stack
    or are blocking other blocks from reaching their correct position, plus blocks
    that are blocking blocks required to be clear. Each such block contributes 2
    to the heuristic value, representing the estimated cost of moving it out of the way.

    # Assumptions
    - The goal state specifies the desired configuration of some blocks,
      including their position (on another block or on the table) and which
      blocks should be clear.
    - Blocks not mentioned in the goal can be anywhere.
    - The cost of moving a block out of the way (unstack/pickup + putdown) is estimated as 2 actions.
    - The heuristic counts each block that *needs* to be moved because it is
      misplaced within a goal stack, or because it is on top of a block that is
      misplaced or needs to be clear.
    - The heuristic value is 0 if and only if the state is the goal state.

    # Heuristic Initialization
    - Parses the goal state to determine the desired position for each block
      involved in an `(on X Y)` or `(on-table X)` goal predicate.
    - Stores these as `goal_on` (mapping block to block below) and `goal_on_table` (set of blocks that should be on the table).
    - Stores the set of blocks that must be clear in the goal state (`goal_clear`).
    - Identifies all blocks that are relevant to the goal (appear in any goal predicate).

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is calculated based on the current state as follows:

    1.  **Parse Current State:** Determine the current position of each block
        (`current_on`, `current_on_table`, `current_holding`), which blocks are clear
        (`current_clear`), and which block is directly on top of another
        (`current_on_top`). Identify if the arm is empty.

    2.  **Identify Correctly Placed Blocks within Goal Stacks:** For each block `B`
        that is part of a goal stack definition (i.e., appears as a key in `goal_on`
        or in `goal_on_table`), recursively determine if it is in its correct goal
        position *and* everything below it in the goal stack is also in its correct
        position. This check is memoized. A block `B` is correctly placed within
        its goal stack if:
        - Its goal is `(on-table B)` and `(on-table B)` is true in the current state.
        - Its goal is `(on B A)` and `(on B A)` is true in the current state, AND block `A` is correctly placed within its goal stack.
        - Blocks not part of a goal stack definition are considered vacuously correctly placed *for this check*.

    3.  **Count Misplaced Blocks and Blocks Blocking Them:** Initialize heuristic `h = 0`.
        Use a set `counted_blocks` to track blocks whose movement cost has already been added.
        Iterate through all blocks `B` that are part of a goal stack definition (keys in `goal_on` or in `goal_on_table`):
        - If `B` is *not* correctly placed within its goal stack (using the recursive check):
            - If `B` has not been counted, add 2 to `h` and add `B` to `counted_blocks`.
            - Trace upwards from `B` in the *current* stack using `current_on_top`. For each block `X` on top of `B`:
                - If `X` has not been counted, add 2 to `h` and add `X` to `counted_blocks`.

    4.  **Count Blocks Blocking Clear Goals:** Iterate through all blocks `W` in `goal_clear`:
        - If `W` is *not* clear in the current state:
            - Trace upwards from `W` in the *current* stack using `current_on_top`. For each block `X` on top of `W`:
                - If `X` has not been counted, add 2 to `h` and add `X` to `counted_blocks`.

    5.  **Handle Held Block:** If `(arm-empty)` is an explicit goal and the arm is holding a block `H`:
        - If `H` has not been counted, add 2 to `h` and add `H` to `counted_blocks`.

    6.  **Return Heuristic Value:** The final value of `h` is the heuristic estimate.
        A value of 0 is returned if and only if the state is the goal state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals  # Goal conditions.
        # Blocksworld has no static facts relevant to the heuristic calculation.
        # task.static is frozenset()

        # Parse goal state to build goal configuration
        self.goal_on = {} # Maps block -> block_below
        self.goal_on_table = set() # Set of blocks that should be on the table
        self.goal_clear = set() # Set of blocks that should be clear
        # self.goal_blocks is not strictly needed with the current loop structure

        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == "on":
                block, block_below = parts[1], parts[2]
                self.goal_on[block] = block_below
            elif predicate == "on-table":
                block = parts[1]
                self.goal_on_table.add(block)
            elif predicate == "clear":
                block = parts[1]
                self.goal_clear.add(block)
            # (arm-empty) is handled separately and doesn't involve a block parameter.

    def _is_correctly_placed_recursive(self, block, current_on, current_on_table, goal_on, goal_on_table, memo):
        """
        Recursively checks if a block is in its correct goal position within its stack
        and if everything below it in the goal stack is also correctly placed.
        """
        if block in memo:
            return memo[block]

        goal_below = goal_on.get(block)

        if goal_below is None: # Block is not a key in goal_on. Check if it's in goal_on_table.
            if block in goal_on_table:
                # Goal is (on-table block)
                result = (block in current_on_table)
            else:
                # This block is not defined as being on the table or on another block
                # in the goal state. It's not part of a defined goal stack base.
                # It cannot be "incorrectly placed" *within* a goal stack definition.
                result = True # Vacuously true for blocks not part of goal stack definitions

        else: # Goal is (on block goal_below)
            result = (current_on.get(block) == goal_below) and self._is_correctly_placed_recursive(goal_below, current_on, current_on_table, goal_on, goal_on_table, memo)

        memo[block] = result
        return result


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Parse Current State
        current_on = {} # Maps block -> block_below
        current_on_table = set() # Set of blocks on the table
        current_holding = None # The block being held, or None
        current_clear = set() # Set of clear blocks
        current_on_top = {} # Maps block_below -> block_on_top (inverse of current_on)
        arm_empty = False

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "on":
                block, block_below = parts[1], parts[2]
                current_on[block] = block_below
                current_on_top[block_below] = block
            elif predicate == "on-table":
                block = parts[1]
                current_on_table.add(block)
            elif predicate == "holding":
                block = parts[1]
                current_holding = block
            elif predicate == "clear":
                block = parts[1]
                current_clear.add(block)
            elif predicate == "arm-empty":
                arm_empty = True

        # 2. Identify Correctly Placed Blocks within Goal Stacks (Recursive with Memoization)
        memo_correctly_placed = {}

        # 3. Count Misplaced Blocks and Blocks Blocking Them (within goal stacks)
        h = 0
        counted_blocks = set() # Blocks whose movement cost has been added

        # Iterate through all blocks that are part of a goal stack definition (keys in goal_on or in goal_on_table)
        blocks_in_goal_config = set(self.goal_on.keys()).union(self.goal_on_table)

        for block in blocks_in_goal_config:
            if not self._is_correctly_placed_recursive(block, current_on, current_on_table, self.goal_on, self.goal_on_table, memo_correctly_placed):
                # This block is not in its correct place within the goal stack
                if block not in counted_blocks:
                    h += 2 # Cost to move this block
                    counted_blocks.add(block)

                # Count blocks currently on top of this misplaced block
                current = block
                while current_on_top.get(current) is not None:
                    block_on_top = current_on_top[current]
                    if block_on_top not in counted_blocks:
                        h += 2 # Cost to move the block on top out of the way
                        counted_blocks.add(block_on_top)
                    current = block_on_top # Move up the current stack

        # 4. Count Blocks Blocking Clear Goals
        # Iterate through all blocks that must be clear in the goal.
        for block_to_clear in self.goal_clear:
            if block_to_clear not in current_clear:
                # This block needs to be clear but isn't
                current = block_to_clear
                while current_on_top.get(current) is not None:
                    block_on_top = current_on_top[current]
                    if block_on_top not in counted_blocks:
                        h += 2 # Cost to move the block on top out of the way
                        counted_blocks.add(block_on_top)
                    current = block_on_top # Move up the current stack

        # 5. Handle Held Block
        # If arm-empty is a goal and arm is not empty, the held block must be moved.
        arm_empty_goal = "(arm-empty)" in self.goals # Check if arm-empty is an explicit goal
        if current_holding is not None and arm_empty_goal:
             if current_holding not in counted_blocks:
                 h += 2 # Cost to move the held block
                 counted_blocks.add(current_holding)

        # The heuristic should be 0 if and only if the state is the goal state.
        # With positive-only goals typical in Blocksworld, h=0 implies all goal literals are true.
        # If h is 0, it means:
        # - All blocks in goal_on/goal_on_table are correctly placed in their stacks.
        # - All blocks in goal_clear are clear.
        # - If (arm-empty) is a goal, arm is empty.
        # This set of conditions matches the goal state for typical Blocksworld problems.
        # We return the calculated h, which will be 0 if and only if the state is the goal.

        return h
