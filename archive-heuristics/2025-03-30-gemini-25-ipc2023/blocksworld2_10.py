import re
from heuristics.heuristic_base import Heuristic
# Note: fnmatch is not used in this heuristic.

def get_parts(fact_str: str):
    """
    Parses a PDDL fact string into its predicate and arguments.
    Example: "(on b1 b2)" -> ["on", "b1", "b2"]
             "(clear b1)" -> ["clear", "b1"]
             "(arm-empty)" -> ["arm-empty"]
    Handles potential missing arguments for nullary predicates.
    Raises ValueError for invalid formats like empty strings, "()", or "( )".
    """
    fact_str = fact_str.strip()
    if not fact_str.startswith("(") or not fact_str.endswith(")"):
        raise ValueError(f"Invalid fact format: Missing parentheses in '{fact_str}'")
    
    content = fact_str[1:-1].strip()
    if not content:
         raise ValueError(f"Fact with empty content: '{fact_str}'")
        
    parts = content.split()
    # After split, parts should contain at least the predicate
    if not parts:
         # This case should not be reachable if content is not empty
         raise ValueError(f"Internal parsing error resulted in empty parts for: '{fact_str}'")
         
    return parts


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state.
    It first identifies all goal blocks that are "incorrectly placed" based on
    their position, the correctness of the block supporting them (recursively),
    and whether their 'clear' status matches the goal. Then, it identifies all
    other blocks that are currently stacked above these incorrect blocks. The
    heuristic value is primarily twice the total number of blocks that need to
    move (incorrect blocks + blocking blocks), adjusted for the block currently
    held by the arm.

    # Assumptions
    - Goals primarily consist of `on(block, other_block)`, `on-table(block)`,
      and `clear(block)` facts. `arm-empty` goal is implicitly handled by the
      held block adjustment.
    - Blocks not mentioned in the goal configuration are assumed to not interfere
      unless they are physically blocking a goal block that needs to move.
    - Each block identified as needing to move contributes 2 actions to the
      heuristic value (representing an unstack/pickup and a stack/putdown).
    - If the arm holds a block `b`:
        - If `b` needs to move anyway (is in the `must_move` set), the cost is reduced by 1 (pickup/unstack saved).
        - If `b` does not need to move based on the analysis (i.e., its goal
          position is correct once placed), the cost is increased by 1 (for
          the required stack/putdown action).

    # Heuristic Initialization
    - Parses the task's goal conditions (`task.goals`) to build:
        - `goal_config`: Map block -> block_below | 'table' (from `on` and `on-table` goals).
        - `goal_clear`: Set of blocks that must be clear in the goal.
    - Stores `goal_blocks`: a set of all unique blocks involved in any goal condition.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Current State:** Extract the current configuration from `node.state`:
        - `current_config`: Map block -> block below | 'table'.
        - `current_above`: Map block -> block directly above it.
        - `current_clear`: Set of blocks currently clear.
        - `held_block`: The block currently held by the arm, if any.
    2.  **Define Correctness Check (`is_correct`):** Create a recursive function
        with memoization (`memo`) to determine if a block is correctly placed
        relative to the goal state. A block `b` is correct if:
        a) If `b` has a position goal (`on(b, g)` or `on-table(b)`), this goal
           matches its current position (`current_config`).
        b) If its goal is `on(b, g)`, the block `g` below it must also be correct
           (recursive call `is_correct(g)`).
        c) If `clear(b)` is a goal, then `b` must be currently clear (`current_clear`).
        Blocks not mentioned in goals are handled appropriately (e.g., table is base case).
    3.  **Identify Incorrect Root Blocks:** Iterate through all blocks (`b`) mentioned
        in the goals (`goal_blocks`). Use the `is_correct(b)` function to find all
        blocks that are *not* correctly placed according to the recursive definition.
        Store these in a set `incorrect_roots`.
    4.  **Identify All Blocks to Move:** Start with `must_move = incorrect_roots`.
        Perform a search upwards from the blocks in `incorrect_roots`. If block `x`
        is in `must_move`, and block `y` is currently `on(y, x)`, then `y` must
        also eventually move to allow `x` to be corrected. Add `y` to `must_move`
        and continue searching upwards from `y`. Repeat until no more blocking
        blocks are found.
    5.  **Calculate Base Cost:** The total heuristic estimate is initially
        `h = len(must_move) * 2`.
    6.  **Adjust for Held Block:**
        - If the arm is holding `held_block`:
            - If `held_block` is in the `must_move` set, decrement `h` by 1
              (since the pickup/unstack action is saved).
            - If `held_block` is *not* in the `must_move` set, increment `h` by 1
              (since a stack/putdown action is still required).
    7.  **Final Value:** Return the calculated value `h` (ensuring it's non-negative).
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing goal conditions.
        """
        self.goals = task.goals
        self.static = task.static # Usually empty for Blocksworld

        self.goal_config = {} # block -> block_below | 'table'
        self.goal_clear = set()
        self.goal_blocks = set() # All blocks mentioned in on/on-table/clear goals

        for fact in self.goals:
            try:
                parts = get_parts(fact)
                if not parts: continue # Skip if fact is invalid/empty

                predicate = parts[0]
                args = parts[1:]

                if predicate == "on" and len(args) == 2:
                    block, below = args
                    self.goal_config[block] = below
                    self.goal_blocks.add(block)
                    self.goal_blocks.add(below)
                elif predicate == "on-table" and len(args) == 1:
                    block = args[0]
                    self.goal_config[block] = 'table'
                    self.goal_blocks.add(block)
                elif predicate == "clear" and len(args) == 1:
                    block = args[0]
                    self.goal_clear.add(block)
                    self.goal_blocks.add(block)
                # Ignore 'arm-empty' goal for heuristic calculation logic
            except ValueError as e:
                # Log warning for malformed goal facts if necessary
                # print(f"Warning: Skipping invalid goal fact '{fact}': {e}")
                pass


    def __call__(self, node):
        """
        Computes the heuristic value for the given state node.
        """
        state = node.state

        # 1. Parse Current State
        current_config = {} # block -> block_below | 'table'
        current_above = {} # block -> block_above
        current_clear = set()
        held_block = None

        for fact in state:
            try:
                parts = get_parts(fact)
                if not parts: continue

                predicate = parts[0]
                args = parts[1:]

                if predicate == "on" and len(args) == 2:
                    block, below = args
                    current_config[block] = below
                    current_above[below] = block
                elif predicate == "on-table" and len(args) == 1:
                    block = args[0]
                    current_config[block] = 'table'
                elif predicate == "holding" and len(args) == 1:
                    held_block = args[0]
                elif predicate == "clear" and len(args) == 1:
                    current_clear.add(args[0])
                # Ignore arm-empty state predicate
            except ValueError as e:
                 # Log warning for malformed state facts if necessary
                 # print(f"Warning: Skipping invalid state fact '{fact}': {e}")
                 pass

        # Memoization for the recursive correctness check
        memo = {}

        # 2. Define Correctness Check (`is_correct`)
        def is_correct(block):
            # Handle non-block inputs, like 'table' which is the base case
            if not isinstance(block, str) or block == 'table':
                 return True # Table is always considered a correct base

            # Return memoized result if available
            if block in memo:
                return memo[block]

            correct_so_far = True

            # Check a) Position goal match (if applicable)
            if block in self.goal_config:
                goal_pos = self.goal_config[block]
                current_pos = current_config.get(block) # None if held

                if current_pos != goal_pos:
                    correct_so_far = False

                # Check b) Block below is correct (recursive)
                # Only check if position matched so far and goal is not on-table
                if correct_so_far and goal_pos != 'table':
                    if not is_correct(goal_pos): # Recursive call
                        correct_so_far = False
            # else: block has no position goal, its correctness depends only on clear goal (below)

            # Check c) Clear status matches goal (if applicable)
            # This check is independent of position correctness *up to this point*,
            # but the block is only truly correct if all conditions hold.
            if correct_so_far and block in self.goal_clear:
                if block not in current_clear:
                    correct_so_far = False

            memo[block] = correct_so_far
            return correct_so_far

        # 3. Identify Incorrect Root Blocks
        incorrect_roots = set()
        # Evaluate correctness for all blocks mentioned in any goal condition
        for block in self.goal_blocks:
            if not is_correct(block):
                incorrect_roots.add(block)

        # 4. Identify All Blocks to Move (Incorrect roots + blocks physically above them)
        must_move = set(incorrect_roots)
        blocks_to_check = list(incorrect_roots)
        # Use checked_for_above to prevent redundant checks
        checked_for_above = set()

        while blocks_to_check:
            b = blocks_to_check.pop() # Use as stack for DFS-like upward traversal

            if b in checked_for_above:
                continue
            checked_for_above.add(b)

            block_on_top = current_above.get(b)
            if block_on_top is not None:
                # If the block on top isn't already marked to move, mark it
                # and add it to the list to check what's above it.
                if block_on_top not in must_move:
                    must_move.add(block_on_top)
                    # Add to stack only if it hasn't been processed for upward check yet
                    if block_on_top not in checked_for_above:
                         blocks_to_check.append(block_on_top)

        # 5. Calculate Base Cost
        heuristic_value = len(must_move) * 2

        # 6. Adjust for Held Block
        if held_block is not None:
            if held_block in must_move:
                # Block needs moving anyway; holding saves the pickup/unstack action (cost 1).
                # Since it contributed 2 to h, subtract 1.
                heuristic_value -= 1
            else:
                # Block doesn't need moving based on goal analysis (e.g., its final
                # position is clear and supported correctly), but it must be placed
                # somewhere using one action (stack/putdown).
                heuristic_value += 1

        # Ensure heuristic is non-negative (can happen if h=0 and held_block is adjusted)
        heuristic_value = max(0, heuristic_value)

        return heuristic_value
