import sys
import os
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
# Assuming the Task class definition is available, e.g., from a file named representation.py
# from representation import Task

# Helper function to parse PDDL facts represented as strings
def get_parts(fact_str):
    """
    Removes the surrounding parentheses and splits the fact string by spaces.
    Example: "(on b1 b2)" -> ['on', 'b1', 'b2']
    """
    return fact_str[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state.
    It calculates the cost based on two main factors:
    1. Blocks that are not in their final goal position (considering the entire stack below them).
    2. Blocks that are correctly placed but are obstructed by incorrect blocks stacked on top of them.
    The heuristic value is roughly twice the number of blocks that need to be moved
    at least once to achieve the goal configuration.

    # Assumptions
    - The goal is specified as a conjunction of `(on block support)` and `(on-table block)` predicates,
      defining the target towers. `(clear block)` goals are implicitly handled by the tower structure.
    - All blocks relevant to the final configuration are mentioned in the goal predicates.
    - Standard Blocksworld actions (pickup, putdown, stack, unstack) with unit costs are used.
    - The heuristic does not need to be admissible (it's for Greedy Best-First Search).

    # Heuristic Initialization
    - The constructor parses the `task.goals` to build internal data structures:
        - `goal_on`: A dictionary mapping each block `b` to the block `s` it should be on (`on b s`).
        - `goal_on_table`: A set containing blocks that should be on the table in the goal.
    - It also identifies all unique blocks involved in the goal configuration (`goal_blocks`).

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Current State:** Extract the current configuration from `node.state`:
        - `current_on`: Dictionary mapping block `b` to its current support `s` (`on b s`).
        - `current_on_table`: Set of blocks currently on the table.
        - `current_holding`: The block currently held by the arm, or `None`.
        - Identify all blocks present in the current state.
    2.  **Define `check_foundation(block)`:**
        - This is a recursive helper function that determines if a given `block` is correctly positioned
          relative to its goal support, *and* recursively, whether that support (if it's another block)
          is also correctly positioned, all the way down to the table.
        - It uses memoization (`memo`) to store results for already checked blocks, avoiding redundant computations.
        - **Crucially:** A block currently being held by the arm (`current_holding`) is always considered
          *not* to have a correct foundation, as it must be placed somewhere.
        - **Base Cases:**
            - If `block` is 'table', its foundation is correct (True).
            - If `block`'s current support (table or another block) does not match its `goal_support`,
              its foundation is incorrect (False).
        - **Recursive Step:** If `block`'s current support matches its `goal_support` (and it's not the table),
          the result depends on `check_foundation(goal_support)`.
    3.  **Identify Misplaced Blocks:**
        - Iterate through all unique blocks found in the state and the goal.
        - Use `check_foundation` for each block to classify it as either `correctly_placed` or `incorrectly_placed`.
    4.  **Calculate Cost for Misplaced Blocks (`cost1`):**
        - Every block classified as `incorrectly_placed` must be moved at least once.
        - Moving a block typically requires one action to lift it (pickup/unstack) and one action
          to place it in its target location (putdown/stack).
        - Therefore, `cost1 = len(incorrectly_placed_blocks) * 2`.
    5.  **Calculate Cost for Clearing Correct Blocks (`cost2`):**
        - Iterate through the set of `correctly_placed_blocks`.
        - For each correctly placed block `b`:
            - Determine which block `x`, if any, is currently stacked directly on top of `b` (`current_above`).
            - Determine which block `y`, if any, *should* be stacked directly on top of `b` according to the goal (`goal_above`).
            - If `x` exists (`x` is not None) and `x` is different from `y` (i.e., the wrong block is on top of `b`),
              then block `x` is obstructing the goal configuration and must be moved off `b`.
            - Add 2 to `cost2` to account for the actions needed to move `x` off `b` (unstack `x`, then likely putdown `x` temporarily).
            - Note: The cost for eventually moving `x` to its *own* final destination is already accounted for in `cost1`
              (because if `x` is wrongly placed on `b`, `x` itself must be an `incorrectly_placed` block).
              `cost2` specifically represents the cost of *clearing* the correctly placed blocks below `x`.
    6.  **Total Heuristic Value:** The final heuristic estimate is `h = cost1 + cost2`.
        - This sum estimates the total number of move operations (pickup/unstack + putdown/stack pairs) needed.
        - **Goal State Check:** If the current state is the goal state, `check_foundation` will return True for all blocks.
          `incorrectly_placed_blocks` will be empty (`cost1 = 0`). For all correctly placed blocks, the block currently
          above will match the goal block above, so `cost2 = 0`. The total heuristic `h` will be 0, as required.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing the goal configuration.
        """
        self.goals = task.goals
        self.static = task.static # Blocksworld usually has no static facts defined this way

        self.goal_on = {} # block -> support_block
        self.goal_on_table = set() # blocks on table in goal

        # Parse goal facts to understand the target configuration
        for fact in self.goals:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                # Goal is (on block support)
                self.goal_on[parts[1]] = parts[2]
            elif predicate == 'on-table' and len(parts) == 2:
                # Goal is (on-table block)
                self.goal_on_table.add(parts[1])
            # 'clear' goals are generally consequences of the 'on'/'on-table' structure
            # and are not directly stored here.

        # Identify all unique blocks mentioned anywhere in the goal structure
        self.goal_blocks = set(self.goal_on.keys()) | set(self.goal_on.values()) | self.goal_on_table
        if 'table' in self.goal_blocks:
             self.goal_blocks.remove('table') # Ensure 'table' isn't treated as a block


    def get_current_support(self, block, current_on, current_on_table, current_holding):
        """
        Helper function to find what currently supports a given block (or if it's held).
        Returns the support block, 'table', 'holding', or None if not found.
        """
        if block == current_holding:
            return 'holding'
        if block in current_on:
            return current_on[block]
        if block in current_on_table:
            return 'table'
        # This case should ideally not happen in a well-formed state description
        # where all blocks have a defined location (on table, on block, or held).
        return None


    def check_foundation(self, block, current_on, current_on_table, current_holding, memo):
        """
        Recursively checks if a block and the stack below it match the goal configuration.
        Uses memoization to store results.
        Returns True if the block's foundation is correct relative to the goal, False otherwise.
        """
        # Base case: The table itself is always a correct foundation.
        if block == 'table':
            return True
        # Return memoized result if this block has already been checked.
        if block in memo:
            return memo[block]

        # Find what currently supports this block.
        current_support = self.get_current_support(block, current_on, current_on_table, current_holding)

        # A block being held by the arm is never in its final correct position.
        if current_support == 'holding':
            memo[block] = False
            return False
        # If block location is unknown (shouldn't happen), assume foundation is incorrect.
        if current_support is None:
             memo[block] = False
             return False

        # Determine what *should* support this block according to the goal.
        # If the block should be on the table: goal_support = 'table'.
        # If the block should be on another block 's': goal_support = 's'.
        # If the block is not mentioned in the goal 'on' or 'on-table' predicates: goal_support = None.
        goal_support = self.goal_on.get(block, 'table' if block in self.goal_on_table else None)

        # If the block's final position isn't specified in the goal, we assume its current
        # position doesn't violate the goal *unless* it obstructs something else.
        # The obstruction check happens later (cost2). For foundation check, assume it's okay
        # if not explicitly mentioned. This might need adjustment for domains with partial goals.
        if goal_support is None:
             memo[block] = True # Tentatively OK, might be revisited if it obstructs
             return True

        # Check if the block's current support matches its goal support.
        if current_support != goal_support:
            memo[block] = False # Mismatch, foundation is incorrect.
            return False

        # Current support matches the goal support. Now, recursively check the foundation
        # of the supporting object (unless it's the table).
        # The recursive call handles the base case where goal_support is 'table'.
        foundation_below_ok = self.check_foundation(goal_support, current_on, current_on_table, current_holding, memo)
        memo[block] = foundation_below_ok # Store result before returning.
        return foundation_below_ok


    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.
        """
        state = node.state
        h = 0 # Initialize heuristic value

        # --- 1. Parse Current State ---
        current_on = {} # block -> support block
        current_on_table = set()
        current_holding = None
        all_blocks_in_state = set() # Keep track of all blocks mentioned in the state

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block, support = parts[1], parts[2]
                current_on[block] = support
                all_blocks_in_state.add(block)
                all_blocks_in_state.add(support) # Support might be a block name
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                current_on_table.add(block)
                all_blocks_in_state.add(block)
            elif predicate == 'holding' and len(parts) == 2:
                current_holding = parts[1]
                all_blocks_in_state.add(current_holding)

        # Combine blocks from goal and state to get the full set of relevant blocks
        all_blocks = self.goal_blocks.union(all_blocks_in_state)
        if 'table' in all_blocks:
             all_blocks.remove('table') # 'table' is not a block object

        # --- 2. Classify Blocks using check_foundation ---
        memo = {} # Memoization dictionary for check_foundation results
        correctly_placed_blocks = set()
        incorrectly_placed_blocks = set()

        for block in all_blocks:
            if self.check_foundation(block, current_on, current_on_table, current_holding, memo):
                correctly_placed_blocks.add(block)
            else:
                incorrectly_placed_blocks.add(block)

        # --- 3. Calculate Cost 1: Misplaced Blocks ---
        # Each incorrectly placed block needs approximately 2 actions to move.
        cost1 = len(incorrectly_placed_blocks) * 2

        # --- 4. Calculate Cost 2: Clearing Correct Blocks ---
        cost2 = 0
        # Create a map of what block is directly on top of another block
        current_above = {support: block for block, support in current_on.items()}

        for block in correctly_placed_blocks:
            # Find what *is* currently on top of this correctly placed block
            block_currently_above = current_above.get(block) # Returns None if nothing is on top

            # Find what *should* be on top according to the goal
            block_goal_above = None
            for b_goal, support_goal in self.goal_on.items():
                if support_goal == block:
                    block_goal_above = b_goal
                    break # Found the block that should be on top

            # If the block currently on top is not the one that should be on top...
            if block_currently_above != block_goal_above:
                # ...and there *is* a block currently on top...
                if block_currently_above is not None:
                    # ...then this block (`block_currently_above`) is obstructing `block`.
                    # It needs to be moved off. Add 2 actions for this clearing move.
                    cost2 += 2

        # --- 5. Total Heuristic Value ---
        h = cost1 + cost2

        # Ensure heuristic is 0 if the state is a goal state.
        # This should hold because if the state satisfies all goals, check_foundation
        # will return True for all goal blocks, making cost1=0. Also, current_above
        # will match goal_above for all correctly placed blocks, making cost2=0.
        is_goal = self.goals <= state
        if is_goal and h != 0:
            # This case indicates a potential issue in the logic if it ever occurs.
            # For safety, force h=0 if goal is met.
            h = 0
        elif not is_goal and h == 0 and len(all_blocks) > 0 :
             # If not goal state but h=0 (and there are blocks), it might indicate
             # the heuristic isn't informative enough or there's a logic flaw.
             # A simple fix could be to return 1, but let's rely on the main logic.
             # This can happen if only 'clear' goals remain, which aren't directly costed.
             # Let's add a small cost if arm is not empty and goals aren't met?
             # Or if any goal predicate is false?
             # Let's refine: if h=0 but goal not met, check if arm empty is needed.
             needs_arm_empty = any(p.startswith('(arm-empty)') for p in self.goals)
             has_arm_empty = '(arm-empty)' in state
             if not is_goal and h == 0:
                 # Check if any goal predicate is unsatisfied
                 if not self.goals <= state:
                     h = 1 # Give a minimal cost to non-goal states with h=0

        return h

