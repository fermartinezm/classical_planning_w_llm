import itertools
from heuristics.heuristic_base import Heuristic # Assuming this path is correct

# Helper function to extract parts of a PDDL fact
def get_parts(fact_str):
    """Removes parentheses and splits the fact string by spaces."""
    # Example: "(on b1 b2)" -> ["on", "b1", "b2"]
    # Handles potential extra whitespace and removes empty strings
    parts = fact_str.strip()[1:-1].split()
    return [part for part in parts if part]

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    configuration. It primarily counts the number of blocks that are considered
    "misplaced". A block is misplaced if it is not resting on the correct block
    or the table, as specified in the goal, or if any block underneath it in
    its current stack is misplaced. Each misplaced block is estimated to require
    two actions: one to move it (pickup/unstack) and one to place it correctly
    (putdown/stack). If a block is currently being held by the robot arm, the
    heuristic is adjusted because the first action (pickup/unstack) is implicitly
    done or not needed next; only the placement action remains.

    # Assumptions
    - The goal specifies the final position (`on(A,B)` or `on-table(A)`) for
      all relevant blocks that need to be in a specific configuration.
    - Blocks have uniform size and stacking is always possible if the target is clear.
    - Each action (pickup, putdown, stack, unstack) has a cost of 1.
    - The heuristic focuses on achieving the `on` and `on-table` goal predicates.
      `clear` predicates in the goal are implicitly satisfied if the `on`/`on-table`
      predicates are met correctly. The `arm-empty` goal is partially considered
      via the handling of the `holding` state.

    # Heuristic Initialization
    - The constructor (`__init__`) parses the task's goal description (`task.goals`).
    - It builds a `goal_config` dictionary mapping each block mentioned in an
      `on` or `on-table` goal to the object (another block's name or the
      special string 'table') that should be directly beneath it in the goal state.
    - It identifies the set `self.blocks` containing all unique block names
      involved in these goal configurations.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Check Goal State:** If the current state already satisfies all goal
        predicates (`self.goals <= node.state`), the heuristic value is 0.
    2.  **Parse Current State:** Determine the current configuration from `node.state`:
        - Identify which block is held by the arm (`held_block`), if any.
        - Build `current_config`: a dictionary mapping each block `b` to the object
          (block name or 'table') currently directly beneath it.
    3.  **Count Misplaced Blocks:**
        - Initialize `misplaced_count = 0`.
        - Use a memoization dictionary `memo` for the recursive placement check.
        - Iterate through all blocks `b` that are part of the goal configuration (`self.blocks`).
        - For each block `b`:
            - If `b` is the `held_block`, it's considered misplaced from its goal stack. Increment `misplaced_count`.
            - If `b` is not held, call a recursive helper function `_is_correctly_supported(b, current_config, memo)`:
                - **Base Case:** `_is_correctly_supported('table', ...)` returns `True`.
                - **Memoization:** Return stored result if `b` is in `memo`.
                - **Check Goal:** If `b` is not in `self.goal_config`, it's misplaced (or supporting incorrectly). Return `False`.
                - **Check Current Support:** Get `goal_below = self.goal_config[b]` and `current_below = current_config.get(b)`. If `current_below != goal_below`, block `b` is not on the right support. Return `False`.
                - **Recursive Step:** If `b` is on the correct support `goal_below`, recursively call `_is_correctly_supported(goal_below, ...)`. The result determines if `b` is correctly supported all the way down.
                - Store the result in `memo` before returning.
            - If `b` is not held and `_is_correctly_supported(b, ...)` returns `False`, increment `misplaced_count`.
    4.  **Calculate Base Heuristic Value:** The base estimate is `h = 2 * misplaced_count`, assuming two moves per misplaced block.
    5.  **Adjust for Held Block:** If a block is currently held (`held_block` is not None), one of the two estimated moves (the pickup/unstack) is not needed next. Therefore, reduce the heuristic value by 1. Ensure `h` does not become negative. `h = max(0, h - 1)`.
    6.  **Return `h`**.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing goal conditions to understand
        the target configuration.
        """
        self.goals = task.goals
        self.goal_config = {} # Stores {block: object_below} for the goal state
        blocks_in_goal_config = set() # All blocks mentioned in goal structure

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip empty or invalid facts
            predicate = parts[0]

            if predicate == "on" and len(parts) == 3:
                # Goal is (on block_a block_b)
                block_on_top = parts[1]
                block_below = parts[2]
                self.goal_config[block_on_top] = block_below
                blocks_in_goal_config.add(block_on_top)
                blocks_in_goal_config.add(block_below) # Also track blocks that are supports
            elif predicate == "on-table" and len(parts) == 2:
                # Goal is (on-table block_a)
                block_on_table = parts[1]
                self.goal_config[block_on_table] = 'table'
                blocks_in_goal_config.add(block_on_table)
            # Other goals like 'clear' or 'arm-empty' are not directly used
            # to build the target structure map.

        # self.blocks contains all unique blocks involved in the goal structure
        self.blocks = blocks_in_goal_config


    def _is_correctly_supported(self, block, current_config, memo):
        """
        Recursive helper function with memoization to check if a block
        is correctly supported all the way down to the table according
        to the goal configuration.
        Returns True if correctly supported, False otherwise.
        """
        # Base case: The table is the ultimate correct support.
        if block == 'table':
            return True

        # Check memoization cache
        if block in memo:
            return memo[block]

        # Check if the block has a defined goal position below it in the goal config.
        # If not, it cannot be part of a correctly supported goal stack from this point up.
        if block not in self.goal_config:
             memo[block] = False
             return False

        goal_below = self.goal_config[block]
        # Use .get() for current_below as the block might be held or missing from state config
        current_below = current_config.get(block)

        # Check if the block is currently resting on the correct object (or table).
        if current_below != goal_below:
            memo[block] = False
            return False

        # If it's on the correct object, recursively check if that object
        # is itself correctly supported.
        result = self._is_correctly_supported(goal_below, current_config, memo)
        memo[block] = result
        return result

    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.
        h = 2 * (num misplaced blocks) - (1 if holding a block else 0)
        Returns 0 if the state is a goal state.
        """
        state = node.state

        # Check if the current state is a goal state first for h=0
        if self.goals <= state:
             return 0

        held_block = None
        current_config = {} # Stores {block: object_below} for the current state

        # Parse the current state to find block positions and held block
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or invalid facts
            predicate = parts[0]

            if predicate == "on" and len(parts) == 3:
                # state is (on block_a block_b) -> block_a is on block_b
                current_config[parts[1]] = parts[2]
            elif predicate == "on-table" and len(parts) == 2:
                # state is (on-table block_a) -> block_a is on 'table'
                current_config[parts[1]] = 'table'
            elif predicate == "holding" and len(parts) == 2:
                # state is (holding block_a)
                held_block = parts[1]
            # Ignore 'clear' and 'arm-empty' predicates for config building

        # Count blocks that are not correctly supported according to the goal
        misplaced_count = 0
        memo = {} # Memoization for the recursive check within this state evaluation

        # Iterate through all blocks that are part of the goal configuration
        for block in self.blocks:
            if block == held_block:
                # A held block is always considered misplaced from its final stacked position.
                misplaced_count += 1
            elif not self._is_correctly_supported(block, current_config, memo):
                # A block not correctly supported is misplaced.
                misplaced_count += 1

        # Base heuristic: 2 actions per misplaced block
        heuristic_value = 2 * misplaced_count

        # Adjustment: If holding a block, one action (pickup/unstack) is already "done".
        if held_block is not None:
             # Subtract 1, ensuring the result is not negative.
             heuristic_value = max(0, heuristic_value - 1)

        # Final heuristic value. Note that h=0 might occur slightly before the true goal
        # if e.g. arm-empty is a goal but the arm holds the last correctly placed block.
        # This is acceptable for Greedy Best-First Search.
        return heuristic_value
