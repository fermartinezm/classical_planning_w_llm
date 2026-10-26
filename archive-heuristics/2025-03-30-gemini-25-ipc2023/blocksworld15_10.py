import os
import sys
# The user needs to ensure that the 'heuristics' module is in the Python path
# or that this file is placed correctly relative to heuristic_base.py
# Example: Add the parent directory to sys.path if 'heuristics' is a sibling folder.
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Removes parentheses and splits by space.
    Example: "(on b1 b2)" -> ["on", "b1", "b2"]
    """
    # Assumes facts are simple strings like "(predicate arg1 arg2 ...)"
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain designed for Greedy Best-First Search.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state.
    It counts the number of blocks that are not in their final correct position
    (relative to the block below them or the table) and the blocks that are
    currently stacked on top of such misplaced blocks. Each such block is assumed
    to require two actions (one pickup/unstack, one putdown/stack). An adjustment
    is made based on whether the arm is currently holding a block. This heuristic
    is designed for informativeness rather than admissibility, aiming to guide
    Greedy Best-First Search effectively by providing a reasonable estimate of
    the remaining work.

    # Assumptions
    - The goal is specified as a conjunction of (on block block) and (on-table block) predicates.
      'clear' predicates in the goal are ignored by this heuristic as they are typically
      consequences of the final 'on'/'on-table' configuration being achieved. If a goal
      requires a block to be clear but not involved in an 'on' or 'on-table' goal fact,
      this heuristic might not perfectly capture the cost to achieve that 'clear' state.
    - All actions (pickup, putdown, stack, unstack) have a uniform cost of 1.
    - The heuristic does not need to be admissible or consistent.

    # Heuristic Initialization
    - Parses the goal predicates (`on`, `on-table`) provided in the task definition during initialization.
    - Builds a dictionary `goal_below` where `goal_below[block]` stores the object name (string)
      or the special string 'table' that should be directly underneath `block` in the goal state.
    - Identifies the set `goal_blocks` containing all unique block names involved in the
      goal's `on` and `on-table` predicates. This helps in identifying all relevant blocks.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Check for Goal State:** If the current state already satisfies all goal predicates defined in the task, the estimated cost to reach the goal is 0.
    2.  **Parse Current State:** Analyze the set of facts true in the current state (`node.state`) to determine the world configuration:
        - `current_below`: A dictionary mapping each block to the block or 'table' currently directly below it.
        - `current_above`: A dictionary mapping each block to the block currently directly above it (if any). This is derived from the 'on' facts.
        - `current_holding`: Stores the name of the block currently held by the arm, or None if the arm is empty (`(arm-empty)` is true).
        - `all_blocks`: A set containing all unique block names mentioned in the current state or in the goal configuration.
    3.  **Identify Correctly Placed Blocks:** Define a recursive function `is_correctly_placed(block)` that checks if a block is in its final goal position relative to the tower below it. This function uses memoization to avoid redundant computations for blocks within the same tower.
        - A block `B` is considered correctly placed if:
            - The block or table specified by `goal_below[B]` matches the block or table found in `current_below[B]`.
            - AND, if `B` is supposed to be on another block `C` (i.e., `goal_below[B] == C`), then block `C` must also be correctly placed (this forms the recursive check down the tower).
        - Important conditions:
            - A block currently held by the arm (`current_holding`) is never considered correctly placed in its final position.
            - Blocks that exist in the current state but are not mentioned in the goal's `on` or `on-table` structure (`goal_below.get(block)` is None) are considered misplaced, as they likely need to be moved out of the way.
    4.  **Identify Blocks Requiring Moves:** Determine which blocks need to be moved to achieve the goal configuration:
        - `NeedsMoving`: A set containing all blocks `B` for which `is_correctly_placed(B)` returns False. These blocks are either in the wrong location themselves or are resting on a block that is ultimately in the wrong location.
        - `BlocksToClear`: A set containing all blocks that are currently located somewhere above any block listed in `NeedsMoving`. These blocks obstruct access to the misplaced blocks below them and must be moved first. This set is found by iterating upwards (using `current_above`) from each block in `NeedsMoving`.
        - `TotalBlocksToMove`: The union of the `NeedsMoving` and `BlocksToClear` sets. This represents the complete set of blocks that must be picked up or unstacked at some point during the plan to reach the goal configuration.
    5.  **Calculate Base Heuristic Value:**
        - The primary estimate is `h = 2 * len(TotalBlocksToMove)`. This assumes that each block identified in `TotalBlocksToMove` requires approximately two actions: one action to pick it up (using `pickup` or `unstack`) and another action to place it somewhere (using `putdown` or `stack`).
    6.  **Adjust for Arm State:** Refine the estimate based on the block currently held by the arm:
        - If the arm is holding a block `X` (`current_holding == X`):
            - If `X` is in the `TotalBlocksToMove` set: This means `X` needs to be moved eventually. Since it's already held, the pickup/unstack action for this block is effectively completed. Therefore, subtract 1 from `h` to account only for the remaining putdown/stack action needed for this specific block's move.
            - If `X` is NOT in `TotalBlocksToMove`: This implies `X` is considered "correctly placed" relative to its goal support (an unusual situation if it's held, perhaps meaning the block below it was just put in place) or it's a block not part of the goal structure that simply needs to be put down somewhere. In either scenario, one action (putdown/stack) is still required for the held block. Add 1 to `h`.
    7.  **Return Value:** Return the final calculated heuristic value `h`. Crucially, ensure the value is at least 1 if the current state is not a goal state. This prevents the search from terminating prematurely if the calculation happens to yield 0 for a non-goal state (e.g., if `TotalBlocksToMove` is empty but the arm holds a block not needing moving).
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing goal conditions from the task.

        Args:
            task: The planning task object containing goals, operators, initial state, etc.
                  We primarily use `task.goals`.
        """
        self.goals = task.goals
        # Static facts are typically not present or needed in standard Blocksworld domains.
        self.static = task.static

        self.goal_below = {} # Stores goal config: block -> block_below | 'table'
        self.goal_blocks = set() # Stores all blocks mentioned in goal 'on'/'on-table'

        # Parse the goal facts provided in the task to build the goal configuration map.
        for fact in self.goals:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "on":
                # Example: Goal fact "(on b1 b2)" means b1 should be on b2.
                # Store: goal_below[b1] = b2
                if len(parts) == 3:
                    block, below_block = parts[1], parts[2]
                    self.goal_below[block] = below_block
                    self.goal_blocks.add(block)
                    self.goal_blocks.add(below_block)
            elif predicate == "on-table":
                # Example: Goal fact "(on-table b1)" means b1 should be on the table.
                # Store: goal_below[b1] = 'table'
                if len(parts) == 2:
                    block = parts[1]
                    self.goal_below[block] = 'table'
                    self.goal_blocks.add(block)
            # We ignore 'clear' predicates in the goal definition, assuming they
            # will be satisfied if the 'on'/'on-table' structure is correct.

    def _get_current_config(self, state):
        """
        Parses the current state facts to determine the configuration of blocks
        and the status of the arm.

        Args:
            state: A frozenset of strings representing the facts true in the current state.

        Returns:
            A tuple containing:
            - current_below (dict): Map block -> block/table currently below it.
            - current_above (dict): Map block -> block currently directly above it.
            - current_holding (str | None): The name of the block held by the arm, or None.
            - all_blocks (set): A set of all unique block names found in the state or goal.
        """
        current_below = {} # block -> block_below | 'table'
        current_on = {} # block -> block_below (only for 'on' facts)
        current_holding = None
        # Initialize with blocks known from the goal to handle cases where
        # a block might only appear in the goal initially or be cleared/held.
        all_blocks = set(self.goal_blocks)

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or invalid facts
            predicate = parts[0]

            # Determine block configurations based on predicates
            if predicate == "on" and len(parts) == 3:
                block, below_block = parts[1], parts[2]
                current_on[block] = below_block
                current_below[block] = below_block
                all_blocks.add(block)
                all_blocks.add(below_block)
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                current_below[block] = 'table'
                all_blocks.add(block)
            # Determine arm status
            elif predicate == "holding" and len(parts) == 2:
                current_holding = parts[1]
                all_blocks.add(current_holding)
            # Collect all blocks mentioned in any relevant predicate
            elif predicate == "clear" and len(parts) == 2:
                 all_blocks.add(parts[1])
            elif predicate == "arm-empty" and len(parts) == 1:
                pass # This state is implicitly known if current_holding remains None

        # Derive the inverse map: which block is directly above another
        # current_above[block_below] = block_above
        current_above = {below: above for above, below in current_on.items()}

        return current_below, current_above, current_holding, all_blocks

    def __call__(self, node):
        """
        Computes the heuristic value for the given search node.

        Args:
            node: The search node object containing the state (`node.state`) to evaluate.

        Returns:
            An integer representing the estimated cost (number of actions) to reach
            the goal from the node's state.
        """
        state = node.state

        # Optimization: If the current state already satisfies all goals, heuristic value is 0.
        if self.goals <= state:
             return 0

        # Parse the current state configuration
        current_below, current_above, current_holding, all_blocks = self._get_current_config(state)

        # Memoization cache for the recursive placement check to avoid recomputing
        memo_correct = {}

        def is_correctly_placed(block):
            """
            Recursively checks if a block is correctly placed relative to its
            goal position and the block(s) below it. Uses memoization via the
            `memo_correct` dictionary captured from the outer scope.

            Args:
                block (str): The name of the block to check.

            Returns:
                bool: True if the block is considered correctly placed, False otherwise.
            """
            # Return cached result if available
            if block in memo_correct:
                return memo_correct[block]

            # Blocks being held by the arm are never in their final correct place on the table/stack
            if block == current_holding:
                memo_correct[block] = False
                return False

            # Get the target object/location below the block in the goal configuration
            goal_b = self.goal_below.get(block)
            # Get the object/location currently below the block from the current state
            current_b = current_below.get(block)

            # Case 1: Block exists in state but has no defined goal position (not in goal_below).
            # This means it's an extra block or in a position not specified by goal on/on-table.
            # It's considered misplaced as it likely needs to be moved out of the way.
            if goal_b is None:
                memo_correct[block] = False
                return False

            # Case 2: Block has a goal position, but isn't currently on anything (and not held).
            # This might indicate an intermediate state or an issue. Treat as misplaced.
            if current_b is None:
                 memo_correct[block] = False
                 return False

            # Case 3: The object/table currently below the block doesn't match the goal target.
            if goal_b != current_b:
                memo_correct[block] = False
                return False

            # Case 4: Base case for recursion. Block is on the table and should be. Correct so far.
            if goal_b == 'table':
                memo_correct[block] = True
                return True

            # Case 5: Recursive step. Block is on another block 'goal_b' as required by the goal.
            # For the current block to be truly correct, the block below it ('goal_b')
            # must also be correctly placed. Recursively call the check for 'goal_b'.
            result = is_correctly_placed(goal_b)
            memo_correct[block] = result
            return result

        # --- Heuristic Calculation ---

        # 1. Find all blocks that are not correctly placed according to the recursive check.
        needs_moving = set()
        # Iterate through all known blocks to check their status
        for block in all_blocks:
            # Only check blocks that actually exist in the current configuration
            # (i.e., are on the table, on another block, or held by the arm)
            if block in current_below or block == current_holding:
                if not is_correctly_placed(block):
                    needs_moving.add(block)

        # 2. Find all blocks that are currently stacked above any misplaced block.
        blocks_to_clear = set()
        # For each block that needs moving, trace upwards in its current tower
        # and add all blocks found above it to the set of blocks that need clearing.
        for block in needs_moving:
            curr = block
            # Use current_above map to find block directly above 'curr'
            while current_above.get(curr) is not None:
                above_block = current_above[curr]
                blocks_to_clear.add(above_block)
                curr = above_block # Move up one level in the tower

        # 3. Combine the sets: these are all blocks that must be handled (moved).
        total_blocks_to_move = needs_moving | blocks_to_clear

        # 4. Calculate the base heuristic value: estimate 2 actions per block to move.
        h = 2 * len(total_blocks_to_move)

        # 5. Adjust the heuristic based on the arm's status and the held block.
        if current_holding is not None:
            held_block = current_holding
            if held_block in total_blocks_to_move:
                # If holding a block that needs moving, one action (pickup/unstack)
                # is effectively already done. Subtract 1 for this completed part of the move.
                h -= 1
            else:
                # If holding a block that is not in the set needing moves (e.g.,
                # it's considered correctly placed relative to goal support, or not part of goal),
                # it still requires one action (putdown/stack) eventually. Add 1 for this action.
                h += 1

        # 6. Final check: Ensure heuristic is non-negative and at least 1 for non-goal states.
        if h == 0 and not (self.goals <= state):
             # If the calculation resulted in 0 but the state is not the goal,
             # return 1. This ensures the search progresses and doesn't incorrectly
             # identify a non-goal state as having 0 cost to reach the goal.
             return 1
        else:
             # Return the calculated value, ensuring it's not negative due to unforeseen edge cases.
             return max(0, h)

