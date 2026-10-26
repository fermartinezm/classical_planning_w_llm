import sys
import os
from heuristics.heuristic_base import Heuristic

# Ensure the heuristic_base module can be found if running this file directly
# or in a different project structure. Adjust path as necessary.
# try:
#     # Example: Adjust path if file is in heuristics/ folder relative to a root
#     sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#     from heuristics.heuristic_base import Heuristic
# except ImportError:
#     # Fallback if structure is flat or module is already in PYTHONPATH
#     from heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact string.
    Removes parentheses and splits by space.
    Example: "(on b1 b2)" -> ["on", "b1", "b2"]
    """
    # Assumes fact is a non-empty string starting with '(' and ending with ')'
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state.
    It counts the number of blocks that need to be moved to achieve the goal configuration.
    Each block identified as needing to be moved contributes 2 actions (one pick/unstack, one put/stack)
    to the heuristic value, except for a block currently held by the arm, which contributes 1 action (put/stack).

    # Assumptions
    - All blocks present in the initial state are relevant to the goal configuration.
    - The goal is specified primarily by `on`, `on-table`, and `clear` predicates defining the desired block stacks. Goals like `arm-empty` or `holding` are not directly used in the heuristic calculation but affect the final state check.
    - Actions have a uniform cost of 1.

    # Heuristic Initialization
    - The constructor parses the goal state (`task.goals`) to build representations of the target configuration:
        - `goal_on`: A dictionary mapping each block to the object it should be on (another block's name or 'table').
        - `goal_clear`: A set of blocks that should be clear in the goal.
        - `goal_above`: A dictionary mapping each block to the block that should be directly on top of it in the goal (or None if it should be clear).
    - It also identifies all unique block names mentioned in the goals (`all_goal_blocks`).

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Current State:** Extract the current configuration from `node.state`:
        - `current_on`: Maps block to what's below it ('table' or another block).
        - `current_above`: Maps block to what's directly above it (or None).
        - `current_holding`: The block held by the arm (or None).
        - Identify all blocks present in the current state (`current_blocks`).
        - Determine the set of all blocks relevant to the problem (`all_blocks_in_problem`).
    2.  **Identify Correctly Placed Towers:** Determine the set `correct_tower_blocks`. A block `b` is in this set if:
        - It's on the correct target (block or table) as specified in `goal_on`.
        - AND the target below it (if it's a block) is also in `correct_tower_blocks`. This is checked iteratively until no more blocks can be added. Blocks held by the arm cannot be part of a correct tower.
    3.  **Identify Blocks to Move:** Create a set `moved_blocks`. This set stores all blocks that must be moved from their current position to reach the goal configuration.
        - If a block `b` is currently held (`current_holding`), add it to `moved_blocks`.
        - Iterate through all blocks `b`. If `b` is already marked to be moved, skip.
        - If `b` is *not* part of a correctly placed tower (`b not in correct_tower_blocks`), mark `b` and *all blocks currently stacked above it* by adding them to `moved_blocks`. This uses a helper function `mark_tower_for_moving`.
        - If `b` *is* part of a correctly placed tower (`b in correct_tower_blocks`), check if the block currently above it (`c_above`) matches the block that *should* be above it (`g_above`, derived from `goal_on` and `goal_clear`). If they don't match and `c_above` exists and is not held, mark `c_above` and *all blocks currently stacked above `c_above`* by adding them to `moved_blocks`.
    4.  **Calculate Cost:**
        - Initialize `h = 0`.
        - For each block `b` in `moved_blocks`:
            - If `b` is the block currently being held, add 1 to `h` (representing the single 'putdown' or 'stack' action needed).
            - Otherwise, add 2 to `h` (representing one 'pickup'/'unstack' and one 'putdown'/'stack').
    5.  **Handle Goal State:** If the current state satisfies all goal conditions (`self.goals <= state`), return 0. If the calculated cost `h` is 0 but the state is not a goal state (an unlikely edge case with this logic), return 1 to ensure non-zero cost for non-goal states. Otherwise, return the calculated cost `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing goal conditions.
        Args:
            task: The planning task object containing goals, initial state, etc.
        """
        self.goals = task.goals
        # Static facts are not typically used in standard Blocksworld definitions.

        self.goal_on = {} # block -> block_below name or 'table'
        self.goal_clear = set() # blocks that should be clear in the goal
        self.all_goal_blocks = set() # All blocks mentioned in goal facts

        # Parse goal facts to build goal configuration representation
        for fact in self.goals:
            # Ensure fact is a string and non-empty before processing
            if not isinstance(fact, str) or len(fact) < 3:
                continue # Skip invalid facts if any

            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == 'on' and len(parts) == 3:
                block_on_top, block_below = parts[1], parts[2]
                self.goal_on[block_on_top] = block_below
                self.all_goal_blocks.add(block_on_top)
                self.all_goal_blocks.add(block_below)
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                self.goal_on[block] = 'table'
                self.all_goal_blocks.add(block)
            elif predicate == 'clear' and len(parts) == 2:
                block = parts[1]
                self.goal_clear.add(block)
                self.all_goal_blocks.add(block)
            # Ignore other potential goal predicates like 'arm-empty' or 'holding'

        # Precompute which block should be above each block in the goal
        self.goal_above = {b: None for b in self.all_goal_blocks} # Initialize all as clear
        for block_on_top, block_below in self.goal_on.items():
            if block_below != 'table':
                # Handle potential inconsistencies: if goal requires 'clear(b)' and 'on(x, b)',
                # the 'on(x, b)' takes precedence for the structure.
                if block_below in self.goal_clear:
                     self.goal_clear.discard(block_below) # Block cannot be clear if something is on it
                self.goal_above[block_below] = block_on_top


    def __call__(self, node):
        """
        Computes the heuristic value for the given state node.
        Args:
            node: A search node containing the state (node.state).
        Returns:
            An integer estimate of the cost to reach the goal.
        """
        state = node.state

        # Optimization: Check if goal is already reached
        if self.goals <= state:
            return 0

        # --- 1. Parse Current State ---
        current_on = {} # block -> block_below name or 'table'
        current_clear = set() # blocks currently clear
        current_holding = None # block currently held, or None
        on_mapping = {} # block -> block directly on top of it
        current_blocks = set() # All blocks present in the current state facts

        for fact in state:
            # Ensure fact is a string and non-empty before processing
            if not isinstance(fact, str) or len(fact) < 3:
                continue # Skip invalid facts

            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == 'on' and len(parts) == 3:
                block_on_top, block_below = parts[1], parts[2]
                current_on[block_on_top] = block_below
                on_mapping[block_below] = block_on_top
                current_blocks.add(block_on_top)
                current_blocks.add(block_below)
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                current_on[block] = 'table'
                current_blocks.add(block)
            elif predicate == 'clear' and len(parts) == 2:
                block = parts[1]
                current_clear.add(block)
                current_blocks.add(block) # Ensure block is known even if just clear on table
            elif predicate == 'holding' and len(parts) == 2:
                current_holding = parts[1]
                current_blocks.add(current_holding)
            # 'arm-empty' predicate is implicitly handled (current_holding is None)

        # Determine the full set of blocks involved in the problem
        all_blocks_in_problem = self.all_goal_blocks.union(current_blocks)
        if not all_blocks_in_problem:
             # If there are no blocks, and goal wasn't met, goal must be non-empty.
             # If goal is empty (e.g., (and)), h=0 is correct.
             return 0 if not self.goals else 1 # Return 1 if goal non-empty but no blocks

        # Map each block to the block currently above it
        current_above = {b: on_mapping.get(b) for b in all_blocks_in_problem}

        # --- 2. Identify Correctly Placed Towers ---
        # A block is correctly placed if it's on its goal target, and the target
        # below it is also correctly placed (recursively down to the table).
        correct_tower_blocks = set()
        made_progress = True
        # Iteratively identify blocks that form part of a correct goal tower
        while made_progress:
            made_progress = False
            for b in all_blocks_in_problem:
                # Skip blocks already confirmed or held by the arm
                if b in correct_tower_blocks or b == current_holding:
                    continue

                goal_target = self.goal_on.get(b) # What should be below b
                current_target = current_on.get(b) # What is currently below b

                # Check if block b is positioned correctly relative to its target
                if goal_target is not None and current_target == goal_target:
                    # Check if the target itself is correctly placed (base case: table)
                    if goal_target == 'table' or goal_target in correct_tower_blocks:
                        if b not in correct_tower_blocks: # Add only if not already present
                            correct_tower_blocks.add(b)
                            made_progress = True # Found a new correctly placed block

        # --- 3. Identify Blocks to Move ---
        # moved_blocks stores all blocks that are not currently part of their final,
        # correctly supported goal tower, or are covering a block incorrectly.
        moved_blocks = set()

        # Helper function to recursively mark a block and everything above it for moving
        def mark_tower_for_moving(bottom_block):
            curr = bottom_block
            while curr is not None and curr != current_holding:
                if curr not in moved_blocks:
                    moved_blocks.add(curr)
                else:
                    # If we encounter an already marked block, the tower above it is also marked
                    break
                # Move to the block above
                curr = current_above.get(curr)

        # The block held by the arm always needs to be moved (placed)
        if current_holding is not None:
            moved_blocks.add(current_holding)

        # Check every block to see if it or the tower it's part of needs moving
        for b in all_blocks_in_problem:
            if b in moved_blocks: # Skip if already marked
                continue

            # Case 1: Block 'b' is misplaced (not part of a correct tower below it)
            if b not in correct_tower_blocks:
                 mark_tower_for_moving(b) # Mark 'b' and everything above it

            # Case 2: Block 'b' is correctly placed, but might be wrongly covered
            else: # b is in correct_tower_blocks
                c_above = current_above.get(b) # Block currently above b
                g_above = self.goal_above.get(b) # Block that should be above b (None if clear)

                # If what's currently above differs from what should be above...
                if c_above != g_above:
                    # ...and there is actually a block currently above (which isn't held)...
                    if c_above is not None and c_above != current_holding:
                         # ...then that block 'c_above' and everything above it must move.
                         mark_tower_for_moving(c_above)

        # --- 4. Calculate Cost ---
        # Cost is 1 for the held block (needs placing) + 2 for all other blocks that need moving.
        cost = 0
        for b in moved_blocks:
            if b == current_holding:
                cost += 1
            else:
                cost += 2

        # --- 5. Final Value ---
        # Ensure heuristic is non-zero for non-goal states.
        # If cost is 0, it implies moved_blocks is empty. This should only happen
        # if the state matches the goal configuration perfectly.
        # We already checked `self.goals <= state` at the start.
        # If somehow cost is 0 but state is not goal, return 1.
        if cost == 0 and not (self.goals <= state):
             # This might happen if the goal includes e.g. (arm-empty) which is not met,
             # but the block configuration is correct.
             return 1

        return cost
