import sys
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # This should not happen with valid PDDL state facts, but as a safeguard:
        return []
    # Remove outer parentheses and split by whitespace
    return fact[1:-1].split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by counting blocks that are not in their final goal position within the goal
    stack structure, and adding a penalty for blocks that are obstructing
    blocks that are not yet in their final position.

    # Assumptions
    - The goal state defines one or more specific stacks of blocks on the table.
    - The heuristic assumes standard Blocksworld actions (pick-up, put-down, stack, unstack).
    - Each action has a cost of 1.
    - The heuristic is non-admissible and designed to guide a greedy best-first search.

    # Heuristic Initialization
    The heuristic is initialized by parsing the goal conditions from the task.
    It builds two maps representing the desired goal structure:
    - `goal_stack_map`: Maps a block to the block directly below it in the goal stack,
      or to the string 'table' if it should be on the table.
    - `goal_above_map`: Maps a block to the block that should be directly on top of it
      in the goal stack. This helps identify which blocks should be clear.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is computed as follows:

    1.  **Goal Check**: First, check if the current state is the goal state using the task's `goal_reached` method. If it is, the heuristic value is 0.

    2.  **Build Current State Structure**: Parse the facts in the current state to build maps representing the current configuration of blocks:
        -   `current_below`: Maps a block to the block directly below it, or 'table'.
        -   `current_above`: Maps a block to the block directly on top of it.
        -   `holding_block`: Stores the block currently held by the arm, or None if the arm is empty.

    3.  **Determine Final Goal Position Status**: Define a recursive helper function `is_in_final_pos(block)` that determines if a given block is currently in its correct position within its goal stack, relative to the table. This function uses memoization to avoid redundant calculations for the same block. A block is in its final goal position if:
        -   It is not currently held by the arm.
        -   It is on the correct block (or table) as specified by `goal_stack_map`.
        -   If it's on another block (not the table), that block is also in its final goal position.
        -   If nothing should be on this block in the goal stack (`goal_above_map` indicates no block should be above it), then nothing is currently on it.

    4.  **Calculate Misplaced Blocks (H1)**: Iterate through all blocks that are part of any goal stack (i.e., are keys in `goal_stack_map`). Count how many of these blocks are *not* in their final goal position according to the `is_in_final_pos` function. Store the set of these blocks that are not in their final position.

    5.  **Calculate Obstructions (H2)**: Iterate through all blocks that are currently sitting directly on top of another block (i.e., appear as values in `current_above`). For each such block `block_on_top` sitting on `block_below`, check if `block_below` is one of the blocks identified in step 4 (i.e., a block that is not in its final goal position). If it is, `block_on_top` is considered an obstruction that needs to be moved out of the way, and we increment the heuristic count.

    6.  **Total Heuristic**: The final heuristic value is the sum of the counts from step 4 (misplaced blocks) and step 5 (obstructions). This value is 0 if and only if the state is the goal state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the goal structure.

        @param task: The planning task object containing initial state, goals, etc.
        """
        self.goals = task.goals  # Goal conditions.
        # Static facts are empty in Blocksworld, so no processing needed here.
        # task.static is frozenset()

        # Build maps representing the goal stack structure
        self.goal_stack_map = {}  # block -> block below it, or 'table'
        self.goal_above_map = {}  # block -> block above it, or None (implicitly by get)
        self.goal_block_set = set() # Set of all blocks involved in goal stacks

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block, below = parts[1], parts[2]
                self.goal_stack_map[block] = below
                self.goal_above_map[below] = block
                self.goal_block_set.add(block)
                self.goal_block_set.add(below)
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                self.goal_stack_map[block] = 'table'
                self.goal_block_set.add(block)
            # We ignore 'clear' and 'arm-empty' goals for building the stack structure,
            # but 'clear' is implicitly handled by checking goal_above_map in is_in_final_pos.

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.

        @param node: The search node containing the current state.
        @return: The estimated heuristic cost (non-admissible).
        """
        state = node.state  # Current world state (frozenset of strings).

        # 1. Goal Check
        # Use the task's goal_reached method which checks if self.goals <= state
        # This is necessary to ensure h=0 only at goal states.
        # We don't have the task object here, but the search framework does.
        # A heuristic function is typically called by the search node/framework,
        # which should handle the goal check before calling the heuristic,
        # or the heuristic itself can check if all its goal predicates are met.
        # Let's check if all goal predicates defined in self.goals are in the state.
        if self.goals <= state:
             return 0

        # 2. Build Current State Structure
        current_below = {}  # block -> block below it, or 'table'
        current_above = {}  # block -> block above it, or None
        holding_block = None

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block, below = parts[1], parts[2]
                current_below[block] = below
                current_above[below] = block
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                current_below[block] = 'table'
            elif predicate == 'holding' and len(parts) == 2:
                holding_block = parts[1]
            # 'clear' and 'arm-empty' are not needed for building below/above maps

        # 3. Determine Final Goal Position Status (Recursive with Memoization)
        memo_final_pos = {} # Memoization dictionary for is_in_final_pos

        def is_in_final_pos(block):
            """
            Checks if a block is in its correct final position within the goal stack.
            Uses closure to access current_below, current_above, holding_block,
            self.goal_stack_map, self.goal_above_map, and memo_final_pos.
            """
            if block in memo_final_pos:
                return memo_final_pos[block]

            # A block must be part of a goal stack to have a defined final position
            # relative to that stack structure.
            if block not in self.goal_stack_map:
                 # This block is not in any goal stack. It cannot be in a final goal *stack* position.
                 # It might be an obstruction, handled by H2.
                 memo_final_pos[block] = False
                 return False

            desired_below = self.goal_stack_map[block]
            current_b = current_below.get(block) # Use .get() in case block is held or not in state

            # 1. Check immediate support
            # If block is held, it's not on its desired support.
            # If block is not held but not in current_below, state is weird, assume not in final pos.
            # If block is on the wrong support.
            if holding_block == block or current_b is None or current_b != desired_below:
                memo_final_pos[block] = False
                return False

            # Immediate support is correct.

            # 2. Check if the support is in its final position (if it's a block)
            if desired_below != 'table':
                # Recursively check the block below.
                # The recursive call handles the case where desired_below is not in goal_stack_map.
                if not is_in_final_pos(desired_below):
                    memo_final_pos[block] = False
                    return False

            # The stack below this block is correct and in its final position.

            # 3. Check if the block should be clear and is clear.
            # A block should be clear in the goal structure if nothing should be on it.
            desired_above = self.goal_above_map.get(block) # Block that should be on this block in goal, or None
            current_a = current_above.get(block) # Block currently on this block, or None

            if desired_above is None and current_a is not None:
                # Should be clear in the goal, but something is on it now. Not in final pos.
                memo_final_pos[block] = False
                return False

            # If we reach here, the block is on the correct support (which is in final pos if not table),
            # and if it should be clear, it is clear.
            memo_final_pos[block] = True
            return True

        # 4. Calculate Misplaced Blocks (H1)
        h1 = 0
        blocks_not_in_final_pos = set()
        # Iterate through blocks that are keys in goal_stack_map (i.e., blocks that are placed on something or table)
        for block in self.goal_stack_map.keys():
            if not is_in_final_pos(block):
                h1 += 1
                blocks_not_in_final_pos.add(block)

        # 5. Calculate Obstructions (H2)
        h2 = 0
        # Iterate through blocks that are currently on top of other blocks
        # (i.e., blocks that appear as values in current_above)
        for block_below, block_on_top in current_above.items():
            # Check if the block below is one of the blocks that is NOT in its final goal position
            if block_below in blocks_not_in_final_pos:
                # This block_on_top is an obstruction on a block that needs to be moved/fixed.
                h2 += 1

        # 6. Total Heuristic
        return h1 + h2

