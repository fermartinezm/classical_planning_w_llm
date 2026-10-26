# Add necessary imports if not already present in the execution environment
# from task import Operator, Task # Assuming task.py provides these

import sys
import time
from collections import defaultdict
# frozenset is a built-in type, no need to import explicitly

# Assume Operator and Task classes are available from code-file-task
# For self-containment during testing, you might include them or mock them.
# In the final output, only the heuristic class should be present.

class blocksworldHeuristic:
    """
    Domain-dependent heuristic for Blocksworld.

    Summary:
    The heuristic estimates the cost to reach the goal state by counting the number of blocks
    that are not in their correct position relative to the block below them within the goal
    stack structure, or are in the correct position relative to below but have extra blocks
    stacked on top of them that should not be there in the goal state.

    Assumptions:
    - The input state is a frozenset of PDDL facts (strings).
    - The input task contains initial_state and goals as frozensets of facts.
    - The domain is Blocksworld with predicates (on ?x ?y), (on-table ?x), (clear ?x),
      (holding ?x), (arm-empty).
    - Goal states consist of (on ?x ?y), (on-table ?x), and potentially (clear ?x), (arm-empty) facts.
    - The 'table' is a special location, not a block.
    - The 'arm' is a special location for a held block.
    - The heuristic value is 0 if and only if the state is a goal state.
    - The heuristic value is finite for solvable states.

    Heuristic Initialization:
    The constructor precomputes the goal configuration by parsing the goal facts.
    It builds two data structures:
    - goal_below: A dictionary mapping each block to the block it should be directly on
                  in the goal state, or the string 'table' if it should be on the table.
    - goal_above: A dictionary mapping each block to a set of blocks that should be
                  directly on top of it in the goal state.
    It also extracts the set of all blocks involved in the problem.

    Step-By-Step Thinking for Computing Heuristic:
    1. For a given state, parse the facts to determine the current configuration:
       - current_below: A dictionary mapping each block to the block it is currently
                        directly on, or 'table' if on the table, or 'arm' if held.
       - current_above: A dictionary mapping each block to a set of blocks currently
                        directly on top of it.
       - Identify the block being held, if any.
    2. Initialize the heuristic value `h` to 0.
    3. Initialize a memoization dictionary for the `_is_in_correct_stack_from_top` recursive function.
    4. Iterate through each block identified during initialization:
       a. Check if the block has a defined goal position (i.e., is present in `self.goal_below`). Blocks without a goal position are skipped for the main heuristic count.
       b. If the block has a goal position, check if it is part of the correct goal stack structure extending down to the table.
          This is done using a recursive helper function `_is_in_correct_stack_from_top(block, goal_below, current_below, memo)`.
          - Base cases: 'table' and 'arm' are considered part of a correct stack below (as they are fixed locations).
          - Recursive step: A block `B` is in the correct stack from top down if its current position relative to below matches its goal position relative to below, AND the block below it (in the current state, which must match the goal below position) is also in the correct stack from top down. Memoization is used to avoid redundant computations.
       c. If the block is *not* in the correct stack from top down (`_is_in_correct_stack_from_top` returns False), increment `h` by 1. This counts blocks that are in the wrong "column" or whose supporting stack is incorrect.
       d. If the block *is* in the correct stack from top down (`_is_in_correct_stack_from_top` returns True), check if there are any "extra" blocks currently stacked on top of it that should *not* be there according to the goal state.
          This is done using a helper function `_has_extra_blocks_above(block, goal_above, current_above)`.
          - It checks if any block currently on top of `block` is not present in the set of blocks that should be on top of `block` in the goal state (`goal_above[block]`).
       e. If there are extra blocks on top (`_has_extra_blocks_above` returns True), increment `h` by 1. This counts blocks that are in the right "column" but are blocked by unwanted blocks above.
    5. Return the final value of `h`.
    """

    def __init__(self, task):
        self.goals = task.goals
        # Extract all unique objects from initial and goal states
        self.all_objects = self._extract_all_objects(task.initial_state, task.goals)
        # Filter out special locations like 'table' and 'arm' to get just the blocks
        self.blocks = sorted([obj for obj in self.all_objects if obj not in ('table', 'arm')])

        self.goal_below = {}
        self.goal_above = defaultdict(set)

        # Parse goal facts to build goal_below and goal_above
        for fact in self.goals:
            # Facts are strings like '(on b1 b2)'
            parts = fact[1:-1].split() # Remove surrounding brackets and split
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                obj1, obj2 = parts[1], parts[2]
                self.goal_below[obj1] = obj2
                self.goal_above[obj2].add(obj1)
            elif predicate == 'on-table' and len(parts) == 2:
                obj1 = parts[1]
                self.goal_below[obj1] = 'table'
            # Ignore 'clear' and 'arm-empty' goals for building goal_below/above structure

        # Note: Blocks not appearing in goal_below will be skipped in __call__ heuristic calculation.
        # This means blocks that don't have a specified target position in the goal
        # do not contribute to the heuristic cost.

    def _extract_all_objects(self, initial_state, goals):
        """Extracts all unique object names from the initial state and goal facts."""
        objects = set()
        for fact_set in [initial_state, goals]:
            for fact in fact_set:
                # Example fact: '(on b1 b2)' or '(on-table b1)' or '(clear b1)' or '(arm-empty)'
                parts = fact[1:-1].split() # Remove surrounding brackets and split
                # Add all parts after the predicate name
                if len(parts) > 1:
                    objects.update(parts[1:])
        return sorted(list(objects)) # Return a sorted list for consistent ordering

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state: The current state (frozenset of facts).
        @return: The heuristic value (integer).
        """
        # If the state is the goal state, the heuristic is 0
        if self.goals <= state:
            return 0

        current_below = {}
        current_above = defaultdict(set)
        held_block = None

        # Parse current state facts to build current_below and current_above
        for fact in state:
            # Facts are strings like '(on b1 b2)'
            parts = fact[1:-1].split() # Remove surrounding brackets and split
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                obj1, obj2 = parts[1], parts[2]
                current_below[obj1] = obj2
                current_above[obj2].add(obj1)
            elif predicate == 'on-table' and len(parts) == 2:
                obj1 = parts[1]
                current_below[obj1] = 'table'
            elif predicate == 'holding' and len(parts) == 2:
                 held_block = parts[1]
                 current_below[held_block] = 'arm' # Represent held block's position

        h = 0
        memo_stack_from_top = {} # Memoization for _is_in_correct_stack_from_top

        # Iterate over all blocks identified during initialization
        for block in self.blocks:
            # Only consider blocks that have a defined goal position (are part of the goal structure)
            if block not in self.goal_below:
                 continue # Skip blocks not relevant to the goal structure

            # Check if the block is in the correct stack from top down
            # This checks if the block is on the correct block, which is on the correct block, etc., down to the table.
            in_correct_stack = self._is_in_correct_stack_from_top(
                block, self.goal_below, current_below, memo_stack_from_top
            )

            if not in_correct_stack:
                # If the block is not in the correct stack column, it contributes 1 to the heuristic
                h += 1
            else:
                # If the block is in the correct stack column, check if it has extra blocks on top
                if self._has_extra_blocks_above(block, self.goal_above, current_above):
                    # If it has extra blocks on top, it also contributes 1 to the heuristic
                    h += 1

        return h

    def _is_in_correct_stack_from_top(self, block, goal_below, current_below, memo):
        """
        Recursive helper to check if a block is in the correct stack structure
        from the block down to the table, based on goal positions.
        Uses memoization to avoid redundant computations.

        @param block: The block to check.
        @param goal_below: Dictionary mapping blocks to what they should be on in the goal.
        @param current_below: Dictionary mapping blocks to what they are currently on.
        @param memo: Memoization dictionary.
        @return: True if the block is in the correct stack from top down, False otherwise.
        """
        # Base cases: 'table' and 'arm' are fixed locations below blocks, always considered 'correct' in this context
        if block in ('table', 'arm'):
            return True

        # Check memoization
        if block in memo:
            return memo[block]

        # Get the block currently below this block
        current_below_block = current_below.get(block)

        # If the block is not found in current_below, it's not on anything or held.
        # In a valid state, every block is either on table, on another block, or held.
        # If it's not in current_below and not held (checked implicitly by iterating blocks),
        # it's an invalid state representation for this heuristic.
        # Treat as not in correct stack.
        if current_below_block is None:
             memo[block] = False
             return False

        # Get the block that should be below this block in the goal
        # We assume block is in goal_below based on the check in __call__
        goal_pos_below = goal_below[block]

        # Check if the current position relative to below matches the goal position
        pos_matches_goal = (current_below_block == goal_pos_below)

        if not pos_matches_goal:
            # If the block is on the wrong thing, it's not in the correct stack
            memo[block] = False
            return False

        # If the position matches the goal, recursively check the block below it
        # The block below in the *current* state must be the same as the block below
        # in the *goal* state for this block to be in the correct stack.
        # So, we recurse on `current_below_block` (which is equal to `goal_pos_below`).
        if goal_pos_below in ('table', 'arm'):
             # Base case: If the block should be on table/arm and is, the stack below is correct (base of recursion)
             result = True
        else:
             # Recursive step: Check if the block below is also in the correct stack
             result = self._is_in_correct_stack_from_top(
                 current_below_block, goal_below, current_below, memo
             )

        # Store and return the result
        memo[block] = result
        return result

    def _has_extra_blocks_above(self, block, goal_above, current_above):
        """
        Checks if there are any blocks currently on top of 'block' that should
        not be there according to the goal state.

        @param block: The block to check what's on top of it.
        @param goal_above: Dictionary mapping blocks to set of blocks that should be on them in the goal.
        @param current_above: Dictionary mapping blocks to set of blocks currently on them.
        @return: True if there is at least one block currently on top that is not in goal_above, False otherwise.
        """
        # Get the set of blocks currently on top of this block
        current_blocks_above = current_above.get(block, set())

        # Get the set of blocks that should be on top of this block in the goal
        goal_blocks_above = goal_above.get(block, set())

        # Check if any block currently above is not in the set of blocks that should be above
        for b_above in current_blocks_above:
            if b_above not in goal_blocks_above:
                return True # Found an extra block

        return False # No extra blocks found
