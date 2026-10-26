import os
import sys
from fnmatch import fnmatch # Not strictly needed now, but kept for potential future use

# Attempt to import the base class. Adjust path if necessary based on environment.
try:
    # Assumes the script is run from a directory where 'heuristics' is a subdirectory
    # or that the 'heuristics' package is in the Python path.
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Provide a dummy base class if the import fails (e.g., for standalone testing)
    # In a real scenario, ensure the environment allows importing heuristic_base.
    class Heuristic:
        def __init__(self, task): pass
        def __call__(self, node): raise NotImplementedError


def get_parts(fact: str):
    """
    Extracts the predicate and arguments from a PDDL fact string.
    Example: "(on b1 b2)" -> ["on", "b1", "b2"]
             "(clear b1)" -> ["clear", "b1"]
             "(arm-empty)" -> ["arm-empty"]
    """
    content = fact[1:-1].strip()
    if not content:
        return []
    return content.split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state.
    It identifies blocks that are not in their final correct position relative to the
    block or table below them (considering the entire stack recursively). It also
    accounts for blocks that obstruct these misplaced blocks. Each block identified
    as needing to be moved contributes a cost of 2 (representing one move to
    pick/unstack and one move to place/stack). If the block that needs moving is
    already held by the arm, the cost contribution is reduced to 1, as the
    pickup/unstack action is implicitly done or not needed.

    # Assumptions
    - The goal configuration is primarily defined by `(on block_a block_b)` and
      `(on-table block_c)` facts.
    - `(clear block)` goals are implicitly satisfied if the goal structure is achieved.
    - `(arm-empty)` goal is implicitly handled; if the arm holds a block that
      shouldn't be held, that block will be considered unstable.
    - Each action (pickup, putdown, stack, unstack) has a uniform cost of 1.
    - The heuristic is intended for Greedy Best-First Search and does not need
      to be admissible (it can overestimate).

    # Heuristic Initialization
    - The constructor (`__init__`) parses the task's goal conditions (`task.goals`).
    - It builds `goal_below`: a dictionary mapping each block to the object
      (another block name string or the string 'table') that should be directly
      beneath it in the final goal configuration.
    - It collects `goal_blocks`: a set of all unique block names mentioned in
      any goal predicate.
    - Initializes `memo_stable`: a dictionary for memoizing stability checks within
      a single heuristic evaluation.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Check Goal Reached:** If the current state (`node.state`) already satisfies
        all goal conditions (`self.goals <= state`), the heuristic value is 0.
    2.  **Parse Current State:** Determine the current configuration:
        - Use `_get_block_below(block, state)` to find what's under a block.
        - Use `_get_block_on(block, state)` to find what's on top of a block.
        - Use `_get_held_block(state)` to find which block (if any) is held.
        - Use `_get_all_blocks_in_state(state)` to get all block names currently mentioned.
    3.  **Initialize Cost:** Set heuristic value `h = 0`. Create an empty set
        `processed_blocks` to track blocks whose cost has been accounted for. Clear
        the `memo_stable` cache.
    4.  **Identify All Blocks:** Combine blocks from the goal (`self.goal_blocks`) and
        the current state to get `all_blocks_involved`. Include the held block if any.
    5.  **Check Stability and Obstructions:** Iterate through each `block` in
        `all_blocks_involved`:
        a.  If `block` is already in `processed_blocks`, skip it.
        b.  **Stability Check:** Call `_is_stable(block, state)`. This function
            recursively checks if the block is on its correct final support (as defined
            in `self.goal_below`) and if that support is also stable, all the way down
            to the table. It returns `False` if the block is misplaced, on an unstable
            base, held when it shouldn't be, or exists when it shouldn't according
            to the goal structure. Memoization (`self.memo_stable`) is used.
        c.  **Calculate Cost:** If `_is_stable` returns `False`:
            i.  The `block` needs to be moved. Calculate its move cost: 1 if this
                `block` is the `held_block`, otherwise 2. Add this cost to `h`.
            ii. Add `block` to `processed_blocks`.
            iii. **Process Obstructions:** Find the block directly on top of `block`
                 in the *current* state (`b_above = _get_block_on(block, state)`).
                 Enter a loop: while `b_above` exists:
                 - If `b_above` is not already in `processed_blocks`: Add 2 to `h`
                   (cost to unstack `b_above` and place it somewhere). Add `b_above`
                   to `processed_blocks`.
                 - Find the block on top of the current `b_above` to continue up
                   the stack of obstructions.
    6.  **Return Heuristic Value:** Return the calculated value `h`. As a safeguard for
        non-goal states where the logic might incorrectly yield h=0, return `max(1, h)`.
        (The initial check in step 1 ensures 0 is returned for true goal states).
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static # Blocksworld usually has no static facts

        # --- Precompute Goal Information ---
        self.goal_below = {} # block -> block_below_name or 'table'
        self.goal_blocks = set() # All blocks mentioned in goal predicates

        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if not parts: continue
            pred = parts[0]

            if pred == 'on' and len(parts) == 3:
                block_above, block_below = parts[1], parts[2]
                self.goal_below[block_above] = block_below
                self.goal_blocks.add(block_above)
                self.goal_blocks.add(block_below)
            elif pred == 'on-table' and len(parts) == 2:
                block = parts[1]
                self.goal_below[block] = 'table'
                self.goal_blocks.add(block)
            elif pred == 'clear' and len(parts) == 2:
                # Record block mentioned in clear goal
                self.goal_blocks.add(parts[1])
            elif pred == 'holding' and len(parts) == 2:
                 # Record block mentioned in holding goal (less common)
                 self.goal_blocks.add(parts[1])
            # 'arm-empty' is a goal condition checked implicitly/globally

        # Memoization cache for _is_stable, cleared per state evaluation
        self.memo_stable = {}


    # --- Helper methods for state parsing ---
    def _get_block_below(self, block, state):
        """Finds what is directly below 'block' in the current state."""
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == 'on' and parts[1] == block:
                return parts[2] # Returns block name (string)
            if len(parts) == 2 and parts[0] == 'on-table' and parts[1] == block:
                return 'table' # Returns 'table' (string)
        return None # Block might be held or not present

    def _get_block_on(self, block, state):
        """Finds what is directly on top of 'block' in the current state."""
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == 'on' and parts[2] == block:
                return parts[1] # Returns block name (string)
        return None # Block is clear

    def _get_held_block(self, state):
        """Finds which block is held, or None if arm is empty."""
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 2 and parts[0] == 'holding':
                return parts[1] # Returns block name (string)
        return None

    def _get_all_blocks_in_state(self, state):
        """Extracts all unique potential block names mentioned in the state predicates."""
        blocks = set()
        for fact in state:
            parts = get_parts(fact)
            # Add arguments (potential blocks)
            for arg in parts[1:]:
                 # Basic filter: avoid adding 'table' etc. Assumes block names don't clash.
                 # A robust solution would use typing info from the task object if available.
                 if arg != 'table':
                     blocks.add(arg)
        return blocks

    # --- Stability Check ---
    def _is_stable(self, block, state):
        """
        Checks if a block is 'stable': in its final goal position AND the block(s)
        below it are also stable. Uses memoization. Returns False if the block
        needs to be moved.
        """
        if block == 'table': # Base case: the table is always stable
            return True
        # Check memoization cache for this block in this state evaluation
        if block in self.memo_stable:
            return self.memo_stable[block]

        current_support = self._get_block_below(block, state)
        target_support = self.goal_below.get(block) # Target from goal dictionary

        # Case 1: Block has no target position in the goal structure (not in goal_below)
        if target_support is None:
            # It's only "stable" if it effectively doesn't exist in the current state
            # (not on table, not on another block) AND is not held.
            is_held = (self._get_held_block(state) == block)
            result = (current_support is None) and (not is_held)
            self.memo_stable[block] = result
            return result
        # Case 2: Block has a target position in the goal structure.
        else:
            # It's unstable if its current support doesn't match the target support.
            if current_support != target_support:
                self.memo_stable[block] = False
                return False
            # If current support matches target, stability depends on the target support's stability.
            else:
                result = self._is_stable(target_support, state) # Recursive call
                self.memo_stable[block] = result
                return result

    # --- Heuristic Calculation ---
    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.
        """
        state = node.state

        # --- Quick Goal Check ---
        # If the current state satisfies all goal predicates, cost is 0.
        if self.goals <= state:
            return 0

        h = 0 # Initialize heuristic value
        processed_blocks = set() # Track blocks already accounted for
        self.memo_stable.clear() # Clear memoization cache for this state evaluation

        held_block = self._get_held_block(state)

        # Determine all unique blocks involved in this state or the goal
        current_blocks = self._get_all_blocks_in_state(state)
        if held_block: # Ensure held block is included
             current_blocks.add(held_block)
        all_blocks_involved = self.goal_blocks | current_blocks

        # Iterate through all involved blocks to check their status
        for block in all_blocks_involved:
            if block in processed_blocks:
                continue # Skip if already processed (e.g., as an obstruction)

            # Check if the block is stable (in its final position on a stable base)
            if not self._is_stable(block, state):
                # Block is not stable, it needs to be moved.
                # Cost is 1 if held (saves pickup/unstack), 2 otherwise.
                cost_for_block = 1 if block == held_block else 2
                h += cost_for_block
                processed_blocks.add(block) # Mark this block as processed

                # Now, process all blocks currently stacked *above* this misplaced block.
                # These also need to be moved out of the way.
                b_above = self._get_block_on(block, state)
                while b_above is not None:
                    if b_above not in processed_blocks:
                        # Cost is 2 (unstack + place somewhere).
                        # If b_above was held, it wouldn't be 'on' this block.
                        h += 2
                        processed_blocks.add(b_above)

                    # Move up the current stack to find the next obstruction
                    current_block_above = b_above
                    b_above = self._get_block_on(current_block_above, state)

        # For non-goal states, ensure the heuristic value is at least 1.
        # This prevents the search from stalling if the heuristic incorrectly computes h=0.
        return max(1, h)
