import os
import sys
from typing import Dict, Set, Optional, List

# Attempt to import the Heuristic base class.
# This assumes the script is run in an environment where the 'heuristics' package is accessible.
# It tries adjusting the path if the direct import fails.
try:
    from heuristics.heuristic_base import Heuristic
    from planning_tasks.task import Task # Assuming Task class is available for type hinting
except ImportError:
    # Adjust path relative to the current file if the package structure is known
    # (e.g., script is in 'heuristics/domains', base class is in 'heuristics')
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    try:
        from heuristic_base import Heuristic
        # Attempt to import Task from a plausible location if needed for type hints
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        from planning_tasks.task import Task
    except ImportError:
        # If it still fails, define dummy classes for structure.
        # This allows the code to be parsed but will not function in a real planner.
        print("Warning: Heuristic base class or Task class not found. Using dummy classes.", file=sys.stderr)
        class Heuristic:
            def __init__(self, task): pass
            def __call__(self, node): return 0
        class Task: # Dummy Task class for type hinting
             def __init__(self, name, facts, initial_state, goals, operators, static): pass


# Helper function to parse PDDL facts represented as strings
def get_parts(fact: str) -> List[str]:
    """Removes parentheses and splits a PDDL fact string into predicate and arguments."""
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    state from the current state in the Blocksworld domain. It works by
    identifying blocks that are not in their final goal configuration and
    blocks that obstruct others from reaching their goal positions. It assigns
    a cost of 2 actions for each block that needs to be moved (one action to
    pick/unstack, one to put/stack) and 1 action if the arm is currently
    holding a block.

    # Assumptions
    - Standard Blocksworld domain with one robotic arm.
    - All actions (pickup, putdown, stack, unstack) have a cost of 1.
    - The goal is specified by a conjunction of `on(A, B)`, `on-table(C)`,
      and `clear(D)` predicates, defining the target towers and clear blocks.
    - The heuristic is designed for Greedy Best-First Search and does not
      need to be admissible (it can overestimate).

    # Heuristic Initialization
    - The constructor parses the goal conditions (`task.goals`) provided by
      the planner.
    - It stores the target configuration: which block should be on which other
      block (`goal_on[block] = target_below`) or on the table
      (`goal_on[block] = 'TABLE'`).
    - It also stores the set of blocks that must be clear in the goal state
      (`goal_clear`).
    - Memoization dictionaries (`memo_correct`, `memo_incorrect`) are initialized
      to cache the results of correctness checks during heuristic evaluation for
      each call to the heuristic.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Current State:** Extract the current configuration from the
        `node.state`. Determine which block is on which (`current_on`),
        what's on top of each block (`current_on_top`), and if the arm is
        holding a block (`current_holding`).
    2.  **Define Correctness:** Implement a recursive function
        `_check_correctness(block, current_on)` that determines if `block`
        is "correctly placed". A block is correctly placed if it rests on the
        entity (block or table) specified in the `goal_on` map, AND that
        entity below it is also correctly placed (recursively checked down to
        the table). Memoization (`self.memo_correct`, `self.memo_incorrect`)
        is used within each call to `__call__` to avoid redundant checks for
        the same block in the same state evaluation.
    3.  **Initialize Cost:** Set heuristic value `h = 0`. Create a set
        `processed_for_moving` to track blocks already assigned a moving cost
        to prevent double counting within this state evaluation.
    4.  **Arm Cost:** If the arm is holding a block (`current_holding`),
        increment `h` by 1 (for the required `putdown` or `stack` action)
        and add the held block to `processed_for_moving`.
    5.  **Block Placement Cost:** Iterate through all unique blocks involved
        in the current state or goal configuration. For each block `B`:
        a.  If `B` is already in `processed_for_moving`, skip it.
        b.  Check if `B` is "correctly placed" using `_check_correctness`.
        c.  If `B` is *not* correctly placed, it needs to be moved. Increment
            `h` by 2 (estimate for `unstack`/`pickup` + `stack`/`putdown`)
            and add `B` to `processed_for_moving`.
        d.  If `B` *is* correctly placed, check the block `X` currently on
            top of it (`current_on_top.get(B)`). If `X` exists, is not already
            processed, and is *not* the block that should be on `B` according
            to the goal (`goal_on.get(X) != B`), then `X` is obstructing `B`
            or is simply in the wrong place relative to the goal stack.
            Increment `h` by 2 (to move `X`) and add `X` to
            `processed_for_moving`.
    6.  **Clear Goal Cost:** Iterate through all blocks `B` that must be
        `clear` in the goal (`self.goal_clear`). Check the block `X`
        currently on top of `B`. If `X` exists and is not already in
        `processed_for_moving`, then `X` prevents `B` from being clear and
        must be moved. Increment `h` by 2 and add `X` to
        `processed_for_moving`. This step ensures blocks obstructing required
        clear states are penalized, even if they weren't caught in step 5d.
    7.  **Return Value:** Return the total calculated cost `h`. This estimates
        the number of actions needed. If the state is a goal state, h will be 0.
    """

    def __init__(self, task: Task):
        """
        Initializes the heuristic by parsing goal conditions from the task.
        """
        super().__init__(task)
        self.goals: frozenset[str] = task.goals
        # Static facts are generally not needed for basic Blocksworld heuristics.
        # self.static: frozenset[str] = task.static

        # Stores block -> target_below ('TABLE' or another block name) from goal
        self.goal_on: Dict[str, str] = {}
        # Stores blocks that must be clear in the goal state
        self.goal_clear: Set[str] = set()

        for fact in self.goals:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                # Goal: parts[1] should be on parts[2]
                self.goal_on[parts[1]] = parts[2]
            elif predicate == 'on-table' and len(parts) == 2:
                # Goal: parts[1] should be on the table
                self.goal_on[parts[1]] = 'TABLE'
            elif predicate == 'clear' and len(parts) == 2:
                # Goal: parts[1] should be clear
                self.goal_clear.add(parts[1])

        # Memoization caches for the correctness check within a single __call__
        # These need to be cleared at the start of each __call__
        self.memo_correct: Dict[str, bool] = {}
        self.memo_incorrect: Dict[str, bool] = {}

    def _check_correctness(self, block: str, current_on: Dict[str, str]) -> bool:
        """
        Recursively checks if a block is correctly placed relative to the
        goal configuration below it, down to the table. Uses memoization
        stored in self.memo_correct and self.memo_incorrect.
        """
        # Base case: The table is always a correct foundation.
        if block == 'TABLE':
            return True

        # Check memoization caches
        if block in self.memo_correct:
            return True
        if block in self.memo_incorrect:
            return False

        # Get the target entity below this block in the goal configuration.
        goal_target = self.goal_on.get(block)

        # If this block has no specified goal position below it (i.e., it's not
        # part of a goal tower structure like on(block, X) or on-table(block)),
        # it cannot be considered "correctly placed" in the context of building
        # the goal towers from the bottom up.
        if goal_target is None:
            self.memo_incorrect[block] = True
            return False

        # Get the entity currently below this block in the state.
        # If the block isn't in current_on, it might be held or not exist in state.
        # If it's not on anything, it cannot match a goal target.
        current_target = current_on.get(block)
        if current_target is None:
             self.memo_incorrect[block] = True
             return False

        # If the block is not resting on the entity specified in the goal,
        # it's incorrectly placed.
        if goal_target != current_target:
            self.memo_incorrect[block] = True
            return False

        # The block is on the correct entity; now recursively check if the
        # entity below it is also correctly placed.
        if self._check_correctness(goal_target, current_on):
            # If the foundation below is correct, this block is also correctly placed.
            self.memo_correct[block] = True
            return True
        else:
            # If the foundation below is incorrect, this block cannot be correctly placed.
            self.memo_incorrect[block] = True
            return False

    def __call__(self, node) -> int:
        """
        Calculates the heuristic value for the given state node.
        Estimates the number of actions required to reach the goal.
        """
        state: frozenset[str] = node.state

        # --- Parse current state ---
        current_on: Dict[str, str] = {}  # block -> block_below / 'TABLE'
        current_on_top: Dict[str, Optional[str]] = {} # block -> block_on_top / None
        current_holding: Optional[str] = None # block being held, or None
        all_blocks_in_state: Set[str] = set() # Track all blocks mentioned

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            # Basic validation for expected number of parts
            if pred == 'on' and len(parts) == 3:
                block_on_top, block_below = parts[1], parts[2]
                current_on[block_on_top] = block_below
                current_on_top[block_below] = block_on_top
                all_blocks_in_state.add(block_on_top)
                all_blocks_in_state.add(block_below)
            elif pred == 'on-table' and len(parts) == 2:
                block = parts[1]
                current_on[block] = 'TABLE'
                all_blocks_in_state.add(block)
            elif pred == 'clear' and len(parts) == 2:
                # We don't directly use current_clear in the calculation,
                # but knowing it exists is useful for state understanding.
                all_blocks_in_state.add(parts[1]) # Ensure clear blocks are known
            elif pred == 'holding' and len(parts) == 2:
                current_holding = parts[1]
                all_blocks_in_state.add(current_holding)
            elif pred == 'arm-empty':
                pass # This is implicitly handled by current_holding being None

        # Ensure all blocks known in the state have an entry in current_on_top
        # Initialize blocks known only from 'clear' or 'holding' as having nothing on top
        for block in all_blocks_in_state:
            if block not in current_on_top:
                 current_on_top[block] = None # Mark as having nothing on top

        # --- Calculate heuristic ---
        h: int = 0
        # Track blocks already penalized for needing to be moved in this state evaluation
        processed_for_moving: Set[str] = set()

        # 1. Cost for holding a block (requires one action to place it)
        if current_holding is not None:
            h += 1
            processed_for_moving.add(current_holding)

        # Get all unique blocks involved in the problem (state or goal)
        # This ensures we consider blocks needed for the goal even if not present initially
        all_blocks = all_blocks_in_state | set(self.goal_on.keys()) | self.goal_clear
        all_blocks.discard('TABLE') # Make sure 'TABLE' isn't treated as a block

        # Clear memoization caches for this specific state evaluation
        self.memo_correct.clear()
        self.memo_incorrect.clear()

        # 2. Check placement cost for every block
        for block in all_blocks:
            # Skip if already accounted for (e.g., held block, or moved off obstruction)
            if block in processed_for_moving:
                continue

            # Check if the block is correctly placed relative to the goal stack below it
            # Need to handle blocks that might be in the goal but not in the current state's
            # current_on map (e.g., if they should be on table but are held).
            # The _check_correctness handles this by returning False if current_on[block] is None.
            is_block_correct = self._check_correctness(block, current_on)

            if not is_block_correct:
                # Block is misplaced relative to the goal tower structure below it.
                # Estimate 2 actions (pickup/unstack + putdown/stack).
                h += 2
                processed_for_moving.add(block)
            else:
                # Block is correctly placed relative to the tower below it.
                # Check if an incorrect block is currently placed on top of it.
                block_on_top = current_on_top.get(block)
                if block_on_top is not None and block_on_top not in processed_for_moving:
                    # Check if block_on_top *should* be on 'block' according to the goal config.
                    goal_target_for_top = self.goal_on.get(block_on_top)
                    if goal_target_for_top != block:
                        # The block on top is incorrect (or 'block' should be clear in goal).
                        # Estimate 2 actions to move block_on_top (unstack + putdown).
                        h += 2
                        processed_for_moving.add(block_on_top)

        # 3. Final check for blocks obstructing required clear blocks
        # This catches cases where a block B needs to be clear, and the block X
        # on top of it might be "correctly placed" relative to B but still needs moving
        # because B must be clear.
        for block_to_be_clear in self.goal_clear:
            block_on_top = current_on_top.get(block_to_be_clear)
            if block_on_top is not None and block_on_top not in processed_for_moving:
                # This block_on_top prevents 'block_to_be_clear' from being clear.
                # Estimate 2 actions to move it (unstack + putdown).
                h += 2
                processed_for_moving.add(block_on_top) # Mark as processed

        # The final heuristic value estimates the number of actions.
        return h
