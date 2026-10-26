import sys
from pathlib import Path
# Ensure the base class 'Heuristic' is available.
# It might require adjusting sys.path or proper installation.
# Example: sys.path.append(str(Path(__file__).parent.parent.parent))
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy base class if the import fails (e.g., for standalone testing)
    class Heuristic:
        def __init__(self, task): pass
        def __call__(self, node): raise NotImplementedError


def get_parts(fact: str):
    """
    Helper function to parse a PDDL fact string "(predicate arg1 arg2 ...)"
    into a list ["predicate", "arg1", "arg2", ...].
    Returns an empty list if the fact format is invalid (e.g., not starting/ending with parentheses).
    """
    if isinstance(fact, str) and fact.startswith("(") and fact.endswith(")"):
        return fact[1:-1].split()
    return []


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Blocksworld domain, designed for
    use with Greedy Best-First Search.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    from the current state. It primarily works by identifying blocks that are not
    in their correct final position relative to the block (or table) beneath them.
    Each such "misplaced" block generally requires two actions: one to pick it up
    (or unstack it), and one to put it down (or stack it) in the correct place.
    The heuristic value is calculated as twice the number of misplaced blocks,
    with a minor adjustment if the robot arm is currently holding a block (since
    the pickup/unstack action is already implicitly done).

    # Assumptions
    - The goal configuration is defined by a set of `(on block block)` and
      `(on-table block)` predicates. Any `(clear block)` predicates in the goal
      are assumed to be naturally satisfied once the structural goals (`on`,
      `on-table`) are met, and are not directly counted by this heuristic.
    - All actions in the Blocksworld domain have a uniform cost of 1.
    - The heuristic is intended for Greedy Best-First Search and does not need
      to be admissible or consistent.
    - The input PDDL task is assumed to be well-formed (e.g., blocks do not
      rest on themselves, goals are achievable).

    # Heuristic Initialization
    - The `__init__` method parses the goal predicates provided in the `task` object.
    - It builds a dictionary `goal_on` where `goal_on[block]` stores the name
      of the object (either another block name string or the special string 'table')
      that `block` should be resting on in the final goal configuration.
    - It also identifies the set `all_blocks` containing all unique block names
      mentioned in the goal or initial state predicates relevant to block positions
      (on, on-table, clear, holding).

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Current State:** The `__call__` method receives a search `node`.
        It analyzes the `node.state` (a frozenset of fact strings) to determine
        the current configuration:
        - It builds a `state_on` dictionary mapping `block -> block/'table'`
          indicating what each block is currently resting on.
        - It identifies `state_holding`, storing the name of the block held by
          the arm, or `None` if the arm is empty (`(arm-empty)` is true).
    2.  **Check Goal Achievement:** It first checks if the current state already
        satisfies all goal predicates (`self.goals <= state`). If so, the goal
        is reached, and the heuristic value is 0.
    3.  **Identify Correct Blocks:** It uses a recursive helper method `_is_correct`
        to determine for each block `b` whether it is "correctly placed".
        - A block `b` is considered correct if it's currently resting on the
          same support (`state_on.get(b)`) as required by the goal
          (`self.goal_on.get(b)`) AND the support itself (if it's another block)
          is also recursively determined to be correct.
        - The base case for the recursion is a block that should be on the table
          (`goal_on[b] == 'table'`) and is currently on the table
          (`state_on[b] == 'table'`).
        - Blocks currently held by the arm (`state_holding`) are explicitly marked
          as *not* correct during this check.
        - Memoization (using a `memo` dictionary) is employed within the
          recursive check to cache the correctness status (True/False) of each
          block, avoiding redundant computations, especially in deep towers.
    4.  **Count Misplaced Blocks:** After determining the correctness status for all
        blocks (populating the `memo` dictionary), the code iterates through all
        known blocks (`self.all_blocks`). Any block whose status in `memo` is
        `False` is counted as "misplaced". The total count is stored in
        `misplaced_count`.
    5.  **Base Heuristic Value:** The initial heuristic estimate `h` is calculated as
        `h = 2 * misplaced_count`. This reflects the typical two moves (one
        pickup/unstack and one putdown/stack) needed for each block that is not
        in its final correct position relative to the structure below it.
    6.  **Adjust for Arm State:**
        - If the arm is currently holding a block (`state_holding is not None`),
          this held block is always considered misplaced (as determined in step 3
          and reflected in `memo`). Since the block is already held, the
          pickup/unstack action is effectively saved for this block. Therefore,
          the heuristic value is decremented by 1 (`h -= 1`).
    7.  **Final Value:** The method returns the calculated heuristic value `h`.
        The logic ensures `h` is non-negative and returns 0 if and only if the
        state is a goal state. A `max(0, h)` safeguard is included for robustness,
        although the core logic should prevent negative values.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing goal conditions and identifying all blocks.

        Args:
            task: The planning task object, containing task.goals, task.initial_state, etc.
        """
        super().__init__(task) # Call base class constructor if necessary
        self.goals = task.goals
        self.goal_on = {} # Stores goal position: block -> block_below / 'table'
        self.all_blocks = set()

        # Parse goal state to find target positions and identify blocks
        for fact in task.goals:
            parts = get_parts(fact)
            if not parts: continue # Skip potential empty strings or malformed facts
            predicate = parts[0]

            if predicate == 'on' and len(parts) == 3:
                block, under_block = parts[1], parts[2]
                # Basic validation: block cannot be on itself
                if block != under_block:
                    self.goal_on[block] = under_block
                    self.all_blocks.add(block)
                    self.all_blocks.add(under_block)
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                self.goal_on[block] = 'table'
                self.all_blocks.add(block)
            elif predicate == 'clear' and len(parts) == 2:
                 # Also consider blocks mentioned only in 'clear' goals
                 # as they are part of the problem's objects
                 self.all_blocks.add(parts[1])
            # 'arm-empty' goal is implicitly handled by the heuristic logic

        # Ensure all blocks mentioned in the initial state are also included
        # This covers blocks that might start correctly placed or are irrelevant to goals
        for fact in task.initial_state:
             parts = get_parts(fact)
             if not parts: continue
             predicate = parts[0]
             # Add blocks from relevant predicates
             if predicate in ['on', 'on-table', 'clear', 'holding'] and len(parts) >= 2:
                 self.all_blocks.add(parts[1])
             # Add the block below in 'on' predicates
             if predicate == 'on' and len(parts) == 3:
                 # Add block below only if it's different from block above
                 if parts[1] != parts[2]:
                     self.all_blocks.add(parts[2])

        # Optional: Add validation here to ensure every block in all_blocks
        # has a defined goal state in self.goal_on, if necessary for robustness.
        # For standard benchmarks, this is usually guaranteed.

    def _is_correct(self, block, state_on, state_holding, memo):
        """
        Recursively checks if a block is correctly placed relative to its support,
        all the way down to the table. Uses memoization via the 'memo' dictionary.

        Args:
            block (str): The name of the block to check.
            state_on (dict): Current mapping of block -> support (block/'table').
            state_holding (str | None): The block currently held by the arm, or None.
            memo (dict): Dictionary to store computed correctness status (block -> bool).

        Returns:
            bool: True if the block is correctly placed, False otherwise.
        """
        # Return cached result if available
        if block in memo:
            return memo[block]

        # A block currently held by the arm is never considered 'correct'
        # in its final position within the tower structure.
        if block == state_holding:
            memo[block] = False
            return False

        current_support = state_on.get(block)
        # Use .get() for goal_on as well, although init should ensure all blocks are keys
        goal_support = self.goal_on.get(block)

        # If the block's goal position is undefined (e.g., block exists but not mentioned
        # in goal structure), or if it's not on anything in the current state
        # (and not held), it's considered incorrect.
        if goal_support is None or current_support is None:
             memo[block] = False
             return False

        # If the block is resting on something different than its goal support, it's incorrect.
        if current_support != goal_support:
            memo[block] = False
            return False

        # Base case: If the block should be on the table ('table' support) and is,
        # it's considered correct *at this level*.
        if goal_support == 'table':
            memo[block] = True
            return True

        # Recursive step: The block is correct only if its required support (goal_support)
        # is also correctly placed. Since current_support == goal_support at this point,
        # we recurse on this support block to check the tower below.
        result = self._is_correct(goal_support, state_on, state_holding, memo)
        memo[block] = result
        return result

    def __call__(self, node):
        """
        Calculates the heuristic value for the given search node's state.

        Args:
            node: The search node object, containing the state (`node.state`).
                  The state is expected to be a frozenset of PDDL fact strings.

        Returns:
            int: The estimated cost (number of actions) to reach the goal state.
                 Returns 0 if the state in the node is already a goal state.
        """
        state = node.state

        # Check if the current state is already a goal state
        # Assumes task.goals is a frozenset or set of goal facts
        if self.goals <= state:
             return 0

        state_on = {} # block -> block / 'table'
        state_holding = None

        # Parse the current state to find block positions and arm status
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]

            if predicate == 'on' and len(parts) == 3:
                # Basic validation: block cannot be on itself
                if parts[1] != parts[2]:
                    state_on[parts[1]] = parts[2]
            elif predicate == 'on-table' and len(parts) == 2:
                state_on[parts[1]] = 'table'
            elif predicate == 'holding' and len(parts) == 2:
                state_holding = parts[1]
            # We don't explicitly need 'clear' or 'arm-empty' facts for this heuristic calculation

        memo = {}
        # Populate the memoization table by checking correctness for all blocks
        # The results (True/False) are stored in 'memo'
        for block in self.all_blocks:
            self._is_correct(block, state_on, state_holding, memo)

        misplaced_count = 0
        # Count the number of blocks marked as incorrect (misplaced)
        for block in self.all_blocks:
             # If block not in memo (shouldn't happen) or memo[block] is False
             if not memo.get(block, False): # Default to False if key missing
                 misplaced_count += 1

        # Base heuristic value: 2 actions per misplaced block
        h = 2 * misplaced_count

        # Adjust heuristic if the arm is holding a block
        if state_holding is not None:
             # A held block is always considered misplaced (should be False in memo).
             # We save one action (pickup/unstack) because it's already held.
             # Verify it was marked as False in memo for consistency before adjusting.
             if not memo.get(state_holding, True): # Check if memo[state_holding] is False
                 h -= 1
             # else: # This case implies held block was somehow marked correct - internal logic error?
                 # If this happens, maybe don't adjust 'h' or log a warning.
                 # Sticking to h -= 1 assumes _is_correct handles held blocks correctly.
                 # print(f"Warning: Held block {state_holding} not marked as incorrect by _is_correct.")
                 pass

        # Return the final heuristic value, ensuring it's non-negative
        # This safeguard handles potential edge cases, though the logic aims for h >= 0.
        return max(0, h)
