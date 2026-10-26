import re
from heuristics.heuristic_base import Heuristic
# Note: The Task class definition is not needed for the heuristic code itself,
# but the heuristic relies on the structure of the task object passed to __init__
# and the node object passed to __call__.

# Helper function to parse PDDL facts
def get_parts(fact):
    """
    Extracts the predicate and arguments from a PDDL fact string.
    Example: '(on b1 b2)' -> ('on', ['b1', 'b2'])
             '(clear b1)' -> ('clear', ['b1'])
             '(arm-empty)' -> ('arm-empty', [])
    Handles potential extra whitespace.
    Raises ValueError for invalid formats like missing parentheses or empty content.
    """
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
        raise ValueError(f"Invalid fact format (missing parentheses): {fact}")
    content = fact[1:-1].strip()
    # Allow predicates with no arguments like (arm-empty)
    if not content:
         raise ValueError(f"Invalid fact format (empty content): {fact}")
    parts = content.split()
    predicate = parts[0]
    args = parts[1:]
    return predicate, args

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    state by counting the number of blocks that are not in their final
    correct position, considering the entire stack below them. It assigns
    a cost of 2 to each "incorrect" block (representing one pick/unstack
    and one put/stack operation). An additional cost of 1 is added if the
    robot arm is currently holding a block. This heuristic is designed for
    use with Greedy Best-First Search and is not necessarily admissible.

    # Assumptions
    - The goal primarily consists of `on(block, support)` or
      `on-table(block)` predicates defining the target configuration.
    - `clear` predicates in the goal are ignored by this heuristic calculation,
      as `clear` status is usually a consequence of the `on`/`on-table` state
      in a goal configuration.
    - All blocks relevant to the goal structure are mentioned in the goal's
      `on` or `on-table` predicates. Blocks not mentioned are assumed to not
      constrain the goal state for this heuristic's purpose (their final
      position doesn't matter, or they should be implicitly moved out of the way).
    - The heuristic is designed for Greedy Best-First Search and does not
      need to be admissible (it can overestimate the true cost).

    # Heuristic Initialization
    - The constructor (`__init__`) parses the `task.goals` (a set of goal facts).
    - It builds a dictionary `self.goal_on` which maps each block mentioned in an
      `on` or `on-table` goal to the object it should be resting on. The support
      is represented by the name of the block below it, or the special string
      'TABLE' if the block should be on the table.
    - It also collects all unique object (block) names mentioned in these relevant
      goal facts into a set `self.goal_objects`. This helps identify which blocks
      have a defined target position.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Current State:** The `__call__` method receives a `node` object.
        It extracts the current state (`node.state`, a frozenset of fact strings).
        It parses these facts to determine:
        - `current_on`: A dictionary mapping block -> support ('TABLE' or other block name)
          representing the current physical arrangement of blocks.
        - `held_block`: The name of the block currently held by the arm (or `None` if
          the arm is empty).
        - `state_objects`: A set of all object names appearing in the current state facts.
    2.  **Identify All Blocks:** Combine the set of objects mentioned in the goal
        (`self.goal_objects`) with objects found in the current state (`state_objects`)
        to get a complete set `all_blocks` of blocks to consider.
    3.  **Check Goal Reached:** As an optimization, if the current state already satisfies
        all goal conditions (`self.goals <= state`), the heuristic value is 0, and we return
        immediately.
    4.  **Define Correctness (Recursive Check):** A private helper method
        `_is_correct(block, current_on, memo)` recursively determines if a `block`
        is "correct". A block is correct if and only if:
        - It is currently resting on the correct support (as defined in `self.goal_on`).
        - AND the support itself (if it's another block) is also correct.
        - The base case is that the 'TABLE' is always a correct support.
        - Memoization (`memo` dictionary) is used to cache results for blocks already
          checked during the current heuristic calculation, avoiding redundant computations,
          especially in tall towers.
    5.  **Count Incorrect Blocks:** Initialize `incorrect_count = 0`. Iterate through all
        unique blocks identified in `all_blocks`. For each block that is *not* currently
        held by the arm:
        - Check if this block has a defined goal position (i.e., `block` is a key in
          `self.goal_on`).
        - If yes, call `_is_correct` to determine if it and the tower below it are
          correctly positioned according to the goal configuration.
        - If `_is_correct` returns `False`, increment `incorrect_count`. Blocks without
          a defined goal position are ignored in this count, unless they are found to be
          interfering with a goal block during the recursive check.
    6.  **Calculate Base Heuristic:** Set the heuristic value `h = 2 * incorrect_count`.
        This estimates two moves (one pick/unstack, one put/stack) for each block
        identified as not being in its final correct tower position relative to the goal.
    7.  **Arm Adjustment:** If `held_block` is not `None` (meaning the arm is holding a block),
        add 1 to `h`. This accounts for the mandatory `putdown` or `stack` action needed
        to place the held block somewhere.
    8.  **Return Value:** Return the calculated heuristic value `h`. The logic ensures `h`
        is always non-negative.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing goal conditions from the task.

        Args:
            task: The planning task object, containing task.goals.
        """
        self.goals = task.goals
        # self.static = task.static # Static facts are typically not used in Blocksworld heuristics

        # --- Heuristic Initialization ---
        self.goal_on = {} # Maps block -> block/TABLE below it in the goal
        self.goal_objects = set() # Stores all object names mentioned in relevant goals

        for fact in self.goals:
            try:
                predicate, args = get_parts(fact)
                # Parse 'on' goals: map top block to bottom block
                if predicate == 'on' and len(args) == 2:
                    top, bottom = args
                    self.goal_on[top] = bottom
                    self.goal_objects.add(top)
                    self.goal_objects.add(bottom)
                # Parse 'on-table' goals: map block to 'TABLE'
                elif predicate == 'on-table' and len(args) == 1:
                    obj = args[0]
                    self.goal_on[obj] = 'TABLE'
                    self.goal_objects.add(obj)
                # Ignore other goal predicates like 'clear', 'arm-empty', 'holding'
                # as they are either consequences or not directly used in this heuristic logic.
            except ValueError:
                # Silently ignore malformed goal facts, or add logging if needed.
                # print(f"Warning: Could not parse goal fact: {fact}")
                pass

    def _is_correct(self, block, current_on, memo):
        """
        Recursively checks if a block is in its correct final position,
        including the correctness of the stack below it. Uses memoization.

        Args:
            block (str): The name of the block to check.
            current_on (dict): Dictionary mapping block -> support ('TABLE' or block)
                               representing the current stack configuration.
            memo (dict): Dictionary used for memoization of results during a single
                         heuristic calculation.

        Returns:
            bool: True if the block and its support structure down to the table
                  match the goal configuration, False otherwise.
        """
        # Base case: The table is always considered a correct support.
        if block == 'TABLE':
            return True
        # Return memoized result if this block has already been evaluated.
        if block in memo:
            return memo[block]

        # Find what the block should be on according to the goal configuration.
        goal_support = self.goal_on.get(block)
        # Find what the block is currently resting on.
        current_support = current_on.get(block)

        # A block is considered incorrect if:
        # 1. Its goal position is not defined in self.goal_on (i.e., its final
        #    position doesn't matter for the specified goals), OR
        # 2. Its current support object/table does not match its goal support.
        if goal_support is None or goal_support != current_support:
            memo[block] = False
            return False

        # If the block is currently on the correct support object/table,
        # its overall correctness depends recursively on the correctness of that support.
        # Note: goal_support cannot be None here due to the check above.
        result = self._is_correct(goal_support, current_on, memo)
        memo[block] = result
        return result

    def __call__(self, node):
        """
        Calculates the heuristic value for a given state node.

        Args:
            node: The search node containing the state (node.state) to evaluate.
                  node.state is expected to be a frozenset of PDDL fact strings.

        Returns:
            int: The estimated cost (number of actions) to reach the goal state.
                 Returns 0 if the current state is already a goal state.
        """
        state = node.state

        # --- Optimization: Check if goal is already reached ---
        # If all goal facts are present in the current state, the cost is 0.
        if self.goals <= state:
             return 0

        # --- Step 1: Parse Current State ---
        current_on = {} # Maps block -> block/TABLE below it currently
        held_block = None
        state_objects = set()

        for fact in state:
            try:
                predicate, args = get_parts(fact)
                # Collect all objects mentioned in the state facts
                state_objects.update(args)
                # Record 'on' relationships
                if predicate == 'on' and len(args) == 2:
                    current_on[args[0]] = args[1]
                # Record 'on-table' relationships
                elif predicate == 'on-table' and len(args) == 1:
                    current_on[args[0]] = 'TABLE'
                # Record which block is being held, if any
                elif predicate == 'holding' and len(args) == 1:
                    held_block = args[0]
                    # Ensure the held block is included in the set of objects
                    state_objects.add(held_block)
            except ValueError:
                 # Silently ignore malformed state facts, or add logging.
                 # print(f"Warning: Could not parse state fact: {fact}")
                 pass

        # --- Step 2: Identify All Blocks ---
        # Combine objects mentioned in relevant goals and the current state
        # to get the full set of blocks to consider.
        all_blocks = self.goal_objects.union(state_objects)

        # --- Step 5: Count Incorrect Blocks ---
        incorrect_count = 0
        # Memoization dictionary for the _is_correct checks within this call
        memo = {}

        # Iterate through all unique blocks involved in the problem
        for block in all_blocks:
            # The correctness of the held block is not checked directly here;
            # its state is handled by the arm adjustment cost.
            if block == held_block:
                continue

            # Only evaluate blocks that have a specified goal position in self.goal_on.
            # If a block's final position isn't specified in the goal, this heuristic
            # assumes it doesn't need to be moved *unless* it's interfering with
            # another block that *does* need to move (handled by the recursion).
            if block in self.goal_on:
                 # Check if the block is currently in its final correct tower position.
                 # This involves checking the block itself and the stack below it recursively.
                 if not self._is_correct(block, current_on, memo):
                    incorrect_count += 1

        # --- Step 6: Calculate Base Heuristic ---
        # Estimate 2 actions per incorrect block (one to pick/unstack, one to put/stack).
        h = 2 * incorrect_count

        # --- Step 7: Arm Adjustment ---
        # Add 1 to the heuristic cost if the arm is currently holding a block,
        # as at least one 'putdown' or 'stack' action is required for it.
        if held_block is not None:
            h += 1

        # --- Step 8: Return Value ---
        # The calculated heuristic value estimates the remaining actions.
        return h
