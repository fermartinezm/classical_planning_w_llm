import re
from heuristics.heuristic_base import Heuristic # Assuming this base class is available

# Helper function to parse PDDL facts represented as strings
def get_parts(fact_str):
    """
    Extracts the predicate and arguments from a PDDL fact string.

    Args:
        fact_str: A string representing a PDDL fact, e.g., '(on b1 b2)'.

    Returns:
        A list containing the predicate name followed by its arguments.
        Returns an empty list if the fact string is malformed or empty.
        Example: '(on b1 b2)' -> ['on', 'b1', 'b2']
                 '(clear b1)' -> ['clear', 'b1']
                 '(arm-empty)' -> ['arm-empty']
    """
    fact_str = fact_str.strip()
    # Basic check for parentheses and non-empty content
    if not fact_str.startswith("(") or not fact_str.endswith(")") or len(fact_str) <= 2:
        # Handle edge cases like "()" or malformed strings
        if fact_str != "()":
             # Optionally print a warning for unexpected formats
             # print(f"Warning: Malformed fact string encountered: '{fact_str}'")
             pass
        return []
    # Extract content within parentheses and split by whitespace
    content = fact_str[1:-1]
    return content.split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    from a given state in the Blocksworld domain. It works by comparing the current
    arrangement of blocks to the desired goal configuration. It identifies blocks
    that are not in their final correct position relative to the goal towers (where
    a block is considered "correct" only if it's in its goal place and the block
    directly below it is also correct). The heuristic value is primarily based on
    the count of misplaced blocks and blocks that are currently obstructing these
    misplaced blocks (i.e., sitting on top of them). This heuristic is designed
    for guiding Greedy Best-First Search, prioritizing informativeness over
    admissibility (it may overestimate the true cost).

    # Assumptions
    - The goal state is primarily defined by a set of `on(X, Y)` and `on-table(X)`
      predicates, which specify the target towers or blocks on the table.
    - `clear(X)` goals are considered implicitly satisfied if the block `X` is
      at the top of its respective goal tower or correctly placed on the table
      as per the goal structure.
    - All blocks present in the initial state are relevant to the problem. Any
      block not explicitly mentioned in an `on` or `on-table` goal predicate is
      assumed to have the table as its final destination.
    - The cost of each action (pickup, putdown, stack, unstack) is uniform (1).

    # Heuristic Initialization
    - The constructor (`__init__`) parses the goal predicates provided in the task
      definition. It builds a dictionary (`goal_on`) representing the target
      configuration, mapping each block to the object below it (`block_below` or
      the special string `'table'`).
    - It also collects the set of all unique block names mentioned anywhere in
      the goal facts to understand the scope of relevant objects.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Current State:** Given a state node, the `__call__` method first
        parses the set of facts representing the current state. It determines:
        - The current location of each block: Is it on another block, on the
          table, or held by the arm? Stored in `current_on`.
        - Which block is immediately above each other block? Stored in
          `current_above`.
        - Which block, if any, is currently held by the arm (`held_block`).
        - The complete set of all unique blocks involved in the problem (combining
          those from the goal and the current state).
    2.  **Identify Correctly Placed Blocks:** The core of the heuristic involves
        identifying which blocks are already "correctly placed". A block `B` is
        defined as correctly placed if:
        a. Its current position (`current_on[B]`) matches its goal position
           (`goal_on[B]`).
        b. AND, if its goal position is on top of another block `C`
           (`goal_on[B] == C`), then block `C` must *also* be correctly placed.
        This check is performed recursively using the `_is_correctly_placed`
        helper method, which uses memoization (a cache) to avoid redundant
        computations. The set of all correctly placed blocks is stored.
    3.  **Calculate Base Cost:** The heuristic value `h` is initialized to 0.
        - If the arm is currently holding a block (`held_block` is not None), `h`
          is incremented by 1. This accounts for the mandatory action (either
          `putdown` or `stack`) needed to place the held block.
    4.  **Calculate Cost for Misplaced Blocks:** The heuristic iterates through
        all blocks identified in the problem. For each block `B` that is *not*
        in the set of `correctly_placed_blocks`:
        - `h` is incremented by 2. This represents the estimated cost of moving
          this misplaced block: one action to pick it up (or unstack it) and one
          action to place it down (or stack it) in its correct final position.
    5.  **Calculate Cost for Clearing:** The heuristic then considers blocks that
        are obstructing misplaced blocks. It iterates through all blocks `A` that
        are currently stacked on top of another block `B` (i.e., `current_on[A] == B`).
        - If the block `B` underneath `A` is found to be *not* correctly placed,
          it means `A` is currently in the way and must be moved before `B` can
          be corrected.
        - In this case, `h` is incremented by 2. This accounts for the actions
          needed to move the obstructing block `A` temporarily out of the way
          (one `unstack` action, one `putdown` or `stack` action elsewhere).
    6.  **Final Value:** The computed value `h` is returned. A final check ensures
        that if `h` is 0 but the state is not actually a goal state (an edge case,
        perhaps only the arm state differs), the heuristic returns 1 instead of 0,
        as the heuristic value should only be 0 for true goal states.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing the goal configuration.

        Args:
            task: The planning task object, containing goals, initial state, etc.
        """
        self.goals = task.goals
        self.static = task.static # Usually empty for standard Blocksworld

        # Build goal structure: block -> block below it or 'table'
        self.goal_on = {}
        # Keep track of all blocks mentioned in goal facts
        self.all_goal_mentioned_blocks = set()

        for fact in self.goals:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts

            predicate = parts[0]
            args = parts[1:]

            # Parse 'on' goal facts
            if predicate == 'on' and len(args) == 2:
                block_on_top, block_below = args
                self.goal_on[block_on_top] = block_below
                self.all_goal_mentioned_blocks.add(block_on_top)
                self.all_goal_mentioned_blocks.add(block_below)
            # Parse 'on-table' goal facts
            elif predicate == 'on-table' and len(args) == 1:
                block = args[0]
                self.goal_on[block] = 'table'
                self.all_goal_mentioned_blocks.add(block)
            # Track blocks mentioned in 'clear' goals as well
            elif predicate == 'clear' and len(args) == 1:
                 self.all_goal_mentioned_blocks.add(args[0])
            # We typically ignore 'arm-empty' goals for heuristic calculation


    def _get_state_details(self, state):
        """
        Parses the current state facts to extract block positions and relationships.

        Args:
            state: A frozenset of strings representing the facts true in the current state.

        Returns:
            A tuple containing:
            - current_on (dict): Maps block -> block/table/'arm' below it.
            - current_above (dict): Maps block -> block directly on top of it.
            - held_block (str | None): The name of the block held, or None.
            - all_blocks (set): A set of all unique block names found in the state
                                and goals.
        """
        current_on = {}    # block -> block below or 'table' or 'arm'
        current_above = {} # block -> block on top
        held_block = None
        state_blocks = set() # Blocks found in the current state facts

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts

            predicate = parts[0]
            args = parts[1:]

            # Process 'on' facts
            if predicate == 'on' and len(args) == 2:
                block_on_top, block_below = args
                current_on[block_on_top] = block_below
                current_above[block_below] = block_on_top
                state_blocks.add(block_on_top)
                state_blocks.add(block_below)
            # Process 'on-table' facts
            elif predicate == 'on-table' and len(args) == 1:
                block = args[0]
                current_on[block] = 'table'
                state_blocks.add(block)
            # Process 'clear' facts (mainly to discover block names)
            elif predicate == 'clear' and len(args) == 1:
                state_blocks.add(args[0])
            # Process 'holding' fact
            elif predicate == 'holding' and len(args) == 1:
                held_block = args[0]
                current_on[held_block] = 'arm' # Mark the held block's location
                state_blocks.add(held_block)
            # Note 'arm-empty' state if needed, but not directly used here
            elif predicate == 'arm-empty':
                pass

        # Combine blocks found in the state with those known from goals
        all_blocks = self.all_goal_mentioned_blocks.union(state_blocks)

        return current_on, current_above, held_block, all_blocks

    def _is_correctly_placed(self, block, current_on, correctly_placed_cache):
        """
        Recursively checks if a block is correctly placed relative to the goal.
        Uses memoization (cache) to avoid re-computation.

        Args:
            block (str): The name of the block to check.
            current_on (dict): The current position map from _get_state_details.
            correctly_placed_cache (dict): Cache to store results of previous checks.

        Returns:
            bool: True if the block is correctly placed, False otherwise.
        """
        # Return cached result if available
        if block in correctly_placed_cache:
            return correctly_placed_cache[block]

        # Determine the goal position for this block.
        # If not specified in 'on' or 'on-table' goals, assume goal is 'table'.
        goal_target = self.goal_on.get(block, 'table')

        # Get the current position of the block from the state.
        current_target = current_on.get(block)

        # A block cannot be correct if its current position is unknown
        # or if its current position doesn't match its goal position.
        if current_target is None or current_target != goal_target:
            correctly_placed_cache[block] = False
            return False

        # Current position matches the goal position.
        # If the goal was 'table', the block is correctly placed.
        if goal_target == 'table':
            correctly_placed_cache[block] = True
            return True
        else:
            # The goal was to be on top of another block ('goal_target').
            # We must recursively check if the block below is also correctly placed.
            below_block_correct = self._is_correctly_placed(
                goal_target, current_on, correctly_placed_cache
            )
            # Store and return the result based on the block below.
            correctly_placed_cache[block] = below_block_correct
            return below_block_correct


    def __call__(self, node):
        """
        Calculate the heuristic value for the given state node.

        Args:
            node: A node object containing the state (`node.state`). The state is
                  expected to be a frozenset of PDDL fact strings.

        Returns:
            An integer estimate of the cost to reach the goal from the node's state.
        """
        state = node.state

        # If the current state already satisfies all goal conditions, heuristic is 0.
        if self.goals <= state:
            return 0

        # Parse the current state to get block positions and relationships.
        current_on, current_above, held_block, all_blocks = self._get_state_details(state)

        # Cache for memoizing the results of the recursive correctness check.
        correctly_placed_cache = {}

        # Determine the set of correctly placed blocks by checking all blocks.
        # Need to run the check on all blocks first to populate the cache correctly
        # due to recursive dependencies.
        for block in all_blocks:
            self._is_correctly_placed(block, current_on, correctly_placed_cache)

        # Now, build the set of blocks confirmed to be correctly placed.
        correctly_placed_blocks = {
            block for block in all_blocks
            if correctly_placed_cache.get(block, False) # Default to False if check failed
        }

        # Initialize heuristic value.
        h = 0

        # Step 1: Add cost if the arm is holding a block (needs 1 action to place).
        if held_block is not None:
            h += 1

        # Step 2: Add cost for each block that is not correctly placed (needs 2 actions).
        misplaced_blocks = all_blocks - correctly_placed_blocks
        h += len(misplaced_blocks) * 2

        # Step 3: Add cost for clearing blocks that are on top of misplaced blocks.
        # Iterate through all blocks that are currently on top of something.
        for block_on_top in all_blocks:
            block_below = current_on.get(block_on_top)
            # Check if it's actually on another block (not table or arm)
            if block_below is not None and block_below != 'table' and block_below != 'arm':
                 # Check if the block it's sitting on is misplaced
                 if block_below not in correctly_placed_blocks:
                     # If A is on B, and B is misplaced, A needs moving (2 actions).
                     h += 2

        # Final check: Ensure heuristic is non-zero for non-goal states.
        # If h is 0 but the state doesn't satisfy goals, return 1.
        if h == 0 and not (self.goals <= state):
             return 1

        return h
