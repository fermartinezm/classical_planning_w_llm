# Assume Heuristic base class is available
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         # Handle potential non-string inputs gracefully
         return []
    return fact[1:-1].split()

# Helper function to recursively check if a block is in its correct goal stack position
def is_goal_positioned(block, goal_pos, memo, current_state_on):
    """
    Checks if block is in its correct goal position relative to the block below it,
    AND the block below it is also correctly positioned relative to the goal stack.
    Uses memoization to avoid redundant calculations.

    Args:
        block (str): The name of the block to check.
        goal_pos (dict): Map from block to the block it should be on (or 'table') in the goal.
        memo (dict): Memoization dictionary {block: bool}.
        current_state_on (dict): Map from block to the block it is currently on (or 'table', 'holding').

    Returns:
        bool: True if the block is in its correct goal stack position relative to the base, False otherwise.
    """
    if block in memo:
        return memo[block]

    # If block is not part of a specific goal stack defined by goal_pos,
    # it doesn't contribute to this part of the heuristic calculation (Step 1).
    # We consider it "correctly positioned" relative to a goal stack if it's not in one.
    if block not in goal_pos:
         memo[block] = True
         return True

    desired_pos = goal_pos[block]
    current_pos = current_state_on.get(block)

    # If the block is held, it's definitely not in its goal position.
    if current_pos == 'holding':
        memo[block] = False
        return False

    # Check if the block is on the correct block or table.
    if current_pos != desired_pos:
        memo[block] = False
        return False

    # If it's on the table and goal is on table, it's correctly positioned relative to base.
    if desired_pos == 'table':
        memo[block] = True
        return True
    else:
        # If it's on another block, check if that block is goal_positioned.
        # The block it should be on (desired_pos) must itself be correctly positioned.
        result = is_goal_positioned(desired_pos, goal_pos, memo, current_state_on)
        memo[block] = result
        return result

# Helper function to get the current position map
def get_current_state_on_map(state):
    """Maps each block to the block it is on, or 'table', or 'holding'."""
    current_state_on = {}
    for fact in state:
        parts = get_parts(fact)
        if not parts: continue # Skip malformed facts
        if parts[0] == 'on' and len(parts) == 3:
            block, under_block = parts[1], parts[2]
            current_state_on[block] = under_block
        elif parts[0] == 'on-table' and len(parts) == 2:
            block = parts[1]
            current_state_on[block] = 'table'
        elif parts[0] == 'holding' and len(parts) == 2:
            block = parts[1]
            current_state_on[block] = 'holding'
    return current_state_on


class blocksworldHeuristic: # Inherit from Heuristic if available
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the distance to the goal by counting three types of
    discrepancies from the goal state:
    1. Blocks that are part of a goal stack but are not currently in their
       correct position relative to the base of that goal stack.
    2. Blocks that are currently stacked on top of another block, but are
       not supposed to be on that specific block according to the goal state.
    3. The arm is not empty when the goal requires it to be empty.

    # Assumptions
    - The goal state defines one or more stacks of blocks, or blocks on the table.
    - All blocks involved in goal stacks are mentioned in goal 'on' or 'on-table' predicates.
    - The heuristic assumes standard Blocksworld actions (pickup, putdown, stack, unstack).
    - The state representation includes explicit '(clear x)', '(on-table x)', '(on x y)', '(arm-empty)', and '(holding x)' facts.

    # Heuristic Initialization
    - Parses the goal predicates to determine the desired position for each block
      that is part of a goal stack (`self.goal_pos`).
    - Stores the set of all goal `(on x y)` facts for quick lookup (`self.goal_on_facts`).
    - Checks if `(arm-empty)` is a goal condition (`self.goal_arm_empty`).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current position of every block (on another block, on the table, or held).
       Store this in a map `current_state_on`.
    2. Initialize the heuristic value `h` to 0.
    3. **Count blocks not in their goal stack position:**
       - For each block `b` that is required to be in a specific position
         within a goal stack (i.e., `b` is a key in `self.goal_pos`):
         - Recursively check if `b` is in its correct position relative to the
           block below it *and* if the block below it is also correctly positioned
           relative to the goal stack base (using `is_goal_positioned` helper).
         - If `b` is *not* correctly positioned within its goal stack, increment `h`.
       - Memoization is used in the recursive check to handle shared sub-stacks efficiently.
    4. **Count blocks wrongly placed on top:**
       - Iterate through all `(on t b)` facts in the current state.
       - If a fact `(on t b)` exists in the state, but the goal state does *not*
         require `t` to be directly on `b` (i.e., `(on t b)` is not in `self.goal_on_facts`),
         then `t` is wrongly placed on `b`. Increment `h`.
    5. **Penalize non-empty arm if arm-empty is a goal:**
       - If `self.goal_arm_empty` is True and `(arm-empty)` is not true in the state,
         increment `h`.
    6. The total heuristic value is `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal information.

        Args:
            task: The planning task object containing initial state, goals, etc.
        """
        self.goals = task.goals

        # Parse goal predicates to build the desired stack structure
        self.goal_pos = {} # Map block -> block_below_it or 'table'
        self.goal_on_facts = set() # Set of goal (on x y) strings
        self.goal_arm_empty = False

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            if parts[0] == 'on' and len(parts) == 3:
                block, under_block = parts[1], parts[2]
                self.goal_pos[block] = under_block
                self.goal_on_facts.add(goal) # Store the exact goal fact string
            elif parts[0] == 'on-table' and len(parts) == 2:
                block = parts[1]
                self.goal_pos[block] = 'table'
            elif parts[0] == 'arm-empty' and len(parts) == 1:
                 self.goal_arm_empty = True
            # Ignore 'clear' goals for building goal_pos, they are implicitly handled
            # by the structure or by counting wrongly placed blocks on top.

        # Collect all blocks that are part of goal stacks
        self.goal_blocks = set(self.goal_pos.keys())


    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.

        Args:
            node: The search node containing the current state.

        Returns:
            int: The estimated heuristic cost to reach a goal state.
        """
        state = node.state

        # Step 1: Build map of current block positions
        current_state_on = get_current_state_on_map(state)

        h = 0

        # Step 2: Count blocks not in their goal stack position
        # We only care about blocks that are part of the goal stacks
        memo = {} # Memoization for is_goal_positioned
        for block in self.goal_blocks:
             if not is_goal_positioned(block, self.goal_pos, memo, current_state_on):
                 h += 1

        # Step 3: Count blocks wrongly placed on top
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == 'on' and len(parts) == 3:
                # Check if this (on t b) fact is NOT in the goal (on x y) facts
                if fact not in self.goal_on_facts:
                    h += 1

        # Step 4: Penalize non-empty arm if arm-empty is a goal
        if self.goal_arm_empty and "(arm-empty)" not in state:
             h += 1

        return h
