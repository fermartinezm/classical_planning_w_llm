from heuristics.heuristic_base import Heuristic
# Assuming Heuristic base class is available in the environment
# from .heuristic_base import Heuristic # Depending on package structure

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty fact string or non-string input defensively
    if not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by whitespace
    return fact[1:-1].split()

def get_objects_from_fact(fact_parts):
    """Extract objects from parsed fact parts."""
    if not fact_parts:
        return set()
    predicate = fact_parts[0]
    if predicate in ["clear", "on-table", "holding"]:
        return {fact_parts[1]} if len(fact_parts) > 1 else set()
    elif predicate == "on":
        return {fact_parts[1], fact_parts[2]} if len(fact_parts) > 2 else set()
    elif predicate == "arm-empty":
        return set() # arm is not a block object
    return set() # Unknown predicate or malformed fact


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the difficulty of reaching the goal state by counting
    the number of blocks that are not in their correct position within the goal
    stacks and the number of blocks that are blocking correctly placed blocks.
    It also accounts for blocks that simply need to be clear in the goal state.

    # Assumptions
    - The goal state defines specific stacks using `on` and `on-table` predicates
      for a subset of blocks.
    - Any block mentioned in a goal `on` or `on-table` predicate is considered
      part of a goal stack structure.
    - Blocks mentioned only in `clear` goal predicates (and not in `on`/`on-table` goals)
      simply need to have nothing on top of them.
    - All blocks relevant to the goal are present in the initial state.

    # Heuristic Initialization
    The heuristic extracts the following information from the task:
    - `goal_support`: A dictionary mapping each block `b` (that appears in a goal
      `on` or `on-table` fact) to the object it should be directly on in the goal
      state ('table' or another block).
    - `goal_blocks_in_stacks`: A set of all blocks that appear as keys in `goal_support`.
      These are the blocks whose position relative to the block below them is specified
      in the goal.
    - `goal_clear_blocks`: A set of all blocks that must be clear in the goal state.
    - `all_blocks`: A set of all block objects present in the initial state.
      This is used to filter goal-related sets to only include existing blocks.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is computed as follows:

    1.  **Identify Current State Structure:**
        -   Determine the current support for each block (`current_support`):
            -   If `(on b y)` is true, `b` is on `y`.
            -   If `(on-table b)` is true, `b` is on the 'table'.
            -   If `(holding b)` is true, `b` is 'held'.
        -   Determine which block is currently on top of each block (`current_on_top`):
            -   If `(on b_on b_under)` is true, `b_on` is on top of `b_under`.

    2.  **Define Helper Function `is_in_goal_stack(b)`:**
        -   This recursive function checks if block `b` is currently in its correct
            goal position relative to the object immediately below it in the goal
            stack, AND if that object is also correctly placed, recursively down
            to the table.
        -   Memoization is used to avoid redundant calculations and infinite recursion.
        -   Base Case: If `b` should be on the 'table' in the goal (`goal_support[b] == 'table'`),
            it's correctly placed if it's currently on the 'table' (`current_support[b] == 'table'`).
        -   Recursive Step: If `b` should be on `b_under` in the goal (`goal_support[b] == b_under`),
            it's correctly placed if it's currently on `b_under` (`current_support[b] == b_under`)
            AND `b_under` is correctly placed in its goal stack (`is_in_goal_stack(b_under)` is True).
        -   If `b` is currently 'held', it's not in its goal stack position.
        -   If `b` is not in `goal_blocks_in_stacks`, it's not part of a defined goal stack structure,
            so `is_in_goal_stack(b)` is considered False (it cannot be in a goal stack position if it's not part of one).

    3.  **Calculate Heuristic Value:**
        -   Initialize heuristic `h = 0`.
        -   **Term 1: Misplaced blocks in goal stacks:** For each block `b` that is part of a goal stack structure (`b` in `goal_blocks_in_stacks`):
            -   If `is_in_goal_stack(b)` is False, increment `h`. This counts blocks that are not in their correct place relative to the goal stack structure below them.
        -   **Term 2: Blocks blocking correctly placed blocks:** For each block `b` that is part of a goal stack structure (`b` in `goal_blocks_in_stacks`):
            -   Find the block `b_on` currently on top of `b` (if any).
            -   If `b_on` exists AND `b` is correctly placed in its goal stack (`is_in_goal_stack(b)` is True) AND `b_on` is *not* the block that should be on `b` in the goal state (i.e., `b_on` is not in `goal_support` or `goal_support[b_on] != b`), then `b_on` is blocking `b`. Increment `h`. This penalizes blocks that are in the way on top of a correctly positioned block in a goal stack.
        -   **Term 3: Blocks that must be clear but aren't (and are not part of goal stacks):** For each block `b` that must be clear in the goal (`b` in `goal_clear_blocks`):
            -   If `b` is *not* part of a goal stack structure (`b` not in `goal_blocks_in_stacks`), and `(clear b)` is not true in the current state, increment `h`. This handles blocks that only have a `clear` requirement in the goal. (Note: If `b` *is* in `goal_blocks_in_stacks` and must be clear, it's the top of a goal stack. If `is_in_goal_stack(b)` is True but it's not clear, the penalty term (Term 2) handles this).

    4.  **Return `h`:** The total heuristic value is the sum of the counts from the three terms.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions related to stacks
        and clear states, and identifying all blocks.
        """
        self.goals = task.goals  # Goal conditions.
        # static_facts = task.static # Static facts are not needed for this heuristic.

        # 1. Build goal_support mapping and identify blocks in goal stacks
        self.goal_support = {}
        self.goal_clear_blocks = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts:
                continue
            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, under_block = parts[1], parts[2]
                self.goal_support[block] = under_block
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                self.goal_support[block] = 'table'
            elif predicate == "clear" and len(parts) == 2:
                block = parts[1]
                self.goal_clear_blocks.add(block)

        self.goal_blocks_in_stacks = set(self.goal_support.keys())

        # 2. Identify all blocks in the problem instance from the initial state
        self.all_blocks = set()
        for fact in task.initial_state:
            parts = get_parts(fact)
            self.all_blocks.update(get_objects_from_fact(parts))

        # Filter goal sets to only include blocks present in the initial state
        self.goal_blocks_in_stacks = {b for b in self.goal_blocks_in_stacks if b in self.all_blocks}
        self.goal_clear_blocks = {b for b in self.goal_clear_blocks if b in self.all_blocks}
        self.goal_support = {b: support for b, support in self.goal_support.items() if b in self.all_blocks and (support == 'table' or support in self.all_blocks)}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify Current State Structure
        current_support = {}
        current_on_top = {}
        arm_is_empty = False

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, under_block = parts[1], parts[2]
                current_support[block] = under_block
                current_on_top[under_block] = block
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                current_support[block] = 'table'
            elif predicate == "holding" and len(parts) == 2:
                block = parts[1]
                current_support[block] = 'held' # Use 'held' to indicate it's in the arm
            elif predicate == "arm-empty":
                arm_is_empty = True # Not directly used in this heuristic calculation, but good to parse

        # 2. Define Helper Function is_in_goal_stack(b) with memoization
        memo_is_in_goal_stack = {}

        def is_in_goal_stack(b):
            """
            Checks if block b is currently in its correct goal stack position,
            recursively down to the table.
            """
            # If b is not part of a defined goal stack structure, it cannot be in one.
            if b not in self.goal_blocks_in_stacks:
                 # This case should ideally not be reached for blocks in goal_blocks_in_stacks
                 # but handles potential inconsistencies or blocks not in goal_support.
                 # For blocks *in* goal_blocks_in_stacks, we proceed.
                 # For blocks *not* in goal_blocks_in_stacks, this function is not called
                 # in the main loop for Term 1 and Term 2.
                 # If somehow called, it means this block isn't defined in goal stacks,
                 # so it's not "in" a goal stack.
                 return False

            if b in memo_is_in_goal_stack:
                return memo_is_in_goal_stack[b]

            # Find current support for block b
            cs = current_support.get(b)

            # If block is held, it's not in its goal stack position
            if cs == 'held':
                memo_is_in_goal_stack[b] = False
                return False

            # Check against goal support
            goal_s = self.goal_support.get(b)

            # Should not happen if b is in goal_blocks_in_stacks, but defensive check
            if goal_s is None:
                 memo_is_in_goal_stack[b] = False
                 return False

            if goal_s == 'table':
                # Goal is on table, check if current is on table
                result = (cs == 'table')
            else:
                # Goal is on another block, check if current is on that block
                # AND if the block below is correctly placed
                result = (cs == goal_s) and is_in_goal_stack(goal_s) # Recursive call

            memo_is_in_goal_stack[b] = result
            return result

        # 3. Calculate Heuristic Value
        h = 0

        # Term 1: Misplaced blocks in goal stacks
        for block in self.goal_blocks_in_stacks:
            if not is_in_goal_stack(block):
                h += 1

        # Term 2: Blocks blocking correctly placed blocks
        for block in self.goal_blocks_in_stacks:
            # Find the block currently on top of this block
            block_on_top = current_on_top.get(block)

            # If there is a block on top AND this block is correctly placed in its goal stack
            if block_on_top is not None and is_in_goal_stack(block):
                 # Check if the block on top is the *wrong* block according to the goal
                 # It's wrong if it's not supposed to be on this block in the goal state.
                 # This is true if block_on_top is not in goal_support OR its goal_support is not 'block'
                 if block_on_top not in self.goal_support or self.goal_support.get(block_on_top) != block:
                     h += 1 # Penalty for the blocking block

        # Term 3: Blocks that must be clear but aren't (and are not part of goal stacks)
        for block in self.goal_clear_blocks:
            # Check only blocks that are NOT part of the goal stack structure defined by on/on-table
            if block not in self.goal_blocks_in_stacks:
                # Check if the block is currently clear
                # A block is clear if nothing is on top of it
                if block in current_on_top: # If something is on top, it's not clear
                    h += 1

        return h

