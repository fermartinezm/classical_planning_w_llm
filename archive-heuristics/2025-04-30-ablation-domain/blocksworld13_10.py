# from heuristics.heuristic_base import Heuristic # Assuming this path is correct in the environment

# If the base class is not provided externally, define a placeholder
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError("Heuristic base class not found.")

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty string or malformed fact defensively
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the "disorder" in the current state compared to the goal state
    by counting the number of blocks that are either not on their correct support
    (the block or table they should be on according to the goal) OR have a block
    stacked immediately on top of them that should not be there according to the goal.
    Additionally, it penalizes states where the arm is holding a block if the goal
    requires the arm to be empty. This captures the idea that a block is misplaced
    if it's in the wrong place or if something immediately above it is wrong,
    and that the arm state matters for achieving the goal.

    # Assumptions
    - The goal state defines a specific configuration of blocks stacked on each other or the table.
    - The heuristic focuses on achieving the correct relative positioning of blocks within stacks.
    - The standard Blocksworld actions (pick-up, put-down, stack, unstack) are used.
    - The goal state includes predicates like `on`, `on-table`, and potentially `clear` and `arm-empty`.

    # Heuristic Initialization
    - Parses the goal conditions to determine:
        -   `goal_support`: A mapping from each block to the block or 'table' it should be directly on in the goal state.
        -   `goal_block_above`: A mapping from each block (that acts as a support in the goal) to the block that should be immediately on top of it in the goal state. Blocks that should be clear in the goal are mapped to `None`.
    - Identifies all blocks involved in the goal configuration.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Initialize the heuristic value `h` to 0.
    2.  Parse the current state to determine:
        -   `current_support`: A mapping from each block to the block, 'table', or 'arm' it is currently on.
        -   `current_block_above`: A mapping from each block (that acts as a support in the current state) to the block that is immediately on top of it.
        -   `holding_block`: The block currently held by the arm, or `None`.
        -   `arm_is_empty`: Boolean indicating if the arm is empty.
    3.  Iterate through every block identified as part of the goal configuration (`self.goal_blocks`).
    4.  For each block `B`:
        -   Retrieve `goal_sup = self.goal_support.get(B)`.
        -   Retrieve `goal_above = self.goal_block_above.get(B, None)`.
        -   Retrieve `current_sup = current_support.get(B, None)`.
        -   Determine `current_above`: Find the block `C` such that `(on C B)` is in the current state. If no such block exists, `current_above` is `None`. This is efficiently done using the `current_block_above` map built in step 2.
    5.  Check if block `B` is "wrongly placed" relative to the goal stack structure:
        -   If `current_sup` is different from `goal_sup`, increment `h`.
        -   Otherwise (if `current_sup` is the same as `goal_sup`), check what's above `B`. If `current_above` is different from `goal_above`, increment `h`.
    6.  After iterating through all goal blocks, check if `(arm-empty)` is a goal predicate. If it is, and the arm is currently *not* empty (`holding_block is not None`), increment `h` by 1. This accounts for the cost of freeing the arm if required by the goal.
    7.  Return the final value of `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and identifying blocks.
        """
        self.goals = task.goals

        # Parse goal conditions to build goal support and goal block above mappings.
        self.goal_support = {} # block -> support_block_or_table
        self.goal_block_above = {} # block -> block_above (only for blocks that are supports in goal)
        goal_on_table = set() # blocks that should be on the table

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue

            predicate = parts[0]
            if predicate == "on":
                block, support = parts[1], parts[2]
                self.goal_support[block] = support
                self.goal_block_above[support] = block # block is directly above support
            elif predicate == "on-table":
                block = parts[1]
                self.goal_support[block] = 'table'
                goal_on_table.add(block)
            # 'clear' predicates in goal imply nothing should be on that block.
            # We handle this by inferring goal_block_above later.

        # Identify all blocks involved in the goal configuration.
        # These are blocks that are keys or values in goal_support, or in goal_on_table.
        all_goal_objects = set(self.goal_support.keys()) | set(self.goal_support.values()) | goal_on_table
        # Filter out 'table' from objects
        all_goal_blocks = {obj for obj in all_goal_objects if obj != 'table'}

        # For any block X that is a goal block, if it's not a support in an 'on' goal,
        # it means nothing should be on it in the goal. Set its goal_block_above to None.
        for block in all_goal_blocks:
            if block not in self.goal_block_above:
                self.goal_block_above[block] = None # Nothing should be on this block in the goal

        # Store the list of blocks we care about for the heuristic calculation.
        self.goal_blocks = sorted(list(all_goal_blocks))


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state.

        # Parse current state to build mappings
        current_support = {} # block -> support (block, table, or arm)
        current_block_above = {} # block -> block_above (only for blocks that are supports)
        holding_block = None
        arm_is_empty = False

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "on":
                block, support = parts[1], parts[2]
                current_support[block] = support
                current_block_above[support] = block # block is directly above support
            elif predicate == "on-table":
                block = parts[1]
                current_support[block] = 'table'
                # Note: We don't add to current_block_above['table'] here.
                # Multiple blocks can be on the table, and we only care about the single block
                # directly on top of a support in a stack.
            elif predicate == "holding":
                holding_block = parts[1]
                current_support[holding_block] = 'arm' # Use 'arm' as a special support location
            elif predicate == "arm-empty":
                arm_is_empty = True
            # 'clear' predicates are implicit: if a block is not a key in current_block_above, it's clear.

        h = 0  # Initialize heuristic value.

        # Iterate through blocks that are part of the goal configuration
        for block in self.goal_blocks:
            goal_sup = self.goal_support.get(block) # Can be block or 'table'
            goal_above = self.goal_block_above.get(block, None) # Can be block or None

            # Find current support for this block
            current_sup = current_support.get(block, None) # None if not found (e.g., block not in state? or bug)

            # Find current block immediately above this block
            # A block C is immediately above block B if (on C B) is in the state.
            current_above = current_block_above.get(block, None) # None if nothing is directly on block B

            # Condition 1: Block is not on its goal support
            if current_sup != goal_sup:
                h += 1
            else:
                # Condition 2: Block is on correct support, but something wrong is above it
                # Check if the block currently above B is the same as the block that should be above B in the goal.
                # If goal_above is None, it means B should be clear in the goal.
                if current_above != goal_above:
                     h += 1

        # Add 1 if (arm-empty) is a goal and the arm is not empty.
        # This accounts for the cost of putting down a held block if needed for the goal.
        if "(arm-empty)" in self.goals and not arm_is_empty:
             h += 1

        return h
