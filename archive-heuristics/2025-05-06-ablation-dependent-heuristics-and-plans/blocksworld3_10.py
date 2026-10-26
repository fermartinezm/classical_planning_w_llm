from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Assumes fact is a string like '(predicate arg1 arg2)'
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not in their
    correct position within the goal stacks. A block is considered
    correctly positioned if it is on the correct block (or the table)
    according to the goal, AND the block it is on is also correctly
    positioned (recursively), down to a correctly positioned base on the table.
    It also adds a penalty if the robot's arm is holding a block.

    # Assumptions
    - The goal specifies a set of desired 'on' and 'on-table' relationships
      that form one or more stacks.
    - All blocks mentioned in goal 'on' or 'on-table' predicates are considered
      part of the goal configuration.
    - Blocks not in the goal configuration do not need to be in a specific
      stack position relative to other goal blocks, but they might need to
      be moved if they are blocking a goal block (this heuristic implicitly
      handles this by counting blocks that are *not* correctly stacked).
    - Having a block in the arm adds to the cost as the arm must be freed
      to perform most stacking/unstacking operations.

    # Heuristic Initialization
    - Parse the goal predicates to build the target stack configuration,
      mapping each block in a goal stack to the block it should be directly
      on top of (or 'table'). Identify all blocks involved in the goal stacks.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to determine the current position of each block
       (what it is directly on top of, or if it's on the table or held).
       Identify if the arm is holding a block.
    2. For each block that is part of the goal configuration (identified during
       initialization):
       - Determine if the block is "correctly stacked" using a recursive check
         with memoization.
       - A block B is correctly stacked if:
         - It is not currently held by the arm.
         - AND its current position relative to the block below it (or the table)
           matches its desired position relative to the block below it (or the table)
           according to the goal configuration.
         - AND if its goal base is another block Y, then Y must also be correctly stacked.
       - The base case for the recursion is a block whose goal base is the 'table'.
         Such a block is correctly stacked if it is currently on the table.
    3. The heuristic value is the total count of blocks in the goal configuration
       that are *not* correctly stacked.
    4. Add 1 to the heuristic value if the robot's arm is currently holding a block.
       This accounts for the action needed to free the arm.
    5. Ensure the heuristic value is 0 if the current state is the goal state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the goal stack configuration.
        """
        # Call the base class constructor if necessary
        # super().__init__(task)

        self.goals = task.goals

        # Build the goal stack configuration: block -> block_below or 'table'
        self.goal_stack_below = {}
        self.goal_blocks = set()

        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts[0] == 'on-table':
                block = parts[1]
                self.goal_stack_below[block] = 'table'
                self.goal_blocks.add(block)
            elif parts[0] == 'on':
                block_on = parts[1]
                block_under = parts[2]
                self.goal_stack_below[block_on] = block_under
                self.goal_blocks.add(block_on)
                self.goal_blocks.add(block_under)

        # Static facts are not used in this heuristic
        # static_facts = task.static

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # If the state is a goal state, the heuristic is 0.
        if self.goals <= state:
             return 0

        # Build current configuration: block -> block_below or 'table'
        current_stack_below = {}
        currently_held = None
        # is_arm_empty = False # Not needed for this heuristic calculation

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                current_stack_below[parts[1]] = parts[2]
            elif parts[0] == 'on-table':
                current_stack_below[parts[1]] = 'table'
            elif parts[0] == 'holding':
                currently_held = parts[1]
            # elif parts[0] == 'arm-empty': # Not needed
            #     is_arm_empty = True

        # Memoization for correctly stacked status in the current state
        correctly_stacked_memo = {}

        def is_correctly_stacked_recursive(block):
            """
            Recursively check if a block is correctly stacked in the current state
            relative to the goal configuration.
            """
            # If we already computed this, return the result
            if block in correctly_stacked_memo:
                return correctly_stacked_memo[block]

            # A block that is currently held is never correctly stacked in a position.
            if currently_held == block:
                correctly_stacked_memo[block] = False
                return False

            # If the block is not part of the goal configuration, it cannot be
            # correctly stacked *relative to the goal stacks*.
            # Its position doesn't match a goal position defined by 'on' or 'on-table'.
            # So, it's not "correctly stacked" in the sense of contributing to a goal stack.
            # Let's return False for non-goal blocks when checking their status.
            # However, a non-goal block *can* be a base for a goal block.
            # The recursive check `is_correctly_stacked_recursive(goal_base)` needs to handle this.
            # If `goal_base` is not in `self.goal_blocks`, the goal configuration is malformed
            # or the heuristic definition needs adjustment. Assuming valid goals,
            # any block that is a base for a goal block is itself a goal block or 'table'.
            # So, `goal_base` will either be 'table' or in `self.goal_blocks`.

            # Get the block it should be on in the goal
            goal_base = self.goal_stack_below.get(block)

            # If the block is in goal_blocks, it must have a goal_base ('table' or another block)
            # If it's not in goal_blocks, this function shouldn't be called for it as the primary block.
            # It's only called recursively for goal_base.
            # Let's assume this function is only called initially for blocks in self.goal_blocks.
            # The recursive call `is_correctly_stacked_recursive(goal_base)` will handle the base.

            # Get the block it is currently on
            current_base = current_stack_below.get(block)

            # Check if the immediate base is correct
            is_immediate_base_correct = (current_base == goal_base)

            # If the immediate base is not correct, the block is not correctly stacked.
            if not is_immediate_base_correct:
                correctly_stacked_memo[block] = False
                return False

            # If the immediate base is correct, check the base recursively (unless it's the table)
            if goal_base == 'table':
                is_base_stack_correct = True # Table is always correctly stacked as a base
            else: # goal_base is another block Y
                 # Y must be correctly stacked for block to be correctly stacked on Y
                 is_base_stack_correct = is_correctly_stacked_recursive(goal_base) # Recursive call

            # The block is correctly stacked if its immediate base is correct AND the base stack is correct
            correctly_stacked_memo[block] = is_immediate_base_correct and is_base_stack_correct
            return correctly_stacked_memo[block]


        # Compute correctly stacked status for all blocks in the goal configuration
        # We need to compute for all blocks that are *in* the goal_blocks set.
        # The recursive calls will handle blocks lower in the stack.
        for block in self.goal_blocks:
             is_correctly_stacked_recursive(block) # This populates the memo

        # The heuristic is the number of blocks in the goal configuration
        # that are NOT correctly stacked.
        h_value = sum(1 for block in self.goal_blocks if not correctly_stacked_memo.get(block, False))

        # Add 1 if the arm is holding a block. Holding a block prevents other actions.
        # This seems like a reasonable additional cost.
        if currently_held is not None:
             h_value += 1

        # The explicit goal check at the beginning handles the h=0 case.

        return h_value
