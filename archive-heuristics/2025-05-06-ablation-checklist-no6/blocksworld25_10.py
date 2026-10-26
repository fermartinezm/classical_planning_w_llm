from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not in their
    final desired configuration within the goal stacks. A block is in its
    final configuration if it is on the correct block/table, and the block
    below it is also in its final configuration, and the block on top of it
    is the correct one (or it should be clear).

    # Heuristic Initialization
    - Parses the goal state to determine the desired support for each block
      (`goal_support`) and the desired block on top of each block
      (`goal_block_on_top`).
    - Identifies all blocks relevant to the goal configuration.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to determine the current support for each block
       (`current_support`) and the current block on top of each block
       (`current_block_on_top`). Also identify all blocks present in the state.
    2. Define a recursive helper function `is_in_final_configuration(block)`
       that checks if a block is in its correct final position relative to its
       support and the block above it, recursively checking the support.
       - Base case: 'table' is always correctly configured.
       - Recursive step: A block B is correctly configured if:
         - It is on the correct support (checked only if block is in goal stack).
         - The block below it is also in final configuration (recursive, only if block is in goal stack).
         - The block on top of it is the correct one (or it should be clear).
       - Use memoization (caching) to avoid redundant computations.
    3. Initialize the heuristic value `h` to 0.
    4. Iterate through all blocks present in the current state.
    5. For each block, check if it is relevant to the goal configuration
       (i.e., mentioned in goal `on`, `on-table`, or `clear` facts).
    6. If the block is relevant and `is_in_final_configuration(block)` returns False,
       increment `h`.
    7. Return `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        super().__init__(task)
        # goal_support, goal_block_on_top, goal_relevant_blocks are set in the super().__init__

        # Pre-parse goal information from task.goals
        self.goal_support = {}
        self.goal_block_on_top = {}
        self.goal_relevant_blocks = set()

        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == 'on':
                B, A = parts[1], parts[2]
                self.goal_support[B] = A
                self.goal_block_on_top[A] = B
                self.goal_relevant_blocks.add(B)
                self.goal_relevant_blocks.add(A)
            elif predicate == 'on-table':
                B = parts[1]
                self.goal_support[B] = 'table'
                self.goal_relevant_blocks.add(B)
            elif predicate == 'clear':
                B = parts[1]
                # If a block is supposed to be clear, nothing should be on top
                # This is implicitly handled by initializing goal_block_on_top to None
                # and only setting it for 'on' facts.
                # Explicitly setting it here ensures blocks only mentioned in clear goals are relevant.
                self.goal_block_on_top[B] = self.goal_block_on_top.get(B, None) # Ensure it's in the map
                self.goal_relevant_blocks.add(B)
            # Ignore arm-empty goal if present

        # Remove 'table' pseudo-block from relevant blocks
        self.goal_relevant_blocks.discard('table')


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # 1. Parse the current state
        current_support = {}
        current_block_on_top = {}
        all_blocks = set()
        # holding_block = None # Not strictly needed for this heuristic logic

        # First pass to identify all blocks
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate in ['on', 'on-table', 'clear', 'holding']:
                for obj in parts[1:]:
                    all_blocks.add(obj)

        # Initialize current_block_on_top for all blocks (assume clear until proven otherwise)
        for block in all_blocks:
            current_block_on_top[block] = None

        # Second pass to build maps
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'on':
                B, A = parts[1], parts[2]
                current_support[B] = A
                current_block_on_top[A] = B
            elif predicate == 'on-table':
                B = parts[1]
                current_support[B] = 'table'
            elif predicate == 'holding':
                B = parts[1]
                current_support[B] = 'arm' # Block is in arm
                # The block it was on is now clear, this is reflected by the absence of (on B A)
                # and potentially a (clear A) fact. current_block_on_top will be None for A.
            # 'clear' facts confirm current_block_on_top is None, which is the default.

        # 2. Define recursive helper with caching
        final_config_cache = {}

        def is_in_final_configuration(block):
            """
            Checks if a block is in its final desired configuration within the goal stacks.
            A block is in final configuration if:
            - It's on the correct support (checked only if block is in goal stack).
            - The block below it is also in final configuration (recursive, only if block is in goal stack).
            - The block on top of it is the correct one (or it should be clear).
            """
            if block == 'table':
                return True
            if block in final_config_cache:
                return final_config_cache[block]

            desired_support = self.goal_support.get(block)
            current_supp = current_support.get(block) # Use .get() in case block is not in state maps (e.g. goal only block)
            desired_block_on_top = self.goal_block_on_top.get(block)
            current_block_above = current_block_on_top.get(block) # Use .get() in case block is not in state maps

            is_above_correct = (current_block_above == desired_block_on_top)

            if desired_support is not None: # Block is part of a goal stack (on something or table)
                is_support_correct = (current_supp == desired_support)
                # If current_supp is None (e.g. block is held), it's not on the desired support.
                if current_supp is None: is_support_correct = False

                # Recursively check the support, but only if the support is a block (not 'table' or 'arm')
                if desired_support != 'table':
                     support_correctly_configured = is_in_final_configuration(desired_support)
                else: # Desired support is table, which is base case and always True
                     support_correctly_configured = True

                result = is_support_correct and support_correctly_configured and is_above_correct
            else: # Block is NOT part of a goal stack (not in goal `on`/`on-table`)
                # It might be in goal_block_on_top (something on it) or goal_clear.
                # Its support doesn't matter for its final configuration state.
                result = is_above_correct # Only check if the block on top is correct (i.e., if it's clear).

            final_config_cache[block] = result
            return result

        # 3. Initialize heuristic value
        h = 0

        # 4. Iterate through all blocks in the state
        # 5. For each block, check if it's relevant to the goal and not in final config
        for block in all_blocks:
             # Check if the block is relevant to the goal configuration
             # A block is relevant if it's in goal_support (key or value) or goal_block_on_top (key or value)
             is_relevant = block in self.goal_relevant_blocks

             if is_relevant:
                 if not is_in_final_configuration(block):
                     h += 1

        # 6. Return h
        return h
