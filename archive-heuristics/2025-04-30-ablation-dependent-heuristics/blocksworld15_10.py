# The Operator and Task classes are assumed to be available in the execution environment.
# No need to import them explicitly in the final provided code.

# from task import Operator, Task # Example import if needed for local testing

class blocksworldHeuristic:
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by counting
        the number of blocks that are not in their correct position within the
        goal stacks, and adding a penalty for each block that is currently
        sitting on top of a block that is not in its correct goal stack position.
        Each such count is multiplied by 2, approximating the two actions
        (pickup/unstack and putdown/stack) typically needed to move a block.
        The "correct goal stack position" for a block is defined recursively:
        a block B is in its correct goal stack position if it is currently
        on the block A it should be on according to the goal, AND block A is
        also in its correct goal stack position (or A is supposed to be on the
        table and is currently on the table).

    Assumptions:
        - The goal state defines a set of desired (on B A) and (on-table B)
          relationships that form one or more stacks rooted on the table.
        - Blocks not mentioned in goal (on) or (on-table) facts are considered
          misplaced if they are present in the state.
        - The heuristic does not explicitly account for the 'clear' or
          'arm-empty' predicates in the goal, assuming they are implicitly
          handled by achieving the correct stack structure.

    Heuristic Initialization:
        The constructor pre-processes the goal facts from the Task object.
        It builds two data structures:
        - self.goal_on_map: A dictionary mapping a block B to the block A it
          should be on in the goal state, derived from (on B A) goal facts.
        - self.goal_table_set: A set of blocks that should be on the table
          in the goal state, derived from (on-table B) goal facts.
        It also collects all unique blocks mentioned in the goal structure into
        self.goal_blocks.

    Step-By-Step Thinking for Computing Heuristic:
        1. Parse the current state facts to determine the current configuration
           of blocks:
           - state_on_map: Maps a block B to the block A it is currently on.
           - state_table_set: Set of blocks currently on the table.
           - state_holding: Set containing the block currently held by the arm (if any).
           - state_blocks: Set of all unique blocks present in the state facts.
        2. Identify all blocks relevant to the problem, which are those present
           in either the goal structure or the current state.
        3. For each relevant block, determine if it is in its "correct goal
           stack position" using a recursive helper function `_is_in_goal_stack_position`.
           This function uses memoization to be efficient. A block is correct
           if its current support matches its goal support AND its support is
           also correct (recursively), or if it's correctly on the table.
           Blocks not part of the goal structure are marked as incorrect by this function.
        4. Initialize the heuristic value `h` to 0.
        5. Iterate through all relevant blocks. If a block B is NOT in its
           correct goal stack position (i.e., `is_correct_map[B]` is False), increment `h` by 2.
           This counts blocks that need to be moved from their desired stack position,
           estimating 2 actions (pickup/unstack + putdown/stack) per block.
        6. Iterate through all (on C B) facts in the current state. If block B
           (the block being supported) is NOT in its correct goal stack position
           (i.e., `is_correct_map[B]` is False), increment `h` by 2. This counts
           blocks C that are on top of a misplaced block B, representing the
           cost of clearing B, estimating 2 actions (unstack + putdown/stack) per block.
        7. The final value of `h` is the heuristic estimate.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by pre-processing goal facts.

        Args:
            task: The planning task object.
        """
        self.goal_on_map = {}
        self.goal_table_set = set()
        self.goal_blocks = set()

        # Parse goal facts
        for fact_str in task.goals:
            parts = fact_str.strip('()').split()
            if parts[0] == 'on' and len(parts) == 3:
                block, block_below = parts[1], parts[2]
                self.goal_on_map[block] = block_below
                self.goal_blocks.add(block)
                self.goal_blocks.add(block_below)
            elif parts[0] == 'on-table' and len(parts) == 2:
                block = parts[1]
                self.goal_table_set.add(block)
                self.goal_blocks.add(block)
            # Ignore 'clear' and 'arm-empty' goals for this heuristic's structure calculation

        # Static facts are ignored as per blocksworld domain definition
        # (task.static is available but empty for blocksworld)


    def __call__(self, state, static_facts=frozenset()):
        """
        Computes the heuristic value for the given state.

        Args:
            state: The current state (frozenset of fact strings).
            static_facts: Static facts (ignored in this domain).

        Returns:
            An integer heuristic value.
        """
        # 1. Parse current state facts
        state_on_map = {}
        state_table_set = set()
        state_holding = set()
        state_blocks = set()
        state_on_facts = [] # Store (C, B) for step 6

        for fact_str in state:
            parts = fact_str.strip('()').split()
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'on' and len(args) == 2:
                block, block_below = args[0], args[1]
                state_on_map[block] = block_below
                state_blocks.add(block)
                state_blocks.add(block_below)
                state_on_facts.append((block, block_below))
            elif predicate == 'on-table' and len(args) == 1:
                block = args[0]
                state_table_set.add(block)
                state_blocks.add(block)
            elif predicate == 'holding' and len(args) == 1:
                block = args[0]
                state_holding.add(block)
                state_blocks.add(block)
            # Ignore 'clear' and 'arm-empty' state facts for structure parsing

        # 2. Identify all relevant blocks
        all_blocks = self.goal_blocks | state_blocks

        # 3. Compute is_correct status for each block
        is_correct_map = {}
        memo = {} # Memoization for the recursive function
        for block in all_blocks:
             is_correct_map[block] = self._is_in_goal_stack_position(
                 block, state_on_map, state_table_set,
                 self.goal_on_map, self.goal_table_set, memo)

        # 4. Initialize heuristic
        h = 0

        # 5. Add cost for blocks not in correct goal stack position
        for block in all_blocks:
            if not is_correct_map.get(block, False): # Use .get with default False for safety
                h += 2 # Cost estimate for moving the block itself

        # 6. Add cost for blocks on top of misplaced blocks
        for C, B in state_on_facts:
             if not is_correct_map.get(B, False): # Check if B is in our map and is incorrect
                 h += 2 # Cost estimate for moving C to clear B

        return h

    def _is_in_goal_stack_position(self, block, state_on_map, state_table_set, goal_on_map, goal_table_set, memo):
        """
        Recursive helper to check if a block is in its correct goal stack position.
        Uses memoization.
        """
        if block in memo:
            return memo[block]

        # Find goal support for this block
        goal_support = None
        if block in goal_table_set:
            goal_support = 'table'
        elif block in goal_on_map:
            goal_support = goal_on_map[block]
        else:
            # Block is not in the goal structure (not in goal_on_map or goal_table_set)
            # It cannot be in its goal stack position.
            memo[block] = False
            return False

        # Find current support for this block
        current_support = None
        if block in state_table_set:
            current_support = 'table'
        elif block in state_on_map:
            current_support = state_on_map[block]
        # If block is held, current_support remains None.

        # Check if current support matches goal support
        if current_support != goal_support:
            memo[block] = False
            return False

        # If the support matches, check if the support itself is correctly placed (unless it's the table)
        if goal_support == 'table':
            memo[block] = True # Block is on table, and goal is on table. Correct.
            return True
        else:
            # Block is on A, goal is on A. Check if A is correctly placed.
            # Recursive call for the block below it in the goal stack.
            # Ensure the block below (goal_support) is actually part of the goal structure
            # before recursing. It should be, based on how goal_on_map is built from goal facts.
            result = self._is_in_goal_stack_position(goal_support, state_on_map, state_table_set, goal_on_map, goal_table_set, memo)
            memo[block] = result
            return result
