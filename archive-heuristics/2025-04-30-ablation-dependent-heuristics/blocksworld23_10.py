class blocksworldHeuristic:
    """
    Domain-dependent heuristic for Blocksworld.

    Summary:
    This heuristic estimates the distance to the goal state by counting two types of
    misplacements:
    1. Blocks that are not in their correct position within the desired goal stacks
       (h_stack). A block is considered correctly stacked only if it is on the
       correct block (or table) as specified by the goal, AND the block below it
       is also correctly stacked (recursively).
    2. Blocks that are currently on top of another block where they shouldn't be
       (h_obstructing). This occurs if the block below should be clear according
       to the goal, or if a different block should be on top of the block below
       according to the goal, AND the current (on) relationship is not itself a goal fact.

    The total heuristic value is calculated as h_stack + 2 * h_obstructing.
    The weight of 2 for h_obstructing reflects that removing an obstructing block
    typically requires at least two actions (unstack/pickup and then putdown/stack
    it elsewhere).

    Assumptions:
    - The input state and goal are represented as frozensets of strings following
      the PDDL fact format (e.g., '(on b1 b2)', '(clear b1)').
    - The domain is Blocksworld with standard predicates (on, on-table, clear,
      arm-empty, holding) and actions (pickup, putdown, stack, unstack).
    - Goal facts primarily specify (on ?x ?y), (on-table ?x), and (clear ?x).
    - All blocks involved in goal stacks have either an (on) or (on-table) goal fact
      specifying their desired location. Blocks not mentioned in (on) or (on-table)
      goal facts are assumed not to have a fixed position requirement within a stack.
    - The heuristic is designed for greedy best-first search and does not need to
      be admissible.

    Heuristic Initialization:
    The __init__ method pre-processes the goal facts from the task to build data
    structures that describe the desired goal configuration:
    - self.goal_location: A dictionary mapping each block to its desired location
      (another block name or the string 'table') based on (on) and (on-table)
      goal facts.
    - self.goal_blocks_above: A dictionary mapping each block to the block that
      should be directly on top of it according to (on) goal facts.
    - self.goal_clear: A set of blocks that should be clear according to (clear)
      goal facts.
    - self.goal_on_facts: A set of (on ?x ?y) goal facts for quick lookup.
    - self.all_blocks: A set containing the names of all blocks present in either
      the initial state or the goal state.
    Static facts are ignored as per the domain definition (they are empty).

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state is the goal state using task.goal_reached(state).
       If it is, the heuristic value is 0.
    2. Parse the current state to determine the current block positions and build
       a dictionary `current_block_above` mapping each block to the block currently
       on top of it (if any).
    3. Compute `h_stack`: Initialize a memoization dictionary `memo_correct_stack`.
       Iterate through all blocks identified during initialization. For each block,
       call the recursive helper function `_is_correctly_stacked`. This function
       checks if the block is in its goal location (on the correct block or table)
       AND if the block below it (in the goal stack) is also correctly stacked.
       The recursion stops at the table or when a block is found to be incorrectly
       placed. Memoization prevents redundant calculations. `h_stack` is the count
       of blocks that have a specific goal location (`block in self.goal_location`)
       but for which `_is_correctly_stacked` returns False.
    4. Compute `h_obstructing`: Iterate through the `current_block_above` dictionary.
       For each block `block_below` that has a block `block_above` currently on it,
       check if this configuration is undesirable. It is undesirable if `block_above`
       is not the block that should be on `block_below` according to `self.goal_blocks_above`,
       OR if `block_below` should be clear according to `self.goal_clear`. Additionally,
       ensure that the current `(on block_above block_below)` fact is *not* itself
       a goal fact. Count the number of such obstructing `block_above` instances.
    5. The final heuristic value is `h_stack + 2 * h_obstructing`.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.initial_state = task.initial_state
        self.all_blocks = set()
        self.goal_location = {}
        self.goal_blocks_above = {}
        self.goal_clear = set()
        self.goal_on_facts = set() # Store (on ?x ?y) goal facts for quick lookup

        # Pre-process goal facts
        for fact_string in self.goals:
            predicate, args = self._parse_fact(fact_string)
            if predicate == 'on-table':
                block = args[0]
                self.goal_location[block] = 'table'
                self.all_blocks.add(block)
            elif predicate == 'on':
                block_above, block_below = args
                self.goal_location[block_above] = block_below
                self.goal_blocks_above[block_below] = block_above
                self.goal_on_facts.add(fact_string) # Store the goal on fact string
                self.all_blocks.add(block_above)
                self.all_blocks.add(block_below)
            elif predicate == 'clear':
                block = args[0]
                self.goal_clear.add(block)
                self.all_blocks.add(block)
            # Ignore arm-empty goal if present

        # Add blocks from initial state that might not be in goal facts
        for fact_string in self.initial_state:
             predicate, args = self._parse_fact(fact_string)
             if predicate in ('on-table', 'clear', 'holding'):
                 if args: self.all_blocks.add(args[0])
             elif predicate == 'on':
                 self.all_blocks.add(args[0])
                 self.all_blocks.add(args[1])
             # Ignore arm-empty

        # Static facts are empty in this domain, but checklist requires handling.
        # No specific data structure needed for empty static facts here.
        # If static facts existed and were relevant (e.g., fixed supports),
        # they would be processed here.

    def _parse_fact(self, fact_string):
        """Helper to parse a fact string into predicate and arguments."""
        # Remove surrounding brackets and split by space
        parts = fact_string[1:-1].split()
        predicate = parts[0]
        args = parts[1:]
        return predicate, args

    def _is_correctly_stacked(self, block, state, memo):
        """
        Recursively checks if a block is in its correct goal stack position,
        including checking the blocks below it in the goal stack.
        Uses memoization to avoid redundant calculations.
        """
        if block in memo:
            return memo[block]

        # If the block doesn't have a specific goal location, it's not part of a required stack structure
        if block not in self.goal_location:
            memo[block] = True
            return True

        goal_loc = self.goal_location[block]

        if goal_loc == 'table':
            result = '(on-table ' + block + ')' in state
            memo[block] = result
            return result
        else: # goal_loc is another block (Y)
            block_below_goal = goal_loc
            # Check if block is currently on the correct block below it
            is_on_correct_block = '(on ' + block + ' ' + block_below_goal + ')' in state
            # Check if the block below is also correctly stacked
            is_below_correct = self._is_correctly_stacked(block_below_goal, state, memo)
            result = is_on_correct_block and is_below_correct
            memo[block] = result
            return result

    def __call__(self, state, task):
        """
        Computes the heuristic value for the given state.
        """
        if task.goal_reached(state):
            return 0

        # 2. Parse current state to find current block positions and blocks above.
        current_block_above = {}
        for fact_string in state:
            predicate, args = self._parse_fact(fact_string)
            if predicate == 'on':
                block_above, block_below = args
                current_block_above[block_below] = block_above

        # 3. Compute h_stack
        h_stack = 0
        memo_correct_stack = {}
        for block in self.all_blocks:
             # Only count blocks that *should* be in a specific location according to the goal
            if block in self.goal_location and not self._is_correctly_stacked(block, state, memo_correct_stack):
                h_stack += 1

        # 4. Compute h_obstructing
        h_obstructing = 0
        for block_below, block_above in current_block_above.items():
            desired_block_above = self.goal_blocks_above.get(block_below)
            should_be_clear = block_below in self.goal_clear
            current_on_fact = '(on ' + block_above + ' ' + block_below + ')'

            # An obstruction exists if:
            # 1. The current (on) relationship is not a goal fact itself.
            # 2. AND (
            #    The block currently on top is not the one desired by the goal
            #    OR the block below should be clear according to the goal
            # )
            # Note: If desired_block_above is None, it means nothing specific should be on block_below.
            # In this case, if block_above exists, it's an obstruction if block_below
            # should be clear, or if block_above shouldn't be on block_below according to the goal (which is always true if desired_block_above is None and current_on_fact is not a goal fact).

            is_undesired_configuration = (desired_block_above != block_above) or should_be_clear

            if is_undesired_configuration and current_on_fact not in self.goal_on_facts:
                 h_obstructing += 1


        # 5. Combine
        # The weight 2 is a heuristic choice. It implies unstacking + moving.
        # A block counted in h_stack might also be counted in h_obstructing.
        # E.g., Block A should be on B (goal), but is on C (state).
        # A is counted in h_stack (not correctly stacked).
        # If A is on C, and C should be clear or have something else on it, A is obstructing C.
        # This double counting is intentional; it reflects multiple issues that need fixing.
        # A block in the wrong place (h_stack) needs to be moved.
        # A block blocking something (h_obstructing) needs to be moved.
        # These are often the same block or related blocks.
        # The sum captures the total "mess".

        return h_stack + 2 * h_obstructing
