class blocksworldHeuristic:
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
        This heuristic estimates the cost to reach the goal by counting the
        number of blocks that are currently in a "wrong" position or have a
        "wrong" block on top of them, relative to the desired configuration
        in the goal state. A block is considered "wrong" if either the block
        it is currently resting on (or the table/hand) is not the one it
        should be resting on according to the goal, OR the block currently
        on top of it is not the one that should be on top according to the
        goal (including the case where it should be clear but isn't).

    Assumptions:
        - The heuristic is designed for the standard STRIPS Blocksworld domain
          as defined in the provided PDDL.
        - States are represented as frozensets of strings.
        - Goal states are defined by a set of facts, primarily specifying
          `on` and `on-table` relationships, and `clear` status for the top
          blocks of goal stacks.
        - The heuristic does not need to be admissible.
        - The heuristic should be 0 if and only if the state is a goal state.
        - The heuristic value should be finite for solvable states.

    Heuristic Initialization:
        The heuristic's constructor pre-processes the goal facts from the
        Task object to build three internal data structures:
        - `self.goal_under`: A dictionary mapping each block `b` to the block
          `y` it should be directly on top of (`(on b y)` in goal), or the
          string 'table' (`(on-table b)` in goal).
        - `self.goal_above`: A dictionary mapping each block `y` to the block
          `b` that should be directly on top of it (`(on b y)` in goal). If
          no block should be on `y` (i.e., `(clear y)` is a goal fact and no
          `(on x y)` is a goal fact), `y` will not be a key in this dictionary,
          or its corresponding goal top can be considered None.
        - `self.goal_blocks`: A set of all blocks mentioned in the goal's
          `on`, `on-table`, or `clear` facts. These are the blocks whose
          positions or clear status are constrained by the goal structure.

    Step-By-Step Thinking for Computing Heuristic:
        1. Parse the current state facts to build temporary dictionaries
           representing the current configuration:
           - `current_under`: Maps each block `b` to the block `y` it is
             currently on (`(on b y)` in state), or 'table' (`(on-table b)`
             in state).
           - `current_above`: Maps each block `y` to the block `b` that is
             currently on top of it (`(on b y)` in state).
           - Identify the `holding_block` if any (`(holding b)` in state).
        2. Handle the block being held: If a block is being held, its current
           "under" status is effectively the hand. Update `current_under`
           for the held block to a special value like 'hand'.
        3. Initialize a counter `wrong_count` to 0.
        4. Iterate through each block `b` in the pre-computed set of
           `self.goal_blocks`. These are the blocks whose positions or clear
           status are relevant to the goal.
        5. For each block `b`, check if its current configuration matches
           its goal configuration relative to the block below and the block above:
           - Check the block underneath: Determine the current base of `b`
             using `current_under.get(b, 'hand' if block == holding_block else 'unknown_base')`.
             Determine the goal base of `b` using `self.goal_under.get(b)`.
             A block's base is considered "wrong" if the goal specifies a base
             (`block in self.goal_under`) and the current base does not match
             the goal base.
           - Check the block above: Determine the current block on top of `b`
             using `current_above.get(b, None)` (None if clear). Determine the
             block that should be on top of `b` according to the goal using
             `self.goal_above.get(b, None)` (None if it should be clear).
             The top relationship is considered "wrong" if the current block
             on top does not match the block that should be on top according
             to the goal. This correctly identifies cases where a block should
             be clear but isn't, or has the wrong block on top.
           - If either the 'under' relationship (when specified by the goal)
             or the 'above' relationship for block `b` does not match the goal,
             increment `wrong_count`.
        6. The final `wrong_count` is the heuristic value.

        This heuristic counts how many blocks relevant to the goal structure
        are not in their correct immediate context (what's below/above them).
        Moving a block or clearing a block on top typically requires one or
        more actions, so this count provides a rough estimate of the work
        needed. It is not admissible because achieving the correct context
        for one block might disrupt the correct context for another, and
        the cost of clearing stacks is not fully captured. However, it
        provides a better estimate than simply counting unsatisfied goal facts
        by considering the structural dependencies.
    """
    def __init__(self, task):
        """
        Initializes the heuristic by pre-processing goal information.

        @param task: The planning task object.
        """
        self.goal_under = {}
        self.goal_above = {}
        self.goal_blocks = set()

        # Build goal_under and goal_above maps from goal facts
        for fact in task.goals:
            if fact.startswith('(on '):
                parts = fact.strip('()').split()
                if len(parts) == 3: # Ensure correct format
                    b, y = parts[1], parts[2]
                    self.goal_under[b] = y
                    self.goal_above[y] = b
                    self.goal_blocks.add(b)
                    self.goal_blocks.add(y)
            elif fact.startswith('(on-table '):
                parts = fact.strip('()').split()
                if len(parts) == 2: # Ensure correct format
                    b = parts[1]
                    self.goal_under[b] = 'table'
                    self.goal_blocks.add(b)
            # Note: (clear ?) goal facts are implicitly handled by goal_above
            # If (clear b) is a goal and no (on x b) is a goal, then b
            # will not be a key in goal_above, which means goal_above.get(b) is None.

        # Ensure blocks mentioned only in (clear ?) goals are also considered goal_blocks
        for fact in task.goals:
             if fact.startswith('(clear '):
                 parts = fact.strip('()').split()
                 if len(parts) == 2: # Ensure correct format
                     b = parts[1]
                     self.goal_blocks.add(b)


    def __call__(self, state, static_info):
        """
        Computes the heuristic value for the given state.

        @param state: The current state (frozenset of facts).
        @param static_info: Static facts (not used in this heuristic).
        @return: The heuristic value (integer).
        """
        # If the state is a goal state, the heuristic is 0.
        # The calculation below should result in 0 for goal states,
        # but an explicit check can sometimes be slightly faster if
        # the goal check is very efficient. However, relying on the
        # calculation ensures consistency.

        current_under = {}
        current_above = {}
        holding_block = None

        # Parse current state facts to build current configuration maps
        for fact in state:
            if fact.startswith('(on '):
                parts = fact.strip('()').split()
                if len(parts) == 3:
                    b, y = parts[1], parts[2]
                    current_under[b] = y
                    current_above[y] = b
            elif fact.startswith('(on-table '):
                parts = fact.strip('()').split()
                if len(parts) == 2:
                    b = parts[1]
                    current_under[b] = 'table'
            elif fact.startswith('(holding '):
                parts = fact.strip('()').split()
                if len(parts) == 2:
                    b = parts[1]
                    holding_block = b # Assume only one block can be held
            # Ignore (clear ?) facts in state as current_above map implicitly tracks clearness

        # Handle the block being held - its 'under' is the hand
        if holding_block:
             current_under[holding_block] = 'hand'

        wrong_count = 0

        # Count blocks in goal_blocks that are in a 'wrong' configuration
        for block in self.goal_blocks:
            # Check if the block is on the correct base (only if goal specifies a base)
            base_is_wrong = False
            if block in self.goal_under:
                 # Get current base, defaulting to 'hand' if held, or 'unknown_base' if not in current_under (shouldn't happen in valid state)
                 current_base = current_under.get(block, 'hand' if block == holding_block else 'unknown_base')
                 goal_base = self.goal_under[block]
                 if current_base != goal_base:
                     base_is_wrong = True

            # Check if the correct block is on top (or if it should be clear but isn't)
            # Get current top, defaulting to None if clear (not in current_above and not held)
            current_top = current_above.get(block, None)
            # Get goal top, defaulting to None if it should be clear (not in goal_above map)
            goal_top = self.goal_above.get(block, None)

            top_is_wrong = (current_top != goal_top)

            # If either the base (when specified by goal) or the top is wrong, increment the count
            if base_is_wrong or top_is_wrong:
                 wrong_count += 1

        # The heuristic value is the count of 'wrong' blocks
        return wrong_count

