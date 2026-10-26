class blocksworldHeuristic:
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
    This heuristic estimates the number of actions required to reach the goal
    state by counting blocks that are not on their correct immediate support
    according to the goal state, and adding a penalty for blocks stacked on
    top of these misplaced blocks.

    Assumptions:
    - The goal state is represented by a set of facts including `(on ?x ?y)`,
      `(on-table ?x)`, and potentially `(clear ?x)`.
    - The heuristic focuses on satisfying the `(on ?x ?y)` and `(on-table ?x)`
      goal facts as the primary measure of progress towards building the
      goal stacks.
    - The heuristic assumes that any block not explicitly given an `on` or
      `(on-table ?x)` goal position relative to a support does not contribute
      to the "misplaced support" count based on its own position, but can
      contribute as a penalty if it's blocking a misplaced block that *does*
      have a goal support.
    - The heuristic value is 0 if and only if the state is the goal state.
    - The heuristic value is finite for solvable states.
    - The heuristic is non-admissible and designed to guide greedy best-first search.
    - Object names do not contain spaces or parentheses.

    Heuristic Initialization:
    The constructor `__init__` parses the goal facts to build a mapping
    `self.goal_support` where `self.goal_support[block]` is the block or
    'table' that `block` should be immediately on top of in the goal state.
    It also collects all unique object names mentioned in the initial state
    and goal facts into `self.all_objects`. Static facts are processed if
    they exist, although the standard blocksworld domain has none.

    Step-By-Step Thinking for Computing Heuristic:
    The `__call__` method computes the heuristic for a given state:
    1. It first checks if the input `state` is the goal state using `self.task.goal_reached(state)`. If it is, the heuristic is 0.
    2. It parses the current `state` to determine the immediate support for each
       block (the block it is on, or 'table'). This information is stored in
       `state_support`. It also identifies the block being held (`holding_block`)
       and builds `state_above`, a mapping where `state_above[block]` is the
       set of blocks currently stacked directly on top of `block`.
    3. It initializes the heuristic value `h` to 0.
    4. It iterates through all blocks identified during initialization (`self.all_objects`).
    5. For each block `B`:
       a. Retrieve its goal support (`goal_sup = self.goal_support.get(B)`). This will be `None` if `B` is not mentioned in any goal `(on ?x ?y)` or `(on-table ?x)` fact.
       b. If `goal_sup is not None` (meaning `B` has a defined goal position relative to its support):
          i. Retrieve its current support (`current_sup`). This is 'table' if `(on-table B)` is in the state, the block `A` if `(on B A)` is in the state. If `(holding B)` is in the state, `current_sup` is set to 'holding'. If the block is not found in `state_support` and is not being held, its current support is effectively `None` (not on anything).
          ii. If the `current_sup` is not equal to the `goal_sup`:
              - Add 1 to the heuristic (representing the need to move this block).
              - If the block is not currently being held (`current_sup != 'holding'`), add the number of blocks currently stacked directly on top of `B` in the state (`len(state_above.get(B, []))`) to `h`. This penalizes blocks that are buried under others when they are already in the wrong place.
   c. If `goal_sup is None`, the block `B` does not have a specific required support relation in the goal. Its position does not directly contribute to the heuristic count based on being misplaced relative to a goal support. (It can still contribute indirectly if it is on top of a block that *is* misplaced and has a goal support).
    6. The total value of `h` after iterating through all blocks is returned as the heuristic estimate.
    """

    def __init__(self, task):
        """
        Heuristic Initialization.

        Parses the goal state to determine the desired support (the block or
        the table) for each block. Collects all objects from initial state and goals.

        Args:
            task: The planning task object containing initial state, goals, etc.
        """
        self.task = task
        self.goal_support = {}
        self.all_objects = set()

        # Extract goal support relations and collect all objects from goals
        for fact in task.goals:
            if fact.startswith('(on '):
                # Fact is like '(on b1 b2)'
                parts = fact.strip('()').split()
                block = parts[1]
                support = parts[2]
                self.goal_support[block] = support
                self.all_objects.add(block)
                self.all_objects.add(support)
            elif fact.startswith('(on-table '):
                # Fact is like '(on-table b1)'
                parts = fact.strip('()').split()
                block = parts[1]
                self.goal_support[block] = 'table'
                self.all_objects.add(block)
            # Ignore (clear ?) goals for goal_support mapping

        # Collect all objects from initial state facts
        for fact in task.initial_state:
             if fact.startswith('(on '):
                parts = fact.strip('()').split()
                self.all_objects.add(parts[1])
                self.all_objects.add(parts[2])
             elif fact.startswith('(on-table '):
                parts = fact.strip('()').split()
                self.all_objects.add(parts[1])
             elif fact.startswith('(clear '):
                parts = fact.strip('()').split()
                self.all_objects.add(parts[1])
             elif fact.startswith('(holding '):
                parts = fact.strip('()').split()
                self.all_objects.add(parts[1])
             elif fact == '(arm-empty)':
                 pass # arm-empty doesn't involve objects

        # Static facts are ignored as per domain definition (none exist)
        # If static facts defined objects, we would process them here.
        # Example: for fact in task.static: ...

    def __call__(self, state):
        """
        Step-By-Step Thinking for Computing Heuristic.

        Args:
            state: The current state as a frozenset of fact strings.

        Returns:
            An integer representing the estimated cost to reach the goal.
        """
        # 1. Check if goal reached
        if self.task.goal_reached(state):
            return 0

        # 2. & 3. Parse state to get current support and blocks above
        state_support = {}
        state_above = {}
        # Identify blocks being held - they have no support below them
        holding_block = None

        for fact in state:
            if fact.startswith('(on '):
                # Fact is like '(on b1 b2)'
                parts = fact.strip('()').split()
                block = parts[1]
                support = parts[2]
                state_support[block] = support
                state_above.setdefault(support, set()).add(block)
            elif fact.startswith('(on-table '):
                # Fact is like '(on-table b1)'
                parts = fact.strip('()').split()
                block = parts[1]
                state_support[block] = 'table'
            elif fact.startswith('(holding '):
                 parts = fact.strip('()').split()
                 holding_block = parts[1]
                 # A held block has no support below it in the stack structure
            # Ignore (clear ?) and (arm-empty)

        # 4. Initialize heuristic
        h = 0

        # 5. Iterate through all objects
        for block in self.all_objects:
            goal_sup = self.goal_support.get(block)

            # 5.b. If block has a defined goal support
            if goal_sup is not None:
                # 5.b.i. Find current support
                current_sup = state_support.get(block)
                # Handle the case where the block is being held
                if block == holding_block:
                    current_sup = 'holding' # Use a special value that won't match any goal_sup

                # 5.b.ii. If current support is not the desired goal support
                if current_sup != goal_sup:
                    # Add 1 for the misplaced block itself
                    h += 1
                    # Add penalty for blocks on top of this misplaced block
                    # If the block is held, nothing is on top.
                    if current_sup != 'holding':
                         h += len(state_above.get(block, []))
            # 5.c. If block does not have a defined goal support, it doesn't
            # contribute to the heuristic based on its own position.

        return h
