import collections

class blocksworldHeuristic:
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
        This heuristic estimates the number of blocks that are not part of a
        correctly built stack segment that matches the goal configuration and
        is rooted correctly on the table. A block is considered "correctly placed"
        if it is on the table and the goal requires it to be on the table, OR
        if it is on another block A, the goal requires it to be on A, AND A is
        itself correctly placed. The heuristic counts the total number of blocks
        minus the number of correctly placed blocks.

    Assumptions:
        - Assumes standard Blocksworld problems where goals primarily define
          the desired stack configurations using `on` and `on-table` predicates.
        - Assumes goal states typically imply `arm-empty` and `clear` for the
          top blocks of goal stacks.
        - Assumes all blocks involved in the goal configuration are present
          in the initial state.
        - The heuristic value is 0 if and only if the state is a goal state
          (under standard Blocksworld assumptions).

    Heuristic Initialization:
        The constructor processes the goal facts from the task to build data
        structures representing the target configuration:
        - `goal_on`: A dictionary mapping a block to the block it should be
          directly on top of in the goal state (`{block_on_top: block_below}`).
        - `goal_on_table`: A set of blocks that should be directly on the table
          in the goal state.
        - `goal_above`: A dictionary mapping a block to the list of blocks
          that should be directly on top of it in the goal state
          (`{block_below: [block_on_top1, ...]}`). This is pre-calculated
          for efficient lookup during heuristic computation.
        - `all_blocks`: A set of all unique blocks present in the initial state
          or goal facts.
        - `total_blocks`: The total count of unique blocks.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state:
        1.  Parse the current state facts to determine the current position of
            each block:
            - `current_on`: Dictionary mapping a block to the block it is
              currently on (`{block_on_top: block_below}`).
            - `current_on_table`: Set of blocks currently on the table.
            - `current_held`: The block currently held by the arm, or None.
              (Note: `current_held` is parsed but not directly used in the
               core structural count, as held blocks are implicitly not
               part of `current_on` or `current_on_table` and thus not
               counted as correctly placed by the BFS).
        2.  Initialize an empty set `correctly_placed_blocks` to store blocks
            that are part of a correctly built stack segment.
        3.  Initialize a queue `q` with blocks that are correctly placed on the
            table according to the goal and state: Iterate through blocks in
            `goal_on_table`. If a block B is in `goal_on_table` AND is also in
            `current_on_table`, add B to `correctly_placed_blocks` and enqueue B.
        4.  Perform a Breadth-First Search (BFS) starting from the blocks in the
            queue:
            - While the queue `q` is not empty, dequeue a block `base_block`.
            - Look up blocks that should be directly on top of `base_block`
              in the goal using the pre-calculated `goal_above` map.
            - For each such `block_on_top`:
                - Check if `block_on_top` is currently directly on `base_block`
                  in the state (`current_on.get(block_on_top) == base_block`).
                - If the condition is met AND `block_on_top` is not already
                  in `correctly_placed_blocks`, add `block_on_top` to
                  `correctly_placed_blocks` and enqueue `block_on_top`.
        5.  After the BFS completes, `correctly_placed_blocks` contains all
            blocks that are part of stack segments matching the goal structure
            and rooted correctly on the table.
        6.  The heuristic value is calculated as the total number of blocks
            minus the number of blocks in `correctly_placed_blocks`. This
            counts the blocks that are "out of place" relative to the goal
            stack structure.
    """
    def __init__(self, task):
        # Store goal configuration
        self.goal_on = {} # {block_on_top: block_below}
        self.goal_on_table = set() # {block}
        self.goal_above = collections.defaultdict(list) # {block_below: [block_on_top1, ...]}
        self.all_blocks = set()

        # Parse goal facts
        for fact_str in task.goals:
            pred, args = self.parse_fact(fact_str)
            if pred == 'on':
                # Ensure args has at least 2 elements
                if len(args) == 2:
                    block_on_top, block_below = args
                    self.goal_on[block_on_top] = block_below
                    self.goal_above[block_below].append(block_on_top)
                    self.all_blocks.add(block_on_top)
                    self.all_blocks.add(block_below)
            elif pred == 'on-table':
                 # Ensure args has at least 1 element
                 if len(args) == 1:
                    self.goal_on_table.add(args[0])
                    self.all_blocks.add(args[0])
            # Ignore 'clear' and 'arm-empty' facts in goal for structural heuristic calculation

        # Parse initial state facts to get all blocks (in case some are only in init)
        # This ensures we know the total number of blocks in the problem.
        for fact_str in task.initial_state:
             pred, args = self.parse_fact(fact_str)
             # Add all arguments that are not the 'arm-empty' predicate itself
             if pred != 'arm-empty':
                 for arg_item in args:
                     self.all_blocks.add(arg_item)

        self.total_blocks = len(self.all_blocks)

    def parse_fact(self, fact_str):
        """Helper to parse a PDDL fact string into predicate and arguments."""
        # Remove parentheses and split by space
        parts = fact_str[1:-1].split()
        predicate = parts[0]
        args = tuple(parts[1:])
        return predicate, args

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state: A frozenset of strings representing the current state facts.
        @return: An integer heuristic value.
        """
        # If there are no blocks defined in the problem, the heuristic is always 0.
        if self.total_blocks == 0:
            return 0

        # Parse current state configuration
        current_on = {} # {block_on_top: block_below}
        current_on_table = set() # {block}
        # current_held = None # Not strictly needed for this heuristic logic

        for fact_str in state:
            pred, args = self.parse_fact(fact_str)
            if pred == 'on':
                # Ensure args has at least 2 elements before accessing args[0], args[1]
                if len(args) == 2:
                    current_on[args[0]] = args[1]
            elif pred == 'on-table':
                 # Ensure args has at least 1 element before accessing args[0]
                 if len(args) == 1:
                    current_on_table.add(args[0])
            # We don't need to track 'holding' or 'clear' explicitly for this structural heuristic

        # Find correctly built stack segments
        correctly_placed_blocks = set()
        q = collections.deque() # Use deque for efficient queue operations

        # Step 3: Identify initial blocks on table that match goal
        for block in self.goal_on_table:
            if block in current_on_table:
                correctly_placed_blocks.add(block)
                q.append(block)

        # Step 4: Extend segments upwards using BFS
        while q:
            base_block = q.popleft() # Dequeue

            # Find blocks that should be on base_block in the goal
            # Use the pre-calculated goal_above map
            blocks_to_place_on_base = self.goal_above.get(base_block, [])

            for block_on_top in blocks_to_place_on_base:
                # Check if block_on_top is currently on base_block
                if current_on.get(block_on_top) == base_block:
                    # Check if block_on_top is already counted
                    if block_on_top not in correctly_placed_blocks:
                        correctly_placed_blocks.add(block_on_top)
                        q.append(block_on_top)

        # Step 6: Calculate heuristic value
        h_value = self.total_blocks - len(correctly_placed_blocks)

        # Ensure heuristic is 0 only for goal states.
        # The calculation `total_blocks - correctly_placed_blocks` results in 0
        # if and only if all blocks are part of a stack structure that perfectly
        # matches the goal structure from the table up. For standard blocksworld
        # problems, this state is indeed the goal state (as it implies all
        # (on X Y) and (on-table Z) goal facts are met, no block is held,
        # and required (clear X) facts for stack tops are met).
        # We rely on this property for the "h is 0 only for goal states" requirement.

        return h_value
