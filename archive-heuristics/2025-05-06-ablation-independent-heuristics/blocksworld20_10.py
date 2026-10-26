from heuristics.heuristic_base import Heuristic
from task import Task


class blocksworldHeuristic(Heuristic):
    """
    Summary:
    A domain-dependent, non-admissible heuristic for the Blocksworld domain,
    designed to guide a greedy best-first search.
    It estimates the number of actions required to reach the goal state
    by summing two components:
    1. For each block that is not on its correct goal support (either another
       block or the table), add 1 (for the action to move it) plus the number
       of blocks currently stacked on top of it (cost to clear it).
    2. For each block that is required to be clear in the goal state, but is
       not currently clear, and is not required to be a support for any other
       block in the goal state, add the number of blocks currently stacked
       on top of it (cost to clear it).

    Assumptions:
    - The input state is a valid Blocksworld state (blocks are either on
      another block, on the table, or held by the arm).
    - The goal state is reachable.
    - The PDDL domain follows the standard Blocksworld predicates (on,
      on-table, clear, holding, arm-empty).
    - Objects are represented as strings (e.g., 'b1').
    - Facts are represented as strings (e.g., '(on b1 b2)').

    Heuristic Initialization:
    The constructor processes the task's initial state and goal state to
    extract:
    - The set of all blocks involved in the problem (`self.all_blocks`).
    - The goal support for each block (which block it should be on, or 'table')
      stored in `self.goal_support_map`. Blocks not mentioned in goal 'on' or
      'on-table' facts do not have an entry.
    - The set of blocks that must be clear in the goal state (`self.goal_clear_set`).
    - The set of blocks that act as supports in goal 'on' facts (`self.goal_supports_set`).
    These are stored for efficient access during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state is the goal state by verifying if all goal
       facts are present in the state. If yes, return 0.
    2. Parse the current state to build data structures representing the
       current configuration:
       - `current_support_map`: Maps each block to its immediate support
         ('table', another block, or 'holding').
       - `current_stacked_on_map`: Maps each block to the block immediately
         on top of it (used for counting blocks above).
       - `current_clear_set`: Set of blocks that are currently clear.
    3. Initialize the heuristic value `h` to 0.
    4. Component 1 (Misplaced Support): Iterate through all blocks identified
       during initialization (`self.all_blocks`):
       - Get the block's goal support (`goal_sup`) from `self.goal_support_map`.
       - Get the block's current support (`current_sup`) from `current_support_map`.
       - If the block has a specific goal support (`goal_sup is not None`) and
         its `current_sup` is different from its `goal_sup`:
         - Increment `h` by 1 (representing the cost to move this block).
         - Calculate the number of blocks currently stacked on top of this block
           using `current_stacked_on_map` (iteratively). Add this count to `h`.
           This represents the cost to clear the block so it can be moved.
    5. Component 2 (Independent Clear Goals): Iterate through the set of blocks
       that must be clear in the goal state (`self.goal_clear_set`):
       - For each block `X` in `self.goal_clear_set`:
         - If `X` is not currently clear (`X` not in `current_clear_set`):
           - Check if `X` is a "top goal block" or its clear status is an
             independent goal. This is true if `X` is not a support for any
             goal 'on' fact (i.e., `X` is not in `self.goal_supports_set`).
           - If `X` is not in `self.goal_supports_set`:
             - Calculate the number of blocks currently stacked on top of `X`
               using `current_stacked_on_map` (iteratively). Add this count to `h`.
               This represents the cost to clear the block to satisfy the clear goal.
    6. Return the final value of `h`.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.goals = task.goals
        self.initial_state = task.initial_state

        self.goal_support_map = {} # block -> support (block or 'table')
        self.goal_clear_set = set()
        self.goal_supports_set = set() # blocks that are supports in goal 'on' facts
        self.all_blocks = set()

        # Extract all blocks and goal information
        all_facts = set(self.initial_state) | set(self.goals)
        for fact_str in all_facts:
            predicate, args = self._parse_fact(fact_str)
            if predicate == 'on':
                if len(args) == 2:
                    block_on_top, block_below = args
                    self.goal_support_map[block_on_top] = block_below
                    self.goal_supports_set.add(block_below)
                    self.all_blocks.add(block_on_top)
                    self.all_blocks.add(block_below)
            elif predicate == 'on-table':
                if len(args) == 1:
                    block = args[0]
                    self.goal_support_map[block] = 'table'
                    self.all_blocks.add(block)
            elif predicate == 'clear':
                if len(args) == 1:
                    block = args[0]
                    self.goal_clear_set.add(block)
                    self.all_blocks.add(block)
            elif predicate in ('holding', 'arm-empty'):
                 # Objects in holding might not appear elsewhere, add them.
                 self.all_blocks.update(args)

    def _parse_fact(self, fact_str):
        """Helper to parse a PDDL fact string."""
        # Remove leading '(' and trailing ')'
        fact_str = fact_str[1:-1]
        parts = fact_str.split()
        predicate = parts[0]
        args = parts[1:]
        return predicate, args

    def _count_blocks_on_top(self, block, stacked_on_map):
        """Iteratively counts blocks stacked on top of the given block."""
        count = 0
        current = block
        # stacked_on_map maps block_below -> block_on_top
        while current in stacked_on_map:
            current = stacked_on_map[current]
            count += 1
        return count


    def __call__(self, node):
        state = node.state

        # If goal is reached, heuristic is 0
        if self.goals <= state:
             return 0

        # Build current state data structures
        current_support_map = {} # block -> support ('table', block, or 'holding')
        current_stacked_on_map = {} # block_below -> block_on_top
        current_clear_set = set()
        # current_holding_block = None # Not strictly needed for heuristic calculation

        for fact_str in state:
            predicate, args = self._parse_fact(fact_str)
            if predicate == 'on':
                if len(args) == 2:
                    block_on_top, block_below = args
                    current_support_map[block_on_top] = block_below
                    current_stacked_on_map[block_below] = block_on_top # Correct mapping
            elif predicate == 'on-table':
                if len(args) == 1:
                    block = args[0]
                    current_support_map[block] = 'table'
            elif predicate == 'clear':
                if len(args) == 1:
                    block = args[0]
                    current_clear_set.add(block)
            elif predicate == 'holding':
                 if len(args) == 1:
                     block = args[0]
                     current_support_map[block] = 'holding' # Block is held, no support below
                     # current_holding_block = block # Not strictly needed

        h = 0

        # Component 1: Misplaced blocks based on support
        for block in self.all_blocks:
            goal_sup = self.goal_support_map.get(block)
            current_sup = current_support_map.get(block)

            # If the block has a specific goal support (on another block or table)
            # and its current support is different.
            if goal_sup is not None and current_sup != goal_sup:
                 h += 1 # Cost to move the block
                 # Add cost to clear the block if something is on top of it.
                 # If the block is currently held ('holding'), it's clear,
                 # and current_stacked_on_map won't have it as a key.
                 if block in current_stacked_on_map: # Check if something is on top
                     h += self._count_blocks_on_top(block, current_stacked_on_map)


        # Component 2: Blocks that need to be clear in goal but aren't,
        # and are top blocks in the goal structure (i.e., not supports).
        for block_to_clear in self.goal_clear_set:
            if block_to_clear not in current_clear_set:
                # Block needs to be clear but isn't.
                # Check if it's a top block in the goal structure (i.e., not a support for any goal 'on' fact).
                is_top_goal_block = block_to_clear not in self.goal_supports_set

                if is_top_goal_block:
                     # Add cost to clear this block.
                     # If the block is currently held, it's clear, so this branch
                     # would not have been reached (block_to_clear in current_clear_set).
                     if block_to_clear in current_stacked_on_map:
                         h += self._count_blocks_on_top(block_to_clear, current_stacked_on_map)

        return h
