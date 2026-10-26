import re
from collections import deque
from heuristics.heuristic_base import Heuristic
# Assuming the Heuristic base class is available in the specified path

class blocksworldHeuristic(Heuristic):
    """
    Summary:
    A domain-dependent heuristic for the blocksworld domain.
    It estimates the cost to reach the goal by identifying blocks that are
    not part of correctly built goal stack segments and blocks that should
    be clear but are not. This heuristic is designed for greedy best-first
    search and does not need to be admissible.

    Assumptions:
    - The goal state consists of conjunctions of (on ?x ?y), (on-table ?z),
      and potentially (clear ?w) predicates.
    - The goal predicates define one or more disjoint stacks of blocks
      or single blocks on the table.
    - Blocks mentioned in (clear ?w) goals are typically the top blocks
      of the goal stacks and are also mentioned in (on ?w ?) or (on-table ?w)
      predicates in the goal.

    Heuristic Initialization:
    The constructor pre-processes the goal state (`task.goals`) to build
    data structures representing the desired configuration:
    - self.goal_on: A dictionary mapping a block name to the name of the
      block it should be directly on top of in the goal state (e.g.,
      {'b1': 'b2'} if (on b1 b2) is a goal).
    - self.goal_on_table: A set of block names that should be directly on the
      table in the goal state (e.g., {'b3'} if (on-table b3) is a goal).
    - self.goal_clear: A set of block names that should be clear (have nothing
      on top) in the goal state (e.g., {'b1'} if (clear b1) is a goal).
    - self.goal_blocks: A set of all block names that are mentioned in the
      goal_on or goal_on_table predicates.
    - self.goal_above: A dictionary mapping a block name to the set of block
      names that should be directly on top of it in the goal state. This is
      the reverse mapping of self.goal_on, precomputed for efficient lookup
      during heuristic calculation.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state (`node.state`), the heuristic value is computed as follows:
    1. Parse the current state facts to identify the current (on ?x ?y),
       (on-table ?z), (clear ?w), and (holding ?x) relationships. Store these
       in temporary dictionaries/sets (current_on, current_on_table,
       current_clear, current_holding).
    2. Determine which blocks are part of a "correctly stacked" segment
       according to the goal. A block X is considered correctly stacked if:
       a) (on-table X) is a goal and (on-table X) is true in the current state, OR
       b) (on X Y) is a goal, (on X Y) is true in the current state, AND Y is
          also correctly stacked.
       This is computed iteratively using a queue (`collections.deque`). The process
       starts with blocks that should be on the table in the goal and are
       currently on the table. It then propagates "correctness" upwards through
       the goal stacks based on the current state configuration. A dictionary
       `is_correctly_stacked` tracks this status for each block in
       `self.goal_blocks`.
    3. Calculate the heuristic value as the sum of two penalty terms:
       a) Penalty 1: The number of blocks in `self.goal_blocks` for which
          `is_correctly_stacked` is False. These are blocks that are part of
          the goal configuration but are not currently positioned such that
          they form a correct stack segment from themselves down to the table.
          This captures the main structural errors in the goal stacks.
       b) Penalty 2: The number of blocks X that are in `self.goal_clear`
          (should be clear in the goal) but are NOT in `current_clear`
          (are not clear in the current state), PROVIDED that
          `is_correctly_stacked[X]` is True. This penalty applies when a block
          is otherwise in a correct stack segment (from itself downwards) but
          has unwanted blocks on top that need to be removed.
    4. Return the total heuristic value (Penalty 1 + Penalty 2).
    The heuristic is 0 if and only if all goal blocks are correctly stacked
    from the bottom up (`is_correctly_stacked` is True for all `goal_blocks`)
    and all blocks that should be clear are indeed clear. This corresponds
    to the goal state.
    """

    def __init__(self, task):
        super().__init__()
        self.goal_on = {}
        self.goal_on_table = set()
        self.goal_clear = set()
        self.goal_blocks = set()

        # Build goal structure from goal facts
        for goal_fact_str in task.goals:
            predicate, objects = self._parse_fact(goal_fact_str)
            if predicate == 'on':
                block_above, block_below = objects
                self.goal_on[block_above] = block_below
                self.goal_blocks.add(block_above)
                self.goal_blocks.add(block_below)
            elif predicate == 'on-table':
                block = objects[0]
                self.goal_on_table.add(block)
                self.goal_blocks.add(block)
            elif predicate == 'clear':
                block = objects[0]
                self.goal_clear.add(block)
            # Ignore other goal predicates like 'arm-empty'

        # Build reverse goal_on mapping for efficient lookup in __call__
        self.goal_above = {}
        for b_above, b_below in self.goal_on.items():
            self.goal_above.setdefault(b_below, set()).add(b_above)

    def _parse_fact(self, fact_str):
        """Helper function to parse a PDDL fact string like '(predicate obj1 obj2)'."""
        # Use regex to find sequences of word characters or hyphens, effectively
        # ignoring parentheses and spaces.
        parts = re.findall(r'[\w-]+', fact_str)
        if not parts:
            return None, [] # Handle empty or invalid fact strings
        predicate = parts[0]
        objects = parts[1:]
        return predicate, objects

    def __call__(self, node):
        state = node.state

        # Parse current state facts
        current_on = {}
        current_on_table = set()
        current_clear = set()
        current_holding = None # Only one block can be held at a time

        for fact_str in state:
            predicate, objects = self._parse_fact(fact_str)
            if predicate == 'on':
                if len(objects) == 2:
                    block_above, block_below = objects
                    current_on[block_above] = block_below
            elif predicate == 'on-table':
                if len(objects) == 1:
                    block = objects[0]
                    current_on_table.add(block)
            elif predicate == 'clear':
                if len(objects) == 1:
                    block = objects[0]
                    current_clear.add(block)
            elif predicate == 'holding':
                if len(objects) == 1:
                    current_holding = objects[0]
            # Ignore arm-empty and other facts not relevant to this heuristic

        # --- Compute is_correctly_stacked ---
        # is_correctly_stacked[block] is True if the block is the top of a
        # correctly built goal stack segment from itself down to the table.
        is_correctly_stacked = {block: False for block in self.goal_blocks}
        q = deque()

        # Base cases: blocks that should be on the table in the goal AND are currently on the table
        for block in self.goal_on_table:
            if block in current_on_table:
                is_correctly_stacked[block] = True
                q.append(block)

        # Propagate correctness upwards through goal stacks
        while q:
            block_below = q.popleft()

            # Find blocks that should be directly on top of block_below in the goal
            blocks_above_in_goal = self.goal_above.get(block_below, set())

            for block_above in blocks_above_in_goal:
                # Check if block_above is currently directly on block_below
                if current_on.get(block_above) == block_below:
                    # If this block hasn't been marked correctly stacked yet, mark it and queue it
                    if not is_correctly_stacked[block_above]:
                        is_correctly_stacked[block_above] = True
                        q.append(block_above)

        # --- Calculate heuristic value ---
        h_value = 0

        # Penalty 1: Count blocks in the goal structure that are NOT correctly stacked
        # from their position down to the table.
        for block in self.goal_blocks:
            if not is_correctly_stacked[block]:
                h_value += 1

        # Penalty 2: Count blocks that should be clear in the goal but aren't,
        # provided they are otherwise correctly stacked below them.
        for block in self.goal_clear:
            if block not in current_clear:
                 # Only add this penalty if the block is otherwise correctly stacked below it.
                 # If is_correctly_stacked[block] is False, the first penalty already covers it.
                 # This penalty specifically targets the cost of clearing the top of a correct stack segment.
                 if block in is_correctly_stacked and is_correctly_stacked[block]:
                     h_value += 1

        return h_value

