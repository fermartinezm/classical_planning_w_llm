import re
from heuristics.heuristic_base import Heuristic
from task import Task


class blocksworldHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
    The heuristic estimates the distance to the goal state by summing two components:
    1. The number of blocks that are not in their correct position relative to the
       goal stack structure, starting from the table up.
    2. The number of blocks that are required to be clear in the goal state but
       are not clear in the current state.
    The heuristic is 0 if and only if the state is a goal state.

    Assumptions:
    - The domain is Blocksworld as defined in the provided PDDL file.
    - The state is represented as a frozenset of PDDL fact strings.
    - PDDL fact strings are in the format '(predicate arg1 arg2 ...)' or '(predicate)'.
    - The goal state is defined by a set of facts including (on ?x ?y),
      (on-table ?x), and (clear ?x).
    - The heuristic is used for greedy best-first search and does not need to be admissible.

    Heuristic Initialization:
    In the constructor, the heuristic pre-processes the task information:
    - It identifies all blocks present in the domain by parsing all possible facts
      defined in task.facts.
    - It parses the goal facts (task.goals) to build the desired goal structure:
        - `goal_below`: A dictionary mapping each block to the block it should be
          stacked on top of in the goal, or 'table' if it should be on the table.
        - `goal_clear_blocks`: A set of blocks that must be clear in the goal state.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Check if the state is the goal state. If yes, the heuristic is 0.
    2. Parse the current state facts to build the current structure:
        - `current_below`: A dictionary mapping each block to the block it is
          currently stacked on top of, or 'table' if it is on the table. Blocks
          that are currently held are not included in this mapping.
        - `current_clear_blocks`: A set of blocks that are currently clear.
    3. Compute the set of blocks that are "correctly stacked from the table up".
       A block B is correctly stacked from the table up if:
       - The goal requires B to be on the table (`goal_below[B] == 'table'`) AND
         B is currently on the table (`current_below[B] == 'table'`).
       - OR, the goal requires B to be on B_under (`goal_below[B] == B_under`) AND
         B is currently on B_under (`current_below[B] == B_under`) AND B_under is
         already determined to be correctly stacked from the table up.
       This is computed iteratively:
       - Initialize a set `correctly_stacked` with all blocks that are correctly
         on the table according to both the goal and the current state.
       - Repeatedly iterate through all blocks that have a defined goal base
         and are not yet in `correctly_stacked`. If a block B is currently on
         the block B_under that it should be on according to the goal, AND B_under
         is in `correctly_stacked`, add B to `correctly_stacked`. Repeat until
         no new blocks are added in a pass.
    4. Calculate the first component of the heuristic (h1): This is the count of
       blocks that have a defined goal base (`goal_below` contains the block) but
       are NOT in the `correctly_stacked` set. These blocks are not in their
       correct place within the goal stack structure.
    5. Calculate the second component of the heuristic (h2): This is the count of
       blocks that are in the `goal_clear_blocks` set but are NOT in the
       `current_clear_blocks` set. These are the required clear conditions that
       are not met.
    6. The total heuristic value is h1 + h2.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task

        # --- Heuristic Initialization ---
        # Extract all objects (blocks) from all possible facts
        self.all_blocks = set()
        for fact_str in task.facts:
            self.all_blocks.update(self._extract_objects_from_fact_string(fact_str))

        # Build goal structure: goal_below and goal_clear_blocks
        self.goal_below = {}  # block -> block_below or 'table'
        self.goal_clear_blocks = set()  # blocks that must be clear
        for goal_fact_str in task.goals:
            parts = self._extract_fact_parts(goal_fact_str)
            if not parts:
                continue
            predicate = parts[0]
            if predicate == 'on':
                # Fact is like '(on b1 b2)'
                if len(parts) == 3:
                    block_above, block_below = parts[1], parts[2]
                    self.goal_below[block_above] = block_below
            elif predicate == 'on-table':
                # Fact is like '(on-table b1)'
                if len(parts) == 2:
                    block = parts[1]
                    self.goal_below[block] = 'table'
            elif predicate == 'clear':
                # Fact is like '(clear b1)'
                if len(parts) == 2:
                    block = parts[1]
                    self.goal_clear_blocks.add(block)
            # Ignore other goal predicates if any (like arm-empty, holding)

    def _extract_fact_parts(self, fact_str):
        # Helper to parse fact string into predicate and arguments
        # Handles '(arm-empty)' case
        parts = fact_str.strip('()').split()
        return parts

    def _extract_objects_from_fact_string(self, fact_str):
        # Helper to extract objects from a fact string
        parts = self._extract_fact_parts(fact_str)
        if not parts:
            return []
        predicate = parts[0]
        # Objects are arguments for these specific predicates
        if predicate in ['on', 'on-table', 'clear', 'holding']:
            return parts[1:]
        return []

    def __call__(self, node):
        state = node.state

        # Check if goal is reached (heuristic is 0)
        if self.task.goal_reached(state):
            return 0

        # --- Step-By-Step Thinking for Computing Heuristic ---

        # 1. Build current state structure: current_below and current_clear_blocks
        current_below = {}  # block -> block_below or 'table'
        current_clear_blocks = set()  # blocks that are clear
        # We don't strictly need current_holding or arm_empty for this heuristic logic

        for fact_str in state:
            parts = self._extract_fact_parts(fact_str)
            if not parts:
                continue
            predicate = parts[0]
            if predicate == 'on':
                # Fact is like '(on b1 b2)'
                if len(parts) == 3:
                    block_above, block_below = parts[1], parts[2]
                    current_below[block_above] = block_below
            elif predicate == 'on-table':
                # Fact is like '(on-table b1)'
                if len(parts) == 2:
                    block = parts[1]
                    current_below[block] = 'table'
            elif predicate == 'clear':
                # Fact is like '(clear b1)'
                if len(parts) == 2:
                    block = parts[1]
                    current_clear_blocks.add(block)
            # Ignore other state predicates (holding, arm-empty) for current_below/clear

        # 2. Compute blocks correctly stacked from the table up
        correctly_stacked = set()
        # Initialize with blocks correctly on table according to goal AND state
        for block in self.goal_below: # Only consider blocks that have a goal base
            goal_base = self.goal_below[block]
            current_base = current_below.get(block) # Use get here as block might be held or not in state

            if goal_base == 'table' and current_base == 'table':
                correctly_stacked.add(block)

        # Propagate correctness upwards
        changed = True
        while changed:
            changed = False
            # Iterate only over blocks that have a goal base and are not yet correctly stacked
            # Create a list copy to avoid modifying the set while iterating
            for block in list(self.goal_below.keys()):
                 if block not in correctly_stacked:
                    goal_base = self.goal_below[block]
                    current_base = current_below.get(block)

                    # Check if goal requires block on goal_base, state has block on current_base,
                    # and current_base is the same as goal_base and goal_base is correctly stacked.
                    # We only propagate if the block is currently on *some* base (not held)
                    if goal_base != 'table' and current_base is not None and current_base != 'table' and current_base == goal_base and goal_base in correctly_stacked:
                         correctly_stacked.add(block)
                         changed = True

        # 3. Calculate heuristic components
        # Component 1: Blocks that have a goal base but are NOT correctly stacked from table up
        # These are blocks that are part of the desired stack structure but are misplaced relative to it.
        h1 = len([block for block in self.goal_below if block not in correctly_stacked])

        # Component 2: Unmet (clear B) goals where B is in goal_clear_blocks
        # These are blocks that need to be clear in the goal but are currently blocked.
        h2 = 0
        for block in self.goal_clear_blocks:
            # Check if the clear fact is NOT in the current state
            if '(clear ' + block + ')' not in state:
                 h2 += 1

        # 4. Total heuristic is the sum
        h_value = h1 + h2

        return h_value

