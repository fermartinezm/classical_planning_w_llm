# blocksworldHeuristic.py

class blocksworldHeuristic:
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
    This heuristic estimates the number of blocks that need to be moved
    to reach the goal state. It counts blocks that are currently on a base
    (another block or the table) that is not their required goal base,
    plus any blocks currently stacked on top of them. It also counts blocks
    that need to be cleared according to the goal but are not clear, plus
    any blocks stacked on top of them. Finally, it adds 1 if the arm needs
    to be empty according to the goal but is currently holding a block.

    Assumptions:
    - The input task is a valid Blocksworld planning task.
    - The state representation is a frozenset of strings as described.
    - The goal is a conjunction of facts.
    - The heuristic is intended for greedy best-first search and does
      not need to be admissible.
    - The problem is solvable (heuristic value is finite).

    Heuristic Initialization:
    The constructor precomputes information from the task's goal:
    - `goal_on_facts`: A set of goal facts of the form `(on X Y)`.
    - `goal_on_table_facts`: A set of goal facts of the form `(on-table X)`.
    - `goal_clear_blocks`: A set of blocks X for which `(clear X)` is a
      goal fact.
    - `goal_arm_empty`: A boolean indicating if `(arm-empty)` is a goal fact.
    This precomputation avoids redundant parsing of the goal in every
    heuristic call.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Parse the current state facts to build data structures representing
       the current configuration:
       - `current_config`: Maps each block X to the block Y it is currently
         on, or 'table' if on the table, or 'holding' if held.
       - `current_stack_above`: Maps each block Y to the block X currently
         on top of it (if any).
       - `current_stack_below`: Maps each block X to the block Y it is
         currently on (if any block is below it).
       - `held_block`: The block currently held by the arm, or None.
    2. Identify `blocks_on_wrong_base`: A set of blocks X such that X is
       currently on a block Y, and `(on X Y)` is not a goal fact, OR X is
       currently on the table, and `(on-table X)` is not a goal fact.
       Blocks that are held are not considered here.
    3. Identify `blocks_to_clear_now`: The subset of `goal_clear_blocks`
       that are not currently clear in the state.
    4. Initialize `blocks_that_must_be_moved`: A set starting with the union
       of `blocks_on_wrong_base` and `blocks_to_clear_now`. These blocks are
       either on the wrong base or need to be cleared.
    5. Add blocks currently stacked on top of any block already in
       `blocks_that_must_be_moved`. This is done by iterating through
       currently stacked blocks and checking if they are on top of a block
       in the set using a recursive helper function `_is_on_block_set`
       with memoization.
    6. The heuristic value `h` is the total number of unique blocks in
       `blocks_that_must_be_moved`.
    7. If `goal_arm_empty` is True and the arm is not currently empty
       (`held_block` is not None), increment `h` by 1 (for the putdown action).
    8. Return `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic with goal information from the task.

        @param task: The planning task object.
        """
        self.goal_on_facts = set()
        self.goal_on_table_facts = set()
        self.goal_clear_blocks = set()
        self.goal_arm_empty = False

        # Precompute goal information
        for goal_fact in task.goals:
            if goal_fact.startswith('(on '):
                self.goal_on_facts.add(goal_fact)
            elif goal_fact.startswith('(on-table '):
                self.goal_on_table_facts.add(goal_fact)
            elif goal_fact.startswith('(clear '):
                _, block = self._parse_fact(goal_fact)
                self.goal_clear_blocks.add(block)
            elif goal_fact == '(arm-empty)':
                self.goal_arm_empty = True

    def _parse_fact(self, fact_str):
        """Parses a fact string into a tuple (predicate, arg1, arg2, ...)."""
        # Remove parentheses and split by space
        parts = fact_str.strip('()').split()
        if not parts:
            return () # Should not happen for valid PDDL facts
        return tuple(parts)

    def _is_on_block_set(self, block, block_set, current_stack_below_map, memo):
        """
        Recursive helper to check if a block is currently stacked on top of
        any block in the given set.
        """
        if block in memo:
            return memo[block]

        if block in block_set:
            memo[block] = True
            return True

        below = current_stack_below_map.get(block)
        if below is None or below == 'table' or below == 'holding':
            # Block is on the table or held, not on another block.
            memo[block] = False
            return False

        # Block is on 'below'. Check if 'below' is in the set or on something in the set.
        result = self._is_on_block_set(below, block_set, current_stack_below_map, memo)
        memo[block] = result
        return result

    def __call__(self, state, task):
        """
        Computes the heuristic value for the given state.

        @param state: The current state (frozenset of facts).
        @param task: The planning task object (needed for goal facts check).
                     Note: task is passed by the search algorithm, although
                     most goal info is precomputed in __init__. We use it
                     here to check current state against goal clear facts.
        @return: The heuristic value (integer).
        """
        # 1. Parse current state
        current_config = {}
        current_stack_above = {}
        current_stack_below = {}
        held_block = None

        for fact in state:
            if fact.startswith('(on '):
                _, block, below = self._parse_fact(fact)
                current_config[block] = below
                current_stack_above[below] = block
                current_stack_below[block] = below
            elif fact.startswith('(on-table '):
                _, block = self._parse_fact(fact)
                current_config[block] = 'table'
            elif fact.startswith('(holding '):
                _, block = self._parse_fact(fact)
                current_config[block] = 'holding'
                held_block = block
            # Note: (clear ...) and (arm-empty) facts don't define position/stack structure

        # 2. Identify blocks on the wrong base
        blocks_on_wrong_base = set()
        # Iterate through all blocks present in the current state config
        for block in current_config:
            current_below = current_config[block]
            if current_below == 'holding': continue # Held blocks are handled by arm-empty goal

            if current_below == 'table':
                # Block is on table. Is this its goal position?
                if f'(on-table {block})' not in self.goal_on_table_facts:
                    blocks_on_wrong_base.add(block)
            else: # Block is on a block Y = current_below
                # Is this its goal position?
                if f'(on {block} {current_below})' not in self.goal_on_facts:
                    blocks_on_wrong_base.add(block)

        # 3. Identify blocks that need clearing for goal and are not clear
        blocks_to_clear_now = {
            block for block in self.goal_clear_blocks
            if f'(clear {block})' not in state
        }

        # 4. Initialize blocks that must be moved
        blocks_that_must_be_moved = set(blocks_on_wrong_base).union(blocks_to_clear_now)

        # 5. Add blocks currently stacked on top of any block already in blocks_that_must_be_moved.
        memo_on_set = {}
        # Iterate through all blocks that are currently on top of something
        all_current_stacked_blocks = set(current_stack_below.keys())
        for block in all_current_stacked_blocks:
            if self._is_on_block_set(block, blocks_that_must_be_moved, current_stack_below, memo_on_set):
                 blocks_that_must_be_moved.add(block)

        # 6. The heuristic value h is the total number of unique blocks in blocks_that_must_be_moved.
        h = len(blocks_that_must_be_moved)

        # 7. Add cost for arm-empty goal if arm is not empty
        if self.goal_arm_empty and held_block is not None:
            h += 1 # Need one putdown action

        # 8. Return h
        return h
