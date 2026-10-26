from heuristics.heuristic_base import Heuristic
from task import Task


class blocksworldHeuristic(Heuristic):
    """
    Domain-dependent heuristic for Blocksworld.

    Summary:
    Estimates the cost by counting blocks that are misplaced relative to their
    goal base, plus blocks that are on top of problematic blocks (either
    misplaced or needing to be clear), plus a penalty if the arm is not empty
    when it should be. This heuristic is designed for greedy best-first search
    and does not need to be admissible. It aims to prioritize states that are
    structurally closer to the goal configuration.

    Assumptions:
    - The input state and goal are valid Blocksworld states according to the
      provided PDDL domain.
    - Each block explicitly mentioned in an 'on' or 'on-table' goal fact has
      a unique desired base (either another block via 'on' or the table via
      'on-table').
    - The goal state does not require a block to be held (i.e., if 'holding'
      appears in the goal, it implies an invalid goal for this domain).
      '(arm-empty)' is the standard goal for the arm state.

    Heuristic Initialization:
    - Parses the goal facts provided in the Task object.
    - Stores desired 'on' relationships in a dictionary (block -> base_block).
    - Stores blocks that should be on the table in a set.
    - Stores blocks that should be clear in a set.
    - Records if '(arm-empty)' is a goal fact.
    - Identifies the set of all blocks explicitly mentioned in the 'on',
      'on-table', or 'clear' goal facts. This set defines the blocks relevant
      to the goal structure and clear conditions.

    Step-By-Step Thinking for Computing Heuristic:
    1. Parse the current state facts to extract the current configuration:
       - Current 'on' relationships (block -> base_block).
       - Blocks currently on the table.
       - Blocks that are currently clear.
       - The block currently being held (if any).
       - The current arm state (empty or holding).
    2. Identify 'wrongly_based_blocks': Iterate through the blocks identified
       in the goal structure during initialization. For each such block,
       determine its desired base (from goal facts) and its current base
       (from state facts - either a block it's on, or the table). If the
       current base does not match the desired base, the block is wrongly based.
       Count these blocks ('wrongly_based_count') and store them in a set
       ('wrongly_based_blocks').
    3. Identify 'problematic_blocks': This set includes blocks that are
       obstacles to achieving the goal structure or clear conditions. A block U
       is problematic if:
       - It is in the set of 'wrongly_based_blocks' (meaning it's part of the
         goal structure but in the wrong place relative to its base).
       - OR, '(clear U)' is a goal fact, but '(clear U)' is false in the
         current state (meaning something is on top of a block that needs to
         be clear).
    4. Count 'blocking_count': Iterate through the current 'on' relationships
       '(on B U)'. If the block below, U, is in the set of 'problematic_blocks',
       then the block on top, B, is considered a blocking block. Count these
       blocking blocks.
    5. Calculate the base heuristic value: This is the sum of
       'wrongly_based_count' and 'blocking_count'. This counts blocks that are
       themselves in the wrong place or are on top of blocks that are in the
       wrong place or need to be cleared.
    6. Add 'arm_empty_penalty': If '(arm-empty)' is a goal fact and the arm is
       not empty in the current state, add 1 to the heuristic value. This
       accounts for the action needed to free the arm.
    7. Return the final heuristic value. The heuristic is 0 if and only if
       all goal facts are satisfied (all blocks in goal structure are correctly
       based, all blocks needing to be clear are clear, and the arm is empty
       if required), ensuring it is 0 only at goal states.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.goal_on = {}
        self.goal_on_table = set()
        self.goal_clear = set()
        self.goal_arm_empty = False
        self.goal_blocks = set()

        # Preprocessing: Parse goal facts
        for fact_str in task.goals:
            parsed_fact = self._parse_fact(fact_str)
            predicate = parsed_fact[0]
            args = parsed_fact[1:]

            if predicate == 'on' and len(args) == 2:
                block, base = args
                self.goal_on[block] = base
                self.goal_blocks.add(block)
                self.goal_blocks.add(base)
            elif predicate == 'on-table' and len(args) == 1:
                block = args[0]
                self.goal_on_table.add(block)
                self.goal_blocks.add(block)
            elif predicate == 'clear' and len(args) == 1:
                block = args[0]
                self.goal_clear.add(block)
                # Add block to goal_blocks even if only in clear goal,
                # as its clear status is part of the goal.
                self.goal_blocks.add(block)
            elif predicate == 'arm-empty':
                self.goal_arm_empty = True
            # Ignore 'holding' in goal as it's not a standard final state

    def _parse_fact(self, fact_str):
        """Helper to parse a fact string into a list [predicate, arg1, ...]."""
        # Remove surrounding brackets and split by space
        # Handles facts like '(predicate arg1 arg2)' or '(predicate arg1)'
        parts = fact_str[1:-1].split()
        return parts

    def __call__(self, node):
        state = node.state

        # Parse current state facts
        current_on = {}
        current_on_table = set()
        current_clear = set()
        current_holding = set()
        current_arm_empty = False
        # current_blocks = set() # Not strictly needed for this heuristic logic

        for fact_str in state:
            parsed_fact = self._parse_fact(fact_str)
            predicate = parsed_fact[0]
            args = parsed_fact[1:]

            if predicate == 'on' and len(args) == 2:
                block, base = args
                current_on[block] = base
                # current_blocks.add(block)
                # current_blocks.add(base)
            elif predicate == 'on-table' and len(args) == 1:
                block = args[0]
                current_on_table.add(block)
                # current_blocks.add(block)
            elif predicate == 'clear' and len(args) == 1:
                block = args[0]
                current_clear.add(block)
                # current_blocks.add(block)
            elif predicate == 'holding' and len(args) == 1:
                block = args[0]
                current_holding.add(block)
                # current_blocks.add(block)
            elif predicate == 'arm-empty':
                current_arm_empty = True

        # 1. Identify wrongly based blocks in goal structure
        wrongly_based_count = 0
        wrongly_based_blocks = set() # Keep track for problematic check

        for block in self.goal_blocks:
            desired_base = None
            if block in self.goal_on:
                desired_base = self.goal_on[block]
            elif block in self.goal_on_table:
                desired_base = 'table' # Use a special value for table base

            # If block is in goal structure and has a desired base location
            if desired_base is not None:
                current_base = None
                if block in current_on:
                    current_base = current_on[block]
                elif block in current_on_table:
                    current_base = 'table'
                # If block is holding, its base is 'hand', which is never a goal base

                if current_base != desired_base:
                    wrongly_based_count += 1
                    wrongly_based_blocks.add(block)

        # 2. Identify problematic blocks
        problematic_blocks = set()
        # Problematic if wrongly based (and in goal structure)
        problematic_blocks.update(wrongly_based_blocks)
        # Problematic if needs to be clear but isn't
        for block in self.goal_clear:
            if block not in current_clear:
                 problematic_blocks.add(block)

        # 3. Count blocks blocking problematic blocks
        blocking_count = 0
        # Iterate through current 'on' relationships (B is on U)
        for block_on_top, block_below in current_on.items():
            if block_below in problematic_blocks:
                blocking_count += 1

        # 4. Calculate initial heuristic
        h = wrongly_based_count + blocking_count

        # 5. Add arm-empty penalty if applicable
        if self.goal_arm_empty and not current_arm_empty:
            h += 1
        # Note: If arm-empty is NOT a goal, we don't penalize holding a block.

        # Heuristic is 0 iff it's a goal state (checked in thought process).

        return h

