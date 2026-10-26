from heuristics.heuristic_base import Heuristic
from task import Task


class blocksworldHeuristic(Heuristic):
    """
    Summary:
    A domain-dependent heuristic for the Blocksworld domain. It estimates the
    number of actions required to reach the goal state by counting blocks that
    are in the wrong position or are obstructing the placement of other blocks,
    and adding a cost for the arm state if it conflicts with the goal. This
    heuristic is non-admissible and designed for greedy best-first search.

    Assumptions:
    - The domain is Blocksworld with standard predicates (on, on-table, clear,
      holding, arm-empty) and actions (pickup, putdown, stack, unstack).
    - Block names are strings, typically starting with 'b'.
    - The goal state is defined by a conjunction of facts involving on, on-table,
      clear, and optionally arm-empty.
    - The heuristic assumes a cost of 2 for moving a block (representing
      pickup/unstack + putdown/stack) and 1 for changing the arm state.

    Heuristic Initialization:
    In the constructor, the heuristic precomputes information about the goal state
    from the task definition:
    - `goal_pos`: A dictionary mapping blocks that appear in 'on' or 'on-table'
      goal facts to their target location (another block name or the string 'table').
    - `goal_pos_inverse`: A dictionary mapping blocks that appear as the 'underob'
      in an 'on' goal fact to the block that should be directly on top of it.
    - `goal_blocks`: A set of all blocks explicitly mentioned in 'on' or 'on-table'
      goal facts.
    - `goal_clear_blocks`: A set of blocks that must be clear in the goal state,
      based on 'clear' goal facts.
    - `goal_arm_empty`: A boolean indicating whether `(arm-empty)` is a goal fact.
    - `all_blocks`: A set of all possible block names in the problem instance,
      extracted from all facts in the task definition. This is used to initialize
      state-dependent dictionaries to cover all relevant objects.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state, the heuristic value `h` is computed as follows:
    1.  Initialize `h` to 0.
    2.  Build a representation of the current state:
        - `current_pos`: A dictionary mapping each block to its current location
          (another block name or 'table'). Blocks that are held will not have
          an entry set here, effectively defaulting to `None` from initialization.
        - `current_above`: A dictionary mapping each block to the block currently
          sitting directly on top of it, or `None` if it is clear.
        - `arm_empty_in_state`: A boolean indicating whether `(arm-empty)` is true
          in the current state.
    3.  Add cost for misplaced goal blocks: Iterate through each block in `goal_blocks`.
        If its current position (`current_pos.get(block)`) is different from its
        goal position (`goal_pos[block]`), add 2 to `h`. This counts blocks that
        are not in their desired final location.
    4.  Add cost for obstructing blocks: Iterate through all blocks that could
        potentially be supporting another block (`all_blocks`). If a block `b_under`
        currently has a block `b_top` on it (`current_above[b_under] is not None`),
        check if this conflicts with the goal state for `b_under`:
        - If `b_under` is in `goal_pos_inverse` (meaning a specific block should be
          on it in the goal) AND the block currently on it (`b_top`) is *not* the
          block that should be on it (`goal_pos_inverse[b_under]`), add 2 to `h`.
          This counts blocks that are obstructing a correct stack.
        - Elif `b_under` is in `goal_clear_blocks` (meaning it should be clear
          in the goal state), add 2 to `h`. This counts blocks that are obstructing
          a block that needs to be clear.
    5.  Add cost for arm state: If `(arm-empty)` is a goal fact (`goal_arm_empty` is True)
        and the arm is not empty in the current state (`arm_empty_in_state` is False),
        add 1 to `h`. This counts the need to put down a held block.
    6.  Return the total heuristic value `h`.

    The heuristic value is 0 if and only if the state is a goal state, as all
    discrepancy counts will be zero in a goal state, and any non-goal state
    will have at least one discrepancy counted (assuming a solvable problem
    and standard Blocksworld goals).
    """

    def __init__(self, task: Task):
        super().__init__()
        self.task = task

        # Precompute goal information
        self.goal_pos = {}
        self.goal_pos_inverse = {}
        self.goal_blocks = set()
        self.goal_clear_blocks = set()
        self.goal_arm_empty = False

        for fact_str in self.task.goals:
            predicate, args = self.parse_fact(fact_str)
            if predicate == 'on':
                block = args[0]
                under_block = args[1]
                self.goal_pos[block] = under_block
                self.goal_pos_inverse[under_block] = block
                self.goal_blocks.add(block)
                self.goal_blocks.add(under_block)
            elif predicate == 'on-table':
                block = args[0]
                self.goal_pos[block] = 'table'
                self.goal_blocks.add(block)
            elif predicate == 'clear':
                block = args[0]
                self.goal_clear_blocks.add(block)
            elif predicate == 'arm-empty':
                self.goal_arm_empty = True

        # Collect all possible block names from task facts
        self.all_blocks = set()
        for fact_str in self.task.facts:
            _, args = self.parse_fact(fact_str)
            for arg in args:
                # Assuming block names start with 'b' and are not predicates
                if arg.startswith('b'):
                    self.all_blocks.add(arg)

    def parse_fact(self, fact_str):
        """Helper to parse a fact string into predicate and arguments."""
        # Example: '(on b1 b2)', '(arm-empty)'
        # Remove outer brackets
        content = fact_str[1:-1]
        parts = content.split()
        predicate = parts[0]
        args = parts[1:]
        return predicate, args

    def __call__(self, node):
        state = node.state
        h = 0

        # Build current state representation
        current_pos = {block: None for block in self.all_blocks}
        current_above = {block: None for block in self.all_blocks}
        arm_empty_in_state = False

        for fact_str in state:
            predicate, args = self.parse_fact(fact_str)
            if predicate == 'on':
                block = args[0]
                under_block = args[1]
                current_pos[block] = under_block
                current_above[under_block] = block
            elif predicate == 'on-table':
                block = args[0]
                current_pos[block] = 'table'
            elif predicate == 'holding':
                # Block is held, its position is implicitly 'hand'.
                # We don't need to store 'hand' in current_pos for this heuristic logic.
                pass
            elif predicate == 'arm-empty':
                arm_empty_in_state = True

        # 1. Add cost for misplaced goal blocks
        for block in self.goal_blocks:
            current_p = current_pos.get(block) # .get(block) handles blocks not in state (e.g. held)
            goal_p = self.goal_pos[block]

            if current_p != goal_p:
                h += 2

        # 2. Add cost for obstructing blocks
        # Iterate through all blocks that *could* be supporting something
        for block_under in self.all_blocks:
            block_top = current_above.get(block_under)
            if block_top is not None: # There is a block on top of block_under
                # Check if block_under is involved in a goal fact that conflicts with block_top being on it
                # Case 1: block_under should have a *different* block on it in the goal
                if block_under in self.goal_pos_inverse:
                    block_top_goal = self.goal_pos_inverse[block_under]
                    if block_top != block_top_goal:
                        h += 2
                # Case 2: block_under should be clear in the goal
                elif block_under in self.goal_clear_blocks:
                     h += 2
                # Note: If block_under is not in goal_pos_inverse and not in goal_clear_blocks,
                # its state of having something on it is not directly conflicting with a goal fact.

        # 3. Add cost for arm state
        if self.goal_arm_empty and not arm_empty_in_state:
            h += 1

        return h
