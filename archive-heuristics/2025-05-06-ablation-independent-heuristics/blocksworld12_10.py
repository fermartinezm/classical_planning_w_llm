# Ensure the base class Heuristic is imported
from heuristics.heuristic_base import Heuristic
# Import Task for docstring type hinting
from task import Task

class blocksworldHeuristic(Heuristic):
    """
    Domain-dependent heuristic for Blocksworld.

    Summary:
    Estimates the cost by counting two types of "disorder" relative to the goal state:
    1. Blocks that are part of a goal stack but are not currently on their correct goal support (another block or the table).
    2. Blocks that are currently on top of a block that is part of a goal stack, but the block on top is not the one that should be there according to the goal stack structure.
    The sum of these two counts is multiplied by 2, as a rough estimate that each necessary correction involves at least two actions (e.g., unstack/pickup and stack/putdown). This heuristic is not admissible but aims to guide greedy best-first search efficiently by prioritizing states where blocks are closer to their goal positions or where fewer blocks are blocking goal configurations.

    Assumptions:
    - The input task is a valid Blocksworld task.
    - Goal states primarily involve specific (on ?x ?y) and (on-table ?x) configurations. (clear ?x) and (arm-empty) goals are not explicitly counted but are implicitly addressed by resolving the structural goals.
    - The set of objects (blocks) is static throughout the plan.
    - The heuristic is used for greedy best-first search and does not need to be admissible.

    Heuristic Initialization:
    The constructor pre-processes the goal facts to build data structures representing the desired goal configuration of blocks:
    - goal_pos: A dictionary mapping each block that is part of a goal stack to its required support (another block name or the string 'table'). This is derived from (on ?x ?y) and (on-table ?x) goal facts.
    - goal_stack_map: A dictionary mapping each block that is part of a goal stack (except the top block) to the block that should be directly on top of it in the goal. This is derived from (on ?x ?y) goal facts.
    - goal_block_set: A set containing all block names that appear as arguments in goal (on ?x ?y) or (on-table ?x) facts.

    Step-By-Step Thinking for Computing Heuristic:
    1. Parse the current state facts to identify the current configuration of blocks:
       - state_pos: A dictionary mapping each block currently on another block, on the table, or being held, to its current support (another block name, the string 'table', or a special value '__holding__').
       - state_stack_map: A dictionary mapping each block currently supporting another block to the block directly on top of it.
    2. Initialize heuristic value `h` to 0.
    3. Calculate `misplaced_support_count`: Iterate through each block in `self.goal_block_set`. Retrieve its required goal support from `self.goal_pos` and its current support from `state_pos`. If the block has a defined goal support (which is true for all blocks in `self.goal_block_set` by construction) and its current support is different from the goal support, increment the count. A block being held is considered to have '__holding__' as support. A block not mentioned in state_pos (which shouldn't happen for blocks from the problem instance) is considered to have None as support.
    4. Calculate `incorrect_on_top_count`: Iterate through each (block_below, block_on_top) pair in `state_stack_map` (representing an (on block_on_top block_below) fact in the state).
       - Check if `block_below` is part of a goal stack (i.e., `block_below` is in `self.goal_block_set`).
       - If it is, find the block that should be on top of `block_below` in the goal stack (`self.goal_stack_map.get(block_below)`).
       - If the block currently on top (`block_on_top`) is not the block that should be there in the goal stack (either because nothing specific should be on it in the goal stack according to `goal_stack_map`, or a different block should be), increment the count.
    5. The final heuristic value is `(misplaced_support_count + incorrect_on_top_count) * 2`.
    6. The heuristic is 0 if and only if the state is a goal state (with respect to the on/on-table configuration). It is finite for all solvable states.
    """
    def __init__(self, task):
        super().__init__()
        self.goals = task.goals

        # Pre-process goal facts to build goal structure maps
        self.goal_pos = {} # block -> goal_support (block or 'table')
        self.goal_stack_map = {} # block_below -> block_above
        self.goal_block_set = set() # all blocks mentioned in goal on/on-table

        for g in self.goals:
            parts = g.strip('()').split()
            predicate = parts[0]
            if predicate == 'on':
                block_above = parts[1]
                block_below = parts[2]
                self.goal_pos[block_above] = block_below
                self.goal_stack_map[block_below] = block_above
                self.goal_block_set.add(block_above)
                self.goal_block_set.add(block_below)
            elif predicate == 'on-table':
                block = parts[1]
                self.goal_pos[block] = 'table'
                self.goal_block_set.add(block)
            # Ignore 'clear' and 'arm-empty' goals for building the core structural goal

    def __call__(self, node):
        state = node.state

        # Parse state facts to build current structure maps
        state_pos = {} # block -> current_support (block or 'table' or special value)
        state_stack_map = {} # block_below -> block_above

        # Define a special value for 'holding' support
        HOLDING_SUPPORT = '__holding__'

        for fact in state:
            parts = fact.strip('()').split()
            predicate = parts[0]
            if predicate == 'on':
                block_above = parts[1]
                block_below = parts[2]
                state_pos[block_above] = block_below
                state_stack_map[block_below] = block_above
            elif predicate == 'on-table':
                block = parts[1]
                state_pos[block] = 'table'
            elif predicate == 'holding':
                block = parts[1]
                state_pos[block] = HOLDING_SUPPORT
            # Ignore 'clear', 'arm-empty' for structural heuristic

        misplaced_support_count = 0
        # Part 1: Count blocks not on their correct goal support
        # Only consider blocks that are part of the goal structure
        for block in self.goal_block_set:
            goal_support = self.goal_pos.get(block)
            # current_support will be None if the block is not in state_pos (e.g., not on/on-table/holding).
            # In a valid blocksworld state, every block is either on another block, on the table, or held.
            # So state_pos should contain all blocks from the problem instance.
            current_support = state_pos.get(block)

            # If block is supposed to be on something or table, check if it is
            # The check `goal_support is not None` is technically redundant because goal_block_set
            # is populated only from 'on' and 'on-table' goals, ensuring goal_pos has an entry.
            if current_support != goal_support:
                misplaced_support_count += 1

        incorrect_on_top_count = 0
        # Part 2: Count blocks that are on top of a block B, where B is part of a goal stack,
        # and the block on top is *not* the one that should be there in the goal stack.
        # Iterate through all blocks that are *on* something in the state.
        for block_below, block_on_top in state_stack_map.items():
            # Is block_below part of a goal stack?
            if block_below in self.goal_block_set:
                # What block should be on block_below in the goal?
                goal_block_on_top = self.goal_stack_map.get(block_below)
                # If there should be no block on top (it should be clear above this point in goal stack)
                # OR if the block on top is the wrong one
                if goal_block_on_top is None or goal_block_on_top != block_on_top:
                    incorrect_on_top_count += 1

        # The heuristic is the sum of the two counts, multiplied by 2.
        # This is a non-admissible estimate of actions needed.
        h = (misplaced_support_count + incorrect_on_top_count) * 2

        return h
