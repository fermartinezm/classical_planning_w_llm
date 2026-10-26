from collections import deque
from heuristics.heuristic_base import Heuristic
from task import Operator, Task # Assuming Task and Operator are available from task.py


class blocksworldHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
        This heuristic estimates the cost to reach the goal by identifying
        blocks that are not part of a correctly built "goal stack prefix"
        and summing the number of blocks currently on top of them plus one
        (for the block itself). It also adds a penalty if the arm should
        be empty in the goal but is currently holding a block.
        A block is part of a "goal stack prefix" if it is on the table
        as required by the goal, or it is on the block required by the goal
        AND the block below it is part of a goal stack prefix. This is
        computed bottom-up from the table.

    Assumptions:
        - The input task is a valid Blocksworld task according to the PDDL
          domain provided.
        - State and goal facts are represented as frozensets of strings
          like '(predicate arg1 arg2)'.
        - Blocks are unique objects.
        - The goal configuration consists of one or more stacks on the table.
        - The heuristic is designed for greedy best-first search and does
          not need to be admissible.

    Heuristic Initialization:
        The constructor parses the goal facts to build the target configuration:
        - `self.goal_below`: A dictionary mapping each block (that should be
          on another block or the table in the goal) to the block or 'table'
          it should be directly on top of.
        - `self.goal_clear`: A set of blocks that should be clear in the goal.
        - `self.goal_arm_empty`: A boolean indicating if the arm must be empty
          in the goal.
        This information is extracted once during initialization.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state:
        1. Parse the current state facts to determine:
           - `on_map`: Who is currently on whom (or on the table).
           - `holding_block`: Which block is being held, if any.
           - `all_blocks_in_state`: The set of all blocks present in the state.
        2. Identify blocks that are part of a "goal stack prefix" in the current state.
           - Initialize a boolean map `is_goal_prefix` for all blocks in the state to False.
           - Use a queue for propagation. Add any block `X` to the queue if it is currently
             on the table AND the goal requires it to be on the table. Mark `is_goal_prefix[X]` as True.
           - While the queue is not empty, take a block `Y`. Find any block `X` that is
             currently on `Y`. If the goal requires `X` to be on `Y` AND `is_goal_prefix[Y]`
             is True AND `is_goal_prefix[X]` is currently False, mark `is_goal_prefix[X]`
             as True and add `X` to the queue.
           - After propagation, `is_goal_prefix[X]` is True if block `X` is in its correct
             position relative to the block below it, AND that block is also in its correct
             position relative to the block below it, and so on, all the way down to a
             block correctly placed on the table according to the goal.
        3. Calculate the current height of each block in its stack.
           - Build an inverse map `on_top_map` from `on_map` (who is directly on top of whom).
           - Calculate the height (number of blocks above) for each block by traversing
             upwards from the block using `on_top_map`.
        4. Calculate the heuristic value `h`.
           - Initialize `h = 0`.
           - For each block `X` in the state:
             If `is_goal_prefix[X]` is False (meaning it's not part of a correctly built
             goal stack prefix):
               Add `1 + height[X]` to `h`. This estimates the cost to move `X` and all
               blocks currently on top of it out of the way.
           - If the goal requires the arm to be empty (`self.goal_arm_empty` is True)
             and the arm is currently holding a block (`holding_block` is not None):
               Add 1 to `h`. This accounts for the action needed to free the arm.
        5. Return `h`. The heuristic is 0 if and only if the state is a goal state.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.goal_below = {}
        self.goal_clear = set()
        self.goal_arm_empty = False
        self._parse_goal_facts(task.goals)

    def _parse_fact(self, fact_str):
        """Helper to parse a fact string into predicate and arguments."""
        # Remove parentheses and split by space
        parts = fact_str.strip('()').split()
        predicate = parts[0]
        args = parts[1:]
        return predicate, args

    def _parse_goal_facts(self, goals):
        """Parses goal facts to build goal configuration maps."""
        for fact_str in goals:
            predicate, args = self._parse_fact(fact_str)
            if predicate == 'on':
                # Goal fact (on A B) means A should be on B
                self.goal_below[args[0]] = args[1]
            elif predicate == 'on-table':
                # Goal fact (on-table A) means A should be on the table
                self.goal_below[args[0]] = 'table'
            elif predicate == 'clear':
                self.goal_clear.add(args[0])
            elif predicate == 'arm-empty':
                self.goal_arm_empty = True

    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # 1. Parse the current state facts
        on_map = {} # block -> block_below or 'table'
        clear_set = set()
        holding_block = None
        arm_empty = False
        all_blocks_in_state = set()

        for fact_str in state:
            predicate, args = self._parse_fact(fact_str)
            if predicate == 'on':
                on_map[args[0]] = args[1]
                all_blocks_in_state.add(args[0])
                all_blocks_in_state.add(args[1])
            elif predicate == 'on-table':
                on_map[args[0]] = 'table'
                all_blocks_in_state.add(args[0])
            elif predicate == 'clear':
                clear_set.add(args[0])
                all_blocks_in_state.add(args[0])
            elif predicate == 'holding':
                holding_block = args[0]
                all_blocks_in_state.add(args[0])
            elif predicate == 'arm-empty':
                arm_empty = True

        # 2. Identify blocks that are part of a "goal stack prefix"
        # A block X is a goal prefix if it's on the table and goal says so,
        # OR it's on Y and goal says so, AND Y is a goal prefix.
        is_goal_prefix = {block: False for block in all_blocks_in_state}
        q_prefix = deque()

        # Initialize queue with blocks correctly placed on the table according to the goal
        for block in all_blocks_in_state:
            # Check if block is on the table in the state
            if on_map.get(block) == 'table':
                # Check if block should be on the table in the goal
                if self.goal_below.get(block) == 'table':
                    is_goal_prefix[block] = True
                    q_prefix.append(block)

        # Propagate goal prefix status upwards
        while q_prefix:
            base_block = q_prefix.popleft()
            # Find blocks currently on top of base_block in the state
            blocks_on_top = [b for b, base in on_map.items() if base == base_block]
            for block_on_top in blocks_on_top:
                # Check if block_on_top should be on base_block in the goal
                if self.goal_below.get(block_on_top) == base_block:
                    # If it's in the correct relative position AND base is a goal prefix
                    # AND it hasn't been marked as goal prefix yet
                    if not is_goal_prefix[block_on_top]:
                        is_goal_prefix[block_on_top] = True
                        q_prefix.append(block_on_top)

        # 3. Calculate the current height of each block
        # Height is the number of blocks currently stacked directly on top of a block.
        height = {block: 0 for block in all_blocks_in_state}
        on_top_map = {} # block_below -> block_on_top
        for block, base in on_map.items():
            if base != 'table':
                on_top_map[base] = block

        # Find blocks that are currently at the top of stacks (not below anything)
        # These are blocks in all_blocks_in_state that are not values in on_top_map
        blocks_that_are_bases = set(on_top_map.keys())
        top_blocks = [block for block in all_blocks_in_state if block not in blocks_that_are_bases and block != holding_block]

        # Calculate height by traversing down from top blocks
        # A block's height is 1 + height of the block directly on top of it.
        # We can compute this bottom-up efficiently.
        # Re-initialize height for bottom-up calculation
        height = {block: 0 for block in all_blocks_in_state}
        q_height = deque([block for block in all_blocks_in_state if on_map.get(block) == 'table']) # Start with blocks on table

        # Add held block to queue if any
        if holding_block:
             q_height.append(holding_block)

        processed_for_height = set()

        while q_height:
            current = q_height.popleft()
            if current in processed_for_height:
                continue
            processed_for_height.add(current)

            # Find blocks currently on top of 'current'
            blocks_on_top_of_current = [b for b, base in on_map.items() if base == current]

            # If there are blocks on top, we need their heights first.
            # Re-add current to queue and add blocks on top if not processed.
            if blocks_on_top_of_current:
                 q_height.append(current) # Process current again after blocks on top are processed
                 for b_on_top in blocks_on_top_of_current:
                     if b_on_top not in processed_for_height:
                         q_height.append(b_on_top)
                 continue # Skip height calculation for now, wait for blocks on top

            # If no blocks are on top, or blocks on top are processed, calculate height
            # Height of current block = max height of blocks on top + 1 (if any block on top)
            max_height_on_top = 0
            for b_on_top in blocks_on_top_of_current:
                 max_height_on_top = max(max_height_on_top, height.get(b_on_top, 0) + 1) # Add 1 for the block itself

            height[current] = max_height_on_top


        # 4. Calculate the heuristic value h
        h = 0
        for block in all_blocks_in_state:
            if not is_goal_prefix.get(block, False):
                # Block is not part of a correctly built goal stack prefix
                # Add cost to move this block and everything on top of it
                # The height calculated is the number of blocks *above* it.
                h += (1 + height.get(block, 0))

        # Add penalty if arm should be empty but isn't
        if self.goal_arm_empty and holding_block is not None:
             h += 1 # Cost to put down or stack the held block

        return h
