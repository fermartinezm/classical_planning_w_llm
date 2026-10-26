# from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # PDDL facts are typically like "(predicate arg1 arg2)"
    if fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    # Fallback for unexpected formats, though unlikely with valid PDDL
    return fact.split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    Estimates the number of blocks that are misplaced within the goal stack structure
    plus the number of blocks that are obstructing goal-relevant locations, plus a penalty for a non-empty arm.

    # Heuristic Initialization
    - Parse goal facts to identify the desired position for each block in a goal stack.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. Parse the goal facts to build the desired stack configuration (`goal_config`)
       mapping each block that should be on top of something or the table to its desired location below.
       Also build an inverse map (`goal_block_on_top`) and track blocks that should be on the table (`goal_block_on_table`).
       Identify all blocks mentioned in goal `on`/`on-table` facts (`goal_blocks`).
    2. Parse the current state facts to build the current configuration (`current_config`),
       mapping each block that is on top of something or the table to its current location below.
       Also build the current stack structure (`current_stacks`) and identify the block being held (`current_holding`).
    3. Compute Cost 1: Iterate through each block that is a key in `goal_config` (i.e., a block that is supposed to be on top of something or the table). Check recursively if this block is in its correct goal position relative to the goal stack below it. Count the number of such blocks that are *not* correctly placed.
    4. Compute Cost 2: Iterate through all blocks that are currently placed on top of something or the table. For each such block `X` currently on `Y`: if `Y` is a goal block, or if `Y` is 'table' and some block should be on the table according to the goal, and `X` is *not* the block that should be directly on `Y` according to the goal, increment an obstructing count.
    5. Compute Cost 3: If the arm is currently holding a block, add 1 to the cost.
    6. The total heuristic value is the sum of Cost 1, Cost 2, and Cost 3.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration.
        """
        super().__init__(task)

        # Parse goal facts to build the desired stack configuration
        # goal_config[block] = block_below_or_table
        self.goal_config = {}
        # goal_block_on_top[block_below] = block_on_top (for non-table)
        self.goal_block_on_top = {}
        # goal_block_on_table = set of blocks that should be on the table
        self.goal_block_on_table = set()
        # goal_blocks = set of all blocks mentioned in goal on/on-table facts
        self.goal_blocks = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'on':
                block_on_top = parts[1]
                block_below = parts[2]
                self.goal_config[block_on_top] = block_below
                self.goal_block_on_top[block_below] = block_on_top
                self.goal_blocks.add(block_on_top)
                self.goal_blocks.add(block_below)
            elif parts[0] == 'on-table':
                block = parts[1]
                self.goal_config[block] = 'table'
                self.goal_block_on_table.add(block)
                self.goal_blocks.add(block)
            # Ignore (clear ?) goals, as they are usually consequences

        # Static facts are not needed for this heuristic in Blocksworld.
        # self.static = task.static # Already done by super().__init__


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse current state facts
        current_config = {} # block -> block_below_or_table
        current_stacks = {} # block_below -> list of blocks directly on top
        current_holding = None

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                block_on_top = parts[1]
                block_below = parts[2]
                current_config[block_on_top] = block_below
                if block_below not in current_stacks:
                    current_stacks[block_below] = []
                current_stacks[block_below].append(block_on_top)
            elif parts[0] == 'on-table':
                block = parts[1]
                current_config[block] = 'table'
            elif parts[0] == 'holding':
                current_holding = parts[1]

        # --- Heuristic Calculation ---

        total_cost = 0

        # Cost 1: Number of blocks in goal stacks that are not in their correct position
        # relative to the goal stack below them.
        correctly_placed_memo = {}
        misplaced_goal_blocks = set()
        # Iterate through blocks that are *supposed* to be on top of something or the table
        for block in self.goal_config.keys():
             if not self._is_correctly_placed_recursive(block, state, self.goal_config, current_config, correctly_placed_memo):
                 misplaced_goal_blocks.add(block)

        total_cost += len(misplaced_goal_blocks)

        # Cost 2: Number of blocks that are currently on top of a block Y (or table),
        # where Y is part of a goal stack (or Y is table and a goal block should be on it),
        # AND the block on top is NOT the block that should be on Y according to the goal.

        obstructing_blocks_count = 0
        all_placed_blocks = set(current_config.keys())
        if current_holding:
            all_placed_blocks.discard(current_holding) # Held block is not "placed" on something

        # Relevant locations below are goal blocks or 'table' if a goal block should be on the table.
        relevant_below_locations = set(self.goal_blocks)
        if self.goal_block_on_table: # If there's any block that should be on the table
            relevant_below_locations.add('table')

        for block_on_top in all_placed_blocks:
            block_below = current_config[block_on_top]

            if block_below in relevant_below_locations:
                # This block is on top of a relevant location.
                # Is it the *correct* block that should be there according to the goal?
                is_obstructing = False
                if block_below == 'table':
                    # If block_on_top is on the table, and it's NOT one of the blocks
                    # that should be on the table according to the goal, it's obstructing.
                    if block_on_top not in self.goal_block_on_table:
                         is_obstructing = True
                else: # block_below is a goal block
                    # If block_on_top is on a goal block, and it's NOT the specific block
                    # that should be on that goal block according to the goal, it's obstructing.
                    desired_on_top = self.goal_block_on_top.get(block_below)
                    if block_on_top != desired_on_top:
                         is_obstructing = True

                if is_obstructing:
                    obstructing_blocks_count += 1

        total_cost += obstructing_blocks_count

        # Cost 3: Penalty for arm not being empty.
        if current_holding is not None:
             total_cost += 1

        return total_cost

    def _is_correctly_placed_recursive(self, block, state, goal_config, current_config, memo):
        """
        Checks if a block is in its correct goal position relative to the goal stack below it.
        Uses memoization.
        """
        if block in memo:
            return memo[block]

        # A block is correctly placed if:
        # 1. It's a base block in the goal and is on the table in the current state.
        # 2. It's on block A in the goal and is on block A in the current state, AND A is correctly placed.

        # If the block is not in the goal configuration, it cannot be correctly placed
        # within a goal stack structure. This check is mostly for safety if called incorrectly.
        if block not in goal_config:
             memo[block] = False
             return False

        desired_below = goal_config[block]
        current_below = current_config.get(block) # Use .get in case block isn't in state (e.g., holding)

        if current_below != desired_below:
            memo[block] = False
            return False

        if desired_below == 'table':
            # Block is on the table, and it should be. Correctly placed.
            memo[block] = True
            return True
        else:
            # Block is on the correct block, now check if the block below is correct.
            # The block below must also be part of the goal configuration structure.
            if desired_below not in goal_config:
                 # This case should ideally not happen if goal_config is built correctly
                 # from connected goal 'on' facts leading down to a 'on-table' fact.
                 # If it happens, it means the desired block below is not part of the
                 # defined goal stacks, so the current block cannot be correctly placed
                 # within a goal stack.
                 memo[block] = False
                 return False

            result = self._is_correctly_placed_recursive(desired_below, state, goal_config, current_config, memo)
            memo[block] = result
            return result
