class blocksworldHeuristic:
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing three components:
        1. The number of blocks that are part of a goal stack but are not currently
           positioned correctly relative to the blocks below them in the goal stack,
           considering the stack structure from the table up.
        2. The number of blocks that are currently resting on top of another block
           which needs to be clear according to the goal state.
        3. A penalty of 1 if the arm is not empty and the goal requires the arm to be empty.

        The heuristic is non-admissible and designed to guide a greedy best-first search
        by prioritizing states where more blocks are in their correct stack positions
        or where fewer blocks are obstructing necessary movements.

    Assumptions:
        - The input task is a valid Blocksworld planning task.
        - Goal states specify desired 'on', 'on-table', and 'clear' relationships,
          and typically require the arm to be empty.
        - Blocks not mentioned as the upper block in goal 'on' facts and not mentioned
          in goal 'on-table' facts do not have a specific required position relative
          to the stack below them for the purpose of the 'not_correctly_stacked_count'
          component.

    Heuristic Initialization:
        The `__init__` method precomputes information from the goal state:
        - `goal_below_map`: A dictionary mapping each block `b` involved as the upper
          block in a goal 'on' fact or the block in a goal 'on-table' fact, to the
          block `u` it should be on, or the string 'table'.
        - `needs_clear`: A set of blocks that must be clear in the goal state,
          either because `(clear b)` is a goal fact or because some block `y`
          needs to be placed on `b` (`(on y b)` is a goal fact).
        - `task_goals`: Stores the original goal set for easy lookup of `(arm-empty)`.

    Step-By-Step Thinking for Computing Heuristic:
        The `__call__(self, state)` method computes the heuristic value for a given state:
        1.  **Identify Current State Structure**: Iterate through the facts in the current
            `state` to build a `current_support_map` (mapping each block to the block
            or 'table' it is currently on) and identify the `held_block` (if any).
        2.  **Compute `not_correctly_stacked_count`**:
            -   Initialize a memoization dictionary `correctly_stacked_memo`.
            -   Define a recursive helper function `is_correctly_stacked_recursive(block, goal_below_map, current_support_map, held_block, memo)`:
                -   Base Case 1: If the result for `block` is already in `memo`, return it.
                -   Base Case 2: If `block` is not a key in `goal_below_map` (meaning it's not the upper block in any goal 'on' fact and not the block in a goal 'on-table' fact), its position relative to the block below it is not constrained by the goal stacks. It is considered "correctly stacked" relative to blocks above it in the goal stack. Return True.
                -   Check if `block` is currently `held_block`. If yes, it's not correctly stacked on a block/table. Return False.
                -   Get the `goal_support` from `goal_below_map` and `current_support` from `current_support_map`.
                -   If `current_support` is None (block not found on table/block/held), it's wrong.
                -   Check current support matches goal support. If `current_support != goal_support`, it's not correctly stacked. Return False.
                -   If goal support is table, and current support is table, it's correctly stacked (base case). Return True.
                -   Recursive Step: If goal support is a block `u`, recursively call `is_correctly_stacked_recursive(u, ...)`. The result for `block` is the result of the recursive call. Store and return the result.
            -   Initialize `not_correctly_stacked_count = 0`.
            -   Iterate through each `block` that is a key in `goal_below_map` (these are the blocks whose position relative to the block below is explicitly specified in the goal). If `is_correctly_stacked_recursive` returns False for the block, increment `not_correctly_stacked_count`.
        3.  **Compute `blocking_count`**:
            -   Initialize `blocking_count = 0`.
            -   Iterate through the facts in the current `state`.
            -   If a fact is `'(on ?x ?b)'`, check if `?b` is in the `needs_clear` set. If yes, increment `blocking_count`.
        4.  **Compute `arm_penalty`**:
            -   Initialize `arm_penalty = 0`.
            -   If `(arm-empty)` is a goal fact and `held_block` is not None, set `arm_penalty = 1`.
        5.  **Calculate Total Heuristic**: The heuristic value is the sum: `not_correctly_stacked_count + blocking_count + arm_penalty`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing goal information.

        Args:
            task: The planning task object (instance of Task class).
        """
        self.task_goals = task.goals
        self.goal_below_map = {}
        self.needs_clear = set()

        # Precompute goal_below_map
        for goal_fact in self.task_goals:
            parts = goal_fact.strip('()').split()
            predicate = parts[0]
            if predicate == 'on':
                block = parts[1]
                support = parts[2]
                self.goal_below_map[block] = support
            elif predicate == 'on-table':
                block = parts[1]
                self.goal_below_map[block] = 'table'

        # Precompute needs_clear
        for goal_fact in self.task_goals:
            parts = goal_fact.strip('()').split()
            predicate = parts[0]
            if predicate == 'clear':
                block = parts[1]
                self.needs_clear.add(block)
            elif predicate == 'on': # If block Y needs to be on block B, B needs to be clear
                 block_below = parts[2]
                 self.needs_clear.add(block_below)


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        Args:
            state: The current state (frozenset of facts).

        Returns:
            An integer representing the estimated cost to reach the goal.
        """
        # 1. Identify Current State Structure
        current_support_map = {}
        held_block = None
        for fact in state:
            parts = fact.strip('()').split()
            predicate = parts[0]
            if predicate == 'on':
                block = parts[1]
                support = parts[2]
                current_support_map[block] = support
            elif predicate == 'on-table':
                block = parts[1]
                current_support_map[block] = 'table'
            elif predicate == 'holding':
                held_block = parts[1]

        # 2. Compute not_correctly_stacked_count
        def is_correctly_stacked_recursive(block, goal_below_map, current_support_map, held_block, memo):
            if block in memo:
                return memo[block]

            # If the block is not a key in goal_below_map, it means no block needs to be on it
            # according to the goal, and it doesn't need to be on the table according to the goal.
            # Its position relative to the block below it is not constrained by the goal stacks.
            # Treat it as correctly stacked for the purpose of blocks *above* it in the goal stack.
            if block not in goal_below_map:
                 memo[block] = True
                 return True

            # If the block is currently held, it's not correctly stacked on a block/table.
            if block == held_block:
                memo[block] = False
                return False

            goal_support = goal_below_map[block]
            current_support = current_support_map.get(block) # None if not on table/block

            # If current support is unknown (block exists in goal_below_map but not in state facts on/on-table/holding), it's wrong.
            if current_support is None:
                 memo[block] = False
                 return False

            # Check current support matches goal support
            if current_support != goal_support:
                memo[block] = False
                return False

            # If goal support is table, and current support is table, it's correctly stacked (base case)
            if goal_support == 'table':
                memo[block] = True
                return True

            # If goal support is a block 'u', check if 'u' is correctly stacked
            u = goal_support
            result = is_correctly_stacked_recursive(u, goal_below_map, current_support_map, held_block, memo)
            memo[block] = result
            return result

        not_correctly_stacked_count = 0
        correctly_stacked_memo = {}
        # Iterate over blocks that are keys in goal_below_map, as these are the blocks
        # whose position relative to the block below is explicitly specified in the goal.
        for block in self.goal_below_map.keys():
            if not is_correctly_stacked_recursive(block, self.goal_below_map, current_support_map, held_block, correctly_stacked_memo):
                 not_correctly_stacked_count += 1


        # 3. Compute blocking_count
        blocking_count = 0
        for fact in state:
            parts = fact.strip('()').split()
            predicate = parts[0]
            if predicate == 'on':
                # block_on_top = parts[1] # Not needed for count
                block_below = parts[2]
                if block_below in self.needs_clear:
                    blocking_count += 1
            # Note: A block in hand doesn't block anything below it.
            # A block on the table doesn't block anything below it (table doesn't need clearing).

        # 4. Compute arm_penalty
        arm_empty_goal = '(arm-empty)' in self.task_goals
        arm_penalty = 0
        if arm_empty_goal and held_block is not None:
             arm_penalty = 1 # It costs one action (putdown or stack) to make arm empty

        # 5. Calculate Total Heuristic
        heuristic_value = not_correctly_stacked_count + blocking_count + arm_penalty

        return heuristic_value
