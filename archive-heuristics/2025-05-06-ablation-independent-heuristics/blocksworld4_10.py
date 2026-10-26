from heuristics.heuristic_base import Heuristic
from task import Task


class blocksworldHeuristic(Heuristic):
    """
    Summary:
        A domain-dependent heuristic for the Blocksworld domain.
        It estimates the cost to reach the goal by counting the number of
        blocks that are not part of a correctly built stack segment rooted
        at the table, according to the goal configuration, and adds the count
        of unsatisfied non-structural goal facts.

    Assumptions:
        - The state representation includes facts specifying the immediate
          support for each block: either (on B A), (on-table B), or (holding B).
        - The goal specifies the desired stack configuration using (on B A)
          and (on-table B) facts. (clear) and (arm-empty) facts in the goal
          are considered non-structural goals.
        - The heuristic assumes a solvable problem instance where the goal
          configuration forms valid stacks rooted on the table.

    Heuristic Initialization:
        - Parses the goal facts to build the target stack configuration.
          A dictionary `goal_config` is created where `goal_config[B] = A`
          if the goal is (on B A), and `goal_config[B] = 'table'` if the
          goal is (on-table B).
        - Identifies the set of all blocks that are part of this goal
          configuration (`goal_blocks`). This includes blocks that should
          be on the table and blocks that other blocks should be on.

    Step-By-Step Thinking for Computing Heuristic:
        1. For a given state, build the current configuration of blocks.
           A dictionary `current_config` is created where `current_config[B] = A`
           if (on B A) is true in the state, `current_config[B] = 'table'`
           if (on-table B) is true, and `current_config[B] = 'arm'` if
           (holding B) is true.
        2. Initialize a memoization dictionary `memo` for the `is_correctly_stacked` helper.
        3. Initialize a counter `correctly_stacked_count` to 0.
        4. For each block `B` in the set of `goal_blocks`:
           a. Call the recursive helper function `is_correctly_stacked(B, current_config, goal_config, memo)`.
           b. If the function returns True, increment `correctly_stacked_count`.
        5. Calculate the number of blocks in the goal configuration that are
           not correctly stacked: `misplaced_structural_blocks = len(goal_blocks) - correctly_stacked_count`.
        6. Count the number of unsatisfied goal facts that are not related
           to the core stack structure (`on` or `on-table` facts). These are
           typically `clear` or `arm-empty` goals. Iterate through `task.goals`
           and count facts not in the current state that are not `on` or
           `on-table` facts.
        7. The heuristic value is the sum of `misplaced_structural_blocks` and
           the count of unsatisfied non-structural goal facts. This sum is 0
           if and only if the state is the goal state.

    Helper function `is_correctly_stacked(block, current_config, goal_config, memo)`:
        - This function recursively checks if a block `block` is part of a
          correctly built goal stack segment in the `current_config`, based
          on the `goal_config`.
        - It uses memoization (`memo`) to avoid redundant computations.
        - Base Case 1: If the result for `block` is already in `memo`, return it.
        - Base Case 2: If `block` is not in `goal_config` (meaning nothing should
          be on this block according to the goal structure), it cannot be
          correctly stacked within the structure we are tracking. Memoize
          False and return False. (This case handles blocks that are values
          in `goal_config` but not keys).
        - Recursive Step:
            - Determine the `target_below` block/table for `block` from `goal_config`.
            - Determine the `current_below` block/table/arm for `block` from `current_config`.
            - If `current_below` is different from `target_below`, `block` is not correctly stacked. Memoize False and return False.
            - If `current_below` is the same as `target_below`:
                - If `target_below` is 'table', `block` is correctly stacked (base case for recursion). Memoize True and return True.
                - If `target_below` is a block `A`, `block` is correctly stacked only if `A` is also correctly stacked. Recursively call `is_correctly_stacked(A, current_config, goal_config, memo)`. Memoize the result and return it.
    """

    def __init__(self, task):
        super().__init__()
        self.goal_config = {}
        self.goal_blocks = set()
        self.goals = task.goals # Store task goals for checking non-structural goals

        # Parse goal facts to build goal_config and goal_blocks
        for fact_str in task.goals:
            predicate, args = self.parse_fact(fact_str)
            if predicate == 'on' and len(args) == 2:
                block, below = args
                self.goal_config[block] = below
                self.goal_blocks.add(block)
                self.goal_blocks.add(below) # Add the block it should be on
            elif predicate == 'on-table' and len(args) == 1:
                block = args[0]
                self.goal_config[block] = 'table'
                self.goal_blocks.add(block)

        # 'table' is not a block, remove it if added
        self.goal_blocks.discard('table')

    def parse_fact(self, fact_str):
        """Helper to parse a PDDL fact string."""
        # Remove outer parentheses and split by space
        parts = fact_str.strip('()').split()
        predicate = parts[0]
        args = parts[1:]
        return predicate, args

    def is_correctly_stacked(self, block, current_config, goal_config, memo):
        """
        Recursively checks if a block is part of a correctly built goal stack segment.
        """
        if block in memo:
            return memo[block]

        # If the block is not part of the goal configuration structure (i.e.,
        # it's not a key in goal_config), it cannot be correctly stacked
        # within the structure we are tracking.
        if block not in goal_config:
             memo[block] = False
             return False

        target_below = goal_config[block]
        current_below = current_config.get(block) # Use .get() in case block is not in state config

        # If block is not in current_config, it's not on anything, not on table, not held.
        # This indicates it's not correctly placed relative to any support.
        if current_below is None:
             memo[block] = False
             return False

        if current_below != target_below:
            memo[block] = False
            return False
        else: # current_below == target_below
            if target_below == 'table':
                memo[block] = True
                return True
            else: # target_below is a block A
                # Check if A is correctly stacked
                result = self.is_correctly_stacked(target_below, current_config, goal_config, memo)
                memo[block] = result
                return result

    def __call__(self, node):
        state = node.state

        # Build current_config from state facts
        current_config = {}
        for fact_str in state:
            predicate, args = self.parse_fact(fact_str)
            if predicate == 'on' and len(args) == 2:
                block, below = args
                current_config[block] = below
            elif predicate == 'on-table' and len(args) == 1:
                block = args[0]
                current_config[block] = 'table'
            elif predicate == 'holding' and len(args) == 1:
                block = args[0]
                current_config[block] = 'arm'
            # Ignore 'clear' and 'arm-empty' for building current_config

        memo = {}
        correctly_stacked_count = 0

        # Count blocks in goal_blocks that are correctly stacked
        for block in self.goal_blocks:
            if self.is_correctly_stacked(block, current_config, self.goal_config, memo):
                correctly_stacked_count += 1

        # Calculate the number of blocks in the goal configuration that are NOT correctly stacked.
        misplaced_structural_blocks = len(self.goal_blocks) - correctly_stacked_count

        # Count unsatisfied non-structural goal facts
        # Non-structural goals are typically 'clear' and 'arm-empty' in blocksworld.
        unsatisfied_non_structural_goals = 0
        for goal_fact_str in self.goals:
             predicate, args = self.parse_fact(goal_fact_str)
             is_structural_predicate = (predicate == 'on' and len(args) == 2) or (predicate == 'on-table' and len(args) == 1)

             if not is_structural_predicate:
                 # It's a non-structural goal fact (e.g., clear, arm-empty)
                 if goal_fact_str not in state:
                     unsatisfied_non_structural_goals += 1

        # The heuristic is the sum of misplaced structural blocks and unsatisfied non-structural goals.
        # This sum is 0 iff the state is the goal state.
        h_value = misplaced_structural_blocks + unsatisfied_non_structural_goals

        return h_value
