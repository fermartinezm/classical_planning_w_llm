from heuristics.heuristic_base import Heuristic
from task import Task

class blocksworldHeuristic(Heuristic):
    """
    Summary:
        A domain-dependent heuristic for the Blocksworld domain.
        It estimates the distance to the goal by counting the number of blocks
        that are not in their correct position within the goal stack structure,
        considering both the block below and the block above. It also penalizes
        the arm holding a block if the goal requires the arm to be empty.

    Assumptions:
        - The problem is a standard Blocksworld instance.
        - The goal defines a clear configuration of blocks in stacks on the table,
          represented by (on ?x ?y) and (on-table ?x) facts.
        - (arm-empty) can be a goal fact.
        - States and goals are well-formed (e.g., no cycles in 'on' relations,
          at most one block on another, at most one block held).

    Heuristic Initialization:
        The constructor parses the goal facts from the task description.
        It builds:
        - self.goal_config: A dictionary mapping each block to the block it should
          be directly on top of in the goal state, or 'table' if it should be
          on the table.
        - self.goal_stack_above: A dictionary mapping each block to the block
          that should be directly on top of it in the goal state.
        - self.all_blocks_in_goal: A set of all blocks mentioned in the goal
          (on or on-table facts).
        - self.goal_arm_empty: A boolean indicating if (arm-empty) is a goal fact.
        Static facts are ignored as they are not relevant for this heuristic
        in Blocksworld.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state:
        1. Parse the current state facts ((on ?x ?y), (on-table ?x), (holding ?x), (arm-empty))
           to build:
           - current_config: A dictionary mapping each block currently on another
             block or the table to the block/table below it.
           - current_stack_above: A dictionary mapping each block currently having
             another block on top to the block above it.
           - current_holding: The block currently held by the arm, or None.
           - current_arm_empty: Boolean indicating if the arm is empty.
        2. Initialize the heuristic value `h` to 0.
        3. Initialize a cache `correctly_placed_below_cache` for memoizing results
           of the recursive helper function.
        4. Define a recursive helper function `_is_stack_below_correct(block)`:
           - This function checks if the stack segment from the table up to `block`
             in the current state matches the goal configuration.
           - It uses the `correctly_placed_below_cache` to avoid redundant computations
             and handle recursion.
           - Base cases include checking the cache, checking if the block is on the
             correct base ('table' or another block), and recursively checking the
             block below if it's supposed to be on another block.
           - Returns True if the stack below is correct according to the goal, False otherwise.
        5. Iterate through each block `b` in `self.all_blocks_in_goal`:
           - Call `_is_stack_below_correct(b)`.
           - If it returns False, increment `h`. This block is misplaced because
             its base or something below it in the stack is wrong.
           - If it returns True (the stack below `b` is correct), check if the
             block currently on top of `b` matches the block that should be on
             top of `b` in the goal (`self.goal_stack_above.get(b)`).
           - If the block on top is incorrect (`current_stack_above.get(b) != self.goal_stack_above.get(b)`),
             increment `h`. This block is misplaced because something wrong is on top.
        6. If `self.goal_arm_empty` is True and `current_holding` is not None,
           increment `h` by 1, as the arm is not in the desired empty state.
        7. Return the final value of `h`.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.goal_config = {}
        self.goal_stack_above = {}
        self.all_blocks_in_goal = set()
        self.goal_arm_empty = False

        for fact_str in task.goals:
            if fact_str == '(arm-empty)':
                self.goal_arm_empty = True
                continue

            # Remove outer parentheses and split
            parts = fact_str[1:-1].split()
            if not parts: # Handle empty fact string if any
                continue

            predicate = parts[0]
            args = parts[1:]

            if predicate == 'on' and len(args) == 2:
                x, y = args
                self.goal_config[x] = y
                self.goal_stack_above[y] = x
                self.all_blocks_in_goal.add(x)
                self.all_blocks_in_goal.add(y)
            elif predicate == 'on-table' and len(args) == 1:
                x = args[0]
                self.goal_config[x] = 'table'
                self.all_blocks_in_goal.add(x)
            # Ignore other goal predicates like 'clear' for this heuristic's structure

    def __call__(self, node):
        state = node.state

        current_config = {}
        current_stack_above = {}
        current_holding = None
        current_arm_empty = False

        for fact_str in state:
            if fact_str == '(arm-empty)':
                current_arm_empty = True
                continue
            if fact_str.startswith('(holding '):
                # Extract block name from '(holding block_name)'
                current_holding = fact_str[len('(holding '):-1]
                continue

            # Remove outer parentheses and split
            parts = fact_str[1:-1].split()
            if not parts: # Handle empty fact string if any
                continue

            predicate = parts[0]
            args = parts[1:]

            if predicate == 'on' and len(args) == 2:
                x, y = args
                current_config[x] = y
                current_stack_above[y] = x
            elif predicate == 'on-table' and len(args) == 1:
                x = args[0]
                current_config[x] = 'table'
            # Ignore other state predicates like 'clear'

        h = 0
        correctly_placed_below_cache = {}

        # Define the recursive helper function within __call__ to access its variables
        def _is_stack_below_correct(block):
            if block in correctly_placed_below_cache:
                return correctly_placed_below_cache[block]

            # If the block is not part of the goal configuration, it cannot be
            # correctly placed within the goal stack structure.
            if block not in self.all_blocks_in_goal:
                 correctly_placed_below_cache[block] = False
                 return False

            desired_below = self.goal_config.get(block)
            current_below = current_config.get(block)

            # Handle the case where the block is currently held
            if current_holding == block:
                current_below = 'arm'

            # Check if the current position relative to below matches the desired position
            if current_below != desired_below:
                result = False
            # Base case: If desired is table and current is table, it's correct below
            elif desired_below == 'table':
                result = True
            # Recursive case: If desired is a block y, check if currently on y AND y is correct below
            # Ensure y is a block that is part of the goal configuration structure
            elif desired_below in self.all_blocks_in_goal:
                y = desired_below
                if current_below == y and _is_stack_below_correct(y):
                     result = True
                else:
                     result = False
            else: # Should not happen in valid goals defining stacks where all blocks below are also in goal config
                 result = False

            correctly_placed_below_cache[block] = result
            return result

        # Calculate heuristic based on misplaced blocks in goal configuration
        for block in self.all_blocks_in_goal:
            # Check if the stack segment from the table up to the block is correct
            if not _is_stack_below_correct(block):
                h += 1
            else:
                # If the stack below is correct, check if the block above is correct
                desired_above = self.goal_stack_above.get(block)
                current_above = current_stack_above.get(block)
                if current_above != desired_above:
                    h += 1

        # Add penalty for arm not empty if goal requires it
        if self.goal_arm_empty and current_holding is not None:
            h += 1

        return h
