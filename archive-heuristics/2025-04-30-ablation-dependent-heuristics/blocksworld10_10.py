import sys

# Assuming the Task class is available in the environment or imported from a local file
# from task import Task

class blocksworldHeuristic:
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
    This heuristic estimates the cost to reach the goal state by counting the number
    of blocks that are not currently in their correct position within their goal stack
    segment, starting from the table. It multiplies this count by 2, as a rough
    estimate that each such block requires at least two actions (e.g., unstack/pickup
    and stack/putdown) to be placed correctly. Additionally, it adds 1 to the heuristic
    if the goal requires the arm to be empty but it is not, representing the cost
    to free the arm.

    Assumptions:
    - The PDDL domain is standard Blocksworld with predicates (clear, on-table, arm-empty, holding, on)
      and actions (pickup, putdown, stack, unstack).
    - The goal state specifies the desired positions for all relevant blocks using
      '(on X Y)' and '(on-table X)' facts. It is assumed that every block whose
      position matters in the goal is included in these facts, and thus will be
      present in the `goal_on_map`. Blocks not in `goal_on_map` are considered
      as not having a specific goal position within a stack structure relevant
      to building the goal configuration.
    - The heuristic is designed for greedy best-first search and does not need to be admissible.

    Heuristic Initialization:
    The heuristic is initialized with the planning task. During initialization,
    it parses the goal facts to build a map (`self.goal_on_map`) indicating the
    desired base for each block (either another block or 'table'). It also collects
    all unique block names present in the initial state and goal state facts
    into `self.all_blocks`.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Define a recursive helper function `is_in_goal_stack(block, state)`:
        -   This function checks if a given `block` is currently in its correct
            position relative to its goal base, AND if its goal base is also
            correctly positioned within its goal stack segment, recursively down
            to the table.
        -   It uses memoization (`self._is_in_goal_stack_memo`) to avoid redundant
            calculations for the same block in the same state.
        -   Base Case 1: If the block's goal position is not specified in `self.goal_on_map`,
            it cannot be part of a goal stack defined by the goal facts, so it returns False.
        -   Base Case 2: If the block's goal base is 'table' according to `self.goal_on_map`,
            it returns True if and only if the fact `(on-table block)` is present in the current `state`.
        -   Recursive Step: If the block's goal base is another block `Y` according to
            `self.goal_on_map`, it returns True if and only if the fact `(on block Y)`
            is present in the current `state` AND `is_in_goal_stack(Y, state)` returns True.
    2.  Initialize the heuristic value `h_value` to 0.
    3.  Reset the memoization dictionary `self._is_in_goal_stack_memo` for the current state.
    4.  Iterate through all blocks that have a specified goal position (i.e., are keys in `self.goal_on_map`).
    5.  For each such block, call `is_in_goal_stack(block, state)`.
    6.  If `is_in_goal_stack` returns False, increment `h_value`. This counts the number
        of blocks that are not part of a correctly built stack segment from the table up.
    7.  Multiply the `h_value` by 2. This estimates that each block not in its goal
        stack requires roughly two actions (one to pick it up/unstack, one to place it).
    8.  Check if `(arm-empty)` is a goal fact and if `(arm-empty)` is not true in the current state.
        If both conditions are met, increment `h_value` by 1, representing the cost
        to free the arm.
    9.  Return the final `h_value`.

    The heuristic is 0 if and only if the state is a goal state (assuming standard
    blocksworld goals specify positions for all relevant blocks and require the arm empty).
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing goal facts and collecting block names.

        @param task: The planning task object.
        """
        self.goal_on_map = {}  # block -> base (block or 'table')
        self.goal_clear_blocks = set() # blocks that must be clear (for completeness, not directly used in h calculation)
        self.all_blocks = set() # All blocks in the problem

        # Helper function to extract objects from a fact string
        def extract_objects_from_fact(fact_str):
            parts = fact_str.strip('()').split()
            # Ignore predicate name (parts[0])
            return set(parts[1:])

        # Collect all blocks from initial state and goal state facts
        for fact in task.initial_state:
            self.all_blocks.update(extract_objects_from_fact(fact))
        for fact in task.goals:
            self.all_blocks.update(extract_objects_from_fact(fact))

        # Parse goal facts to build goal_on_map and goal_clear_blocks
        for fact in task.goals:
            parts = fact.strip('()').split()
            predicate = parts[0]
            if predicate == 'on':
                block = parts[1]
                base = parts[2]
                self.goal_on_map[block] = base
            elif predicate == 'on-table':
                block = parts[1]
                self.goal_on_map[block] = 'table'
            elif predicate == 'clear':
                block = parts[1]
                self.goal_clear_blocks.add(block)

        # Memoization dictionary for is_in_goal_stack, reset per state
        self._is_in_goal_stack_memo = {}

    def is_in_goal_stack(self, block, state):
        """
        Checks if a block is currently in its correct position relative to its
        goal stack segment, recursively down to the table.

        @param block: The block to check.
        @param state: The current state (frozenset of facts).
        @return: True if the block is in its goal stack segment, False otherwise.
        """
        if block in self._is_in_goal_stack_memo:
            return self._is_in_goal_stack_memo[block]

        # If block's goal position is not specified, it cannot be in a goal stack
        # defined by goal_on_map.
        if block not in self.goal_on_map:
             self._is_in_goal_stack_memo[block] = False
             return False

        goal_base = self.goal_on_map[block]

        result = False
        if goal_base == 'table':
            # Goal is (on-table block)
            if f'(on-table {block})' in state:
                result = True
        else:
            # Goal is (on block goal_base)
            # Check if (on block goal_base) is true AND goal_base is in its goal stack
            if f'(on {block} {goal_base})' in state:
                 # Recursive call
                 if self.is_in_goal_stack(goal_base, state):
                     result = True

        self._is_in_goal_stack_memo[block] = result
        return result


    def __call__(self, state, task):
        """
        Computes the domain-dependent heuristic value for the given state.

        @param state: The current state (frozenset of facts).
        @param task: The planning task object.
        @return: The heuristic value (integer).
        """
        # Reset memoization for the current state
        self._is_in_goal_stack_memo = {}

        h_value = 0

        # Count blocks that have a specified goal position but are not in their goal stack
        for block in self.goal_on_map.keys():
            if not self.is_in_goal_stack(block, state):
                h_value += 1

        # Multiply by 2 as a rough estimate of actions per misplaced block
        h_value = h_value * 2

        # Add 1 if arm should be empty but isn't
        if '(arm-empty)' in task.goals and '(arm-empty)' not in state:
            h_value += 1

        return h_value

# Example usage (assuming Task class is defined elsewhere and a task object is available):
# task = ... # Load your blocksworld task here
# heuristic = blocksworldHeuristic(task)
# current_state = ... # Get the current state
# h_value = heuristic(current_state, task)
# print(f"Heuristic value: {h_value}")
