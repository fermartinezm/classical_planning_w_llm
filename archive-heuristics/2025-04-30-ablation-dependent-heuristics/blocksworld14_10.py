# No external modules needed

class blocksworldHeuristic:
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
    This heuristic estimates the cost to reach the goal state by counting
    the number of blocks that are not in their correct position relative
    to the goal configuration, considering the stack structure. It recursively
    checks if a block is on its required support (another block or the table)
    according to the goal, and if that support is itself correctly placed.
    A block is considered "incorrectly stacked" if it's not on its goal support,
    or if its goal support is a block that is itself incorrectly stacked.
    The heuristic value is the total count of such incorrectly stacked blocks
    that are part of the goal stacks, plus a penalty if the arm is not empty.

    Assumptions:
    - The input state and task are valid representations for the blocksworld domain.
    - Goal states specify the desired configuration using 'on' and 'on-table' predicates,
      and possibly 'clear' for the top blocks and 'arm-empty'.
    - Blocks not mentioned as being 'on' or 'on-table' in the goal are not strictly
      part of the goal structure being measured by this heuristic's core count.
      However, their presence might make a goal block incorrectly stacked (e.g.,
      a goal block is supposed to be clear but has a non-goal block on it).
      The recursive definition handles this implicitly: if a goal block X is on
      a non-goal block Y, its current support (Y) won't match its goal support,
      making X incorrectly stacked.

    Heuristic Initialization:
    The heuristic is initialized with a Task object. During initialization,
    it parses the goal facts to build a map (`goal_on_map`) representing the
    desired support for each block involved in the goal stacks. For a goal
    fact `(on X Y)`, `goal_on_map[X]` is set to `Y`. For a goal fact
    `(on-table Z)`, `goal_on_map[Z]` is set to the special value `'table'`.
    It also identifies the set of blocks (`goal_blocks`) that are keys in
    `goal_on_map`. Static facts from the task are also processed, although
    the blocksworld domain provided has no static facts.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Given a state, first check if it is the goal state using `self.task.goal_reached(state)`. If it is, the heuristic is 0. This ensures the heuristic is zero only at the goal.
    2.  Pre-process the current `state` to create a lookup map (`current_support_map`) that allows quick determination of what each block is currently on (another block, the table, or the arm). This involves iterating through the state facts once.
    3.  Define a recursive helper function `_is_correctly_stacked(block, support_map, goal_on_map, memo)` that determines if a given `block` is correctly placed relative to the goal structure in the current `state`. It takes the pre-computed `support_map` and a memoization dictionary `memo`.
    4.  Inside `_is_correctly_stacked`:
        a.  If the result for `block` is already in `memo`, return the stored value.
        b.  If the `block` is not a key in `goal_on_map` (meaning it's not part of the goal stack structure), it cannot be correctly stacked according to this definition. Store `False` in `memo` and return `False`. This handles cases where a goal block is supposed to be on a non-goal block.
        c.  Find the block or table that `block` should be on according to `goal_on_map` (`goal_below`).
        d.  Find the block or table that `block` is currently on using the `support_map` (`current_support`).
        e.  If `goal_below` is `'table'`: The block is correctly stacked if and only if `current_support` is also `'table'`.
        f.  If `goal_below` is a block `Y`: The block is correctly stacked if and only if `current_support` is `Y` AND `Y` is itself correctly stacked (recursive call: `_is_correctly_stacked(Y, support_map, goal_on_map, memo)`).
        g.  Store the computed result for `block` in `memo` before returning.
    5.  Initialize a counter `incorrectly_stacked_count` to 0.
    6.  Initialize an empty memoization dictionary `memo`.
    7.  Iterate through each `block` in the set of `goal_blocks` (blocks that are keys in `goal_on_map`).
    8.  For each `block`, call `_is_correctly_stacked(block, current_support_map, self.goal_on_map, memo)`.
    9.  If the function returns `False`, increment `incorrectly_stacked_count`.
    10. The base heuristic value is `incorrectly_stacked_count`.
    11. Add a penalty of 1 if the arm is not empty (`'(arm-empty)'` is not in the `state`). This accounts for the final `putdown` action often required to clear the arm.
    12. Return the total heuristic value.
    """

    def __init__(self, task):
        self.task = task
        self.goal_on_map = {}
        # Although static facts are empty in this domain,
        # the structure to process them is shown here.
        # self.static_info = task.static # Not used in this heuristic

        # Parse goal facts to build goal_on_map
        for fact_string in task.goals:
            pred, args = self._parse_fact_string(fact_string)
            if pred == 'on':
                # Ensure fact has two arguments for 'on'
                if len(args) == 2:
                    self.goal_on_map[args[0]] = args[1]
                # else: handle invalid fact format? Assume valid PDDL.
            elif pred == 'on-table':
                 # Ensure fact has one argument for 'on-table'
                 if len(args) == 1:
                    self.goal_on_map[args[0]] = 'table'
                 # else: handle invalid fact format? Assume valid PDDL.
            # Ignore 'clear' and 'arm-empty' for goal_on_map

        # goal_blocks are the blocks that are keys in goal_on_map
        self.goal_blocks = set(self.goal_on_map.keys())


    def _parse_fact_string(self, fact_string):
        """Helper to parse a fact string into (predicate, [args])."""
        # Remove surrounding parentheses and split
        # Handles cases like '(arm-empty)' correctly resulting in args=[]
        parts = fact_string.strip("()").split()
        if not parts: # Handle empty string case, though unlikely for facts
            return None, []
        predicate = parts[0]
        args = parts[1:]
        return predicate, args

    # _get_current_support is now implemented efficiently within __call__
    # using a pre-computed map.

    # _is_correctly_stacked is now implemented efficiently within __call__
    # using a pre-computed map and memoization.

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.
        """
        # Heuristic is 0 iff state is a goal state
        if self.task.goal_reached(state):
             return 0

        # Pre-process state to quickly find current support for any block
        current_support_map = {}
        for fact_string in state:
            pred, args = self._parse_fact_string(fact_string)
            if pred == 'on-table' and len(args) == 1:
                current_support_map[args[0]] = 'table'
            elif pred == 'on' and len(args) == 2:
                current_support_map[args[0]] = args[1]
            elif pred == 'holding' and len(args) == 1:
                 current_support_map[args[0]] = 'arm'
        # Blocks not in current_support_map are not on table, on another block, or held.
        # This implies an invalid state representation if a block from goal_blocks
        # is not found here, but we assume valid states.

        # Define the recursive helper function locally to access current_support_map
        def _is_correctly_stacked(block, support_map, goal_on_map, memo):
            """
            Recursively checks if a block is correctly stacked according to the goal.
            Uses memoization.
            """
            if block in memo:
                return memo[block]

            # A block not part of the goal structure is not correctly stacked relative to it.
            # This case is reached if a goal block is supposed to be on a non-goal block Y,
            # and we recursively call _is_correctly_stacked(Y, ...).
            if block not in goal_on_map:
                 memo[block] = False
                 return False

            goal_below = goal_on_map[block]
            current_support = support_map.get(block) # O(1) lookup

            if goal_below == 'table':
                result = (current_support == 'table')
            else: # goal_below is a block Y
                # Check if current support is Y AND Y is correctly stacked
                result = (current_support == goal_below) and _is_correctly_stacked(goal_below, support_map, goal_on_map, memo)

            memo[block] = result
            return result

        incorrectly_stacked_count = 0
        memo = {} # Memoization dictionary for the recursive checks

        # Count blocks in goal_blocks that are not correctly stacked
        for block in self.goal_blocks:
            if not _is_correctly_stacked(block, current_support_map, self.goal_on_map, memo):
                incorrectly_stacked_count += 1

        heuristic_value = incorrectly_stacked_count

        # Add penalty if arm is not empty
        if '(arm-empty)' not in state:
             heuristic_value += 1

        return heuristic_value
