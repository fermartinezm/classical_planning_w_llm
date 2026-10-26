class blocksworldHeuristic:
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
    This heuristic estimates the number of actions required to reach the goal state
    by counting the number of blocks that are not in their correct position within
    the goal stack structure, plus a penalty if the arm is not empty when it
    should be according to the goal. A block is considered "correctly placed"
    if it is on the correct block or the table according to the goal, AND the
    block below it (if any) is also correctly placed, recursively down to the table.
    This captures the dependency structure of building stacks.

    Assumptions:
    - The goal state primarily consists of (on ?x ?y) and (on-table ?x) facts,
      defining desired stack configurations. (clear ?x) and (arm-empty) goals
      are handled secondarily.
    - The goal configuration forms valid, acyclic stacks.
    - The heuristic focuses on placing blocks into their target positions within
      these goal stacks.

    Heuristic Initialization:
    The `__init__` method precomputes the goal position for each block that is
    explicitly mentioned in an `(on ?x ?y)` or `(on-table ?x)` goal fact. This
    mapping, `self.goal_pos`, stores for each block `B`, the block `Y` it should
    be on, or the string 'table' if it should be on the table. It also checks
    if `(arm-empty)` is a required goal fact. Static facts are also processed
    if they were relevant, but the example shows an empty static set, so no
    specific static information is extracted into data structures in this case.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Check if the current state is the goal state using `self.task.goal_reached(state)`. If it is, the heuristic value is 0.
    2.  Initialize a counter `misplaced_count` to 0.
    3.  Initialize a memoization dictionary `memo` to store results of the
        `is_correctly_placed` helper function to avoid redundant calculations
        in the recursive calls.
    4.  Identify the set of blocks that have a specific goal position defined
        in the `self.goal_pos` mapping (these are the keys of `self.goal_pos`).
    5.  For each block `block` in this set:
        a.  Call the recursive helper function `self.is_correctly_placed(block, state, memo)`
            to determine if the block is correctly placed relative to the goal stack
            structure below it.
        b.  If `is_correctly_placed` returns `False`, increment `misplaced_count`.
    6.  Initialize an `arm_penalty` to 0.
    7.  If `(arm-empty)` is a goal fact (checked during initialization) and the
        current state does not contain the fact `'(arm-empty)'`, set `arm_penalty` to 1.
    8.  The final heuristic value is `misplaced_count + arm_penalty`.

    The `is_correctly_placed(block, state, memo)` helper function works as follows:
    - It first checks the `memo` dictionary. If the result for `block` is already
      computed, it returns the memoized value.
    - If the block does not have an explicit goal position in `self.goal_pos`,
      it is considered "correctly placed" relative to the goal stacks we are tracking,
      and the function returns `True` (and memoizes this).
    - If the block's goal is to be on the table (`self.goal_pos[block] == 'table'`),
      it checks if the fact `'(on-table block)'` is present in the current `state`.
      The result is memoized and returned.
    - If the block's goal is to be on another block `support_block` (`self.goal_pos[block] == support_block`),
      it checks if the fact `'(on block support_block)'` is present in the current `state`
      AND recursively calls `is_correctly_placed(support_block, state, memo)` to check
      if the block below it is also correctly placed. The combined boolean result
      is memoized and returned.
    """
    def __init__(self, task):
        """
        Initializes the heuristic by precomputing goal positions and static info.

        @param task: The planning task object.
        """
        self.task = task
        self.goal_pos = {}
        self._arm_empty_is_goal = False

        # Parse goal facts to determine goal positions and check for arm-empty goal
        for goal_fact_string in task.goals:
            predicate, args = self.parse_fact(goal_fact_string)
            if predicate == 'on':
                block, support = args
                self.goal_pos[block] = support
            elif predicate == 'on-table':
                block = args[0]
                self.goal_pos[block] = 'table'
            elif goal_fact_string == '(arm-empty)':
                 self._arm_empty_is_goal = True

        # Process static facts if any. The example shows static as empty.
        # If there were static facts relevant to the heuristic, they would be
        # processed here, e.g., storing block properties.
        # for static_fact_string in task.static:
        #     predicate, args = self.parse_fact(static_fact_string)
        #     # Process static info based on predicate/args


    def parse_fact(self, fact_string):
        """
        Parses a PDDL fact string into (predicate, args).

        @param fact_string: The string representation of a PDDL fact.
        @return: A tuple containing the predicate name (string) and a list of arguments (strings).
        """
        # Remove leading '(' and trailing ')'
        content = fact_string[1:-1]
        # Split by whitespace
        parts = content.split()
        predicate = parts[0]
        args = parts[1:]
        return predicate, args

    def is_correctly_placed(self, block, state, memo):
        """
        Checks if a block is correctly placed according to the goal stack structure,
        including the blocks below it. Uses memoization.

        A block is correctly placed if:
        - It is not required to be in a specific position by the goal (implicitly true).
        - OR its goal is 'on-table' and it is on the table in the state.
        - OR its goal is 'on Y' and it is on Y in the state AND Y is correctly placed.

        @param block: The block to check (string).
        @param state: The current state (frozenset of fact strings).
        @param memo: Dictionary for memoization {block: boolean result}.
        @return: True if the block is correctly placed relative to the goal structure, False otherwise.
        """
        if block in memo:
            return memo[block]

        # If the block doesn't have a specific goal position defined in on/on-table facts,
        # it's considered correctly placed relative to the goal stacks we care about.
        # This handles blocks that might be in the state but are not part of the
        # desired goal configuration structure defined by 'on' and 'on-table' goals.
        if block not in self.goal_pos:
            memo[block] = True
            return True

        goal = self.goal_pos[block]
        is_correct = False

        if goal == 'table':
            # Goal is to be on the table
            is_correct = f'(on-table {block})' in state
        else:
            # Goal is to be on another block (support_block)
            support_block = goal
            # Check if the block is on the correct support block AND the support block is correctly placed
            is_correct = f'(on {block} {support_block})' in state and \
                         self.is_correctly_placed(support_block, state, memo)

        memo[block] = is_correct
        return is_correct

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state: The current state (frozenset of fact strings).
        @return: The estimated number of actions to reach the goal state.
        """
        # If the goal is reached, the heuristic is 0.
        if self.task.goal_reached(state):
            return 0

        misplaced_count = 0
        memo = {} # Memoization dictionary for is_correctly_placed

        # Count blocks that have a goal position but are not correctly placed
        # relative to the goal stack structure.
        # We only iterate through blocks that are explicitly given a position
        # in the goal's 'on' or 'on-table' facts.
        for block in self.goal_pos.keys():
            if not self.is_correctly_placed(block, state, memo):
                misplaced_count += 1

        # Add a penalty if the arm should be empty according to the goal
        # but is not empty in the current state.
        arm_penalty = 0
        if self._arm_empty_is_goal and '(arm-empty)' not in state:
             arm_penalty = 1
             # This penalty accounts for the action needed to clear the arm.
             # If the arm is holding a block, that block is necessarily not
             # in its final 'on' or 'on-table' position, so it would likely
             # already contribute to the misplaced_count if it has a goal_pos.
             # The arm_penalty adds a small cost specific to the arm state.

        return misplaced_count + arm_penalty
