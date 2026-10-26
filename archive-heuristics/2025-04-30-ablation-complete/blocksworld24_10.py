from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # PDDL facts are expected to be strings like "(predicate arg1 arg2)"
    # Handle potential malformed input defensively
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         # Depending on expected input, could return [], raise error, or log warning
         # Assuming valid PDDL fact strings from the planner
         return []
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed by counting the number of blocks that are NOT part of a correctly built stack segment leading up to their goal position. A block is considered part of a correct stack segment if it is in its correct position relative to the block below it in the goal, AND the block below it is also part of a correct stack segment (recursively), or the block below it is the table and the block is correctly on the table.

    # Assumptions
    - The goal defines a specific configuration of blocks, primarily using `on` and `on-table` predicates.
    - `clear` and `arm-empty` goals are implicitly handled as they are consequences of achieving the desired stack structure.
    - The heuristic assumes a block's goal position is either on a specific block or on the table, as defined by the goal state.
    - The goal predicates `on` and `on-table` define a valid set of goal stacks where every block involved in an `on` predicate as the first argument, or in an `on-table` predicate, has its base defined, ultimately tracing down to the table.

    # Heuristic Initialization
    - Determine the target base for each block (either another block or the table) based on the goal state's `on` and `on-table` predicates. This creates a mapping from a block to its desired base. Only blocks explicitly mentioned as being `on` something or `on-table` in the goal will have a defined goal base.

    # Step-By-Step Thinking for Computing Heuristic
    1.  For each block, determine its desired position in the goal state (on which block or on the table). This mapping (`self.goal_base`) is pre-calculated during initialization by parsing the goal facts.
    2.  Define a recursive function `is_correctly_stacked(block, state, memo)` that checks if a block is in its correct goal position relative to its base, AND if that base (if it's another block) is also correctly stacked. Use memoization (`memo`) to store results for already checked blocks within a single heuristic evaluation call, preventing redundant computation and infinite recursion.
        - If the block's goal base is 'table', check if the fact `(on-table block)` exists in the current `state`.
        - If the block's goal base is another block `U`, check if the fact `(on block U)` exists in the current `state`. If this fact is present, recursively call `is_correctly_stacked(U, state, memo)` to check if the base block `U` is correctly stacked. The block `block` is correctly stacked only if both conditions are met.
        - If a block does not have a defined goal base in `self.goal_base`, it is not considered part of a goal stack segment that contributes positively to reducing the heuristic in this model. The function should handle this case (though the main loop in `__call__` ensures we only check blocks *with* a defined goal base).
    3.  Initialize a counter `correctly_stacked_count` to 0.
    4.  Iterate through all blocks that have a defined goal base (i.e., the keys in `self.goal_base`).
    5.  For each such block, call `is_correctly_stacked` with the current state and the memoization dictionary.
    6.  If the function returns `True`, increment `correctly_stacked_count`.
    7.  The heuristic value is the total number of blocks that have a defined goal base (`len(self.goal_base)`) minus the number of these blocks that are currently correctly stacked (`correctly_stacked_count`). This value represents the number of blocks that are "out of place" relative to the desired goal stacks.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal base for each block.
        """
        self.goals = task.goals

        # Determine the goal base for each block that needs to be on something or on the table.
        # This dictionary maps a block name to its desired base ('table' or another block name).
        self.goal_base = {}
        # Parse goal facts to find where each block should be
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts and parts[0] == 'on':
                # (on ?x ?y) means ?x should be on ?y
                block, base = parts[1], parts[2]
                self.goal_base[block] = base
            elif parts and parts[0] == 'on-table':
                # (on-table ?x) means ?x should be on the table
                block = parts[1]
                self.goal_base[block] = 'table'
            # Ignore (clear ?x) and other potential goal predicates for this heuristic's structure logic.

        # self.goal_base now contains entries for all blocks whose position
        # relative to a base is explicitly specified in the goal.
        # The keys of this dictionary are the blocks we care about being "correctly stacked".


    def is_correctly_stacked(self, block, state, memo):
        """
        Recursively checks if a block is in its correct goal position relative to its base,
        and if that base (if it's another block) is also correctly stacked.
        Uses memoization to store results for already checked blocks.

        Args:
            block (str): The name of the block to check.
            state (frozenset): The current state (set of facts).
            memo (dict): Memoization dictionary {block_name: boolean_result}.

        Returns:
            bool: True if the block is correctly stacked according to the goal structure, False otherwise.
        """
        # If the block's correctness has already been computed for this state, return the stored result.
        if block in memo:
            return memo[block]

        # Get the desired base for this block from the pre-calculated goal structure.
        # If a block is requested that doesn't have a defined goal base (shouldn't happen
        # if called only from the loop in __call__), it cannot be "correctly stacked"
        # in the sense of contributing to a goal stack segment.
        goal_base = self.goal_base.get(block)

        # This check is defensive; blocks passed to this function should have a goal_base.
        if goal_base is None:
             # This block is not part of a goal stack we are trying to build.
             # It cannot be "correctly stacked" according to our definition.
             # Store False in memo and return.
             memo[block] = False
             return False

        is_correct = False
        if goal_base == 'table':
            # The goal is for this block to be on the table. Check if the fact exists in the state.
            is_correct = f"(on-table {block})" in state
        else:
            # The goal is for this block to be on another block (goal_base).
            # Check if the fact (on block goal_base) exists in the state.
            if f"(on {block} {goal_base})" in state:
                # If it's on the correct block, recursively check if the base block is correctly stacked.
                # The current block is correctly stacked only if its base is also correctly stacked.
                is_correct = self.is_correctly_stacked(goal_base, state, memo)
            # Else: block is not on the correct base, so it's not correctly stacked.

        # Store the computed result in the memoization dictionary before returning.
        memo[block] = is_correct
        return is_correct


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        The heuristic value is the number of blocks that have a defined goal
        position but are not currently part of a correctly formed goal stack
        segment from the table up.

        Args:
            node: The current search node, containing the state.

        Returns:
            int: The estimated number of actions to reach the goal.
        """
        state = node.state

        # Check if the current state is the goal state. If so, the heuristic is 0.
        # This is a required condition for heuristics used in goal-directed search.
        if self.goals <= state:
             return 0

        # Initialize memoization dictionary for the recursive correctness check.
        # This ensures each block's correctness is computed at most once per state evaluation.
        memo = {}
        correctly_stacked_count = 0

        # Iterate through all blocks that have a defined goal position (either on another
        # block or on the table) according to the goal state. These are the blocks
        # whose correct placement contributes to achieving the goal stack structure.
        for block in self.goal_base.keys():
             # Check if the block is correctly stacked relative to its goal base,
             # and if the stack below it (in the goal) is also correctly stacked.
             if self.is_correctly_stacked(block, state, memo):
                 correctly_stacked_count += 1

        # The heuristic value is the total number of blocks that are part of the
        # goal stack structure (those with a defined goal base) minus the number
        # of those blocks that are currently correctly positioned within that structure.
        # This value represents the number of blocks that are "out of place" relative
        # to the desired goal stacks.
        heuristic_value = len(self.goal_base) - correctly_stacked_count

        # The heuristic value should always be non-negative.
        # It is 0 only when correctly_stacked_count equals len(self.goal_base),
        # which happens precisely when all blocks with a defined goal base are
        # correctly stacked according to the goal structure, implying the goal is met.
        return heuristic_value
