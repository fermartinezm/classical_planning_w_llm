from fnmatch import fnmatch
# Assuming heuristic_base.py exists and defines a Heuristic class with __init__ and __call__
# If running this code standalone, you might need a mock Heuristic class:
# class Heuristic:
#     def __init__(self, task):
#         pass
#     def __call__(self, node):
#         raise NotImplementedError

# Helper functions outside the class as seen in examples
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle facts like '(arm-empty)'
    if fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    # Should not happen for valid PDDL facts representing ground predicates
    return [fact]

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    Estimates the number of actions needed to reach the goal state.
    The heuristic counts the number of clear blocks that are not in their
    correct goal position (relative to the block below them, recursively)
    and multiplies this count by 2 (representing pickup/unstack + putdown/stack).
    If the arm is holding a block, it adds 1 (representing the action to
    put it down or stack it).

    # Heuristic Initialization
    - Extracts the goal configuration to determine the desired parent for each block.
      Blocks not explicitly placed on another block or the table in the goal
      are assumed to have a goal of being on the table.
    - Infers the set of all objects from the initial state and goal facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. In `__init__`:
       a. Infer the set of all objects involved in the problem from initial state and goal facts.
       b. Build a map `goal_parent_map` from the goal facts, indicating which block
          should be directly under each block (or 'table'). Blocks not explicitly
          placed as the first argument of an 'on' or 'on-table' goal are assumed
          to have a goal of being on the table.
    2. In the `__call__` method for a given state:
       a. Build a map `current_parent_map` from the state facts, indicating
          which block is directly under each block (or 'table').
       b. Identify the block being held (if any) and the set of clear blocks
          (blocks that are not currently supporting any other block and are not held).
       c. Define a helper function `is_in_goal_position(block, current_parent_map, memo)`
          that recursively checks if the block is on its correct goal parent, and if
          that parent is also in its correct goal position, all the way down to the table.
          Uses the pre-computed `goal_parent_map` and the current `current_parent_map`.
          Memoization is used for efficiency.
       d. Initialize heuristic value `h = 0`.
       e. Check if the arm is holding a block. If yes, add 1 to `h` (cost to put it down/stack).
       f. Iterate through all blocks that are currently clear (on the table or another block).
          For each clear block, check if it is in its goal position using `is_in_goal_position`.
          If a clear block is NOT in its goal position, add 2 to `h` (representing the minimum
          actions needed to move it: pickup/unstack + putdown/stack).
       g. Return `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration and objects.
        """
        self.goals = task.goals

        # Infer objects from initial state and goals
        potential_objects = set()
        # Known predicates and literals to exclude when inferring objects
        known_predicates = {'clear', 'on-table', 'arm-empty', 'holding', 'on', 'object'}
        known_literals = {'table'}

        # Extract potential objects from initial state facts
        for fact in task.initial_state:
            parts = get_parts(fact)
            for part in parts[1:]: # Arguments after the predicate
                 if part not in known_predicates and part not in known_literals and not part.isdigit():
                     potential_objects.add(part)

        # Extract potential objects from goal facts
        for goal in self.goals:
            parts = get_parts(goal)
            for part in parts[1:]: # Arguments after the predicate
                 if part not in known_predicates and part not in known_literals and not part.isdigit():
                     potential_objects.add(part)

        self.objects = sorted(list(potential_objects)) # Store as sorted list for consistency

        # Build goal_parent_map: block -> parent_block_or_table
        self.goal_parent_map = {}
        # Track blocks that are explicitly mentioned as the first argument of 'on' or 'on-table' goals
        blocks_with_explicit_goal_parent = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'on' and len(parts) == 3:
                block, parent = parts[1], parts[2]
                self.goal_parent_map[block] = parent
                blocks_with_explicit_goal_parent.add(block)
            elif parts[0] == 'on-table' and len(parts) == 2:
                block = parts[1]
                self.goal_parent_map[block] = 'table'
                blocks_with_explicit_goal_parent.add(block)
            # Ignore 'clear' and 'arm-empty' goals for parent mapping

        # Assume blocks not explicitly placed on something or table in goal should be on the table.
        # This handles blocks that might be under goal stacks but whose own position isn't specified
        # as the first argument of an 'on' or 'on-table' goal fact.
        for obj in self.objects:
             if obj not in blocks_with_explicit_goal_parent:
                  self.goal_parent_map[obj] = 'table'


    def is_in_goal_position(self, block, current_parent_map, memo):
        """
        Checks if a block is in its correct goal position recursively with memoization.
        A block B is in goal position if it's on its goal parent A, and A is
        in its goal position (down to the table).
        """
        # Base case: 'table' is always in goal position
        if block == 'table':
            return True

        # Check memoization cache
        if block in memo:
            return memo[block]

        # Get current parent. If None, the block is held or missing from the state.
        # A block not on a table or another block cannot be in a goal position
        # that requires it to be on a table or block.
        current_p = current_parent_map.get(block)
        if current_p is None:
             memo[block] = False
             return False

        # Get the desired parent from the goal
        # Default to 'table' if the block's goal position isn't specified (handled in init)
        goal_parent = self.goal_parent_map.get(block, 'table')

        # Check if the immediate parent is correct AND the parent is in its goal position
        if goal_parent == 'table':
            # Goal is (on-table block)
            is_correct = (current_p == 'table')
        else:
            # Goal is (on block goal_parent)
            is_correct = (current_p == goal_parent) and \
                         self.is_in_goal_position(goal_parent, current_parent_map, memo)

        # Store result in memoization cache
        memo[block] = is_correct
        return is_correct


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Build current_parent_map and find held block and clear blocks
        current_parent_map = {}
        held_block = None
        # Find blocks that are parents (i.e., have something on top of them)
        blocks_that_are_parents = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on' and len(parts) == 3:
                block, parent = parts[1], parts[2]
                current_parent_map[block] = parent
                blocks_that_are_parents.add(parent) # The parent block now has something on it
            elif parts[0] == 'on-table' and len(parts) == 2:
                block = parts[1]
                current_parent_map[block] = 'table'
            elif parts[0] == 'holding' and len(parts) == 2:
                held_block = parts[1]
            # 'clear' and 'arm-empty' facts are used to determine clear_blocks and held_block/arm_empty state

        # A block is clear if nothing is on it and it's not held.
        clear_blocks = {obj for obj in self.objects if obj not in blocks_that_are_parents and obj != held_block}

        total_cost = 0
        memo = {} # Memoization dictionary for is_in_goal_position

        # Cost for the block being held
        if held_block:
             # If holding a block, we must perform an action (putdown or stack) to free the arm.
             # This costs 1 action.
             total_cost += 1
             # The cost of moving the held block to its final destination (if it's not already there)
             # is implicitly captured by the clear_blocks loop *if* the block becomes clear
             # and remains misplaced after being put down/stacked in the next state.
             # This is a reasonable approximation for a non-admissible heuristic.

        # Cost for clear blocks on table/other blocks
        # These are blocks we can potentially pick up/unstack.
        for block in clear_blocks:
             # Check if the clear block is in its goal position
             if not self.is_in_goal_position(block, current_parent_map, memo):
                 # If a clear block is not in its goal position, it needs to be moved.
                 # Minimum actions to move a clear block: pickup/unstack (1) + putdown/stack (1) = 2.
                 total_cost += 2

        return total_cost

