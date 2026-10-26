from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

# Helper function to parse facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by counting the number of blocks that are not in their correct position
    relative to their base in the goal stack, recursively defined. Each such
    misplaced block contributes 2 to the heuristic value, representing the
    estimated cost of unstacking/picking it up and stacking/putting it down.

    # Assumptions
    - The goal specifies the desired configuration of blocks in stacks or on the table.
    - Blocks not explicitly mentioned in goal 'on' or 'on-table' predicates are assumed
      to belong on the table in the goal state.
    - Each move of a block (unstack/pick + stack/put-down) costs approximately 2 actions.
    - The heuristic focuses on block positions and does not explicitly penalize
      the arm holding a block unless that block is also misplaced.

    # Heuristic Initialization
    - Identify all objects (blocks) present in the initial state.
    - Parse the goal conditions to determine the desired base for each block
      (which block it should be on, or if it should be on the table).
    - Build a data structure (`goal_base`) representing the goal stacks.
    - Handle implicit goals for blocks not mentioned in explicit goal stacks (they go on the table).

    # Step-By-Step Thinking for Computing Heuristic
    1. In the `__init__` method:
       - Extract all block names from the initial state facts (`on`, `on-table`, `clear`, `holding`).
       - Parse the goal facts to build a mapping `goal_base` where `goal_base[B] = A`
         if `(on B A)` is a goal, and `goal_base[B] = 'table'` if `(on-table B)` is a goal.
       - For any block identified in the initial state but not present as a key in `goal_base`,
         assume its goal is `(on-table B)` and add `goal_base[B] = 'table'`.
       - Store the set of all objects.
    2. In the `__call__` method for a given state:
       - Parse the current state facts to build a mapping `current_base` where
         `current_base[B] = X` if `(on B X)` is true, and `current_base[B] = 'table'`
         if `(on-table B)` is true. A block being held has no base in this map (represented as None).
       - Define a recursive helper function `is_correctly_positioned(block, current_base, goal_base, memo)`:
         - This function checks if a block is in its correct position relative to its base,
           and if that base is also correctly positioned, recursively.
         - It uses the `goal_base` and `current_base` maps.
         - Base case: If the block's goal is `(on-table B)`, it's correctly positioned if `(on-table B)` is true in the current state (`current_base.get(B) == 'table'`).
         - Recursive step: If the block's goal is `(on B A)`, it's correctly positioned if `(on B A)` is true in the current state (`current_base.get(B) == A`) AND `A` is correctly positioned (`is_correctly_positioned(A, ...)`).
         - Use memoization (`memo`) to avoid redundant calculations for blocks that are bases for multiple stacks or appear in different recursive calls.
         - Handle cases where a block is being held (its current base is None).
       - Initialize heuristic value `h = 0`.
       - Initialize a memoization dictionary `correctly_positioned_memo = {}`.
       - Iterate through all blocks identified in `__init__`:
         - If `is_correctly_positioned(block, current_base, self.goal_base, correctly_positioned_memo)` returns `False`, increment `h`.
       - Return `h * 2`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal configuration and objects."""
        self.goals = task.goals
        self.initial_state = task.initial_state

        # Extract all objects from the initial state facts
        self.objects = set()
        for fact in self.initial_state:
            parts = get_parts(fact)
            # Consider predicates that involve objects
            if parts[0] in ['on', 'on-table', 'clear', 'holding']:
                # Add all arguments as objects
                for obj in parts[1:]:
                    self.objects.add(obj)

        # Parse goal predicates to build goal configuration maps
        self.goal_base = {}  # block -> block_below or 'table'

        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == 'on':
                block, base = parts[1], parts[2]
                self.goal_base[block] = base
            elif predicate == 'on-table':
                block = parts[1]
                self.goal_base[block] = 'table'
            # Ignore clear and arm-empty goals for the base structure

        # Add implicit goals for blocks not mentioned in goal stacks
        # Assume they should be on the table
        for obj in self.objects:
            if obj not in self.goal_base:
                 self.goal_base[obj] = 'table'

    def is_correctly_positioned(self, block, current_base, goal_base, memo):
        """
        Recursively check if a block is in its correct goal position relative to its base,
        and if its base is also correctly positioned.
        """
        if block in memo:
            return memo[block]

        # A block not in our object list cannot be correctly positioned in a relevant stack.
        # This case should ideally not be reached for objects in self.objects.
        if block not in self.objects:
             memo[block] = False
             return False

        target_base = goal_base.get(block)
        current_base_val = current_base.get(block) # Will be None if holding

        # Check if the current base matches the target base
        if current_base_val != target_base:
            memo[block] = False
            return False

        # If the target base is the table, and the current base is the table, it's correctly positioned.
        if target_base == 'table':
            memo[block] = True
            return True

        # If the target base is another block, check if that block is correctly positioned.
        # current_base_val == target_base (which is a block)
        # Ensure the target_base block is also one of the objects we are tracking
        if target_base in self.objects:
             result = self.is_correctly_positioned(target_base, current_base, goal_base, memo)
             memo[block] = result
             return result
        else:
             # This means the goal stack involves a block not in the initial state objects.
             # This is an invalid problem instance or domain assumption mismatch.
             # Treat this path as incorrect positioning.
             memo[block] = False
             return False


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse current state predicates
        current_base = {} # block -> block_below or 'table' or None (if holding)
        # current_on_top = {} # Not needed for this version of heuristic
        # current_holding = None # Not explicitly needed, base=None implies holding

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'on':
                block, base = parts[1], parts[2]
                current_base[block] = base
            elif predicate == 'on-table':
                block = parts[1]
                current_base[block] = 'table'
            elif predicate == 'holding':
                block = parts[1]
                current_base[block] = None # Block being held has no base
            # Ignore clear and arm-empty

        h = 0
        correctly_positioned_memo = {}

        # Count blocks not correctly positioned in their goal stack
        for obj in self.objects:
            # is_correctly_positioned handles the case where current_base[obj] is None (holding)
            if not self.is_correctly_positioned(obj, current_base, self.goal_base, correctly_positioned_memo):
                 h += 1

        # The heuristic is the number of blocks not in their correct recursive position, times 2
        # (representing pick/unstack + put/stack).
        return h * 2
