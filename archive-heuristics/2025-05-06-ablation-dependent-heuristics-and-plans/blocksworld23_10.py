from fnmatch import fnmatch
# Assuming heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not provided
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # print("Warning: heuristics.heuristic_base not found. Using dummy base class.")
    class Heuristic:
        def __init__(self, task):
            self.task = task
        def __call__(self, node):
            raise NotImplementedError

# Helper functions (adapted from examples)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string before slicing
    if not isinstance(fact, str):
        return [] # Return empty list for non-string input
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of args
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    by summing three components:
    1. The number of blocks that are not in their correct position within the goal stack structure.
    2. The number of blocks that are currently on top of a block identified in component 1 (blocking its movement).
    3. 1 if the arm is holding a block, 0 otherwise.

    # Assumptions
    - The goal specifies a set of stacks of blocks on the table.
    - Blocks must be moved one at a time.
    - Actions have unit cost.

    # Heuristic Initialization
    - Extract the desired stacking configuration from the goal facts,
      mapping each block to the block it should be on, or 'table'.
    - Identify all blocks that are part of the goal configuration.

    # Step-By-Step Thinking for Computing Heuristic
    1. Determine the desired support for each block based on the goal state.
       Create a mapping `desired_on[X] = Y` if `(on X Y)` is a goal, or
       `desired_on[X] = 'table'` if `(on-table X)` is a goal.
    2. Determine the current support for each block based on the current state.
       Create a mapping `current_on[X] = Y` if `(on X Y)` is in the state, or
       `current_on[X] = 'table'` if `(on-table X)` is in the state.
       Also, identify the block currently held by the arm.
    3. Identify the set of blocks that are "correctly placed" in their goal stack.
       A block X is correctly placed if:
       - `desired_on[X]` is 'table' and `(on-table X)` is true in the state.
       - `desired_on[X]` is Y, `(on X Y)` is true in the state, AND Y is correctly placed.
       This can be computed iteratively starting from blocks desired on the table.
    4. Identify the set of blocks that are "misplaced". These are blocks that
       are part of the goal configuration but are not correctly placed.
       Calculate `C1 = |Misplaced|`.
    5. Identify the set of blocks that are "blocking" a misplaced block.
       These are blocks Z such that `(on Z X)` is true in the current state
       and X is in the `Misplaced` set.
       Calculate `C2 = |Blocking|`.
    6. Check if the arm is holding a block. If so, this block needs to be
       put down or stacked.
       Calculate `C3 = 1` if `(holding X)` is true for some X, else `0`.
    7. The heuristic value is `C1 + C2 + C3`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the desired stacking
        configuration from the goal facts.
        """
        super().__init__(task) # Call base class constructor

        self.goals = task.goals

        # 1. Determine the desired support for each block based on the goal state.
        self.desired_on = {}
        self.goal_blocks = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "on": # Check if parts is not empty
                block, support = parts[1], parts[2]
                self.desired_on[block] = support
                self.goal_blocks.add(block)
                self.goal_blocks.add(support)
            elif parts and parts[0] == "on-table": # Check if parts is not empty
                block = parts[1]
                self.desired_on[block] = 'table'
                self.goal_blocks.add(block)
            # Ignore (clear X) goals for determining stack structure

        # Remove 'table' from goal_blocks if it was added
        self.goal_blocks.discard('table')

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        task = node.task # Access task from node to check goal_reached

        # Check if goal is reached first for efficiency and correctness
        if task.goal_reached(state):
             return 0

        # 2. Determine the current support for each block based on the current state.
        current_on = {}
        holding_block = None
        # Also build current_above map here for efficiency in step 5
        current_above = {}
        
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts

            if parts[0] == "on":
                block, support = parts[1], parts[2]
                current_on[block] = support
                current_above.setdefault(support, set()).add(block)
            elif parts[0] == "on-table":
                block = parts[1]
                current_on[block] = 'table'
            elif parts[0] == "holding":
                holding_block = parts[1]

        # 3. Identify the set of blocks that are "correctly placed" in their goal stack.
        correctly_placed = set()
        
        # Blocks desired to be on the table
        desired_on_table = {b for b, support in self.desired_on.items() if support == 'table'}

        # Blocks currently on the table
        currently_on_table = {b for b, support in current_on.items() if support == 'table'}

        # Initially correctly placed are those desired on table and currently on table
        correctly_placed.update(desired_on_table.intersection(currently_on_table))

        # Iteratively add blocks correctly placed on top of already correctly placed blocks
        while True:
            newly_correctly_placed = set()
            # Consider blocks that are desired to be on a block (not table)
            blocks_desired_on_block = {b for b, support in self.desired_on.items() if support != 'table'}

            for block in blocks_desired_on_block:
                if block not in correctly_placed:
                    desired_support = self.desired_on.get(block) # Use get in case block is not in desired_on (shouldn't happen for blocks_desired_on_block)
                    current_support = current_on.get(block) # Use get in case block is held or not in state

                    # Check if block is on its desired support AND desired support is correctly placed
                    if desired_support is not None and desired_support != 'table' and \
                       current_support == desired_support and desired_support in correctly_placed:
                         newly_correctly_placed.add(block)

            if not newly_correctly_placed:
                break # No new blocks were added in this iteration
            correctly_placed.update(newly_correctly_placed)

        # Now correctly_placed contains all blocks that are in the right place relative to the table base.

        # 4. Identify the set of blocks that are "misplaced".
        # These are blocks in the goal configuration that are not correctly placed.
        # We only care about blocks that are part of the goal structure.
        misplaced = {b for b in self.goal_blocks if b not in correctly_placed}
        C1 = len(misplaced)

        # 5. Identify the set of blocks that are "blocking" a misplaced block.
        blocking = set()
        # current_above map is already built

        for block in misplaced:
            # Find blocks currently on top of this misplaced block
            blocks_on_top = current_above.get(block, set())
            blocking.update(blocks_on_top)

        C2 = len(blocking)

        # 6. Check if the arm is holding a block.
        C3 = 1 if holding_block is not None else 0

        # 7. The heuristic value is C1 + C2 + C3.
        heuristic_value = C1 + C2 + C3

        return heuristic_value
