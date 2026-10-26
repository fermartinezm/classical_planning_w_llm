from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not running within the planner environment
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError
        def __str__(self):
            return self.__class__.__name__
        def __repr__(self):
            return f"<{self.__class__.__name__}>"


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle cases like '(arm-empty)' which has no arguments
    content = fact[1:-1].strip()
    if not content:
        return (content,)
    return tuple(content.split())


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    by counting blocks that are not in their correct position within the goal
    stacks and adding a penalty for blocks that are currently on top of them.
    It assigns a cost of 2 for each block that needs to be moved because it's
    misplaced or obstructing a misplaced block. This is a non-admissible
    heuristic designed to guide a greedy best-first search.

    # Assumptions
    - The goal state defines specific stacks of blocks on the table.
    - Blocks not mentioned in the goal 'on' or 'on-table' facts do not have
      a specific required position relative to the goal stacks (this heuristic
      focuses only on blocks whose position is specified in the goal).
    - The cost of moving a block (pickup/unstack + putdown/stack) is approximately 2 actions.
    - Clearing a block by moving one block off it also costs approximately 2 actions.

    # Heuristic Initialization
    - Parses the goal facts to build the target stack structure:
      - `goal_parent`: A dictionary mapping a block to the block it should be on in the goal.
      - `goal_on_table`: A set of blocks that should be on the table in the goal.
      - `blocks_with_goal_position`: A set of all blocks whose position is explicitly
        defined in the goal (either on another block or on the table).

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to determine the current stack structure:
       - `current_parent`: A dictionary mapping a block to the block it is currently on.
       - `current_on_table`: A set of blocks currently on the table.
    2. Initialize the heuristic cost `h` to 0.
    3. Initialize a memoization dictionary `memo` for the `_is_correctly_stacked` recursive function.
    4. Iterate through each block `b` in the set `self.blocks_with_goal_position`.
    5. For each block `b`, check if it is "correctly stacked" using a recursive helper function `_is_correctly_stacked(b, ..., memo)`.
       - A block `X` is correctly stacked if:
         - Its goal is `(on-table X)` AND it is currently `(on-table X)`.
         - OR its goal is `(on X Y)` AND it is currently `(on X Y)` AND `Y` is also correctly stacked (recursively checked).
       - Memoization is used to avoid redundant calculations for blocks that are part of multiple checks.
    6. If a block `b` is *not* correctly stacked according to the goal:
       - Add 2 to the heuristic cost `h` (representing the estimated cost to move `b` itself).
       - Count the number of blocks currently stacked directly or indirectly on top of `b` in the current state using a helper function `_count_blocks_on_top(b, current_parent)`.
       - Add 2 times this count to `h` (representing the estimated cost to clear `b` by moving the blocks on top).
    7. Return the total heuristic cost `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal stack information.
        """
        self.goals = task.goals

        # Build goal stack structure
        self.goal_parent = {}
        self.goal_on_table = set()
        self.blocks_with_goal_position = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: # Skip empty facts like '()'
                continue
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'on':
                if len(args) == 2:
                    block, parent = args
                    self.goal_parent[block] = parent
                    self.blocks_with_goal_position.add(block)
                    # The parent block's position might also be specified as a goal
                    # We don't add parent to blocks_with_goal_position here,
                    # it will be added if it appears as a key in goal_parent or in goal_on_table.
            elif predicate == 'on-table':
                if len(args) == 1:
                    block = args[0]
                    self.goal_on_table.add(block)
                    self.blocks_with_goal_position.add(block)
            # Ignore (clear ?x) goals for this heuristic based on stack structure

    def _count_blocks_on_top(self, block, current_parent):
        """
        Recursively counts the number of blocks stacked directly or indirectly
        on top of the given block in the current state.
        """
        # Find the block directly on top of 'block'
        block_on_top = None
        for b, parent in current_parent.items():
            if parent == block:
                block_on_top = b
                break

        if block_on_top is None:
            return 0 # No block directly on top
        else:
            # 1 (for the block directly on top) + blocks on top of that block
            return 1 + self._count_blocks_on_top(block_on_top, current_parent)

    def _is_correctly_stacked(self, block, goal_parent, goal_on_table, current_parent, current_on_table, memo):
        """
        Recursively checks if a block is in its correct position relative
        to the goal stack it belongs to.
        """
        if block in memo:
            return memo[block]

        result = False

        # Check if the block's goal is to be on the table
        if block in goal_on_table:
            # It's correctly stacked if it's currently on the table
            result = block in current_on_table
        else:
            # The goal is (on block goal_pos)
            goal_pos = goal_parent.get(block)
            if goal_pos is not None: # Ensure block is part of an 'on' goal
                 # Find the block currently under 'block'
                current_pos = current_parent.get(block)

                # It's correctly stacked if it's on the correct block AND
                # the block below it is also correctly stacked.
                if current_pos == goal_pos:
                    # Recursive call for the block below. The block below (goal_pos)
                    # must also have its position specified in the goal for this
                    # recursive check to make sense in building a goal stack.
                    # We assume well-formed blocksworld goals where goal parents
                    # are also part of the goal stack structure.
                    if goal_pos in self.blocks_with_goal_position:
                         result = self._is_correctly_stacked(goal_pos, goal_parent, goal_on_table, current_parent, current_on_table, memo)
                    else:
                         # If the block below is NOT in blocks_with_goal_position,
                         # it means its position is not specified in the goal.
                         # This case is unusual for standard blocksworld goal stacks.
                         # If it happens, let's consider the current block correctly
                         # stacked relative to its parent if it's on the correct parent,
                         # assuming the parent's position doesn't matter for this stack segment.
                         # However, the definition of 'correctly stacked' implies the whole stack segment must match.
                         # Let's stick to the strict definition: parent must also be correctly stacked.
                         # If goal_pos is not in blocks_with_goal_position, it cannot be correctly stacked
                         # according to our recursive definition's base cases (on_table or on correctly_stacked).
                         # So, result remains False if goal_pos is not in blocks_with_goal_position.
                         result = False # Strict interpretation
                else:
                    # It's on the wrong block or on the table when it shouldn't be
                    result = False
            # If block is not in goal_on_table and not a key in goal_parent,
            # it shouldn't be passed to this function based on the loop in __call__.

        memo[block] = result
        return result

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Build current stack structure
        current_parent = {}
        current_on_table = set()
        # current_clear = set() # Not needed for this heuristic
        # current_holding = None # Not needed for this heuristic

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'on':
                if len(args) == 2:
                    block, parent = args
                    current_parent[block] = parent
            elif predicate == 'on-table':
                if len(args) == 1:
                    block = args[0]
                    current_on_table.add(block)
            # elif predicate == 'clear':
            #     if len(args) == 1:
            #         current_clear.add(args[0])
            # elif predicate == 'holding':
            #      if len(args) == 1:
            #          current_holding = args[0]


        h = 0
        memo = {} # Memoization for _is_correctly_stacked

        # Iterate only over blocks whose position is specified in the goal
        for block in self.blocks_with_goal_position:
            if not self._is_correctly_stacked(block, self.goal_parent, self.goal_on_table, current_parent, current_on_table, memo):
                # This block is not in its correct goal stack position relative to the goal
                h += 2 # Estimated cost to move this block itself

                # Add estimated cost for clearing blocks on top of this block
                # Each block on top needs to be moved, costing ~2 actions.
                h += 2 * self._count_blocks_on_top(block, current_parent)

        # This heuristic does not explicitly count costs for clearing
        # the target location if it's a block that is not clear.
        # The assumption is that if the target block is part of a goal stack
        # below the current block, its 'not correctly stacked' status and
        # the blocks on top of it will contribute to the heuristic cost
        # through the loop iterating over blocks_with_goal_position.

        # The heuristic is 0 iff all blocks with a specified goal position
        # are correctly stacked according to the recursive definition, which
        # implies the goal stack configuration is achieved. This should be 0
        # only at the goal state for standard blocksworld problems.

        return h
