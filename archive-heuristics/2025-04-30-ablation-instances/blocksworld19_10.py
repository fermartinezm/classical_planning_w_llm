from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Assumes fact is like '(predicate arg1 arg2)'
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    by counting:
    1. Blocks that are not in their correct position within the goal stacks
       (considering the stack structure from the bottom up).
    2. Blocks that are currently obstructing blocks that are required to be clear
       in the goal state but are not.
    3. Whether the arm is holding a block when the goal requires it to be empty.

    # Assumptions
    - The goal state defines one or more stacks of blocks on the table,
      plus possibly an empty arm and clear top blocks.
    - The heuristic counts necessary moves, considering dependencies for
      building stacks from the bottom and clearing obstructions.

    # Heuristic Initialization
    - Parses the goal facts to determine the required parent block (or table)
      for each block that is part of a goal stack or is required to be on the table.
      This builds the `self.goal_parent` map.
    - Identifies all blocks that are required to be clear in the goal state.
    - Checks if arm-empty is a goal.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Determine the current parent (block, table, or arm) for every block
       in the state. Store this in `current_parent` map.
    2. Check if the arm is currently empty.
    3. Identify which blocks are "in-place". A block `B` is "in-place" if:
       - Its goal is `(on-table B)` and `(on-table B)` is true in the state.
       - OR its goal is `(on A B)` and `(on A B)` is true in the state AND
         block `B` is itself "in-place".
       This is computed recursively and memoized. The base case for recursion
       is a block whose goal is `(on-table ...)`. If a block's target parent
       is a block `Y` which is not itself a child in any goal fact, `Y` is
       considered in-place if it is on the table.
    4. Count the number of blocks that are part of a goal stack/on-table goal
       (i.e., are keys in `self.goal_parent`) but are *not* "in-place".
       Add this count to the heuristic value `h`. These blocks need to be moved
       to their correct position relative to the correctly built stack below them.
    5. Count the number of blocks `T` that are currently on top of a block `B`
       where `(clear B)` is a goal fact AND `(clear B)` is not true in the state.
       Add this count to `h`. These blocks `T` need to be moved out of the way.
    6. If `(arm-empty)` is a goal fact AND the arm is not empty in the state,
       add 1 to `h`. The held block needs to be put down or stacked.
    7. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal structure.
        """
        self.goals = task.goals

        # Build the goal_parent map: block -> parent_block (or 'table')
        self.goal_parent = {}
        # Identify blocks that must be clear in the goal
        self.goal_clear_blocks = set()
        # Check if arm-empty is a goal
        self.goal_arm_empty = False

        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'on':
                block, parent = parts[1], parts[2]
                self.goal_parent[block] = parent
            elif parts[0] == 'on-table':
                block = parts[1]
                self.goal_parent[block] = 'table'
            elif parts[0] == 'clear':
                block = parts[1]
                self.goal_clear_blocks.add(block)
            elif parts[0] == 'arm-empty':
                self.goal_arm_empty = True

    def _is_in_place(self, block, goal_parent_map, current_parent_map, memo):
        """
        Recursive helper to check if a block is in its correct goal position
        relative to the correctly built stack below it.
        Uses memoization. Assumes 'block' is a key in goal_parent_map.
        """
        if block in memo:
            return memo[block]

        target_parent = goal_parent_map[block]
        current_p = current_parent_map.get(block)

        # If current_p is None, the block is not in the state in a standard way.
        # This shouldn't happen in valid blocksworld states. Treat as not in-place.
        if current_p is None:
             memo[block] = False
             return False

        if current_p != target_parent:
            memo[block] = False
            return False

        if target_parent == 'table':
            memo[block] = True
            return True
        else: # target_parent is a block
            # Block is on the correct parent, now check if the parent is in-place.
            # The parent block must also be a key in goal_parent_map to be part
            # of the goal stack structure we are checking recursively.
            # If the parent is not in goal_parent_map, it means the parent is
            # the base of a goal stack, and its goal is implicitly on-table.
            # Check if the parent is currently on the table.
            if target_parent not in goal_parent_map:
                 is_parent_in_place = (current_parent_map.get(target_parent) == 'table')
                 memo[block] = is_parent_in_place
                 return is_parent_in_place
            else:
                 # target_parent is a block and is also a child in some goal fact.
                 memo[block] = self._is_in_place(target_parent, goal_parent_map, current_parent_map, memo)
                 return memo[block]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Determine current parent for every block and check arm status
        current_parent = {}
        is_arm_empty = False
        state_clear_blocks = set()

        # Build current_parent and state_clear_blocks
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                block, parent = parts[1], parts[2]
                current_parent[block] = parent
            elif parts[0] == 'on-table':
                block = parts[1]
                current_parent[block] = 'table'
            elif parts[0] == 'holding':
                block = parts[1]
                current_parent[block] = 'arm' # Indicate it's held
            elif parts[0] == 'arm-empty':
                is_arm_empty = True
            elif parts[0] == 'clear':
                block = parts[1]
                state_clear_blocks.add(block)

        h = 0
        memo = {}
        misplaced_blocks = set()

        # 4. Count blocks not in-place
        # Iterate only over blocks that are children in goal facts (keys in goal_parent)
        for block in self.goal_parent.keys():
             # Ensure the block is present in the state (either on/on-table/held)
             # If not present, it's an invalid state for this domain.
             # The get() in _is_in_place handles the case where block is not a key in current_parent
             if block in current_parent:
                if not self._is_in_place(block, self.goal_parent, current_parent, memo):
                    misplaced_blocks.add(block)
                    h += 1
             # else: block is not in state at all? Should not happen in valid instances.

        # 5. Count blocks on top of blocks that should be clear but aren't
        # Build current_child map for efficient lookup
        current_child = {}
        for block, parent in current_parent.items():
            if parent not in ['table', 'arm']:
                if parent not in current_child:
                    current_child[parent] = set()
                current_child[parent].add(block)

        for block_to_be_clear in self.goal_clear_blocks:
            if block_to_be_clear not in state_clear_blocks:
                 # This block should be clear but isn't. Something is on it.
                 # Find what is on it.
                 if block_to_be_clear in current_child:
                     # All blocks on top need to be moved.
                     # In blocksworld, only one block can be directly on another.
                     # So current_child[block_to_be_clear] will have at most one element.
                     for blocker_block in current_child[block_to_be_clear]:
                         h += 1 # Count the blocker block

        # 6. Add cost if arm is not empty and should be
        if self.goal_arm_empty and not is_arm_empty:
            h += 1

        return h
