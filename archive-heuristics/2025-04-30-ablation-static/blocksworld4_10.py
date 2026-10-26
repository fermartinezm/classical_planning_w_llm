# Assume heuristics.heuristic_base.Heuristic is available
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the difficulty of reaching the goal state by counting
    the number of blocks that are not in their correct position within the goal
    stack structure, the number of blocks that should be clear but are not,
    and adding a penalty if the robot's arm is holding a block.

    # Assumptions
    - The goal state defines specific stacks of blocks and blocks that must be clear.
    - The heuristic assumes standard Blocksworld actions (pickup, putdown, stack, unstack).

    # Heuristic Initialization
    The heuristic pre-processes the goal state to build data structures representing:
    - The desired base for each block involved in a goal stack (`goal_base_map`).
    - The set of blocks that must be clear in the goal state (`goal_clear_blocks`).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:
    1.  Parse the current state to determine:
        -   The block currently on top of each other block or the table (`current_base_map`).
        -   The set of blocks that are currently clear (`current_clear_blocks`).
        -   The block currently held by the arm (`holding_block`).
    2.  Define a recursive helper function `_is_correctly_stacked(block)` that checks if a block is in its final goal position relative to the stack below it. This function uses memoization to be efficient. A block `B` is correctly stacked if:
        -   It is currently on the block (or table) that it should be on according to `goal_base_map`, AND
        -   The block it is on (if it's a block, not the table) is also correctly stacked.
        -   A block being held is never correctly stacked relative to a base.
    3.  Initialize the heuristic value `h = 0`.
    4.  **Component 1 (Misstacked Blocks):** Iterate through all blocks that are part of a goal stack (i.e., are keys in `goal_base_map`). If a block is currently held, it is considered misstacked. Otherwise, call `_is_correctly_stacked` for the block. For each block that is not correctly stacked, add 1 to `h`.
    5.  **Component 2 (Uncleared Goal Blocks):** Iterate through all blocks that must be clear in the goal state (`goal_clear_blocks`). If a block is not in the set of currently clear blocks (`current_clear_blocks`), add 1 to `h`.
    6.  **Component 3 (Busy Arm):** If the robot's arm is currently holding any block (`holding_block` is not None), add 1 to `h`. This penalizes having the arm busy when it could be used for a goal-relevant action.
    7.  Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals

        # Data structures to represent the goal configuration
        self.goal_base_map = {} # block -> base (block or 'table')
        self.goal_clear_blocks = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'on':
                block, base = parts[1], parts[2]
                self.goal_base_map[block] = base
            elif parts[0] == 'on-table':
                block = parts[1]
                self.goal_base_map[block] = 'table'
            elif parts[0] == 'clear':
                block = parts[1]
                self.goal_clear_blocks.add(block)

        # Note: Blocksworld has no static facts relevant to this heuristic.
        # task.static is available but not used here.

    def _is_correctly_stacked(self, block, current_base_map, holding_block, memo):
        """
        Recursive helper to check if a block is in its final goal position
        relative to the stack below it, using memoization.
        Assumes 'block' is a key in self.goal_base_map.
        """
        if block in memo:
            return memo[block]

        # If the block is currently held, it's not correctly stacked relative to a base.
        if holding_block == block:
            memo[block] = False
            return False

        goal_base = self.goal_base_map[block]
        current_base = current_base_map.get(block) # Use .get in case block is not on anything (e.g. off-table in some domains, though not blocksworld)

        # Check if it's on the correct base
        if current_base != goal_base:
            memo[block] = False
            return False

        # If it's on the correct base, check the base recursively
        if goal_base == 'table':
            memo[block] = True
            return True
        else:
            # goal_base is a block name here. It must also be a key in goal_base_map.
            # The recursion will eventually hit the 'table' base case.
            memo[block] = self._is_correctly_stacked(goal_base, current_base_map, holding_block, memo)
            return memo[block]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Data structures to represent the current state configuration
        current_base_map = {} # block -> base (block or 'table')
        current_clear_blocks = set()
        holding_block = None

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                block, base = parts[1], parts[2]
                current_base_map[block] = base
            elif parts[0] == 'on-table':
                block = parts[1]
                current_base_map[block] = 'table'
            elif parts[0] == 'clear':
                current_clear_blocks.add(parts[1])
            elif parts[0] == 'holding':
                holding_block = parts[1]

        # Memoization for _is_correctly_stacked calls within this state evaluation
        memo = {}

        h = 0

        # Component 1: Blocks in goal stacks not correctly stacked
        # Iterate over all blocks that are keys in goal_base_map
        for block in self.goal_base_map:
             # A held block is never correctly stacked relative to a base
             if holding_block == block:
                 h += 1
                 continue

             # Check if the block is correctly stacked relative to its goal base and the stack below it
             if not self._is_correctly_stacked(block, current_base_map, holding_block, memo):
                 h += 1

        # Component 2: Goal clear predicates not satisfied
        for block in self.goal_clear_blocks:
            if block not in current_clear_blocks:
                h += 1

        # Component 3: Arm holding a block
        if holding_block is not None:
            h += 1

        return h
