# Assume this base class exists or is replaced by a suitable definition
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # This works for facts with or without arguments, including arm-empty
    return fact[1:-1].split()

# Define the Blocksworld heuristic class
# class blocksworldHeuristic(Heuristic): # Use this if inheriting from a base class
class blocksworldHeuristic: # Use this if no base class is provided
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing several components:
    1. The number of blocks that are part of the goal stack structure but are not in their correct position relative to their support within that structure.
    2. The number of blocks that are currently on top of a block that is not in its correct position relative to its goal stack structure.
    3. The number of blocks that are required to be clear in the goal state but are not clear in the current state.
    4. A penalty if the robot arm is holding a block.

    # Assumptions
    - The goal state defines a specific configuration of blocks stacked on each other or on the table.
    - The heuristic assumes standard Blocksworld actions (pickup, putdown, stack, unstack).
    - The heuristic is not guaranteed to be admissible but aims to be informative for greedy best-first search.

    # Heuristic Initialization
    - Parses the goal conditions to identify:
        - `goal_config`: A mapping from a block to the block it should be directly on top of, or 'table'.
        - `goal_blocks`: The set of all blocks explicitly mentioned in the `on` or `on-table` goal predicates.
        - `goal_clear`: The set of blocks that must be clear in the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value for a given state is computed as follows:

    1.  **Parse Current State:** Determine the current position of each block (on another block, on the table, or held by the arm) and which blocks are clear.
        - `current_config`: Mapping from a block to its current support (block below or 'table').
        - `current_on_top`: Mapping from a block below to the block currently on top of it (assuming only one block on top).
        - `current_holding`: The block currently held by the arm, or `None`.
        - `current_clear`: The set of blocks that are currently clear.

    2.  **Identify Correctly Placed Blocks:** Determine the set of blocks (`correctly_placed_set`) that are currently in their correct position *relative to their goal support*, and whose support is also correctly placed (recursively).
        - Start with blocks in `goal_blocks` that are currently on the table and whose goal is also to be on the table.
        - Iteratively add blocks `B` from `goal_blocks` whose current support `A` matches their goal support (`goal_config[B] == A`), and `A` is already in `correctly_placed_set`.

    3.  **Calculate Component 1 (Misplaced Goal Blocks):** Count the number of blocks in `goal_blocks` that are *not* in the `correctly_placed_set`. These blocks are either on the wrong support or are part of an incorrect stack structure.

    4.  **Calculate Component 2 (Blocks on Incorrect Stacks):** Count the number of blocks `X` such that `(on X Y)` is true in the current state, and `Y` is *not* in the `correctly_placed_set`. These blocks need to be moved out of the way to build the correct stacks.

    5.  **Calculate Component 3 (Unmet Clear Goals):** Count the number of blocks `B` in `goal_clear` such that `(clear B)` is *false* in the current state. These blocks need to be cleared.

    6.  **Calculate Component 4 (Arm Penalty):** Add 1 to the heuristic if the robot arm is currently holding a block. This represents the cost of needing to free the arm.

    7.  **Sum Components:** The total heuristic value is the sum of the counts from steps 3, 4, 5, and 6.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals

        # Parse goal facts to build the goal configuration and identify goal blocks/clearance
        self.goal_config = {}  # block -> block_below or 'table'
        self.goal_blocks = set() # Blocks explicitly mentioned in goal 'on' or 'on-table'
        self.goal_clear = set()  # Blocks that must be clear in the goal

        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'on':
                self.goal_config[parts[1]] = parts[2]
                self.goal_blocks.add(parts[1])
                self.goal_blocks.add(parts[2])
            elif parts[0] == 'on-table':
                self.goal_config[parts[1]] = 'table'
                self.goal_blocks.add(parts[1])
            elif parts[0] == 'clear':
                self.goal_clear.add(parts[1])
            # Ignore (arm-empty) in goal parsing for structure, handle in __call__

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Ensure heuristic is 0 for goal states
        if self.goals <= state:
             return 0

        # 1. Parse Current State
        current_config = {}  # block -> block_below or 'table'
        current_on_top = {}  # block_below -> block_on_top (assuming only one block on top)
        current_holding = None
        current_clear = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                block_on = parts[1]
                block_below = parts[2]
                current_config[block_on] = block_below
                current_on_top[block_below] = block_on
            elif parts[0] == 'on-table':
                block = parts[1]
                current_config[block] = 'table'
            elif parts[0] == 'holding':
                current_holding = parts[1]
            elif parts[0] == 'clear':
                current_clear.add(parts[1])
            # Ignore (arm-empty) in state parsing

        # 2. Identify Correctly Placed Blocks
        correctly_placed_set = set()

        # Add blocks correctly on the table according to the goal
        for block, goal_support in self.goal_config.items():
            if goal_support == 'table' and current_config.get(block) == 'table':
                correctly_placed_set.add(block)

        # Iteratively add blocks correctly stacked on top of already correctly placed blocks
        added_this_iteration = True
        while added_this_iteration:
            added_this_iteration = False
            # Find blocks B such that goal_config[B] == A and current_config.get(B) == A
            # where A is already in correctly_placed_set.
            # Iterate through blocks that have a goal support that is a block.
            for block, goal_support in list(self.goal_config.items()):
                if block not in correctly_placed_set and goal_support != 'table':
                     current_support = current_config.get(block)
                     if current_support is not None and current_support == goal_support:
                         # Check if the support block is correctly placed
                         if goal_support in correctly_placed_set:
                             correctly_placed_set.add(block)
                             added_this_iteration = True

        # 3. Calculate Component 1 (Misplaced Goal Blocks)
        # Count blocks in goal_blocks that are NOT part of the correctly_placed_set
        h1 = sum(1 for block in self.goal_blocks if block not in correctly_placed_set)

        # 4. Calculate Component 2 (Blocks on Incorrect Stacks)
        # Count blocks X such that (on X Y) is true, and Y is NOT in correctly_placed_set.
        # This counts blocks that are on top of any block that is not part of a correctly built stack prefix.
        h2 = sum(1 for block_below in current_on_top if block_below not in correctly_placed_set)

        # 5. Calculate Component 3 (Unmet Clear Goals)
        # Count blocks B in goal_clear such that (clear B) is false in state.
        h_clear_goal = sum(1 for block in self.goal_clear if block not in current_clear)

        # 6. Calculate Component 4 (Arm Penalty)
        # Add 1 if the arm is holding any block.
        h_arm = 1 if current_holding is not None else 0

        # 7. Sum Components
        total_cost = h1 + h2 + h_clear_goal + h_arm

        return total_cost
