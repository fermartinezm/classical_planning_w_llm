# Assuming the Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if isinstance(fact, str) and fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    return [] # Should not happen with valid PDDL facts from the planner

class blocksworldHeuristic: # Inherit from Heuristic if available
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the cost by summing two components that represent
    structural differences between the current state and the goal state:
    1. The number of blocks that are not on their correct goal support (either another block or the table).
    2. The number of blocks that are currently on top of another block, where the block on top is not the one that should be there according to the goal state (or the block below should be clear).

    # Assumptions
    - The goal state defines the desired support for each block (`on` or `on-table`).
    - The goal state implicitly defines which block should be on top of another, or if a block should be clear (`clear`).
    - Each instance of a block being on the wrong support, or a block having the wrong block on top, contributes 1 to the total heuristic cost. This is a simplified count of necessary "corrections".
    - The standard Blocksworld actions (pickup, putdown, stack, unstack) are used.
    - The arm can hold only one block at a time.

    # Heuristic Initialization
    - Extract all block objects involved in the problem from the initial state and goal conditions.
    - Build `goal_on_map`: maps each block (that has a specified support in the goal) to its required support (another block or 'table') based on goal facts (`on`, `on-table`).
    - Build `goal_on_top_map`: maps each block (that is a support for another block in the goal) to the block that should be directly on top of it based on goal facts (`on`). Blocks that should be clear in the goal are implicitly handled by their absence in this map.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic cost to 0.
    2. Build `current_on_map`: maps each block to its current support (another block or 'table') based on state facts (`on`, `on-table`). Identify the block currently being held, if any.
    3. Build `current_on_top_map`: maps each block (that is a support) to the block currently directly on top of it based on state facts (`on`).
    4. Iterate through all known blocks (those found in the initial state or goals):
       - Determine the block's current support. If the block is held, its support is considered 'arm'.
       - Find the block's goal support using `goal_on_map`. If a block is not in `goal_on_map`, it doesn't have a specified goal support, so it doesn't contribute to this component.
       - If the block has a goal support (`block in self.goal_on_map`) AND its current support does not match its goal support, add 1 to the total cost.
    5. Iterate through all known blocks (as potential supports):
       - Determine the block currently on top of the support block. If the support block is being held, nothing is on top of it.
       - Find the block that should be on top of the support block according to the goal using `goal_on_top_map`. If nothing should be on top (i.e., the support block should be clear), this map won't have an entry, equivalent to `None`.
       - If the block currently on top is different from the block that should be on top, add 1 to the total cost. This counts blocks that are incorrectly placed on top of other blocks.
    6. Return the total calculated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal information and objects.
        """
        self.goals = task.goals

        # Extract all blocks from initial state and goals
        self.all_blocks = set()
        for fact in task.initial_state:
             parts = get_parts(fact)
             if len(parts) > 1:
                 # Add all arguments except predicate name
                 self.all_blocks.update(parts[1:])
        for goal in task.goals:
             parts = get_parts(goal)
             if len(parts) > 1:
                 self.all_blocks.update(parts[1:])

        # Build goal_on_map: block -> support (another block or 'table')
        self.goal_on_map = {}
        # Build goal_on_top_map: support (block) -> block (on top)
        self.goal_on_top_map = {}

        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'on' and len(parts) == 3:
                block, support = parts[1], parts[2]
                self.goal_on_map[block] = support
                # Store the block that should be on top of 'support'
                self.goal_on_top_map[support] = block
            elif parts[0] == 'on-table' and len(parts) == 2:
                block = parts[1]
                self.goal_on_map[block] = 'table'
            # (clear B) goals implicitly mean nothing should be on B.
            # Absence from goal_on_top_map implies nothing should be on top.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Build current_on_map: block -> support (another block or 'table')
        current_on_map = {}
        # Build current_on_top_map: support (block) -> block (on top)
        current_on_top_map = {}
        # Track which block is being held, if any
        holding_block = None

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on' and len(parts) == 3:
                block, support = parts[1], parts[2]
                current_on_map[block] = support
                current_on_top_map[support] = block # Assuming only one block on top
            elif parts[0] == 'on-table' and len(parts) == 2:
                block = parts[1]
                current_on_map[block] = 'table'
            elif parts[0] == 'holding' and len(parts) == 2:
                 holding_block = parts[1]


        total_cost = 0

        # Component 1: Blocks not on their goal support
        # Iterate over blocks that have a defined goal position
        for block, goal_support in self.goal_on_map.items():
            current_support = current_on_map.get(block)

            # If the block is currently held, its support is effectively the arm,
            # which is never a goal support in standard blocksworld.
            if holding_block == block:
                 current_support = 'arm' # Use a placeholder not equal to any block or 'table'

            if current_support != goal_support:
                total_cost += 1

        # Component 2: Incorrect blocks on top of other blocks
        # Iterate over blocks that can potentially be supports (all blocks)
        for support_block in self.all_blocks:
             # Check if support_block is the block being held - nothing can be on top of it in this state
             if support_block == holding_block:
                 current_block_on_top = None
             else:
                 current_block_on_top = current_on_top_map.get(support_block)

             goal_block_on_top = self.goal_on_top_map.get(support_block)

             if current_block_on_top != goal_block_on_top:
                  total_cost += 1

        # The heuristic is 0 iff the state is the goal state.
        # If state is goal, all goal facts are true.
        # (on B A) goal true => current_on_map[B]=A, goal_on_map[B]=A. Match.
        # (on-table B) goal true => current_on_map[B]='table', goal_on_map[B]='table'. Match.
        # (holding B) goal true => holding_block = B. If goal_on_map[B] is defined, it won't match 'arm'. This heuristic assumes arm-empty goal.
        # Assuming (arm-empty) is always a goal unless a specific block needs to be held (unusual).
        # If (arm-empty) is a goal and arm is not empty, the held block is misplaced.
        # If state is goal, (arm-empty) is true, holding_block is None.
        # (on B A) goal true => current_on_top_map[A]=B, goal_on_top_map[A]=B. Match.
        # (clear A) goal true => goal_on_top_map[A]=None. If state is goal, (on C A) is false for all C, so current_on_top_map[A]=None. Match.
        # If state is goal, total_cost is 0.
        # If state is not goal, at least one goal fact is false, leading to a mismatch in either component 1 or 2, making cost > 0.

        return total_cost
