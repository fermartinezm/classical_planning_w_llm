from heuristics.heuristic_base import Heuristic

# Helper functions outside the class
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def build_goal_config(goal_facts):
    """Build a map from block to the block below it in the goal state, or 'table'."""
    goal_below = {}
    # Find blocks that should be on other blocks
    for fact in goal_facts:
        parts = get_parts(fact)
        if parts[0] == 'on':
            top, bottom = parts[1], parts[2]
            goal_below[top] = bottom
    # Find blocks that should be on the table
    for fact in goal_facts:
        parts = get_parts(fact)
        if parts[0] == 'on-table':
            block = parts[1]
            # A block cannot be both on another block and on the table in the goal
            if block not in goal_below:
                 goal_below[block] = 'table'
    return goal_below

def build_current_config(state_facts):
    """Build maps for current state: block to block below/table/holding, and block below to block above."""
    current_below = {}
    current_above = {}
    holding_block = None

    for fact in state_facts:
        parts = get_parts(fact)
        if parts[0] == 'on':
            top, bottom = parts[1], parts[2]
            current_below[top] = bottom
            current_above[bottom] = top
        elif parts[0] == 'on-table':
            block = parts[1]
            current_below[block] = 'table'
        elif parts[0] == 'holding':
            holding_block = parts[1]
            current_below[holding_block] = 'holding' # Represent holding state

    return current_below, current_above, holding_block

def count_blocks_on_top(block, current_above_map):
    """Count blocks stacked directly or indirectly on the given block."""
    count = 0
    current = block
    while current in current_above_map:
        count += 1
        current = current_above_map[current]
    return count

def get_all_blocks(state_facts, goal_facts):
    """Collect all unique block names from state and goal facts."""
    blocks = set()
    # Collect from state facts
    for fact_str in state_facts:
        parts = get_parts(fact_str)
        # Predicates like (on x y), (on-table x), (clear x), (holding x)
        if parts[0] in ['on', 'on-table', 'clear', 'holding']:
            blocks.update(parts[1:]) # Add all arguments as potential blocks
    # Collect from goal facts
    for fact_str in goal_facts:
         parts = get_parts(fact_str)
         # Predicates like (on x y), (on-table x), (clear x)
         if parts[0] in ['on', 'on-table', 'clear']:
             blocks.update(parts[1:]) # Add all arguments as potential blocks
         # (arm-empty) has no block arguments
    return list(blocks)


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goal
    configuration by counting blocks that are not in their correct position
    relative to the block below them in the goal stack, plus the cost to clear
    blocks stacked on top of them or on blocks that need to be clear.

    # Assumptions
    - Standard Blocksworld actions (pickup, putdown, stack, unstack) each cost 1.
    - Moving a block from one position to another typically requires at least
      2 actions (pickup/unstack + putdown/stack).
    - Clearing a block on top of another typically requires at least 2 actions
      (unstack + putdown/stack elsewhere).
    - The goal state is fully specified by 'on', 'on-table', 'clear', and 'arm-empty' predicates.

    # Heuristic Initialization
    - Parses the goal facts to build the desired stack configuration (`goal_below_map`).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. If the current state is the goal state, the heuristic is 0.
    2. Parse the current state to build the current stack configuration (`current_below_map`, `current_above_map`) and identify if a block is being held (`holding_block`).
    3. Collect all unique block names present in the current state and goal facts.
    4. Initialize the total estimated cost to 0.
    5. Identify blocks that are part of the goal stack configuration (those appearing as the top object in goal 'on' or 'on-table' predicates).
    6. Iterate through all blocks identified in step 5. For each block:
       - Determine its required position relative to the block below it in the goal (`goal_pos`).
       - Determine its current position relative to the block below it (`current_pos`).
       - If `current_pos` does not match `goal_pos`:
         - Add 2 to the total cost (representing the cost to move this block).
         - Add 2 times the number of blocks currently stacked on top of this block to the total cost (representing the cost to clear them out of the way).
    7. Iterate through all blocks identified in step 3. For each block:
       - Check if `(clear block)` is a goal predicate and `(clear block)` is not true in the current state.
       - If this condition holds AND the block was NOT already counted as being misplaced relative to the block below it in step 6:
         - Add 2 times the number of blocks currently stacked on top of this block to the total cost (representing the cost to clear them to satisfy the clear goal).
    8. Check if `(arm-empty)` is a goal predicate and the robot is currently holding a block. If this condition holds, add 1 to the total cost (representing the cost to put the block down).
    9. Return the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal configuration."""
        self.goals = task.goals
        # Build the goal stack configuration map: block -> block_below_it or 'table'
        self.goal_below_map = build_goal_config(self.goals)
        # All blocks are collected in __call__

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Check for goal state
        if state == self.goals:
            return 0

        # 2. Parse current state
        current_below_map, current_above_map, holding_block = build_current_config(state)

        # 3. Collect all unique block names
        all_blocks = get_all_blocks(state, self.goals)

        # 4. Initialize cost
        total_cost = 0

        # 5. Identify blocks in goal stacks
        goal_stack_blocks = set(self.goal_below_map.keys())

        # 6. Count cost for blocks misplaced relative to below
        misplaced_relative_to_below = set()
        for block in goal_stack_blocks:
            goal_pos = self.goal_below_map[block] # We know it's in the map
            current_pos = current_below_map.get(block)

            if current_pos != goal_pos:
                misplaced_relative_to_below.add(block)
                total_cost += 2 # Cost to move the block itself (pickup/unstack + putdown/stack)
                total_cost += 2 * count_blocks_on_top(block, current_above_map) # Cost to clear blocks on top

        # 7. Count cost for blocks that need to be clear but aren't (and weren't counted in step 6)
        for block in all_blocks:
            is_goal_clear = f'(clear {block})' in self.goals
            is_current_clear = f'(clear {block})' in state

            if is_goal_clear and not is_current_clear:
                 # Block should be clear but isn't. Blocks on top must be moved.
                 # Only add this cost if the block itself wasn't counted as misplaced relative to below.
                 if block not in misplaced_relative_to_below:
                     total_cost += 2 * count_blocks_on_top(block, current_above_map) # Cost to clear blocks on top

        # 8. Check for (arm-empty) goal
        is_goal_arm_empty = '(arm-empty)' in self.goals
        is_current_holding = holding_block is not None

        if is_goal_arm_empty and is_current_holding:
            total_cost += 1 # Cost to put down the held block

        # 9. Return total cost
        return total_cost
