from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact string."""
    # Example: '(on b1 b2)' -> ['on', 'b1', 'b2']
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed by counting
    two types of discrepancies between the current state and the goal state:
    1. Blocks that are not on the correct immediate support (another block or the table)
       as specified in the goal configuration.
    2. Blocks that should be clear in the goal state but currently have other blocks on top of them.
    Each such discrepancy is assigned a cost of 2, representing a simplified estimate
    of the actions (e.g., unstack/pickup + putdown/stack) needed to resolve it.

    # Assumptions:
    - The goal state defines a specific configuration of blocks stacked on each other or on the table.
    - Any block not explicitly placed on another block in the goal is assumed to be clear in the goal. (This assumption is handled by explicitly checking 'clear' goals).
    - The heuristic counts symptoms of being away from the goal, where each symptom roughly corresponds to a block that needs to be moved or cleared.

    # Heuristic Initialization
    - Parses the goal state to build a map of desired immediate supports for each block (`goal_stack_map`).
    - Identifies the set of blocks that should be clear in the goal state (`goal_clear`).
    - Collects all unique block objects present in the initial and goal states.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is the goal state. If yes, return 0.
    2. Initialize heuristic value `h = 0`.
    3. Determine the current position (block below, table, or held) for every block in the state.
    4. Identify the set of blocks that are currently clear.
    5. For each block `B` that has a specific goal position defined in `goal_stack_map`:
       a. Get its current immediate support (`current_pos`).
       b. Get its goal immediate support (`goal_pos`).
       c. If `current_pos` is different from `goal_pos`, increment `h` by 2.
    6. For each block `B` that should be clear in the goal state (`B` is in `goal_clear`):
       a. Check if `B` is currently clear.
       b. If `B` is not currently clear, increment `h` by 2.
    7. Return the total count `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration and objects.
        """
        self.goals = task.goals # Store goal facts for quick lookup

        # Build goal configuration map: block -> block_below (or 'table')
        self.goal_stack_map = {}
        self.goal_clear = set()

        # Collect all unique block objects from initial and goal states
        self.all_blocks = set()

        # Process initial state facts to find all blocks
        for fact in task.initial_state:
             parts = get_parts(fact)
             # Add all arguments except the predicate name
             self.all_blocks.update(parts[1:])

        # Process goal state facts to build goal maps and find all blocks
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            predicate = parts[0]
            if predicate == 'on':
                block, under = parts[1], parts[2]
                self.goal_stack_map[block] = under
                self.all_blocks.add(block)
                self.all_blocks.add(under)
            elif predicate == 'on-table':
                block = parts[1]
                self.goal_stack_map[block] = 'table'
                self.all_blocks.add(block)
            elif predicate == 'clear':
                block = parts[1]
                self.goal_clear.add(block)
            # Ignore 'arm-empty' goals for these structures

        self.all_blocks.discard('table') # 'table' is not a block object


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
             return 0

        # Map current positions and identify clear blocks
        current_pos_map = {} # block -> block_below (or 'table' or 'hand')
        currently_clear = set()

        # Build maps from the current state
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'on':
                block, under = parts[1], parts[2]
                current_pos_map[block] = under
            elif predicate == 'on-table':
                block = parts[1]
                current_pos_map[block] = 'table'
            elif predicate == 'holding':
                block = parts[1]
                current_pos_map[block] = 'hand'
            elif predicate == 'clear':
                block = parts[1]
                currently_clear.add(block)
            # Ignore 'arm-empty'

        h = 0

        # Count blocks not on their correct goal support
        # Iterate through blocks that have a specific goal position
        for block, goal_pos in self.goal_stack_map.items():
            current_pos = current_pos_map.get(block) # Get current position

            # If a block is in the goal_stack_map, it must be in the state
            # (either on something, on table, or held).
            # If it's not found in current_pos_map, the state representation is unusual,
            # but we'll treat it as misplaced.
            if current_pos is None or current_pos != goal_pos:
                 h += 2 # Cost of moving the block

        # Count blocks that should be clear but aren't
        for block in self.goal_clear:
            if block not in currently_clear:
                # Block should be clear but is blocked.
                # Cost of moving the block on top
                h += 2

        return h
