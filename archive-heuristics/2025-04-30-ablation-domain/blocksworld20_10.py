from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact string defensively
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state.
    It counts the number of blocks that are not on their correct base (either another block or the table)
    according to the goal state, multiplying this count by 2 (estimating pick/unstack + place actions).
    It also adds 1 for each block that should be clear in the goal state but is not clear in the current state,
    representing the need to move the block currently on top.

    # Assumptions:
    - Standard Blocksworld actions (pick-up, put-down, stack, unstack).
    - Each block has a unique goal position (either on another block or on the table).
    - The cost of each action is 1.

    # Heuristic Initialization
    - Parses the goal conditions to determine the desired base for each block (`goal_below`)
      and which blocks should be clear (`goal_clear`).

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Parse the current state:
       - Determine the current base for each block (`current_below`). This can be another block, the table, or the arm.
       - Identify which blocks are currently clear (`current_clear`).
       - Identify which block, if any, is currently held by the arm.

    2. Calculate the first component of the heuristic (misplaced bases):
       - Iterate through each block that has a specified goal base in `goal_below`.
       - For each such block, check if its current base in `current_below` matches its goal base.
       - Count the number of blocks where the current base does *not* match the goal base.
       - Multiply this count by 2. This estimates the cost of picking up/unstacking and placing each block that is on the wrong base.

    3. Calculate the second component of the heuristic (blocked goal-clear blocks):
       - Iterate through each block that should be clear according to `goal_clear`.
       - For each such block, check if it is *not* clear in the current state (i.e., it's not in `current_clear`).
       - Count the number of such blocks. This estimates the cost of moving the block that is currently on top of a block that should be clear.

    4. Sum the components:
       - The total heuristic value is the sum of the two components calculated in steps 2 and 3.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals

        # Build mapping from block to its goal base (block below it, or 'table')
        self.goal_below = {}
        # Build set of blocks that should be clear in the goal
        self.goal_clear = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, base = parts[1], parts[2]
                self.goal_below[block] = base
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                self.goal_below[block] = 'table'
            elif predicate == "clear" and len(parts) == 2:
                block = parts[1]
                self.goal_clear.add(block)
            # Ignore other goal predicates if any

        # Static facts are not used in this heuristic.
        # static_facts = task.static


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse current state
        current_below = {}
        current_clear = set()
        # current_holding = None # Not strictly needed for this heuristic logic

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, base = parts[1], parts[2]
                current_below[block] = base
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                current_below[block] = 'table'
            elif predicate == "clear" and len(parts) == 2:
                block = parts[1]
                current_clear.add(block)
            elif predicate == "holding" and len(parts) == 2:
                block = parts[1]
                current_below[block] = 'arm'
                # current_holding = block # Not strictly needed
            # Ignore arm-empty and other facts

        # Heuristic component 1: Blocks not on their correct base
        misplaced_base_count = 0
        for block, goal_base in self.goal_below.items():
            # If a block is not in current_below, it must be held by the arm.
            # Use .get() with 'arm' as default to handle held blocks.
            current_base = current_below.get(block, 'arm')
            if current_base != goal_base:
                misplaced_base_count += 1

        h1 = 2 * misplaced_base_count

        # Heuristic component 2: Blocks blocking a goal-clear block
        blocked_goal_clear_count = 0
        for block in self.goal_clear:
            if block not in current_clear:
                 # This block should be clear but isn't. Something is on top.
                 # We add 1 action to move the block on top.
                 blocked_goal_clear_count += 1

        h2 = blocked_goal_clear_count

        # Total heuristic is the sum of the components
        total_cost = h1 + h2

        return total_cost
