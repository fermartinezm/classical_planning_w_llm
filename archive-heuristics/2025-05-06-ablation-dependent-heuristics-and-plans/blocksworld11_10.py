from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and handle potential leading/trailing whitespace
    fact_str = str(fact).strip()
    if not fact_str.startswith('(') or not fact_str.endswith(')'):
         # Handle known predicates without parameters like 'arm-empty'
         if fact_str == 'arm-empty':
             return ['arm-empty']
         # For other unexpected formats, return empty list
         return []

    return fact_str[1:-1].split()

# Helper function to get stack sequence down to table or arm
def get_current_stack_sequence(block, current_below, current_holding, visited=None):
    """
    Recursively get the sequence of blocks from 'block' down to the table or arm.
    Returns a list like [block, below1, below2, ..., 'table'] or [block, 'arm'].
    Includes cycle detection for robustness against malformed states.
    """
    if visited is None:
        visited = set()
    if block in visited:
        # Cycle detected - indicates a malformed state
        return [block, 'cycle_detected']
    visited.add(block)

    if current_holding == block:
        return [block, 'arm']
    if block not in current_below:
        # Block is not held, not on table, not on another block. Should not happen in valid states.
        # Treat as detached/unknown base for heuristic calculation.
        return [block, 'unknown_base']

    below = current_below[block]
    if below == 'table':
        return [block, 'table']
    else:
        # Recursive call
        return [block] + get_current_stack_sequence(below, current_below, current_holding, visited)


# Helper function to get goal stack sequence down to table
def get_goal_stack_sequence(block, goal_below, visited=None):
    """
    Recursively get the sequence of blocks from 'block' down to the table in the goal.
    Returns a list like [block, goal_below1, goal_below2, ..., 'table'].
    Includes cycle detection for robustness against malformed goals.
    """
    if visited is None:
        visited = set()
    if block in visited:
        # Cycle detected in goal definition - indicates a malformed goal
        return [block, 'cycle_detected_goal']
    visited.add(block)

    if block not in goal_below:
        # This block is not part of a defined goal stack relative to below.
        # It might be a block that just needs to be clear or on the table without
        # anything specific on it. For this heuristic, we only care about blocks
        # explicitly mentioned as being ON or ON-TABLE in the goals.
        # If called for such a block, return a sequence indicating it's not part of a stack.
        return [block, 'not_in_goal_stack']

    below = goal_below[block]
    if below == 'table':
        return [block, 'table']
    else:
        # Recursive call
        return [block] + get_goal_stack_sequence(below, goal_below, visited)


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not in their correct
    position relative to the stack below them, down to the table, as defined by
    the goal state. It counts how many blocks' current stack sequence (down to
    the table or the arm) differs from their desired goal stack sequence (down
    to the table). A block being held is considered not in its final position.

    # Assumptions
    - The goal state defines specific stacks of blocks on the table using
      `(on X Y)` and `(on-table Z)` predicates.
    - Every block in a valid state is either on the table, on another block, or held by the arm.
    - The state and goal representations are consistent (no cycles in stacks).

    # Heuristic Initialization
    - Parses the goal predicates to build a mapping `goal_below` which indicates,
      for each block X, what block Y should be directly below it (`(on X Y)`)
      or if it should be on the table (`(on-table X)`).
    - Identifies all blocks that are explicitly mentioned as being *on* something
      or *on-table* in the goal. These are the blocks whose stack sequence we will check.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current state facts.
    2. Build a mapping `current_below` from the state facts, similar to `goal_below`,
       indicating what is directly below each block currently on the table or another block.
    3. Identify the block currently held by the arm, if any.
    4. Initialize the heuristic cost to 0.
    5. Iterate through each block that is explicitly mentioned as being *on* something
       or *on-table* in the goal (identified during initialization as keys in `goal_below`).
    6. For each such block, determine its current stack sequence down to the table or arm
       using the `current_below` mapping and the held block information.
    7. For the same block, determine its desired goal stack sequence down to the table
       using the `goal_below` mapping.
    8. If the current stack sequence does not match the goal stack sequence for this block,
       increment the heuristic cost by 1.
    9. The total heuristic value is the sum of these counts for all blocks whose
       desired position relative to below is defined in the goal.
       This counts how many blocks are "misplaced" relative to their desired final stack structure.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal stack information.
        """
        self.goals = task.goals  # Goal conditions.

        # Build the goal_below mapping: block -> block_below or 'table'
        self.goal_below = {}
        # Keep track of all blocks that are keys in goal_below (i.e., blocks
        # whose desired position relative to below is explicitly defined).
        self.goal_blocks_to_check = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, below_block = parts[1], parts[2]
                self.goal_below[block] = below_block
                self.goal_blocks_to_check.add(block)
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                self.goal_below[block] = 'table'
                self.goal_blocks_to_check.add(block)
            # Ignore 'clear' goals for building stack structure relevant to this heuristic

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Build current_below mapping and find held block
        current_below = {}
        current_holding = None

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, below_block = parts[1], parts[2]
                current_below[block] = below_block
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                current_below[block] = 'table'
            elif predicate == "holding" and len(parts) == 2:
                current_holding = parts[1]
            # Ignore 'clear' and 'arm-empty' for building stack structure

        total_cost = 0  # Initialize action cost counter.

        # Count blocks whose current stack sequence doesn't match the goal sequence
        for block in self.goal_blocks_to_check:
            current_seq = get_current_stack_sequence(block, current_below, current_holding)
            goal_seq = get_goal_stack_sequence(block, self.goal_below)

            if current_seq != goal_seq:
                total_cost += 1

        # The heuristic is 0 iff the goal is reached based on the sequence comparison logic.
        # If the goal state is reached, all (on X Y) and (on-table X) goal facts
        # for blocks in goal_blocks_to_check must be true, and no block in goal_blocks_to_check
        # can be held. This implies current_seq == goal_seq for all such blocks, resulting in 0.
        # Conversely, if total_cost is 0, it implies current_seq == goal_seq for all
        # blocks in goal_blocks_to_check. This means their positions relative to below
        # match the goal, and they are not held. This satisfies the ON/ON-TABLE goals
        # for these blocks. While this heuristic doesn't explicitly check CLEAR goals,
        # a correct stack configuration usually implies the necessary blocks are clear.
        # This heuristic aims to minimize expanded nodes in greedy search, not strict admissibility.

        return total_cost
