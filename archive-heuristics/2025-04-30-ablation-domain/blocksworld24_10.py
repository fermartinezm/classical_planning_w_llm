from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Use the match function from the example for consistency, although simple string comparison is enough for this heuristic
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    by counting the number of blocks that are not in their correct position
    within the goal stacks, plus a penalty if the arm is holding a block
    and needs to be empty. A block is considered "correctly stacked" if it is
    on the correct block (or table) according to the goal, and the block below
    it is also correctly stacked (recursively down to the table).

    # Assumptions
    - The goal specifies a configuration of blocks using 'on' and 'on-table' predicates.
    - The goal may also require the arm to be empty.
    - Blocks not mentioned in 'on' or 'on-table' goal facts do not have a specific goal location (they can be anywhere, typically on the table).
    - The heuristic counts blocks that are not part of a correctly built stack suffix from the bottom up.

    # Heuristic Initialization
    - Parses the goal state to identify the desired 'on' relationships and blocks that should be 'on-table'.
    - Identifies all blocks that are part of the goal configuration ('goal_blocks').
    - Checks if '(arm-empty)' is explicitly a goal condition.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Parse the current state to determine the current 'on' relationships,
       blocks 'on-table', and whether the arm is holding a block.
    2. Identify the set of blocks that are "correctly stacked" in the current state.
       A block B is correctly stacked if:
       - The goal requires B to be on the table (on-table B), AND the state has (on-table B).
       - The goal requires B to be on block Y (on B Y), AND the state has (on B Y), AND block Y is already correctly stacked.
       This set is computed iteratively, starting with correctly placed blocks on the table
       and propagating upwards through the goal stacks.
    3. The base heuristic value is the number of blocks in the goal configuration
       ('goal_blocks') that are *not* in the set of correctly stacked blocks.
       This counts how many blocks are misplaced relative to their goal stack position
       or are part of a correct stack segment that is not grounded on a correctly
       stacked block below it.
    4. If the goal requires the arm to be empty ('arm-empty' is a goal fact)
       and the arm is currently holding a block, add 1 to the heuristic value.
       This accounts for the action needed to put down the held block.
    5. The total heuristic value is the sum from steps 3 and 4.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal information.

        Args:
            task: The planning task object containing initial state, goals, etc.
        """
        self.goals = task.goals

        # Parse goal facts to build goal structure and identify goal blocks
        self.goal_on = {}  # Map: block_above -> block_below
        self.goal_on_table = set() # Set of blocks that should be on the table
        self.goal_blocks = set()   # Set of all blocks mentioned in goal 'on' or 'on-table' facts
        self.goal_arm_empty = False

        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == 'on':
                block_above, block_below = parts[1], parts[2]
                self.goal_on[block_above] = block_below
                self.goal_blocks.add(block_above)
                self.goal_blocks.add(block_below)
            elif predicate == 'on-table':
                block = parts[1]
                self.goal_on_table.add(block)
                self.goal_blocks.add(block)
            elif predicate == 'arm-empty':
                self.goal_arm_empty = True
            # Ignore 'clear' goals for this heuristic calculation

        # Note: We don't need static facts for this heuristic in blocksworld.

    def __call__(self, node):
        """
        Compute the heuristic estimate for the given state.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer representing the estimated cost to reach the goal.
        """
        state = node.state

        # Parse current state facts
        current_on = {}  # Map: block_above -> block_below
        current_on_table = set() # Set of blocks currently on the table
        is_holding = False

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'on':
                block_above, block_below = parts[1], parts[2]
                current_on[block_above] = block_below
            elif predicate == 'on-table':
                block = parts[1]
                current_on_table.add(block)
            elif predicate == 'holding':
                is_holding = True
            # Ignore 'clear' and 'arm-empty' for stack structure parsing

        # --- Compute correctly_stacked blocks ---
        # A block is correctly stacked if it's in its goal position relative
        # to the block below it (or table) AND the block below it is also
        # correctly stacked. The table is always considered correctly stacked.

        correctly_stacked = set()

        # Start with blocks that should be on the table in the goal
        for block in self.goal_on_table:
            if block in current_on_table:
                correctly_stacked.add(block)

        # Iteratively add blocks that are correctly stacked on top of
        # already correctly stacked blocks according to the goal structure
        while True:
            newly_stacked = set()
            # Check blocks that should be on another block according to the goal
            for block_above, block_below_goal in self.goal_on.items():
                # If the block below is already correctly stacked
                if block_below_goal in correctly_stacked:
                    # Check if the block above is currently on the correct block below
                    if current_on.get(block_above) == block_below_goal:
                         # This block_above is now correctly stacked
                         newly_stacked.add(block_above)

            # Remove blocks that were already counted as correctly stacked
            newly_stacked -= correctly_stacked

            if not newly_stacked:
                break # No new blocks were added in this iteration

            correctly_stacked.update(newly_stacked)

        # --- Calculate heuristic value ---

        # Count blocks in the goal configuration that are NOT correctly stacked
        # These are blocks that need to be moved or have blocks below them that need fixing
        h = len(self.goal_blocks) - len(correctly_stacked)

        # Add a penalty if the arm is holding a block and the goal requires it to be empty
        if self.goal_arm_empty and is_holding:
            h += 1

        return h
