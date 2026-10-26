from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if running standalone for testing
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.task = task
            self.goals = task.goals
            self.static = task.static

        def __call__(self, node):
            raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or non-fact string gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we have the same number of parts as args for a meaningful match
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required by counting the number
    of blocks that are not currently on their correct goal base (either another
    block or the table) and multiplying this count by 2. This is based on the
    idea that moving a block to its correct base typically requires at least two
    actions (pickup/unstack and putdown/stack), assuming preconditions are met.

    # Assumptions
    - The goal specifies the desired 'on' and 'on-table' relationships for a subset
      of blocks. Blocks not mentioned in these goal predicates are "don't care"
      regarding their final position, unless they are blocking a goal-related block.
      This heuristic primarily focuses on the blocks explicitly mentioned in goal
      'on' or 'on-table' predicates.
    - Standard Blocksworld actions (pickup, putdown, stack, unstack) are used.
    - The cost of each action is 1.

    # Heuristic Initialization
    - The heuristic extracts the desired base for each block that appears in a goal
      '(on ?x ?y)' or '(on-table ?x)' predicate. This mapping from block to its
      goal base is stored.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify all blocks that are explicitly mentioned in the goal predicates
       '(on ?x ?y)' or '(on-table ?x)'. These are the blocks whose final position
       is specified.
    2. For each identified block, determine its *goal base*:
       - If '(on BlockX BlockY)' is a goal predicate, the goal base for BlockX is BlockY.
       - If '(on-table BlockZ)' is a goal predicate, the goal base for BlockZ is 'table'.
       Store these goal bases in a map (e.g., `goal_base_map`).
    3. For each identified block, determine its *current base* in the given state:
       - If '(on BlockX BlockY)' is true in the state, the current base for BlockX is BlockY.
       - If '(on-table BlockX)' is true in the state, the current base for BlockX is 'table'.
       - If '(holding BlockX)' is true in the state, the current base for BlockX is 'arm'.
       Store these current bases in a map (e.g., `current_base_map`).
    4. Initialize a counter `misplaced_count` to 0.
    5. Iterate through the set of blocks identified in step 1 (those with a specified goal base).
    6. For each block in this set, compare its current base (`current_base_map`) with its goal base (`goal_base_map`).
    7. If the current base does *not* match the goal base, increment `misplaced_count`.
    8. The heuristic value is the final count multiplied by 2. This estimates that each block not on its correct base requires at least two actions (one to pick it up/unstack it, and one to put it down/stack it on the correct base), ignoring the cost of clearing blocks that might be in the way.

    The heuristic is 0 if and only if all blocks with a specified goal base are currently on that base, which is a necessary condition for the goal state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the goal base for each block
        mentioned in the goal predicates.
        """
        super().__init__(task)

        # Map block to its goal base (block name or 'table')
        self.goal_base_map = {}
        # Set of blocks explicitly mentioned in goal 'on' or 'on-table' predicates
        self.goal_blocks = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: # Skip malformed facts
                continue
            predicate = parts[0]

            if predicate == "on" and len(parts) == 3:
                block, base = parts[1], parts[2]
                self.goal_base_map[block] = base
                self.goal_blocks.add(block)
                self.goal_blocks.add(base) # The base block is also relevant

            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                self.goal_base_map[block] = 'table'
                self.goal_blocks.add(block)

        # Note: 'clear' and 'arm-empty' goal predicates are not directly used
        # to determine goal base, as they describe the state of the block/arm,
        # not its physical support. The heuristic focuses on the structural
        # 'on' and 'on-table' goals.

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Determine the current base for each block in the state
        current_base_map = {}

        # First pass to find 'holding' and 'on' relations
        for fact in state:
            parts = get_parts(fact)
            if not parts: # Skip malformed facts
                continue
            predicate = parts[0]

            if predicate == "holding" and len(parts) == 2:
                held_block = parts[1]
                current_base_map[held_block] = 'arm'

            elif predicate == "on" and len(parts) == 3:
                block, base = parts[1], parts[2]
                current_base_map[block] = base

        # Second pass to find 'on-table' for blocks not held or on another block
        # We only care about blocks that are in the goal_blocks set
        for block in self.goal_blocks:
             # If we already found its base (on another block or held), skip
            if block in current_base_map:
                continue

            # Check if it's on the table
            if f"(on-table {block})" in state:
                 current_base_map[block] = 'table'
            # else: The block's location is unknown or it's a 'don't care' block
            # not currently on a block or table (e.g. in transit, but holding covers that)
            # For blocks in goal_blocks, they must be on something or held in a valid state.
            # If a block is in goal_blocks but not in current_base_map after this,
            # it implies it's not on table, not on another block, and not held.
            # This shouldn't happen in valid blocksworld states for blocks that exist.
            # We can treat its current_base as None or some indicator that it's not
            # where it should be. Let's default to None if not found.
            # If a block is in goal_blocks but not in current_base_map, it's definitely misplaced.
            pass # current_base_map[block] will remain undefined if not found


        # Calculate the number of misplaced blocks
        misplaced_count = 0
        for block in self.goal_base_map: # Iterate through blocks whose goal base is specified
            goal_base = self.goal_base_map[block]
            current_base = current_base_map.get(block) # Use .get to handle blocks not found in state facts

            # If the block is not found in the state at all, it's misplaced relative to its goal
            if current_base is None:
                 misplaced_count += 1
            # If the block is found, check if its base matches the goal base
            elif current_base != goal_base:
                misplaced_count += 1

        # The heuristic is 2 * the number of blocks not on their correct goal base.
        # This is a non-admissible estimate of the actions needed.
        return misplaced_count * 2

