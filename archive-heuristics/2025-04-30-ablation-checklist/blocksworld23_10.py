from fnmatch import fnmatch
# Assuming Heuristic base class is available from the environment
# from heuristics.heuristic_base import Heuristic

# Dummy Heuristic base class for testing purposes if the actual one isn't provided directly
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            pass


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# The heuristic doesn't strictly need the match function, but keeping it
# as it was provided in the example heuristics might be expected.
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    Estimates the number of actions needed to achieve the goal configuration.
    The heuristic counts the number of blocks that are not resting on the
    correct block or the table according to the goal state, and multiplies
    this count by 2.

    The logic is based on the observation that moving a block from one base
    (the block it's currently on, or the table) to another base (the block
    it should be on in the goal, or the table) typically requires at least
    two actions: one to pick it up (pickup or unstack) and one to put it
    down (putdown or stack).

    The heuristic identifies, for each block that has a specified goal position
    relative to another block or the table (i.e., appears in an 'on' or
    'on-table' goal fact), whether its current base matches its goal base.
    Each block whose current base does not match its goal base is counted as
    'misplaced' in terms of its support. The total heuristic value is twice
    this count.

    This heuristic is not admissible as it ignores potential costs for clearing
    blocks that are in the way, but it provides a reasonable estimate of the
    work required to reposition blocks and is efficiently computable. It aims
    to guide a greedy best-first search effectively by prioritizing states
    where more blocks are resting on their correct supports.

    # Heuristic Initialization
    - Parses goal facts to determine the required support (block or table) for
      each block that appears in an 'on' or 'on-table' goal fact.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the goal support (the block it should be on, or 'table') for
       every block that is part of an 'on' or 'on-table' goal fact. Store this
       mapping (e.g., in `self.goal_below`).
    2. For the current state, determine the current support for every block.
       This can be another block (from an 'on' fact), the 'table' (from an
       'on-table' fact), or the 'arm' (from a 'holding' fact). Store this
       mapping (e.g., in `current_config`).
    3. Initialize a counter for 'misplaced bases' to 0.
    4. Iterate through each block that has a defined goal support (from step 1).
    5. For each such block, check its current support (from step 2).
    6. If the block's current support is different from its goal support,
       increment the 'misplaced bases' counter. Blocks in the 'arm' are
       considered to have a different base than any 'on' or 'on-table' goal.
    7. The heuristic value is the 'misplaced bases' counter multiplied by 2.
       This represents the minimum two actions needed to correct the base
       for each block that is currently resting on the wrong thing.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal positions for each block.

        Args:
            task: The planning task object containing initial state, goals, etc.
        """
        # self.goals = task.goals # Storing goals is not strictly necessary after parsing
        self.goal_below = {} # Maps block -> block_below_in_goal or 'table'

        # Parse goal facts to build the goal_below map
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'on':
                # Goal is (on block below)
                block, below = parts[1], parts[2]
                self.goal_below[block] = below
            elif parts[0] == 'on-table':
                # Goal is (on-table block)
                block = parts[1]
                self.goal_below[block] = 'table'
            # Ignore 'clear' or 'arm-empty' goals for this heuristic as they
            # are typically consequences of achieving the 'on'/'on-table' goals.

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer estimate of the remaining cost to reach the goal.
        """
        state = node.state

        # Build current configuration map: block -> block_below or 'table' or 'arm'
        current_config = {}
        # Find which block is holding something, if any
        holding_block = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                block, below = parts[1], parts[2]
                current_config[block] = below
            elif parts[0] == 'on-table':
                block = parts[1]
                current_config[block] = 'table'
            elif parts[0] == 'holding':
                holding_block = parts[1]
                current_config[holding_block] = 'arm' # Mark the block as being in the arm
            # Ignore 'clear' or 'arm-empty' state facts for this heuristic

        misplaced_base_count = 0

        # Iterate through blocks that have a specific location goal (on or on-table)
        for block, goal_target in self.goal_below.items():
            # Check if the block's current base matches its goal base
            # A block must be in the current_config if it's in the goal_below map,
            # assuming valid states where blocks are always somewhere (on, on-table, or holding).
            current_target = current_config.get(block)

            # If the block is found in the current state configuration
            if current_target is not None:
                # If its current base is different from its goal base
                if current_target != goal_target:
                     misplaced_base_count += 1
            # If current_target is None, it means the block is in the goal_below map
            # but not found in the current state's 'on', 'on-table', or 'holding' facts.
            # This indicates an invalid state representation for Blocksworld, but
            # we handle it gracefully by not counting it.

        # Each block that is not on its correct base needs at least 2 actions:
        # pickup/unstack + putdown/stack.
        # This count represents the number of blocks whose immediate support
        # needs to be changed to match the goal configuration.
        heuristic_value = 2 * misplaced_base_count

        # The heuristic does not explicitly add cost for a busy arm or for
        # clearing blocks that are in the way. The factor of 2 per misplaced
        # base implicitly covers the pickup/unstack and putdown/stack actions.
        # While this might underestimate in some complex scenarios requiring
        # intermediate moves, it provides a simple, efficient, and reasonably
        # informative estimate for greedy search.

        return heuristic_value
