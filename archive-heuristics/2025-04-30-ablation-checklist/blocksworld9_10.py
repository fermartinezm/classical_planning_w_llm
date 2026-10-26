from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class BlocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    This heuristic estimates the cost based on the number of blocks that are not
    in their correct goal position relative to the goal structure. A block is
    considered "in place" structurally if it is on the table and the goal
    requires it to be on the table, OR if it is on block Y and the goal requires
    it to be on block Y, AND block Y is also considered "in place" structurally.

    The heuristic value is calculated as 2 times the number of blocks that are
    NOT in place structurally. This is because moving a block typically requires
    at least two actions: one to get it into the arm (pickup or unstack) and
    one to get it out of the arm (putdown or stack). Blocks that are on top of
    misplaced blocks are also implicitly counted as not in place structurally,
    as their support is incorrect relative to the goal, which aligns with the
    need to move them first.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the goal configuration and
        identifying all blocks involved in the problem.
        """
        self.goals = task.goals
        self.initial_state = task.initial_state

        # Build goal configuration: map block -> block_below (or 'table')
        # We only care about 'on' and 'on-table' goals for structural placement.
        self.goal_config = {}
        # Collect all blocks mentioned in goals or initial state
        self.all_blocks = set()

        # Parse goals to build goal_config and collect blocks
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block, support = parts[1], parts[2]
                self.goal_config[block] = support
                self.all_blocks.add(block)
                self.all_blocks.add(support)
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                self.goal_config[block] = 'table'
                self.all_blocks.add(block)
            # 'clear' and 'arm-empty' goals are handled implicitly by the structural count

        # Add blocks from initial state that might not be explicitly in goals
        for fact in self.initial_state:
             parts = get_parts(fact)
             if not parts: continue
             predicate = parts[0]
             if predicate in ['on', 'on-table', 'clear', 'holding'] and len(parts) > 1:
                 # For 'on', parts[1] and parts[2] are blocks
                 if predicate == 'on' and len(parts) == 3:
                     self.all_blocks.add(parts[1])
                     self.all_blocks.add(parts[2])
                 # For 'on-table', 'clear', 'holding', parts[1] is the block
                 elif predicate in ['on-table', 'clear', 'holding'] and len(parts) == 2:
                     self.all_blocks.add(parts[1])

        # Ensure all blocks mentioned in goal_config are in all_blocks
        self.all_blocks.update(self.goal_config.keys())
        self.all_blocks.update(self.goal_config.values()) # Add blocks that are supports in goals

        # Remove 'table' if it was mistakenly added as a block name
        self.all_blocks.discard('table')


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Build current configuration: map block -> block_below (or 'table' or 'arm')
        current_config = {}
        # Initialize all blocks to have unknown support
        for block in self.all_blocks:
            current_config[block] = 'unknown'

        # Populate current_config based on state facts
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block, support = parts[1], parts[2]
                current_config[block] = support
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                current_config[block] = 'table'
            elif predicate == 'holding' and len(parts) == 2:
                block = parts[1]
                current_config[block] = 'arm' # Represent block in arm

        # Identify blocks that are in their correct goal position relative to the goal structure
        # A block B is in_place if:
        # 1. Its goal is (on-table B) AND state has (on-table B)
        # 2. Its goal is (on B Y) AND state has (on B Y) AND Y is in_place
        in_place = set()
        changed = True

        # Add blocks that are correctly on the table according to the goal
        for block in self.all_blocks:
             goal_support = self.goal_config.get(block)
             current_support = current_config.get(block)
             if goal_support == 'table' and current_support == 'table':
                 in_place.add(block)

        # Iteratively add blocks that are correctly stacked on blocks already in_place
        # This loop continues as long as new blocks are added to the in_place set
        while changed:
            changed = False
            # Iterate over blocks that are not yet in_place
            # Create a list copy to allow modification of the set during iteration
            for block in list(self.all_blocks - in_place):
                goal_support = self.goal_config.get(block)
                current_support = current_config.get(block)

                # Check if the block has a specified goal position that is not the table
                if goal_support is not None and goal_support != 'table':
                    # Check if the block is currently on its goal support
                    if current_support == goal_support:
                        # Check if the goal support block is already in_place
                        if goal_support in in_place:
                            in_place.add(block)
                            changed = True

        # The heuristic is 2 * the number of blocks that are NOT in_place structurally.
        # This counts every block that is either in the wrong location or is sitting
        # on top of a block that is in the wrong location (and thus needs to be handled).
        # The factor of 2 is a rough estimate of actions per block move.
        heuristic_value = 2 * (len(self.all_blocks) - len(in_place))

        # Note: This heuristic focuses on the structural goal (on/on-table).
        # It implicitly handles 'clear' goals to some extent because a block
        # that needs to be cleared but isn't clear implies something is on it,
        # and that something is likely not in its goal structural position
        # (unless it's the top block of a goal tower that is itself misplaced).
        # For greedy best-first search, this structural focus is often effective.

        return heuristic_value

