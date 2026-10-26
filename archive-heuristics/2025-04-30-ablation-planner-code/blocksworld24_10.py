from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goal
    configuration of blocks by counting how many blocks are not on their correct
    goal support (either another block or the table) and multiplying by two.
    It is non-admissible and designed to guide a greedy best-first search.

    # Assumptions
    - The goal specifies the desired support for each block using `on` or `on-table` predicates.
    - Every block has a defined goal support in the goal state.
    - The primary cost comes from moving blocks that are not on their correct support.
    - Each such block requires at least two actions (pickup/unstack and stack/putdown)
      to be moved towards its correct position.
    - Achieving `clear` and `arm-empty` goals is implicitly handled by correctly
      positioning blocks.

    # Heuristic Initialization
    - Extracts the goal configuration for each block from the task's goal predicates.
      A dictionary `goal_support` is created where `goal_support[block]` is the
      block it should be `on`, or the string 'table' if it should be `on-table`.
    - The set of all objects (blocks) is also stored.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Parse the current state to determine the current support for each block.
       - Iterate through the state facts.
       - If `(on X Y)` is in the state, the current support for block X is Y.
       - If `(on-table X)` is in the state, the current support for block X is 'table'.
       - If `(holding X)` is in the state, the current support for block X is 'arm'.
       - A dictionary `current_support` is built mapping each block to its current support.

    2. Initialize a counter `misplaced_count` to zero.

    3. Iterate through all blocks identified during initialization.

    4. For each block, compare its current support (`current_support[block]`)
       with its goal support (`goal_support[block]`).

    5. If the current support does not match the goal support, increment `misplaced_count`.
       Blocks being held (`'arm'`) are always considered misplaced relative to a
       standard blocksworld goal.

    6. The heuristic value is `2 * misplaced_count`. This estimates that each
       misplaced block requires at least two actions (one to pick it up/unstack
       it from the wrong place, and one to put it down/stack it in the correct place)
       to contribute to achieving the goal configuration.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configurations.

        Args:
            task: The planning task object containing goals and objects.
        """
        self.goals = task.goals
        self.objects = task.objects  # Get the list of all objects (blocks)

        # Build the goal support map: block -> support (block or 'table')
        self.goal_support = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts:
                continue # Skip empty or malformed goal facts
            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, support = parts[1], parts[2]
                self.goal_support[block] = support
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                self.goal_support[block] = 'table'
            # Note: We ignore (clear X) and (arm-empty) goals for the support map,
            # as they don't define a block's physical support.

        # Ensure every block has an entry in goal_support, even if its goal is
        # only specified by (clear X) or if it's not mentioned in the goal.
        # For blocks not mentioned in 'on' or 'on-table' goals, their goal support
        # is implicitly 'unknown' or 'irrelevant' for this heuristic.
        # However, standard blocksworld problems usually specify a location for all blocks.
        # We'll assume all blocks in task.objects are potentially relevant and
        # might appear in goal_support. If a block isn't in goal_support,
        # it won't contribute to the misplaced count based on support.

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer estimate of the remaining cost to reach the goal.
        """
        state = node.state

        # Build the current support map: block -> support (block, 'table', or 'arm')
        current_support = {}
        holding_block = None

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue # Skip empty or malformed state facts
            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, support = parts[1], parts[2]
                current_support[block] = support
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                current_support[block] = 'table'
            elif predicate == "holding" and len(parts) == 2:
                holding_block = parts[1]
                current_support[holding_block] = 'arm' # Special support 'arm'

        misplaced_count = 0

        # Iterate through all blocks to check their support
        for obj in self.objects:
            # If a block is not in the current_support map, it's an inconsistency
            # or it's the block being held. We handle 'holding' explicitly.
            # If a block is not in goal_support, it means its goal location
            # wasn't specified by 'on' or 'on-table'. We assume such blocks
            # don't contribute to the 'misplaced support' count.
            # This heuristic focuses only on blocks with explicit goal supports.

            if obj in self.goal_support:
                goal_sup = self.goal_support[obj]
                curr_sup = current_support.get(obj) # Use .get() to handle blocks not found (e.g., held)

                # If the block is being held, its current support is 'arm'.
                # A block being held is never on its goal support (unless the goal is holding, which is not standard).
                # So, if holding_block is obj, it's misplaced w.r.t. its goal support.
                if holding_block == obj:
                     # If the block is held, its current support is 'arm'.
                     # If the goal support is not 'arm' (standard case), it's misplaced.
                     # We assume goal support is never 'arm'.
                     misplaced_count += 1
                elif curr_sup is not None and curr_sup != goal_sup:
                    # Block is not held, is found in state, and its support is wrong
                    misplaced_count += 1
                elif curr_sup is None and holding_block != obj:
                     # Block is not held, but not found in on/on-table facts.
                     # This indicates an inconsistent state representation if it's not the held block.
                     # For robustness, we might treat this as misplaced if it has a goal support.
                     # However, assuming valid states, this case shouldn't happen for blocks with goal_support.
                     pass # Assuming valid state representation

        # The heuristic is twice the number of blocks not on their goal support.
        # This is a simple estimate of the minimum moves needed.
        # Each misplaced block needs to be picked up/unstacked (1 action)
        # and then stacked/putdown (1 action).
        return 2 * misplaced_count

