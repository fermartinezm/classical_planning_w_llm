# from heuristics.heuristic_base import Heuristic # Assuming this is provided by the environment

from fnmatch import fnmatch

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
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

# Heuristic class definition
# Inherit from Heuristic in the actual environment
# class blocksworldHeuristic(Heuristic):
class blocksworldHeuristic:
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not in their correct final position
    within their goal stack, considering both the block below and the block above.
    A block is considered "correctly placed in its stack" if it is on its correct goal support,
    has the correct block (or nothing) immediately on top according to the goal, AND
    the block below it (if any) is also correctly placed in its stack, AND the block above it
    (if any) is also correctly placed in its stack. The heuristic counts the number of blocks
    for which this recursive condition is false.

    # Assumptions
    - The goal state defines a specific configuration of stacks of blocks on the table.
    - Every block mentioned in the goal is part of a goal stack (either on another block or on the table).
    - All objects listed in the problem instance are relevant to the goal configuration.
    - The state representation is consistent (e.g., a block is either on something, on the table, or held).

    # Heuristic Initialization
    - Parses the goal facts (`task.goals`) to build maps representing the desired block configuration:
      - `goal_support_map[block]` stores the block that `block` should be on top of (or 'table').
      - `goal_on_top_map[block]` stores the block that should be directly on top of `block` (or None if `block` should be clear).
    - Extracts the set of all blocks from the task objects (`task.objects`).

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the goal facts to create `goal_support_map` and `goal_on_top_map`.
       - Initialize `goal_on_top_map` with None for all blocks.
       - For each goal fact `(on X Y)`, set `goal_support_map[X] = Y` and `goal_on_top_map[Y] = X`.
       - For each goal fact `(on-table Z)`, set `goal_support_map[Z] = 'table'`.
       - Goal facts like `(clear W)` are implicitly handled by the initial None value in `goal_on_top_map`.
    2. Parse the current state facts (`node.state`) to create `current_support_map` and `current_on_top_map`.
       - Initialize `current_support_map` and `current_on_top_map` with None for all blocks.
       - For each state fact `(on X Y)`, set `current_support_map[X] = Y` and `current_on_top_map[Y] = X`.
       - For each state fact `(on-table Z)`, set `current_support_map[Z] = 'table'`.
       - For each state fact `(holding H)`, set `current_support_map[H] = 'arm'`.
       - After parsing, any block in `self.objects` not found in `current_support_map` is assumed to be on the table.
    3. Define a recursive helper function `is_correctly_placed_in_stack(block, current_support_map, current_on_top_map, goal_support_map, goal_on_top_map, memo)`.
       - Base cases: If `block` is None, 'table', or 'arm', return True. If `block` is not a recognized object, return True.
       - Memoization: If result for `block` is in `memo`, return it.
       - Get the current and goal support and the blocks currently and goal-wise on top.
       - Check if the current support matches the goal support for `block`. If not, return False.
       - Check if the block currently on top matches the block that should be on top according to the goal. If not, return False.
       - If both immediate support and block on top match, recursively check the block below (`goal_support_map[block]`) and the block above (`goal_on_top_map[block]`). The block is correctly placed in the stack if its immediate configuration is correct AND the parts of the stack it connects to (below and above) are also correct.
       - Store the result in `memo` before returning.
    4. Initialize the heuristic value to 0.
    5. Iterate through all blocks in the problem instance (`self.objects`).
    6. For each block, call `is_correctly_placed_in_stack`. If it returns False, increment the heuristic value.
    7. Return the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal configuration and objects."""
        self.goals = task.goals
        # Assuming task object has an 'objects' attribute which is a frozenset of object names
        self.objects = task.objects

        # Data structures to store the goal configuration
        self.goal_support_map = {}
        # Initialize all blocks should be clear in the goal unless specified otherwise
        self.goal_on_top_map = {obj: None for obj in self.objects}

        # Parse goal facts
        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == "on":
                block, support = parts[1], parts[2]
                self.goal_support_map[block] = support
                self.goal_on_top_map[support] = block # Y should have X on top
            elif predicate == "on-table":
                block = parts[1]
                self.goal_support_map[block] = 'table'
            # (clear X) facts are implicitly handled by initializing goal_on_top_map with None

        # Optional: Check if all objects in self.objects are in self.goal_support_map
        # If not, it might indicate a problem definition issue or blocks not part of goal stacks.
        # Assuming valid problems where all objects are in goal stacks.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Data structures to store the current configuration
        current_support_map = {}
        current_on_top_map = {obj: None for obj in self.objects}
        # held_block = None # Not strictly needed here

        # Parse current state facts
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "on":
                block, support = parts[1], parts[2]
                current_support_map[block] = support
                current_on_top_map[support] = block
            elif predicate == "on-table":
                block = parts[1]
                current_support_map[block] = 'table'
            elif predicate == "holding":
                held_block = parts[1]
                current_support_map[held_block] = 'arm'

        # For any block not found in current_support_map after parsing 'on' and 'holding',
        # it must be on the table. This handles cases where '(on-table X)' might be omitted
        # or implied by the absence of other location facts.
        for obj in self.objects:
            if obj not in current_support_map:
                current_support_map[obj] = 'table'


        # Memoization dictionary for the recursive function
        memo = {}

        def is_correctly_placed_in_stack(block):
            """
            Recursive helper to check if a block is in its correct goal position
            relative to its support and the block on top, and if those are also correct.
            """
            # Base cases: 'table' and 'arm' are conceptual supports, not blocks to check. None is also a valid "block" above.
            if block is None or block == 'table' or block == 'arm':
                return True
            # If block is not a recognized object, it's not a block we care about.
            if block not in self.objects:
                 return True # Should not happen with valid inputs/goals

            # Memoization
            if block in memo:
                return memo[block]

            # Get current and goal configurations
            # Use .get() for safety, although for objects in self.objects,
            # current_support_map should have an entry after the parsing loop.
            current_supp = current_support_map.get(block)
            current_on_top = current_on_top_map.get(block, None)
            goal_supp = self.goal_support_map.get(block, None)
            goal_on_top = self.goal_on_top_map.get(block, None)

            # Check immediate position and block on top
            if current_supp != goal_supp:
                memo[block] = False
                return False
            if current_on_top != goal_on_top:
                memo[block] = False
                return False

            # If immediate configuration is correct, check recursively
            # Recurse on the block below (goal_supp)
            is_below_correct = is_correctly_placed_in_stack(goal_supp)

            # Recurse on the block above (goal_on_top), if one exists in the goal
            is_above_correct = True
            if goal_on_top is not None:
                 is_above_correct = is_correctly_placed_in_stack(goal_on_top)

            result = is_below_correct and is_above_correct
            memo[block] = result
            return result

        # Compute the heuristic value
        heuristic_value = 0
        # Iterate through all objects defined in the problem instance
        for obj in self.objects:
            # Only consider objects that are part of the goal configuration
            # (i.e., they have a defined goal support).
            if obj in self.goal_support_map:
                if not is_correctly_placed_in_stack(obj):
                    heuristic_value += 1
            # Blocks not in goal_support_map are ignored.

        return heuristic_value
