# Required imports
from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if running standalone for testing
# In the actual environment, the provided base class will be used.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static
        def __call__(self, node):
            raise NotImplementedError
        # Add a dummy task class for testing
        class DummyTask:
            def __init__(self, goals, static):
                self.goals = goals
                self.static = static

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

# Helper function to match PDDL facts with patterns (optional, but good practice)
# This function is not strictly needed for the final heuristic implementation
# but was used in the thought process and examples. Keep it for completeness
# or remove if not used.
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
    immediate support (another block or the table) according to the goal state.
    For each block that is not on its correct support, it estimates 2 actions:
    one to move it off its current support, and one to move it onto its goal support.
    Blocks that are currently held are also considered misplaced relative to any
    goal support (since goals are never to be held), incurring a cost of 2
    (putdown + move to goal support).

    # Assumptions
    - The goal state specifies the desired immediate support for every block involved in a stack
      using `(on ?x ?y)` or `(on-table ?x)` predicates.
    - The cost of clearing blocks that are in the way is implicitly handled or ignored
      in this non-admissible heuristic.
    - Each misplaced block requires at least two moves (off and on).
    - The arm state (`arm-empty`, `holding`) is handled by considering a held block as misplaced.
    - All blocks mentioned as the first argument in an `on` or `on-table` goal predicate
      are assumed to exist and be in a supported or held state in any reachable state.

    # Heuristic Initialization
    - Parses the goal state to determine the desired immediate support for each block
      that is explicitly positioned in the goal.
      This is stored in the `goal_support` dictionary, mapping a block to the block
      it should be on, or the string 'table'.
    - Identifies all blocks that are explicitly given a position in the goal state
      (i.e., appear as the first argument in an `on` or `on-table` goal predicate).
      These are the blocks whose position we track for the heuristic.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to determine the immediate support for each block.
       This creates a `current_support` dictionary, mapping a block to the block
       it is currently on, or the string 'table'.
    2. Identify which block, if any, is currently being held by the arm.
    3. Initialize the heuristic value `h` to 0.
    4. Iterate through each block that has a specified goal support (i.e., each block
       that appeared as the first argument in an `on` or `on-table` goal predicate).
    5. For each such block:
       - Get its goal support (`goal_pos`) from the pre-calculated `self.goal_support` map.
       - Check if the block is currently held (`held_block == block`).
       - If the block is currently held:
         - It is misplaced relative to any goal support (since goals are never 'held').
         - Add 2 to `h` (estimated cost: putdown + move to goal support).
       - If the block is not currently held:
         - Get its current support (`current_pos`) from the `current_support` dictionary.
         - If the current support is different from the goal support (`current_pos != goal_pos`):
           - The block is misplaced relative to its base.
           - Add 2 to `h` (estimated cost: move off current support + move onto goal support).
         # Note: If current_pos is None here, it implies the block is not held and not supported.
         # This shouldn't happen in valid Blocksworld states derived from a valid initial state.
         # The check `current_pos != goal_pos` correctly handles `None != goal_pos`.
    6. The total heuristic value is the sum of these costs for all blocks whose position
       is explicitly defined in the goal and is currently incorrect.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the goal support for blocks
        explicitly positioned in the goal state.
        """
        super().__init__(task)
        self.goals = task.goals

        # Build the goal_support map: block -> block_below or 'table'
        self.goal_support = {}
        # Keep track of blocks whose position is explicitly set in the goal
        self.blocks_with_goal_pos = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]
            if predicate == "on":
                # (on ?x ?y) means ?x should be on ?y
                if len(parts) == 3:
                    block_on_top, block_below = parts[1], parts[2]
                    self.goal_support[block_on_top] = block_below
                    self.blocks_with_goal_pos.add(block_on_top)
            elif predicate == "on-table":
                # (on-table ?x) means ?x should be on the table
                if len(parts) == 2:
                    block_on_table = parts[1]
                    self.goal_support[block_on_table] = 'table'
                    self.blocks_with_goal_pos.add(block_on_table)
            # Ignore (clear ?x) and (arm-empty) goals for support mapping

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state  # frozenset of fact strings

        # Build the current_support map: block -> block_below or 'table'
        current_support = {}
        held_block = None

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]
            if predicate == "on":
                if len(parts) == 3:
                    block_on_top, block_below = parts[1], parts[2]
                    current_support[block_on_top] = block_below
            elif predicate == "on-table":
                if len(parts) == 2:
                    block_on_table = parts[1]
                    current_support[block_on_table] = 'table'
            elif predicate == "holding":
                 if len(parts) == 2:
                     held_block = parts[1]

        h = 0
        # Iterate through blocks that have a defined goal position
        for block in self.blocks_with_goal_pos:
            goal_pos = self.goal_support[block] # Goal support must exist for these blocks

            if held_block == block:
                 # Block is currently held, but goal is on something or table. Misplaced.
                 # Cost: putdown (1) + move to goal support (1) = 2
                 h += 2
            else:
                # Block is not held. Find its current support.
                # If a block is in blocks_with_goal_pos but not in current_support and not held,
                # it implies an inconsistency in the state representation (e.g., block is missing).
                # Assuming valid states, if not held, it must be in current_support.
                current_pos = current_support.get(block)

                # Check if current support matches goal support
                if current_pos != goal_pos:
                    # Block is not held, and its current support is wrong.
                    # Cost: move off current support (1) + move onto goal support (1) = 2
                    h += 2

        # The heuristic value is the total count.
        return h
