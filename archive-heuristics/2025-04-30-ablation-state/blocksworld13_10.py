from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions adapted from provided examples
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
    # Basic check: number of parts must match number of args unless args contains wildcards
    if len(parts) != len(args) and '*' not in args:
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are misplaced within the
    goal stack structure, plus the number of blocks that are stacked on top of
    correctly placed blocks in the goal structure when they shouldn't be there.
    It guides the search towards building the goal stacks from the bottom up
    and clearing any blocks obstructing this process.

    # Assumptions
    - The goal specifies the desired position (on another block or on the table)
      for all blocks that are relevant to the final configuration.
    - Blocks not mentioned in goal 'on' or 'on-table' predicates are considered
      irrelevant to the goal structure for the purpose of the 'in place' check.
    - Standard Blocksworld rules apply (one block on another, arm holds one block).
    - The initial state contains all blocks mentioned in the goal.

    # Heuristic Initialization
    - Parses the goal state to build a dictionary mapping each block to its
      desired support (another block or 'table'). This defines the goal stack
      structure.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to determine:
       - The current support for each block (which block it is on, or 'table').
       - The block currently held by the arm.
       - The block currently on top of each block (inverse of support).
    2. Compute the first term: Number of blocks X in the goal configuration
       (those mentioned in goal 'on' or 'on-table' predicates) that are *not*
       "in place".
       a. A block X is "in place" if:
          - Its goal is (on-table X) AND it is currently on the table.
          - OR its goal is (on X Y) AND it is currently on Y AND Y is "in place".
          - A block being held is never "in place" relative to a support.
       b. Use memoization (a cache) within the recursive "is_in_place" function
          to efficiently determine the "in place" status for each block.
    3. Compute the second term: Number of blocks Y that are currently on top of
       some block X, where X is part of the goal configuration AND X is
       determined to be "in place" by the recursive check, AND Y is *not* the
       block that should be on X according to the goal. These are "extra" blocks
       that need to be removed from correctly built goal stack segments.
    4. The total heuristic value is the sum of the first and second terms.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the goal stack structure.
        """
        self.goals = task.goals

        # Build the goal position mapping: block -> desired_support ('table' or another block)
        self.goal_position = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "on":
                block, support = parts[1], parts[2]
                self.goal_position[block] = support
            elif parts[0] == "on-table":
                block = parts[1]
                self.goal_position[block] = 'table'

        # We only care about blocks explicitly mentioned in the goal positions
        self.goal_blocks = set(self.goal_position.keys())

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state
        state_facts = set(state)

        # Parse the current state
        current_position = {} # block -> support ('table' or another block)
        held_block = None
        blocks_currently_on_top_map = {} # block_below -> block_on_top (assuming only one)

        for fact in state_facts:
            parts = get_parts(fact)
            if parts[0] == "on":
                block, support = parts[1], parts[2]
                current_position[block] = support
                blocks_currently_on_top_map[support] = block # Store the block on top
            elif parts[0] == "on-table":
                block = parts[1]
                current_position[block] = 'table'
            elif parts[0] == "holding":
                held_block = parts[1]

        # Memoization cache for the recursive is_in_place check
        in_place_cache = {}

        def is_in_place(block):
            """
            Recursively check if a block is in its correct goal position,
            relative to its support, and if that support is also in place.
            """
            if block in in_place_cache:
                return in_place_cache[block]

            # If the block is not part of the goal structure, it's trivially "in place"
            # with respect to the goal structure we care about.
            if block not in self.goal_position:
                 in_place_cache[block] = True
                 return True

            goal_support = self.goal_position[block]
            current_support = current_position.get(block) # None if not found (e.g., held)

            # A held block is never in place relative to a support
            if held_block == block:
                 in_place_cache[block] = False
                 return False

            # Check if the current support matches the goal support
            if current_support != goal_support:
                in_place_cache[block] = False
                return False

            # If the support is 'table', the block is in place if it's on the table
            if goal_support == 'table':
                result = True # Already checked current_support == goal_support
            else:
                # If the support is another block, it's in place only if the support is also in place
                result = is_in_place(goal_support)

            in_place_cache[block] = result
            return result

        # Term 1: Count blocks in goal_blocks that are not in place
        not_in_place_count = 0
        for block in self.goal_blocks:
            # Only consider blocks that are actually present in the current state's position facts or held
            if block in current_position or held_block == block:
                 if not is_in_place(block):
                    not_in_place_count += 1
            # else: block is in goal_blocks but not in state? Should not happen in valid problems.

        # Term 2: Count blocks that are on top of correctly placed goal blocks,
        # unless they are the correct block that should be there according to the goal.
        extra_blocks_on_correctly_placed_goal_blocks_count = 0

        # Iterate through all blocks that are currently supporting something (excluding table)
        # The keys of blocks_currently_on_top_map are the blocks that have something on them
        for block_below in blocks_currently_on_top_map:
             if block_below != 'table':
                 block_on_top = blocks_currently_on_top_map[block_below]

                 # Check if the block below is a goal block and is in place
                 if block_below in self.goal_blocks and is_in_place(block_below):
                     # Check if the block on top is NOT the block that should be there according to the goal
                     # If block_on_top is not in goal_position, it definitely shouldn't be there.
                     # If block_on_top is in goal_position, check if its goal_support is block_below.
                     if self.goal_position.get(block_on_top) != block_below:
                          extra_blocks_on_correctly_placed_goal_blocks_count += 1

        # The heuristic is the sum of the two terms
        return not_in_place_count + extra_blocks_on_correctly_placed_goal_blocks_count
