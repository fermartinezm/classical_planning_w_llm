from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts represented as strings
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
    # Ensure we don't go out of bounds if fact has fewer parts than args
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not in their
    correct final position within the goal stacks. A block is in its
    correct final position if it is on the correct block (or the table)
    according to the goal, AND the block immediately below it is also
    in its correct final position.

    # Assumptions
    - The goal state consists of one or more stacks of blocks, or single
      blocks on the table.
    - Each block involved as the subject of an 'on' or 'on-table' goal fact
      has a unique goal position (either on another specific block or on the table).
    - The goal state forms valid stacks (no cycles).
    - All blocks present in the initial state that are relevant to the goal
      are accounted for in the goal facts as the subject of an 'on' or 'on-table'
      predicate, or they are blocks that other goal blocks are placed upon.
    - Standard Blocksworld goals do not require the arm to be non-empty;
      the arm state is transient towards achieving block arrangements.
      The heuristic focuses on block arrangement.

    # Heuristic Initialization
    - Extract the goal positions for each block from the task's goal facts.
      This creates a mapping from each block (that is the subject of an 'on'
      or 'on-table' goal fact) to the object it should be directly on top of
      in the goal state (another block or 'table').
    - Identify the set of blocks whose final position is explicitly defined
      as the subject of a goal fact.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Pre-process the current state to create a map of the current position
       of each block. A block can be 'on-table', 'on' another block, or 'holding'.
       Store this in a dictionary mapping block name to its position ('table',
       block_below name, or 'holding').
    2. Initialize the heuristic value `misplaced_count` to 0.
    3. Initialize a memoization dictionary `memo` to store the result of
       `is_in_final_goal_pos` checks for each block.
    4. Define a recursive helper function `is_in_final_goal_pos(block)` that
       checks if a given `block` is in its correct final position according
       to the goal and the current state.
       - If the result for `block` is already in `memo`, return the memoized value.
       - If `block` is not one whose final position is explicitly defined in the goal
         (i.e., it's not in `self.blocks_with_explicit_goal_pos`), consider it
         correctly placed for the purpose of checking blocks above it in the goal stacks.
         Memoize `True` for this block and return `True`.
       - Get the desired `goal_below` object for this `block` from `self.goal_pos`.
       - Get the `current_pos` of the `block` from the state map created in step 1.
       - If `current_pos` is 'holding' or if the block is not found in the state map
         (which shouldn't happen in valid states, but handled defensively), the block
         is not in its goal position. Memoize `False` and return `False`.
       - Check if the `current_pos` matches the `goal_below`. If they do not match,
         the block is not in the correct position relative to the object below it.
         Memoize `False` and return `False`.
       - If the `current_pos` matches the `goal_below`:
         - If `goal_below` is 'table', the block is correctly placed relative to the table,
           and the table is always in its final position. Memoize `True` and return `True`.
         - If `goal_below` is another block, recursively call `is_in_final_goal_pos`
           on `goal_below`. Memoize and return the result of this recursive call.
    5. Iterate through each `block` in the set `self.blocks_with_explicit_goal_pos`.
       For each block, if `is_in_final_goal_pos(block)` returns `False`, increment `misplaced_count`.
    6. Return the final `misplaced_count` as the heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal positions for each block.
        """
        self.goals = task.goals  # Goal conditions.
        # Static facts are not needed for this heuristic in Blocksworld.
        # static_facts = task.static

        # Store goal positions for each block that is the subject of an 'on' or 'on-table' goal fact.
        # goal_pos[block] = block_below_in_goal or 'table'
        self.goal_pos = {}
        # Set of blocks whose final position is explicitly defined in the goal
        self.blocks_with_explicit_goal_pos = set()

        for goal in self.goals:
             predicate, *args = get_parts(goal)
             if predicate == "on":
                 block, block_below = args
                 self.goal_pos[block] = block_below
                 self.blocks_with_explicit_goal_pos.add(block)
             elif predicate == "on-table":
                 block = args[0]
                 self.goal_pos[block] = 'table'
                 self.blocks_with_explicit_goal_pos.add(block)

        # Note: Blocks that are only mentioned as being *under* another block
        # in the goal (e.g., block B in (on A B) where B is not the subject
        # of any other goal fact) are not added to self.blocks_with_explicit_goal_pos.
        # Their final position is implicitly their position in the initial state
        # or whatever position allows the blocks above them to be correctly placed.
        # The heuristic considers them "correct" for dependency checking purposes
        # if they are not in the self.blocks_with_explicit_goal_pos set.


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state.

        # Pre-process state for faster lookups: map block -> current_position ('table', block_below, or 'holding')
        current_pos_map = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                current_pos_map[parts[1]] = parts[2] # block -> block_below
            elif parts[0] == 'on-table':
                current_pos_map[parts[1]] = 'table' # block -> 'table'
            elif parts[0] == 'holding':
                current_pos_map[parts[1]] = 'holding' # block -> 'holding'
            # Ignore other facts like 'clear', 'arm-empty' for position mapping

        # Memoization dictionary for is_in_final_goal_pos
        # Maps block -> boolean (True if in final goal pos, False otherwise)
        memo = {}

        def is_in_final_goal_pos(block):
            """
            Recursive helper function to check if a block is in its final goal position.
            Uses memoization.
            """
            if block in memo:
                return memo[block]

            # Base case 1: If the block is not one whose final position is explicitly defined in the goal,
            # it is considered correctly placed for dependency checking of blocks above it.
            if block not in self.blocks_with_explicit_goal_pos:
                 memo[block] = True
                 return True

            # Get the desired goal position below this block
            goal_below = self.goal_pos[block]

            # Get the current position of the block from the pre-processed map
            current_pos = current_pos_map.get(block)

            # If block is being held or not found in state map, it's not in the goal position.
            if current_pos is None or current_pos == 'holding':
                 memo[block] = False
                 return False

            # Check if current position matches goal position below
            is_correct_relative_to_below = (current_pos == goal_below)

            if not is_correct_relative_to_below:
                memo[block] = False
                return False

            # If correct relative to below, check if the block below is in its final goal position
            if goal_below == 'table':
                # Table is always in its final goal position
                memo[block] = True
                return True
            else:
                # Recursively check the block below
                result = is_in_final_goal_pos(goal_below)
                memo[block] = result
                return result

        # Count blocks not in their final goal position
        misplaced_count = 0
        # We only care about blocks whose final position is explicitly defined in the goal
        for block in self.blocks_with_explicit_goal_pos:
             if not is_in_final_goal_pos(block):
                 misplaced_count += 1

        # The heuristic value is the number of blocks not in their final goal position.
        # This represents the number of blocks that need to be moved into their
        # correct place within the goal stacks.

        return misplaced_count
