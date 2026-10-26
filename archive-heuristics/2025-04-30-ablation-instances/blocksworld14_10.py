from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not in their
    correct final position relative to the block directly below them (or the table),
    considering the desired stack structure defined by the goal. It counts the
    number of blocks that are not part of a correctly built stack segment
    starting from the table and matching the goal configuration upwards.
    Essentially, it counts how many blocks are "misplaced" within the goal
    stack structures.

    # Assumptions
    - The goal state defines a specific configuration of blocks on the table
      and stacked on top of each other using `(on-table ?x)` and `(on ?x ?y)`
      predicates.
    - Standard Blocksworld actions (pickup, putdown, stack, unstack).
    - The heuristic assumes unit cost for all actions.
    - The goal predicates only involve `on` and `on-table` for blocks.

    # Heuristic Initialization
    - Parses the goal facts to build a mapping from each block to its
      required support (the block it should be on, or 'table') in the goal state.
    - Identifies all blocks that are part of the goal stack structure defined
      by the `on` and `on-table` goal predicates.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state facts to determine the current support for each block
       (which block it is on, or 'table'). Also identify if any block is being held
       by the arm.
    2. Initialize a memoization dictionary to store the 'correctly_stacked' status
       for each block to avoid redundant calculations and handle dependencies
       within stacks.
    3. Initialize a counter for misplaced blocks to 0.
    4. For each block that is part of the goal stack structure (identified during initialization):
       a. Check if the 'correctly_stacked' status for this block has already been computed
          (i.e., is in the memoization dictionary). If so, skip to the next block.
       b. If not memoized, recursively determine if the block is 'correctly_stacked'
          relative to the goal:
          i. A block `B` is correctly stacked if its current support (the block it's on,
             or the table) is the same as its goal support AND the block below it
             (its goal support, if it's a block) is also correctly stacked.
          ii. The base case is a block `B` whose goal support is 'table'. It is
              correctly stacked if its current support is also 'table'.
          iii. If the current support is different from the goal support (including
               being held by the arm, which acts as a temporary support), the block
               is not correctly stacked.
          iv. The result of the check for the current block is stored in the memoization
              dictionary.
       c. If the block is determined to be *not* correctly stacked, increment the
          misplaced blocks counter.
    5. The final count of misplaced blocks is the heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal support relationships
        and identifying all blocks involved in the goal.
        """
        self.goals = task.goals

        # Map block -> goal_support (block or 'table')
        self.goal_support_map = {}
        # Set of all blocks mentioned in the goal placement facts
        self.goal_blocks = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'on':
                block, support = parts[1], parts[2]
                self.goal_support_map[block] = support
                self.goal_blocks.add(block)
                # Add the support block to goal_blocks if it's not 'table'
                if support != 'table':
                    self.goal_blocks.add(support)
            elif parts[0] == 'on-table':
                block = parts[1]
                self.goal_support_map[block] = 'table'
                self.goal_blocks.add(block)

        # Ensure 'table' is not in goal_blocks set if it was accidentally added
        self.goal_blocks.discard('table')


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Map block -> current_support (block or 'table')
        current_support_map = {}
        holding_block = None

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                current_support_map[parts[1]] = parts[2]
            elif parts[0] == 'on-table':
                current_support_map[parts[1]] = 'table'
            elif parts[0] == 'holding':
                holding_block = parts[1]

        # Memoization dictionary for correctly_stacked status
        memo = {}

        def is_correctly_stacked(block):
            """
            Recursive helper to check if a block is correctly stacked according to the goal.
            Uses memoization.
            """
            if block in memo:
                return memo[block]

            # Get the required support for this block from the goal
            goal_sup = self.goal_support_map.get(block)

            # If the block is not in the goal_support_map, it means it's not the upper
            # block in any goal (on block X) fact, and not in any goal (on-table block) fact.
            # This implies it's not part of the specific stack structures defined in the goal
            # that we are trying to build from the bottom up. We don't count such blocks
            # as "misplaced" relative to the goal structure definition itself.
            # This case should ideally not be reached for blocks in self.goal_blocks.
            if goal_sup is None:
                 memo[block] = True # Not misplaced relative to the goal structure definition
                 return True

            # Find the current support for this block
            current_sup = current_support_map.get(block)
            if holding_block == block:
                current_sup = 'arm' # Special support for held blocks

            # If current support doesn't match goal support, it's not correctly stacked
            if current_sup != goal_sup:
                memo[block] = False
                return False

            # If current support matches goal support, check the support recursively
            if goal_sup == 'table':
                # Base case: correct support is the table
                memo[block] = True
                return True
            else:
                # Recursive case: correct support is another block.
                # Check if the block below is correctly stacked.
                result = is_correctly_stacked(goal_sup)
                memo[block] = result
                return result

        misplaced_count = 0
        # Only count blocks that are part of the goal stack structure (those in self.goal_blocks)
        for block in self.goal_blocks:
             if not is_correctly_stacked(block):
                 misplaced_count += 1

        return misplaced_count
