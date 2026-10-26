# Removed fnmatch import as it's not used by get_parts or the main logic.
from heuristics.heuristic_base import Heuristic

# Helper function from Logistics example
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to reach the goal
    by counting the number of blocks that are not in their correct position
    relative to the block below them in the goal stack, considering the
    recursive nature of correct stacking.

    # Assumptions
    - The goal state is defined by a set of (on ?x ?y) and (on-table ?x)
      predicates specifying the desired stack configuration.
    - The heuristic focuses only on achieving these positional goals.
    - Blocks not mentioned in the goal predicates are not considered by the
      heuristic (implicitly assumed their final position doesn't affect goal
      achievement, or they can be anywhere).

    # Heuristic Initialization
    - Parses the goal predicates to build a mapping `self.goal_on` where
      `self.goal_on[block]` is the block it should be on, or 'table' if it
      should be on the table.
    - Identifies the set of all blocks that are explicitly mentioned in the
      goal predicates (`self.goal_blocks`).

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Parse the current state to determine the immediate support for each block.
       Create a mapping `current_on` where `current_on[block]` is the block
       it is currently on, 'table' if it's on the table, or 'arm' if it's being held.
       Only consider blocks that are part of the goal (`self.goal_blocks`).

    2. Initialize a boolean flag `correctly_stacked` for each block in
       `self.goal_blocks` to `False`. This flag indicates whether the block
       is in its correct goal position AND the block below it (if any) is
       also in its correct goal position (recursively).

    3. Iteratively update the `correctly_stacked` flags. Repeat until no
       new blocks are marked as correctly stacked in an iteration:
       - Set `changed = False` at the start of the iteration.
       - For each block `B` in `self.goal_blocks`:
         - If `correctly_stacked[B]` is already `True`, skip.
         - Find what `B` is currently on (`current_support = current_on.get(B)`)
         - Find what `B` should be on according to the goal (`goal_support = self.goal_on.get(B)`) # This will be 'table' or a block name

         - If `goal_support == 'table':`
           - If `current_support == 'table':`
             - `correctly_stacked[block] = True`
             - `changed = True`
         # goal_support must be a block name if not 'table'
         elif current_support == goal_support and correctly_stacked.get(goal_support, False):
              correctly_stacked[block] = True
              changed = True

    4. The heuristic value is the total number of blocks in `self.goal_blocks`
       for which `correctly_stacked` is still `False` after the iterative process.
       These are the blocks that are not in their final desired position relative
       to their support, considering the entire stack structure.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the goal stack structure.
        """
        self.goals = task.goals

        # Build the goal stack structure: block -> support ('table' or block)
        self.goal_on = {}
        # Keep track of all blocks involved in the goal
        self.goal_blocks = set()

        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == 'on':
                block, support = parts[1], parts[2]
                self.goal_on[block] = support
                self.goal_blocks.add(block)
                self.goal_blocks.add(support)
            elif predicate == 'on-table':
                block = parts[1]
                self.goal_on[block] = 'table'
                self.goal_blocks.add(block)
            # Ignore other goal predicates like (clear ?) or (arm-empty)

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Determine the current support for each block
        current_on = {}
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'on':
                block, support = parts[1], parts[2]
                current_on[block] = support
            elif predicate == 'on-table':
                block = parts[1]
                current_on[block] = 'table'
            elif predicate == 'holding':
                block = parts[1]
                current_on[block] = 'arm' # Block is held, not on anything

        # Initialize correctly_stacked flags for goal blocks
        correctly_stacked = {block: False for block in self.goal_blocks}

        # Iteratively mark blocks that are correctly stacked from the bottom up
        changed = True
        while changed:
            changed = False
            for block in self.goal_blocks:
                # If already marked, no need to re-check
                if correctly_stacked[block]:
                    continue

                current_support = current_on.get(block)
                goal_support = self.goal_on.get(block) # This will be 'table' or a block name

                # Case 1: Block should be on the table
                if goal_support == 'table':
                    if current_support == 'table':
                        correctly_stacked[block] = True
                        changed = True
                # Case 2: Block should be on another block (goal_support is a block name)
                # Check if it's on the correct block AND that block is correctly stacked
                # We need to ensure goal_support is itself a goal block to check correctly_stacked[goal_support]
                # Based on how goal_blocks is built, goal_support will be in goal_blocks if it's a block.
                elif current_support == goal_support and correctly_stacked.get(goal_support, False):
                     correctly_stacked[block] = True
                     changed = True

        # The heuristic is the number of blocks not correctly stacked
        h = sum(1 for block in self.goal_blocks if not correctly_stacked[block])

        return h
