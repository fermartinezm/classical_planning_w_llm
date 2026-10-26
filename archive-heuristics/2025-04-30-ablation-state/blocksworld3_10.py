from fnmatch import fnmatch
# Assuming heuristics.heuristic_base exists and defines a Heuristic base class
# If not, a simple base class might be needed or the inheritance removed.
# For this problem, we assume it exists as per the problem description and examples.
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the "distance" to the goal by counting the number
    of blocks that are either in the wrong position relative to their support
    (table or another block) or are incorrectly "blocked" (i.e., have the wrong
    block on top or incorrect clearance status) relative to the goal configuration.
    Each such discrepancy for a block contributes 1 to the heuristic value.

    # Assumptions
    - The goal state specifies the desired `on` relationships and which blocks
      should be on the table.
    - Blocks not mentioned as being `on` or `on-table` in the goal are assumed
      to have a goal of being on the table.
    - The heuristic counts discrepancies at each block's level: its support
      and the block directly on top of it.
    - The arm being empty is typically a consequence of achieving the goal
      block configuration, and its state is implicitly handled by the block
      positions. A held block is considered misplaced relative to its goal support.

    # Heuristic Initialization
    - Parse the goal conditions to determine the desired support (`on` or `on-table`)
      for each block and which block should be directly on top of each block
      in the goal state.
    - Identify all blocks present in the problem instance from the initial state.
    - Determine which blocks should be clear in the goal state (those not
      required to have another block on top).
    - Store these relationships in dictionaries (`goal_support`, `goal_block_on_top`)
      and a set (`goal_clear`).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state to determine the current support (`on` or `on-table`)
       for each block, which block is directly on top of each block, and which
       blocks are currently clear. Also identify if any block is being held.
       Store these in dictionaries (`current_support`, `current_block_on_top`)
       and a set (`current_clear`).
    2. Initialize the heuristic value `h` to 0.
    3. Iterate through all blocks identified during initialization.
    4. For each block `X`:
       a. Determine its current support (`current_supp`) and goal support (`goal_supp`).
          If a block is held, its current support is considered 'arm'. If a block
          is not in the goal `on`/`on-table` predicates, its goal support is 'table'.
       b. If `current_supp != goal_supp`:
          Increment `h` by 1. This block is in the wrong place relative to its support.
       c. If `current_supp == goal_supp`:
          This block is on the correct support. Now check if it's correctly "blocked"
          or "clear" relative to the goal configuration above it.
          i. Determine if the block is currently clear (`current_is_clear`) and
             if it should be clear in the goal (`goal_is_clear`).
          ii. If `current_is_clear` is True and `goal_is_clear` is False:
              Increment `h` by 1. The block should have something on top but is clear.
          iii. If `current_is_clear` is False and `goal_is_clear` is True:
               Increment `h` by 1. The block should be clear but has something on top.
          iv. If `current_is_clear` is False and `goal_is_clear` is False:
              The block is not clear and should not be clear. Check if the block
              currently on top is the correct one according to the goal.
              Determine the block currently on top (`current_top`) and the block
              that should be on top in the goal (`goal_top`).
              If `goal_top is not None` and `current_top != goal_top`:
                  Increment `h` by 1. The block has the wrong block on top.
    5. Return the final value of `h`.

    This heuristic is non-admissible. It counts multiple types of "errors" or
    "misplacings" for each block, aiming to prioritize states where more blocks
    are closer to their final configuration or less obstructed.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration details and
        identifying all blocks in the problem.
        """
        self.goals = task.goals

        # Identify all blocks in the problem from the initial state facts
        self.all_problem_blocks = set()
        for fact in task.initial_state:
             parts = get_parts(fact)
             # Add all arguments of predicates as potential blocks
             if len(parts) > 1:
                 self.all_problem_blocks.update(parts[1:])

        # Parse goals to build goal configuration maps
        self.goal_support = {} # block -> block_below (or 'table')
        self.goal_block_on_top = {} # block_below -> block_on_top

        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'on':
                block, below = parts[1], parts[2]
                self.goal_support[block] = below
                self.goal_block_on_top[below] = block
            elif parts[0] == 'on-table':
                block = parts[1]
                self.goal_support[block] = 'table'
            # Ignore 'clear' and 'arm-empty' goals for building support/top maps

        # Determine which blocks should be clear in the goal state
        # A block should be clear in the goal if no other block should be on top of it.
        blocks_that_should_have_something_on_top_in_goal = set(self.goal_block_on_top.keys())
        self.goal_clear = self.all_problem_blocks - blocks_that_should_have_something_on_top_in_goal

        # Add explicitly mentioned clear goals just in case (though standard goals make this redundant)
        # This loop might add blocks that should have something on top if the goal is malformed,
        # but the heuristic logic handles this by prioritizing the goal_block_on_top map.
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == 'clear':
                 if parts[1] in self.all_problem_blocks: # Only add if it's a known block
                     self.goal_clear.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse current state
        current_support = {} # block -> block_below (or 'table')
        current_block_on_top = {} # block_below -> block_on_top
        current_clear = set()
        held_block = None

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                block, below = parts[1], parts[2]
                current_support[block] = below
                current_block_on_top[below] = block
            elif parts[0] == 'on-table':
                block = parts[1]
                current_support[block] = 'table'
            elif parts[0] == 'clear':
                block = parts[1]
                current_clear.add(block)
            elif parts[0] == 'holding':
                held_block = parts[1]
            # Ignore arm-empty

        h = 0

        for block in self.all_problem_blocks:
            # Determine current support
            if block == held_block:
                current_supp = 'arm'
            else:
                # If block is not held, it must be on something or on the table
                # Look up its support, default to 'table' if not found (shouldn't happen in valid state)
                current_supp = current_support.get(block, 'table')

            # Determine goal support (default to table if block not in goal on/on-table)
            goal_supp = self.goal_support.get(block, 'table')

            # --- Check position relative to support ---
            if current_supp != goal_supp:
                h += 1
            else: # current_supp == goal_supp
                # --- Check clearance and block on top ---
                current_is_clear = block in current_clear
                goal_is_clear = block in self.goal_clear

                if current_is_clear and not goal_is_clear:
                    # Should have something on top, but is clear
                    h += 1
                elif not current_is_clear and goal_is_clear:
                    # Should be clear, but has something on top
                    h += 1
                elif not current_is_clear and not goal_is_clear:
                    # Should have something on top, and does. Check if it's the right block.
                    current_top = current_block_on_top.get(block, None)
                    goal_top = self.goal_block_on_top.get(block, None)
                    # If goal_top is None, it means the block should be clear in the goal.
                    # But we are in the branch where not goal_is_clear is true, meaning goal_top should NOT be None.
                    # So we only need to check if current_top matches goal_top when goal_top is not None.
                    if goal_top is not None and current_top != goal_top:
                         h += 1
                    # Note: The case where goal_top is None (should be clear) but current_top is not None (not clear)
                    # is handled by the 'not current_is_clear and goal_is_clear' branch above.

        return h
