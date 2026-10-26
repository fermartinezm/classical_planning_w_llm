from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to extract parts of a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to reach the goal state
    by counting the number of blocks that are not on their correct goal base
    (either another block or the table), plus a penalty for blocks that are
    incorrectly blocked (have something on top that shouldn't be there),
    plus one if the arm is holding a block. It prioritizes getting blocks
    onto their correct foundation and clearing blocks that are blocking others.

    # Assumptions:
    - All blocks mentioned in the initial state or goal state are relevant.
    - Blocks not specified in the goal state's `on` or `on-table` predicates
      are assumed to need to be on the table and clear in the goal state.
    - The heuristic value is 0 if and only if the state is a goal state.

    # Heuristic Initialization
    - Extracts all block names from the initial state and goal state facts.
    - Parses the goal facts to determine the required base location (block or table)
      for each block in the goal state. Blocks not explicitly given an `on` or
      `on-table` goal are assigned a goal base of 'table'.
    - Stores the goal facts in a set for efficient lookup of goal top requirements.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Initialize the heuristic value `h` to 0.
    2. Check if the arm is currently holding any block (i.e., `(arm-empty)` is false).
       If the arm is holding a block, increment `h` by 1. This accounts for the
       block being in transit and not in a stable location.
    3. Iterate through each block identified in the problem:
       a. Determine the block's required base location in the goal state (either
          a specific block it should be `on` or the `table`). This information
          is pre-calculated during initialization (`self.goal_bases`).
       b. Determine the block's current base location in the current state. This is
          either the block it is currently `on`, the `table` if it is `on-table`.
          (Blocks being held are handled by step 2 and skipped here).
       c. If the block's current base location is different from its required goal
          base location, increment `h` by 1. This counts blocks that need to be
          moved to their correct foundation.
       d. Check if the block is incorrectly blocked. Find the block currently
          directly on top of this block in the current state. Then, check the
          goal facts to see if this block should be clear or have a specific
          block on top in the goal state.
          - If there is a block on top in the current state, AND
          - If the goal requires this block to be `clear`, OR
          - If the goal requires a *different* specific block to be on top,
          then increment `h` by 1. This accounts for the need to move the
          blocking block first.
    4. Return the total value of `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and block information."""
        self.goals = task.goals
        self.initial_state = task.initial_state

        # Convert goals to a set for faster lookup
        self.goal_facts_set = frozenset(self.goals)

        # 1. Identify all blocks
        self.blocks = set()
        # Collect blocks from initial state and goal state
        for fact in list(self.initial_state) + list(self.goals):
            parts = get_parts(fact)
            if not parts: continue
            # Consider predicates that involve blocks
            if parts[0] in ["on", "on-table", "clear", "holding"]:
                for obj in parts[1:]:
                    self.blocks.add(obj)

        # 2. Determine goal base for each block
        self.goal_bases = {}
        # Initialize all blocks to have goal base 'table' by default
        for block in self.blocks:
             self.goal_bases[block] = 'table'

        # Override default if block is specified in an 'on' goal
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            if parts[0] == "on" and len(parts) == 3:
                block_on_top = parts[1]
                block_below = parts[2]
                # Ensure the block on top is one we are tracking
                if block_on_top in self.blocks:
                     self.goal_bases[block_on_top] = block_below

        # Blocks mentioned in (on-table X) goals already have default 'table' base.
        # No need to explicitly process (on-table) goals for goal_bases.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        h = 0

        # 1. Check if arm is holding a block
        if "(arm-empty)" not in state:
             h += 1 # Cost for the block being held

        # 2. Count blocks not on their goal base and incorrectly blocked
        for block in self.blocks:
            # Find current base
            current_base = None
            if f"(on-table {block})" in state:
                current_base = 'table'
            else:
                # Check if it's on another block
                for fact in state:
                    parts = get_parts(fact)
                    if parts and parts[0] == "on" and len(parts) == 3 and parts[1] == block:
                        current_base = parts[2]
                        break
            # If current_base is None, the block must be held.
            # Its cost is accounted for by the arm check. Skip further checks for held blocks.
            if current_base is None:
                 continue

            goal_base = self.goal_bases.get(block, 'table') # Should always find the block

            # Check if B is on its correct goal base.
            if current_base != goal_base:
                h += 1 # Block needs to be moved to its correct base.

            # Check if B is incorrectly blocked.
            current_block_on_B = None
            for fact in state:
                parts = get_parts(fact)
                if parts and parts[0] == "on" and len(parts) == 3 and parts[2] == block:
                    current_block_on_B = parts[1]
                    break

            # Determine what should be on top of B in the goal
            goal_block_on_B = None
            goal_B_should_be_clear = False
            # Check goal facts for (on ?c B) or (clear B)
            for goal_fact in self.goal_facts_set:
                parts = get_parts(goal_fact)
                if not parts: continue
                if parts[0] == "on" and len(parts) == 3 and parts[2] == block:
                    goal_block_on_B = parts[1]
                    break # Found the block that should be on top
                if parts[0] == "clear" and len(parts) == 2 and parts[1] == block:
                    goal_B_should_be_clear = True
                    break # Found that it should be clear

            if current_block_on_B is not None: # Something is on B
                if goal_B_should_be_clear:
                    h += 1 # B should be clear, but isn't.
                elif goal_block_on_B is not None and current_block_on_B != goal_block_on_B:
                    h += 1 # Wrong block is on B.
                # If goal_block_on_B is None and goal_B_should_be_clear is False, it means the goal
                # doesn't specify what should be on B. Having something on B is okay in this heuristic's view.

        return h
