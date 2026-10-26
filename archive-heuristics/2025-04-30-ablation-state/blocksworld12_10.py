from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to reach the goal
    by counting the number of blocks that are in the wrong position relative
    to their goal parent AND the number of blocks that have the wrong block
    on top relative to their goal child. Each type of discrepancy for each
    relevant block contributes 1 to the heuristic value.

    # Assumptions
    - The goal state defines a specific configuration of blocks stacked
      on each other or on the table.
    - The heuristic focuses on achieving the correct relative positions
      between blocks and the table as defined by the goal.
    - Standard Blocksworld goals are assumed, where (clear X) goals
      typically correspond to blocks that are the top of a goal stack.
    - The 'table' is treated as a special location/parent.
    - The 'holding' state is treated as a special parent type.

    # Heuristic Initialization
    - The constructor processes the goal predicates (`task.goals`) to build
      a representation of the desired goal configuration:
        - `goal_parent`: Maps a block to the block or 'table' it should be on.
          A value of `None` means the block is only mentioned in a `clear` goal
          without a specified parent structure.
        - `goal_child`: Maps a block to the block that should be immediately
          on top of it. A value of `None` means the block should be clear on top
          in the goal.
        - `goal_blocks`: The set of all blocks explicitly mentioned in the
          goal configuration (either in `on`, `on-table`, or `clear` predicates).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state (`node.state`):
    1. Parse the current state facts to determine the current configuration:
       - `current_parent`: Maps a block to the block, 'table', or 'holding'
         it is currently on or held by.
       - `current_child`: Maps a block to the block currently immediately
         on top of it (only one child is possible in Blocksworld).
         A value of `None` means the block is currently clear on top.
       - `current_holding`: The block currently held by the arm, or None.
    2. Initialize the heuristic value `h = 0`.
    3. Iterate through each block `B` that is part of the goal configuration (`self.goal_blocks`).
    4. For each block `B`, compare its current position/parent with its goal parent:
       - Get the goal parent (`goal_p`) from `self.goal_parent.get(B)`.
       - Get the current parent (`current_p`) from `current_parent.get(B)`.
       - If `B` has a specific goal parent (`goal_p` is not None, meaning it appeared in an `on` or `on-table` goal predicate), and its `current_p` is different from `goal_p`, increment `h`. This counts blocks that are on the wrong support, held when they shouldn't be, on the table when they shouldn't be on something else, or on something else when they should be on the table.
    5. For each block `B`, compare what is currently on top of it with what should be on top according to the goal:
       - Get the goal child (`goal_c`) from `self.goal_child.get(B)`. `None` means `B` should be clear on top in the goal.
       - Get the current child (`current_c`) from `current_child.get(B)`. `None` means `B` is currently clear on top.
       - If `goal_c` is `None` (B should be clear on top) but `current_c` is not `None` (something is on B), increment `h`.
       - If `goal_c` is a specific block `C` (C should be on B) but `current_c` is not `C` (either nothing or the wrong block is on B), increment `h`.
    6. The total value of `h` is the heuristic estimate.
    """

    def _process_goal(self, goals):
        """
        Process goal predicates to build the desired goal configuration structure.
        """
        self.goal_parent = {}  # block -> parent_block_or_table (None if only in clear goal)
        self.goal_child = {}   # parent_block -> child_block (None if parent should be clear on top)
        self.goal_blocks = set() # blocks involved in goal stacks or explicitly in clear goals

        # First pass: Process 'on' and 'on-table' to build the stack structure
        for goal in goals:
            parts = get_parts(goal)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block, parent = parts[1], parts[2]
                self.goal_parent[block] = parent
                self.goal_child[parent] = block # Assumes only one block on top in goal
                self.goal_blocks.add(block)
                self.goal_blocks.add(parent)
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                self.goal_parent[block] = 'table'
                self.goal_blocks.add(block)

        # Second pass: Process 'clear' goals and infer top blocks
        for goal in goals:
             parts = get_parts(goal)
             if not parts: continue
             predicate = parts[0]
             if predicate == 'clear' and len(parts) == 2:
                 block = parts[1]
                 # If a block must be clear, it cannot have a goal_child.
                 # If it was set as a child in the first pass, the goal is inconsistent,
                 # but we prioritize the 'clear' requirement for the heuristic.
                 self.goal_child[block] = None
                 self.goal_blocks.add(block) # Add to goal_blocks even if only in clear goal

        # Infer goal_child for blocks in goal_blocks that were not parents in 'on' goals
        # These blocks are the tops of their respective goal stacks (or standalone blocks)
        for block in list(self.goal_blocks): # Iterate over a copy if modifying the set
             if block not in self.goal_child:
                 self.goal_child[block] = None # Represents being clear on top


    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # 1. Parse current state facts
        current_parent = {} # block -> parent_block_or_table_or_holding
        current_child = {}  # parent_block -> child_block (only one child possible)
        current_holding = None

        # Build current config from state facts
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block, parent = parts[1], parts[2]
                current_parent[block] = parent
                current_child[parent] = block
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                current_parent[block] = 'table'
            elif predicate == 'holding' and len(parts) == 2:
                block = parts[1]
                current_holding = block
                current_parent[block] = 'holding' # Special parent type
            # 'clear' facts are not needed to build parent/child structure

        # 2. Initialize heuristic value
        h = 0

        # 3. Iterate through each block that is part of the goal configuration
        for block in self.goal_blocks:
            # Get goal and current parent/child, using .get() for safety
            # If a block is in goal_blocks but not in the current state config
            # (e.g., not on anything, not on table, not held - implies off-stage,
            # which shouldn't happen in standard BW), current_parent/child will be None.
            # This is handled correctly by the comparisons.
            goal_p = self.goal_parent.get(block)
            current_p = current_parent.get(block)

            goal_c = self.goal_child.get(block)
            current_c = current_child.get(block)

            # 4. Check parent position
            # Count if the block has a specific goal parent AND the current parent is different
            # If goal_p is None, it means the block was only in a 'clear' goal,
            # and we don't have a specific parent requirement from the goal structure.
            if goal_p is not None and current_p != goal_p:
                 h += 1

            # 5. Check child position (what's on top)
            # Count if the block should be clear but isn't
            if goal_c is None: # Block should be clear on top in the goal
                if current_c is not None: # But something is on it in the current state
                    h += 1
            # Count if the block should have a specific block on top, but doesn't have that specific block
            elif current_c != goal_c: # goal_c is not None, so it's a specific block C
                 h += 1

        # 6. Return the total heuristic value
        return h
