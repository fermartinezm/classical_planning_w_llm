from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    by counting two types of discrepancies from the goal configuration:
    1. Blocks that are not on their correct goal base (table or another block).
    2. Blocks that are currently stacked on top of another block, but should not
       be there according to the goal configuration.
    Each discrepancy is counted as 1, representing at least one action needed
    to resolve that specific structural issue.

    # Assumptions
    - The goal specifies the desired base for each block (either on the table
      or on a specific block) using `(on ?x ?y)` and `(on-table ?x)` predicates.
    - The heuristic counts each misplaced block and each incorrectly stacked
      block on top as requiring at least one action to fix. These counts are
      summed up.
    - All objects involved in the problem are blocks.

    # Heuristic Initialization
    - Extracts the desired base for each block from the goal predicates ((on ?x ?y)
      and (on-table ?x)). This creates the `self.goal_base` mapping.
    - Builds an inverse mapping (`self.goal_stack_above`) to quickly find which
      block should be directly on top of another block in the goal configuration.
    - Identifies all unique blocks involved in the problem by examining all
      parameters in relevant predicates (`on`, `on-table`, `holding`, `clear`)
      found in both the initial state and the goal.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value `h` to 0.
    2. Build temporary mappings for the current state:
       - `current_base`: block -> its current base ('table', another block, or 'arm').
       This is done by iterating through the state facts `(on ?x ?y)`, `(on-table ?x)`, and `(holding ?x)`.
    3. Iterate through all unique blocks identified during initialization:
       - For each block `b`, get its desired base (`desired_base`) from `self.goal_base`.
       - Get its current base (`current_base`) from the temporary `current_base` mapping. If a block is not found in `current_base` (meaning it's not on/on-table/holding), its current base is effectively 'nowhere'.
       - If the block `b` has a desired base defined in the goal (meaning it's part of the goal structure) AND its `current_base` is different from the `desired_base`, increment `h` by 1. This counts blocks that are not in their target location relative to their base.
    4. Iterate through all facts in the current state:
       - If a fact is of the form `(on x y)`:
         - `x` is the block currently on top, `y` is the base block.
         - Find the block that *should* be directly on top of `y` in the goal configuration using `self.goal_stack_above.get(y)`.
         - If the block currently on top (`x`) is NOT the block that should be on top of `y` in the goal (`block_on_top_goal`), increment `h` by 1. This counts blocks that are blocking or incorrectly placed on top of others.
    5. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal structure and blocks.
        """
        self.goals = task.goals
        self.initial_state = task.initial_state

        # Build goal structure mappings: desired base and desired block on top
        self.goal_base = {} # block -> base (block or 'table')
        self.goal_stack_above = {} # base (block) -> block_on_top (only for block bases)

        # Set of all blocks in the problem
        self.all_blocks = set()

        # Predicates whose parameters are blocks in blocksworld
        block_predicates = {'on', 'on-table', 'holding', 'clear'}

        # Extract goal structure and identify blocks from goals
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'on':
                block, base = parts[1], parts[2]
                self.goal_base[block] = base
                self.goal_stack_above[base] = block # Map base block to the block that should be on it
                self.all_blocks.add(block)
                self.all_blocks.add(base)
            elif parts[0] == 'on-table':
                block = parts[1]
                self.goal_base[block] = 'table'
                self.all_blocks.add(block)
            # Add blocks from other goal predicates like (clear ?x)
            elif parts[0] in block_predicates and len(parts) > 1:
                 for obj in parts[1:]:
                     self.all_blocks.add(obj)


        # Identify blocks from initial state
        for fact in self.initial_state:
             parts = get_parts(fact)
             if parts[0] in block_predicates and len(parts) > 1:
                 for obj in parts[1:]:
                     self.all_blocks.add(obj)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0

        # Build current state mappings: current base
        current_base = {} # block -> base (block, 'table', or 'arm')

        # Populate current_base
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                block, base = parts[1], parts[2]
                current_base[block] = base
            elif parts[0] == 'on-table':
                block = parts[1]
                current_base[block] = 'table'
            elif parts[0] == 'holding':
                block = parts[1]
                current_base[block] = 'arm'
            # Ignore 'clear' and 'arm-empty' for base structure

        # 1. Count blocks not on their correct base
        for block in self.all_blocks:
            desired_base = self.goal_base.get(block)
            current_b = current_base.get(block) # Can be None if block is not on/on-table/holding

            # If the block is part of the goal structure (has a desired base)
            # AND its current base is different from the desired base
            if desired_base is not None and current_b != desired_base:
                h += 1

        # 2. Count blocks that are currently on top of another block incorrectly
        # Iterate through all (on x y) facts in the current state
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                block_on_top_current = parts[1]
                base_current = parts[2]

                # Find what block should be on top of base_current in the goal
                # Note: goal_stack_above only maps block bases to the block on top.
                # If base_current is 'table', nothing should be on it in the goal structure.
                block_on_top_goal = self.goal_stack_above.get(base_current)

                # If the block currently on top (block_on_top_current) is NOT the block
                # that should be on top in the goal (block_on_top_goal)
                if block_on_top_current != block_on_top_goal:
                     h += 1

        return h
