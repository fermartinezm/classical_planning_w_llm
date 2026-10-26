from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(on b8 b9)" -> ["on", "b8", "b9"]
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the distance to the goal state by combining two components:
    1. A count of blocks that are not part of a correctly built stack segment
       starting from the table, based on the goal configuration.
    2. A count of goal facts that are not satisfied in the current state.
    The sum of these two counts provides the heuristic value. This heuristic
    is designed for greedy best-first search and does not need to be admissible.

    # Assumptions
    - The goal state consists of one or more stacks of blocks on the table,
      defined by `on` and `on-table` predicates.
    - The goal may also include `clear` predicates for the top blocks of stacks
      and `arm-empty`.
    - The heuristic assumes a block is correctly placed relative to its base
      only if it is on the correct base AND that base is recursively correctly
      placed down to the table.

    # Heuristic Initialization
    - Parse the goal facts (`task.goals`).
    - Build a mapping `self.goal_base` where `self.goal_base[block]` is the
      block or 'table' that `block` should be directly on top of in the goal state.
      Blocks that are bases of goal stacks but not explicitly mentioned with
      `on-table` are assumed to have 'table' as their goal base.
    - Identify `self.all_goal_blocks`, the set of all blocks whose position
      is specified within the goal stack structure (either as a block on top
      or a base).

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current state's configuration:
       - Create a mapping `current_bases` where `current_bases[block]` is the
         block or 'table' that `block` is currently on, or 'arm' if it's held.
         This is done by iterating through `on`, `on-table`, and `holding` facts
         in the current state.
    2. Compute the first component of the heuristic (h8):
       - Initialize `h8 = 0`.
       - For each block `b` in `self.all_goal_blocks`:
         - Use a recursive helper function `is_correctly_placed(b, current_bases)`
           to check if `b` is on its goal base and if that base is recursively
           correctly placed down to the table according to the goal structure.
         - If `is_correctly_placed` returns False, increment `h8`.
       - The recursive helper `is_correctly_placed(block, current_bases)`:
         - Base case: If `block` is the string 'table', return True.
         - Determine the desired base for `block` from `self.goal_base`. If `block`
           is not a key in `self.goal_base` (meaning it's a base of a goal stack
           but not explicitly listed with `on-table`), its desired base is 'table'.
         - Determine the current base for `block` from `current_bases`.
         - If the block is not found in `current_bases` (e.g., state is malformed)
           or its current base does not match the desired base, return False.
         - Otherwise (current base matches desired base), recursively call
           `is_correctly_placed` on the desired base.
    3. Compute the second component of the heuristic (unsatisfied goals):
       - Initialize `unsatisfied_goals_count = 0`.
       - For each goal fact `g` in `self.goals`:
         - If `g` is not present in the current state, increment `unsatisfied_goals_count`.
    4. The total heuristic value is the sum of `h8` and `unsatisfied_goals_count`.
       This sum is 0 if and only if the state is the goal state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the goal structure from the task.
        """
        self.goals = task.goals # Store the full set of goal facts

        # Build the goal_base mapping: block -> desired_base
        self.goal_base = {}
        blocks_involved_in_on = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'on':
                block_on_top, base = parts[1], parts[2]
                self.goal_base[block_on_top] = base
                blocks_involved_in_on.add(block_on_top)
                blocks_involved_in_on.add(base)
            elif parts[0] == 'on-table':
                block = parts[1]
                self.goal_base[block] = 'table'
                blocks_involved_in_on.add(block)

        # Blocks that are values in goal_base but not keys (and not 'table')
        # are the bases of goal stacks whose goal base is implicitly 'table'.
        # Ensure all blocks involved in 'on' goals have a goal_base entry.
        for block in blocks_involved_in_on:
             if block != 'table' and block not in self.goal_base:
                 self.goal_base[block] = 'table' # This block should be on the table

        # Set of all blocks whose position is specified within the goal structure
        # These are the blocks we need to check for correct placement.
        self.all_goal_blocks = set(self.goal_base.keys())
        # Add blocks that are values but not keys (bases of stacks)
        for base in self.goal_base.values():
            if base != 'table':
                 self.all_goal_blocks.add(base)


    def is_correctly_placed(self, block, current_bases):
        """
        Recursive helper to check if a block is in its goal position relative
        to its base, and if that base is also correctly placed, down to the table.
        """
        if block == 'table':
            return True

        # Get the desired base for this block from the goal structure.
        # If block is not a key in goal_base (it's a base of a goal stack),
        # its desired base is implicitly 'table'.
        desired_base = self.goal_base.get(block, 'table')

        # Get the current base for this block from the state.
        # A block might be held ('arm') or not found if state is malformed.
        current_base = current_bases.get(block)

        # If the block is not found in the state (e.g., only exists in goal)
        # or its current base does not match the desired base, it's not correctly placed.
        if current_base is None or current_base != desired_base:
            return False

        # If the current base is correct, recursively check if the base itself
        # is correctly placed.
        return self.is_correctly_placed(desired_base, current_bases)


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Build current_bases mapping from the state
        current_bases = {}
        # We only need 'on' and 'on-table' facts to determine the base.
        # 'holding' means it's not on a base, but we can record it as 'arm'.
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                block_on_top, base = parts[1], parts[2]
                current_bases[block_on_top] = base
            elif parts[0] == 'on-table':
                block = parts[1]
                current_bases[block] = 'table'
            elif parts[0] == 'holding':
                 block = parts[1]
                 current_bases[block] = 'arm' # Block is held

        # Part 1: Count blocks not recursively correctly placed (Heuristic 8 component)
        h8 = 0
        # Iterate through all blocks involved in the goal stack structure
        for block in self.all_goal_blocks:
            if not self.is_correctly_placed(block, current_bases):
                h8 += 1

        # Part 2: Count unsatisfied goal facts
        unsatisfied_goals_count = 0
        for goal in self.goals:
            if goal not in state:
                unsatisfied_goals_count += 1

        # The total heuristic is the sum of the two components.
        return h8 + unsatisfied_goals_count
