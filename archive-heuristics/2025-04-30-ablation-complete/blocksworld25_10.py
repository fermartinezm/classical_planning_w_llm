from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by counting several types of 'misplacements' and 'obstacles' in the current state
    relative to the goal configuration. It considers blocks being on the wrong support,
    blocks having the wrong block on top (even if the base is correct), blocks that
    need to be clear but aren't, and the arm being occupied when it needs to be empty.

    # Assumptions
    - The goal state consists of specific (on X Y), (on-table X), (clear X), and (arm-empty) predicates.
    - All blocks mentioned in the initial state or goal state are relevant.
    - Standard Blocksworld actions (pickup, putdown, stack, unstack) with unit cost.
    - State representation is consistent (e.g., (clear X) is false iff (on Y X) is true for some Y).

    # Heuristic Initialization
    - Parses the goal facts to determine the desired support for each block (`goal_on`)
      and the desired block on top of each support (`goal_above`).
    - Identifies all blocks present in the initial state or goal state.
    - Stores the set of blocks whose final position is specified in the goal (`goal_blocks`).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1.  Build a representation of the current state:
        - `current_on`: Mapping each block to its current support ('table', another block, or 'arm').
        - `current_above`: Mapping each support ('table', block) to the block currently directly on top of it ('nothing' if clear).
        - Identify the block currently held by the arm (if any).
        - Identify blocks that are currently clear.
        - Check if the arm is currently empty.
    2.  Initialize heuristic value `h = 0`.
    3.  Add penalty for blocks that are in the goal (`goal_blocks`) but are not on their correct goal position (on wrong support or held). For each block `B` in `goal_blocks`, if its current support is 'arm' or is different from `goal_on[B]`, add 1 to `h`.
    4.  Add penalty for blocks that are correctly placed on their goal support, but have the wrong block on top (or should be clear but aren't, or should have a block on top but don't). For each block `B` (excluding 'table'), if `B` is currently on its goal support (`current_on[B] == goal_on[B]`), but the block currently on top of `B` (`current_above[B]`) is different from the block that should be on top of `B` in the goal (`goal_above[B]`), add 1 to `h`.
    5.  Add penalty for unsatisfied `(clear X)` goals. For each block `X` that must be clear in the goal, if `(clear X)` is not true in the current state, add 1 to `h`.
    6.  Add penalty if `(arm-empty)` is a goal and the arm is not empty in the current state. Add 1 to `h`.
    7.  The total heuristic value is the sum of these penalties.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration and identifying blocks.
        """
        self.goals = task.goals
        self.goal_on = {} # Maps block -> goal_support ('table' or block)
        self.all_blocks = set()

        # Extract goal positions and identify all blocks from goals
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'on':
                block, support = parts[1], parts[2]
                self.goal_on[block] = support
                self.all_blocks.add(block)
                self.all_blocks.add(support)
            elif parts[0] == 'on-table':
                block = parts[1]
                self.goal_on[block] = 'table'
                self.all_blocks.add(block)
            # (clear X) and (arm-empty) goals are handled separately

        # Identify all blocks from initial state
        for fact in task.initial_state:
             parts = get_parts(fact)
             # Consider predicates that involve blocks
             if parts[0] in ['on', 'on-table', 'clear', 'holding']:
                 if len(parts) > 1: # Avoid issues with arm-empty
                    self.all_blocks.add(parts[1])
             # Assuming only block objects appear in these predicates

        # Remove 'table' if it was added as a block
        self.all_blocks.discard('table')

        # Identify blocks that have a specified goal position (on or on-table)
        self.goal_blocks = set(self.goal_on.keys())

        # Precompute goal_above map
        self.goal_above = {}
        # Initialize all blocks to have 'nothing' above them in goal by default
        for block in self.all_blocks:
             self.goal_above[block] = 'nothing'
        # Then override for blocks that should have something on them
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'on':
                block, support = parts[1], parts[2]
                if support != 'table': # 'table' is not a block support in goal_above map
                     self.goal_above[support] = block


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        current_on = {}
        current_above = {}
        is_holding = None
        current_clear = set()
        arm_is_empty = '(arm-empty)' in state

        # Build current_on and current_above maps, find held block and clear blocks
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                block, support = parts[1], parts[2]
                current_on[block] = support
                current_above[support] = block # Assuming only one block can be on top
            elif parts[0] == 'on-table':
                block = parts[1]
                current_on[block] = 'table'
            elif parts[0] == 'holding':
                 is_holding = parts[1]
                 current_on[is_holding] = 'arm' # Represent held block's location as 'arm'
            elif parts[0] == 'clear':
                 current_clear.add(parts[1])

        # For supports that are clear and nothing is mapped above them yet, mark current_above as 'nothing'
        # Iterate over all blocks and 'table' as potential supports
        all_supports = self.all_blocks | {'table'}
        for support in all_supports:
             if support in current_clear and support not in current_above:
                  current_above[support] = 'nothing'
             # Note: If a support is not clear and nothing is mapped above it,
             # it implies an (on Y support) fact exists but wasn't processed,
             # or the block Y is held. Assuming valid states where (clear X) is false iff (on Y X) is true.


        h = 0
        # Penalty 1: Blocks in goal_blocks not in their goal position (on wrong support or held)
        for block in self.goal_blocks:
            current_support = current_on.get(block, 'unknown') # 'unknown' if block isn't in state (shouldn't happen in valid states)

            if current_support == 'arm':
                 h += 1
            elif current_support != 'unknown' and current_support != self.goal_on[block]:
                 h += 1

        # Penalty 2: Wrong block on top of correctly placed block
        # Iterate over all blocks as potential supports (excluding 'table')
        for block_B in self.all_blocks:
             # Check if block_B is correctly placed relative to its support
             if current_on.get(block_B) == self.goal_on.get(block_B):
                  current_block_on_B = current_above.get(block_B, 'nothing')
                  goal_block_on_B = self.goal_above.get(block_B, 'nothing') # 'nothing' if nothing should be on it

                  if current_block_on_B != goal_block_on_B:
                       h += 1

        # Penalty 3: Unsatisfied (clear X) goals
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == 'clear':
                  block_to_be_clear = parts[1]
                  if f'(clear {block_to_be_clear})' not in state:
                       h += 1

        # Penalty 4: Unsatisfied (arm-empty) goal
        if '(arm-empty)' in self.goals and not arm_is_empty:
             h += 1

        return h
