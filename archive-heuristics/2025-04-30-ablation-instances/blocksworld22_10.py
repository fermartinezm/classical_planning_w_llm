from fnmatch import fnmatch
# Assuming the Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by counting blocks that are not in their correct goal position relative to
    the block below them (or the table), and adding a penalty for blocks that
    are obstructing a correctly placed block. Each misplaced block or obstruction
    is estimated to cost 2 actions (e.g., unstack/pickup + stack/putdown).

    # Assumptions
    - The goal state specifies the desired position (on another block or on the table)
      for a subset of blocks. Blocks not mentioned in the goal are not explicitly
      considered in the base count but may contribute to the obstruction penalty.
    - Moving a block typically requires picking it up and putting it down/stacking it,
      costing at least 2 actions. Clearing an obstructing block also costs actions.
    - The heuristic does not explicitly model the arm state or clear predicates,
      assuming that necessary clearing/arm movements are implicitly covered by
      the estimated cost of moving misplaced/obstructing blocks.

    # Heuristic Initialization
    - The heuristic extracts the goal configuration from the task's goal conditions.
    - It builds a map `goal_below` indicating which block should be immediately below
      each block in the goal stacks, and a set `goal_on_table` for blocks that
      should be directly on the table.
    - It also builds an inverse map `goal_above` to quickly find the block that
      should be on top of a given block in the goal.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state to determine the immediate support for each block
       (`current_below`, `current_on_table`, `current_holding`).
    2. Initialize the heuristic value `h` to 0.
    3. Count cost for blocks not in their correct goal position relative to their support:
       - Iterate through each block `B` that is part of the goal structure (i.e.,
         mentioned in `goal_below` or `goal_on_table`).
       - Determine the goal support (`goal_sup`) for block `B`.
       - Determine the current support (`current_sup`) for block `B`
         (is it on the table, on another block, or being held?).
       - If `current_sup` is different from `goal_sup`, it means block `B` is
         in the wrong place relative to its intended support. Add 2 to `h`
         (estimating the cost of moving this block).
    4. Count cost for blocks that are obstructing a correctly placed block:
       - Iterate through each fact `(on X Y)` in the current state, where `X` is
         on top of `Y`.
       - Check if `Y` is part of the goal structure as a block that should be
         supported (i.e., `Y` is a value in `goal_below` or in `goal_on_table`).
         If not, this stack is irrelevant to goal placement, skip.
       - If `Y` is in the goal structure, check if `Y` is currently in its correct
         goal position relative to its own support (i.e., find `Y`'s current support
         and compare it to `Y`'s goal support).
       - If `Y` is correctly supported, check if the block currently on top of it, `X`,
         is the block that *should* be on top of `Y` in the goal state (`goal_above[Y]`).
       - If `X` is not the correct block that should be on `Y` in the goal, it means
         `X` is obstructing a correctly placed block `Y`. Add 2 to `h` (estimating
         the cost of moving the obstructing block `X`).
    5. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the goal configuration.
        """
        self.goals = task.goals

        # Build goal stack structure
        self.goal_below = {}  # Maps block -> block it should be on (or 'table')
        self.goal_above = {}  # Maps block -> block that should be on it
        self.goal_on_table = set() # Set of blocks that should be on the table

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block_on_top, block_below = parts[1], parts[2]
                self.goal_below[block_on_top] = block_below
                self.goal_above[block_below] = block_on_top
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                self.goal_on_table.add(block)
            # Ignore other goal predicates like (clear) or (arm-empty) for this heuristic

        # Identify all blocks involved in the goal (either as a block or a support)
        self.goal_blocks = set(self.goal_below.keys()) | set(self.goal_above.keys()) | self.goal_on_table
        # Remove 'table' from goal_blocks if it was added from goal_below values
        self.goal_blocks.discard('table')


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse current state structure
        current_below = {}  # Maps block -> block it is currently on
        current_on_table = set() # Set of blocks currently on the table
        current_holding = set() # Set of blocks currently held by the arm
        # current_clear = set() # Not explicitly used in this heuristic logic

        # Pre-parse state facts for quick lookup
        state_facts_map = {}
        for fact in state:
             parts = get_parts(fact)
             if not parts: continue
             predicate = parts[0]
             if predicate not in state_facts_map:
                 state_facts_map[predicate] = []
             state_facts_map[predicate].append(parts)

        # Populate current position maps
        if 'on' in state_facts_map:
            for parts in state_facts_map['on']:
                 if len(parts) == 3:
                     current_below[parts[1]] = parts[2]
        if 'on-table' in state_facts_map:
             for parts in state_facts_map['on-table']:
                 if len(parts) == 2:
                     current_on_table.add(parts[1])
        if 'holding' in state_facts_map:
             for parts in state_facts_map['holding']:
                 if len(parts) == 2:
                     current_holding.add(parts[1])
        # if 'clear' in state_facts_map:
        #      for parts in state_facts_map['clear']:
        #          if len(parts) == 2:
        #              current_clear.add(parts[1])


        h = 0

        # 1. Count cost for blocks not in their correct goal position relative to their support
        for block in self.goal_blocks:
            goal_sup = self.goal_below.get(block, 'table' if block in self.goal_on_table else None)

            # Find current support
            current_sup = None
            if block in current_holding:
                current_sup = 'arm'
            elif block in current_on_table:
                current_sup = 'table'
            elif block in current_below:
                current_sup = current_below[block]
            # If block is not in any known position, it's implicitly misplaced if it has a goal_sup

            if goal_sup is not None and current_sup != goal_sup:
                 h += 2 # Estimate 2 actions (pickup/unstack + putdown/stack) to fix this block's support

        # 2. Count cost for blocks that are obstructing a correctly placed block
        # Iterate through blocks Y that are potential supports in the goal structure
        # Check blocks that are currently supporting something
        if 'on' in state_facts_map:
            for parts in state_facts_map['on']:
                if len(parts) == 3:
                    X, Y = parts[1], parts[2] # X is on Y

                    # Check if Y is part of the goal structure as a block that should be supported
                    goal_sup_Y = self.goal_below.get(Y, 'table' if Y in self.goal_on_table else None)

                    if goal_sup_Y is not None: # Y is a block that should be in the goal structure
                        # Check if Y is correctly supported in the current state
                        current_sup_Y = None
                        if Y in current_holding: # Y cannot be holding something if X is on Y
                            pass # Should not happen
                        elif Y in current_on_table:
                            current_sup_Y = 'table'
                        elif Y in current_below:
                            current_sup_Y = current_below[Y]

                        if current_sup_Y == goal_sup_Y: # Y is correctly supported
                            # Check if X is the block that should be on Y in the goal
                            goal_block_above_Y = self.goal_above.get(Y)

                            if X != goal_block_above_Y: # X is not the correct block
                                h += 2 # Estimate 2 actions to move the obstructing block X

        return h
