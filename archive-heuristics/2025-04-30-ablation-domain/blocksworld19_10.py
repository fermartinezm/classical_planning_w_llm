from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to reach the goal state
    by counting blocks that are not in their correct goal position relative to
    their support, adding a penalty for blocks that are stacked on top of
    misplaced blocks, and adding a penalty if the arm state does not match
    the explicit goal arm state.

    # Assumptions
    - The goal state defines specific stacks of blocks and blocks on the table.
    - The goal state explicitly specifies the desired arm state (either arm-empty or holding a specific block), or implies arm-empty if neither is specified.
    - All blocks mentioned in the goal are relevant. Any other blocks present in the state are considered misplaced.

    # Heuristic Initialization
    - Extract the goal configuration, specifically which block should be on which other block or on the table.
    - Identify the desired state of the arm (empty or holding a specific block) if explicitly stated in the goal.
    - Collect the set of all blocks mentioned in the goal.
    - Static facts are ignored as the Blocksworld domain typically has none.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Parse Goal:
       - Create a map `goal_support` where `goal_support[block]` is the block (or 'table') that `block` should be directly on in the goal.
       - Identify if the goal explicitly requires `(arm-empty)` or `(holding ?block)`.
       - Collect all blocks mentioned in the goal into `all_goal_blocks`.

    2. Parse Current State:
       - Create a map `current_support` where `current_support[block]` is the block (or 'table' or 'arm') that `block` is currently directly on.
       - Create a map `current_on_map` where `current_on_map[block_below]` is the block currently on top of `block_below`.
       - Identify the current arm state (empty or holding a specific block).
       - Collect all blocks present in the current state into `all_current_blocks`.

    3. Identify Misplaced Blocks:
       - A block is considered "misplaced" if it is part of the goal configuration (`goal_support` has an entry for it) but its current support (`current_support`) does not match its goal support.
       - Any block present in the current state (`all_current_blocks`) but *not* part of the goal configuration (`goal_support` does *not* have an entry for it) is also considered misplaced.
       - Collect all such blocks into a set `misplaced_support_blocks`.

    4. Calculate Base Heuristic:
       - The initial heuristic value is the total count of blocks in the `misplaced_support_blocks` set.

    5. Add Blocking Penalty:
       - Iterate through all blocks `block_below` that currently have a block `block_on_top` on them (using `current_on_map`).
       - If `block_on_top` is in the `misplaced_support_blocks` set, it means `block_on_top` is not in its correct position.
       - Being on top of another block when it's misplaced adds a penalty, as it likely needs to be moved. Add 1 to the heuristic for each such `block_on_top`.

    6. Add Arm State Penalty:
       - If the current arm state contradicts an *explicit* goal fact about the arm, add 1 to the heuristic.
       - Specifically:
         - If the goal requires `(arm-empty)` but the arm is currently holding a block: +1.
         - If the goal requires `(holding X)` but the arm is currently empty: +1.
         - If the goal requires `(holding X)` but the arm is currently holding a different block Y: +1.

    7. Return Total Heuristic:
       - The final heuristic value is the sum of the base heuristic, the blocking penalty, and the arm state penalty.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal configuration."""
        self.goals = task.goals  # Goal conditions as a frozenset of strings

        # Parse goal facts to build goal_support map and identify goal arm state
        self.goal_support = {} # block -> support_block_or_table
        self.all_goal_blocks = set()
        self.goal_holding_block = None
        self.goal_arm_empty_specified = False

        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                x, y = parts[1], parts[2]
                self.goal_support[x] = y
                self.all_goal_blocks.add(x)
                self.all_goal_blocks.add(y)
            elif predicate == 'on-table' and len(parts) == 2:
                x = parts[1]
                self.goal_support[x] = 'table'
                self.all_goal_blocks.add(x)
            elif predicate == 'holding' and len(parts) == 2:
                self.goal_holding_block = parts[1]
                self.all_goal_blocks.add(self.goal_holding_block)
            elif predicate == 'arm-empty' and len(parts) == 1:
                self.goal_arm_empty_specified = True

        # Static facts are ignored for this domain
        # static_facts = task.static

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings)

        # Parse current state
        current_support = {} # block -> support_block_or_table_or_arm
        current_on_map = {} # block_below -> block_on_top
        current_holding_block = None
        current_arm_empty = False
        all_current_blocks = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                x, y = parts[1], parts[2]
                current_support[x] = y
                current_on_map[y] = x
                all_current_blocks.add(x)
                all_current_blocks.add(y)
            elif predicate == 'on-table' and len(parts) == 2:
                x = parts[1]
                current_support[x] = 'table'
                all_current_blocks.add(x)
            elif predicate == 'holding' and len(parts) == 2:
                current_holding_block = parts[1]
                current_support[current_holding_block] = 'arm'
                all_current_blocks.add(current_holding_block)
            elif predicate == 'arm-empty' and len(parts) == 1:
                current_arm_empty = True

        # Combine blocks from goal and state
        all_relevant_blocks = self.all_goal_blocks | all_current_blocks

        # Identify blocks that are NOT in their goal position relative to their support
        misplaced_support_blocks = set()
        for block in all_relevant_blocks:
            goal_supp = self.goal_support.get(block)
            current_supp = current_support.get(block) # None if block is not on/on-table/holding

            # Case 1: Block is part of the goal configuration
            if goal_supp is not None:
                if current_supp != goal_supp:
                    misplaced_support_blocks.add(block)
            # Case 2: Block is NOT part of the goal configuration but is in the current state
            # If a block exists in the state but is not mentioned in the goal configuration, it's misplaced.
            elif block in all_current_blocks:
                 misplaced_support_blocks.add(block)


        # Heuristic starts with the count of blocks with misplaced support
        h = len(misplaced_support_blocks)

        # Add penalty for blocks that are on top of blocks that are themselves misplaced
        # Iterate through current 'on' facts (block_on_top is on block_below)
        for block_below, block_on_top in current_on_map.items():
            # If the block on top is in the set of misplaced blocks, add a penalty
            if block_on_top in misplaced_support_blocks:
                h += 1 # Add penalty for the block_on_top being on something when it's misplaced

        # Add penalty for arm state mismatch with explicit goal arm state
        if current_holding_block: # Arm is holding something
            if self.goal_arm_empty_specified:
                # Goal wants arm empty, but it's holding something
                h += 1
            elif self.goal_holding_block and current_holding_block != self.goal_holding_block:
                 # Goal wants arm holding a specific block, but it's holding a different one
                 h += 1
        elif current_arm_empty: # Arm is empty
            if self.goal_holding_block:
                # Goal wants arm holding a block, but it's empty
                h += 1

        return h
