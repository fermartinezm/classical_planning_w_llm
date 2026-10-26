from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Utility function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to reach the goal state
    by counting blocks that are not in their goal position relative to their base,
    plus blocks that are stacked on top of blocks that are not in their goal
    position relative to their base. Each such block is considered 'out of place'
    and requires at least one move operation (pickup/unstack + putdown/stack)
    to potentially correct its position or clear a block below it.

    # Assumptions:
    - The goal state consists of blocks arranged in stacks or on the table,
      defined by `on` and `on-table` predicates.
    - Each block that appears in an `on` or `on-table` goal predicate has a unique goal base (either the table or another block).
    - The set of blocks is typically the same in the initial and goal states.

    # Heuristic Initialization
    - Extracts the goal base for each block (if specified by an `on` or `on-table` goal) from the goal conditions.
    - Identifies all blocks involved in the problem by collecting arguments from initial state and goal facts.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. **Identify all blocks:** Collect all unique block names mentioned in the initial state and the goal state.
    2. **Determine goal bases:** Parse the goal facts (`on` and `on-table`) to create a mapping from each block to its desired base (either 'table' or the block it should be on). Blocks only appearing in `clear` goals are not assigned a goal base by this step.
    3. **Determine current state:** Parse the current state facts (`on`, `on-table`, `holding`) to find the current base for each block (if any) and identify which block, if any, is being held.
    4. **Identify blocks with misplaced bases (`MisplacedBase`):** Iterate through all blocks. A block is considered to have a misplaced base if:
       - It is currently being held (and has a goal base defined).
       - It is not being held, has a goal base defined, has a current base defined (is on the table or another block), and its current base is different from its goal base.
    5. **Identify blocks stacked on misplaced blocks (`BlocksAboveMisplaced`):** Iterate through the current `on` relationships. If block X is currently on block Y, and Y is in the `MisplacedBase` set, then X is added to the `BlocksAboveMisplaced` set.
    6. **Calculate heuristic value:** The heuristic estimate is the sum of the number of blocks in `MisplacedBase` and the number of blocks in `BlocksAboveMisplaced`. This counts the blocks that are either in the wrong place relative to their base or are blocking a block that is in the wrong place relative to its base. Each such block generally requires at least one move operation.
    7. **Handle goal state:** If the current state is the goal state (all goal facts are true), the heuristic value is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal bases and identifying all blocks.
        """
        self.goals = task.goals

        self.goal_bases = {}
        self.all_blocks = set()

        # Collect all blocks and their goal bases from goal facts
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'on':
                block, base = parts[1], parts[2]
                self.goal_bases[block] = base
                self.all_blocks.add(block)
                self.all_blocks.add(base)
            elif parts[0] == 'on-table':
                block = parts[1]
                self.goal_bases[block] = 'table'
                self.all_blocks.add(block)
            # Arguments from 'clear' or 'arm-empty' goals are collected below

        # Collect blocks from initial state and any remaining from goals (e.g. only in clear goals)
        for fact in task.initial_state | self.goals:
             parts = get_parts(fact)
             # Consider predicates that take block arguments
             if parts[0] in ['on', 'on-table', 'clear', 'holding']:
                 for arg in parts[1:]:
                     if arg != 'arm-empty': # arm-empty is not a block object
                         self.all_blocks.add(arg)

    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # If the goal is reached, the heuristic is 0.
        # Use the task's built-in goal check for robustness.
        if self.goals <= state:
             return 0

        current_bases = {}
        current_on_map = {}
        holding_block = None

        # Determine current bases and the block being held
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                block, base = parts[1], parts[2]
                current_bases[block] = base
                current_on_map[block] = base
            elif parts[0] == 'on-table':
                block = parts[1]
                current_bases[block] = 'table'
            elif parts[0] == 'holding':
                holding_block = parts[1]

        # Identify blocks with misplaced bases
        misplaced_base = set()
        for block in self.all_blocks:
            goal_base = self.goal_bases.get(block)
            current_base = current_bases.get(block) # Will be None if block is held or not on table/another block

            # A block is misplaced if:
            # 1. It's the block being held (and has a goal base defined).
            # 2. It's not held, has a goal base defined, has a current base defined (is on the table or another block), and they don't match.
            # Blocks without a goal base (e.g. only appear in clear goals) are not considered misplaced by this logic.
            if goal_base is not None: # Only consider blocks that have a goal base
                if block == holding_block:
                     misplaced_base.add(block)
                elif current_base is not None and current_base != goal_base:
                    misplaced_base.add(block)
                # Note: If goal_base is not None but current_base is None (and not holding),
                # this implies an invalid state where a block is neither held nor on a base.
                # Assuming valid states, current_base will be non-None if not holding.


        # Identify blocks stacked on top of misplaced blocks
        blocks_above_misplaced = set()
        for block, base in current_on_map.items():
            if base in misplaced_base:
                blocks_above_misplaced.add(block)

        # The heuristic is the sum of blocks with misplaced bases and blocks above them.
        # This counts blocks that are 'out of place' in terms of their base or are
        # blocking a block that is 'out of place'.
        h = len(misplaced_base) + len(blocks_above_misplaced)

        return h
