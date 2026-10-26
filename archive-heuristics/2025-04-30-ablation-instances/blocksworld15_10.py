from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or invalid format defensively
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed by counting the number
    of blocks that are not in their correct final position relative to the goal
    configuration. A block is in its final position if it is on the table
    and the goal requires it to be on the table, OR if it is on another block
    that is in its final position and the goal requires it to be on that block.

    # Assumptions
    - The goal specifies the desired position (on another block or on the table)
      for a subset of blocks.
    - All blocks mentioned in the goal are present in the initial state.
    - The goal state typically requires specific stacks of blocks and potentially
      other blocks on the table.
    - The heuristic value is 0 if and only if the state is a goal state.

    # Heuristic Initialization
    - Extract the goal configuration from the task's goal facts. This involves
      mapping each block to its target location (another block or 'table')
      as specified in the goal.
    - Identify the set of all blocks that are part of the goal configuration.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Extract Current Configuration:
       - Determine the current position of every block that is relevant to the goal:
         whether it's on another block, on the table, or being held by the arm.
         Store this in a dictionary mapping blocks to their current base
         ('table', 'holding', or another block). Facts about blocks not
         involved in the goal configuration are ignored.

    2. Identify Blocks in Final Position:
       - Initialize a set `in_final_pos` to store blocks that are correctly placed
         according to the goal configuration and the positions of blocks below them.
       - Add all blocks `B` to `in_final_pos` if the goal requires `B` to be on the
         table (`(on-table B)` is a goal fact) AND `B` is currently on the table
         (`(on-table B)` is true in the current state).
       - Iteratively add blocks `A` to `in_final_pos` if the goal requires `A` to be
         on block `B` (`(on A B)` is a goal fact) AND `A` is currently on `B`
         (`(on A B)` is true in the current state) AND `B` is already in the
         `in_final_pos` set. Repeat this process until no new blocks are added
         in an iteration. This ensures that blocks are considered correctly placed
         only if the entire stack below them (down to the table) is also correctly placed
         according to the goal.

    3. Compute Heuristic Value:
       - The heuristic value is the total number of blocks that are part of the
         goal configuration minus the number of blocks identified as being in their
         final position. This counts the number of blocks that are not yet part
         of the correctly built goal structure.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the goal configuration.
        """
        self.goals = task.goals  # Goal conditions.
        # Static facts are not needed for this heuristic in Blocksworld.
        # static_facts = task.static

        # Map blocks to their target location ('table' or another block) from goal facts.
        self.goal_config = {}
        # Set of all blocks that are mentioned in the goal configuration.
        self.goal_blocks = set()

        for goal_fact_str in self.goals:
            parts = get_parts(goal_fact_str)
            if not parts: # Skip malformed facts
                continue
            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, base = parts[1], parts[2]
                self.goal_config[block] = base
                self.goal_blocks.add(block)
                self.goal_blocks.add(base) # Add the base block too
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                self.goal_config[block] = 'table'
                self.goal_blocks.add(block)

        # Remove 'table' from goal_blocks as it's not a block object
        self.goal_blocks.discard('table')


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        based on the number of blocks not in their final position.
        """
        state = node.state  # Current world state (frozenset of fact strings).

        # 1. Extract Current Configuration
        current_config = {} # Maps block to its base ('table', 'holding', or another block)
        # We only care about blocks that are part of the goal configuration
        # and are explicitly mentioned in the state facts about position/holding.
        # Blocks not mentioned in on/on-table/holding facts are not relevant for current position.
        # (Although in blocksworld, all blocks are always in one of these states).
        for fact_str in state:
            parts = get_parts(fact_str)
            if not parts: # Skip malformed facts
                continue
            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, base = parts[1], parts[2]
                # Only track position for blocks relevant to the goal or blocks they are on
                if block in self.goal_blocks or base in self.goal_blocks:
                     current_config[block] = base
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                if block in self.goal_blocks:
                    current_config[block] = 'table'
            elif predicate == "holding" and len(parts) == 2:
                 block = parts[1]
                 if block in self.goal_blocks:
                     current_config[block] = 'holding'

        # 2. Identify Blocks in Final Position
        in_final_pos = set()

        # Add blocks correctly on the table according to the goal
        # Iterate over goal_config to find blocks whose goal is 'table'
        for block, target in self.goal_config.items():
            if target == 'table':
                # Check if the block is currently on the table
                # Use .get() to handle blocks that might not be in current_config (e.g., malformed state)
                if current_config.get(block) == 'table':
                    in_final_pos.add(block)

        # Iteratively add blocks correctly stacked
        newly_added = True
        while newly_added:
            newly_added = False
            # Iterate over goal_config to find blocks whose target is another block
            for block, target in self.goal_config.items():
                 # Check if the target is a block (not 'table') and is part of the goal blocks
                 # Check if the target block is already identified as being in its final position
                 if target in self.goal_blocks and target in in_final_pos:
                    # Check if the current block is currently on the target block
                    # and if the current block is not already in final_pos
                    # Use .get() for current_config lookup
                    if current_config.get(block) == target and block not in in_final_pos:
                        in_final_pos.add(block)
                        newly_added = True

        # 3. Compute Heuristic Value
        # The heuristic is the number of goal blocks that are NOT in their final position.
        # This is equivalent to total goal blocks minus those in final position.
        # We only consider blocks that are explicitly part of the goal configuration.
        h = len(self.goal_blocks) - len(in_final_pos)

        # Ensure heuristic is non-negative (should be guaranteed by logic)
        return max(0, h)
