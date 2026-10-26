from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact string."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the cost by counting the number of blocks that are
    either not on their correct support block/table according to the goal,
    or are on their correct support but have the wrong block stacked directly on top of them.
    This captures structural differences between the current state and the goal configuration.

    # Assumptions
    - The goal is a conjunction of (on X Y) and (on-table Z) facts defining a specific configuration of blocks.
    - The heuristic only considers blocks that are mentioned in the goal.
    - The state representation is valid (each block is either on another block, on the table, or held).

    # Heuristic Initialization
    - Parses the goal conditions (`self.goals`) to determine the desired support (block or table) for each block
      and the desired block that should be directly on top of each block.
    - Stores this goal configuration in dictionaries (`self.goal_pos`, `self.goal_above`).
    - Identifies all blocks involved in the goal (`self.goal_blocks`).

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value `h` to 0.
    2. Parse the current state (`node.state`) to determine the current support (block, table, or held) for each block
       and the block currently stacked directly on top of each block. Store this in dictionaries
       (`current_pos`, `current_above`). This parsing focuses only on blocks identified in the goal during initialization.
    3. Iterate through each block `B` that is involved in the goal configuration (`self.goal_blocks`).
    4. For each block `B`:
       a. Retrieve its current position (`current_pos.get(B)`) and its goal position (`self.goal_pos.get(B)`).
       b. If the current position is different from the goal position (`current_pos.get(B) != self.goal_pos.get(B)`),
          increment the heuristic value `h` by 1.
       c. If the current position is the same as the goal position, retrieve the block currently on top of `B`
          (`current_above.get(B)`) and the block that should be on top of `B` according to the goal
          (`self.goal_above.get(B)`).
       d. If the block currently on top is different from the goal block on top (`current_above.get(B) != self.goal_above.get(B)`),
          increment the heuristic value `h` by 1.
    5. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration.
        """
        self.goals = task.goals

        # Determine all blocks involved in the goal
        self.goal_blocks = set()
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts[0] in ['on', 'on-table']:
                 # Arguments at index 1 and potentially 2 are blocks
                 self.goal_blocks.add(parts[1])
                 if len(parts) > 2:
                     self.goal_blocks.add(parts[2])

        # Build goal configuration maps
        # goal_pos: block -> block | 'table'
        # goal_above: block -> block | None (None means nothing should be on it)
        self.goal_pos = {}
        self.goal_above = {block: None for block in self.goal_blocks}

        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            predicate = parts[0]
            if predicate == 'on':
                block_on = parts[1]
                block_under = parts[2]
                self.goal_pos[block_on] = block_under
                self.goal_above[block_under] = block_on
            elif predicate == 'on-table':
                block_on_table = parts[1]
                self.goal_pos[block_on_table] = 'table'
            # Ignore other potential goal predicates like (arm-empty) if they exist

        # Note: goal_pos should be defined for all blocks in goal_blocks
        # if the goal is a valid blocksworld configuration.

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # Build current configuration maps from the state for goal blocks
        # current_pos: block -> block | 'table' | 'held'
        # current_above: block -> block | None (None means it is clear)
        current_pos = {}
        current_above = {block: None for block in self.goal_blocks}

        # Parse state facts to populate current_pos and current_above
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'on':
                block_on = parts[1]
                block_under = parts[2]
                # Only track positions/above for blocks relevant to the goal
                if block_on in self.goal_blocks:
                    current_pos[block_on] = block_under
                if block_under in self.goal_blocks:
                    current_above[block_under] = block_on
            elif predicate == 'on-table':
                block_on_table = parts[1]
                if block_on_table in self.goal_blocks:
                    current_pos[block_on_table] = 'table'
            elif predicate == 'holding':
                block_held = parts[1]
                if block_held in self.goal_blocks:
                    current_pos[block_held] = 'held'
            # (clear X) facts are implicitly handled by current_above initialization to None

        # Compute heuristic
        h = 0
        for block in self.goal_blocks:
            # Get current position, default to None if block somehow not found in state facts (shouldn't happen in valid states)
            current_p = current_pos.get(block)
            goal_p = self.goal_pos.get(block) # Should always exist for goal_blocks

            # Compare current position to goal position
            if current_p != goal_p:
                h += 1
            else: # current_p == goal_p
                # If position is correct, check if the block above is correct
                current_ab = current_above.get(block)
                goal_ab = self.goal_above.get(block)
                if current_ab != goal_ab:
                    h += 1

        return h
