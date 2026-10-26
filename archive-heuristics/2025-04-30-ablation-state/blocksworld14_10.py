from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    fact_str = str(fact).strip()
    if fact_str.startswith('(') and fact_str.endswith(')'):
        return fact_str[1:-1].split()
    else:
        # Handle simple atomic facts like "arm-empty"
        return [fact_str] if fact_str else []

# The heuristic class
class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed by counting
    the number of blocks that are not in their correct goal position
    relative to the block below them, plus the number of blocks that
    are in the correct position below but have the wrong block on top,
    plus one if the arm is holding a block. This heuristic aims to
    capture the effort required to place blocks correctly and to clear
    incorrectly placed blocks that are blocking others.

    # Assumptions
    - Standard Blocksworld actions (pickup, putdown, stack, unstack) with cost 1.
    - The goal specifies the desired stack configurations using `on` and `on-table` predicates.
    - All blocks mentioned in the goal are present in the initial state.
    - The heuristic does not need to be admissible.

    # Heuristic Initialization
    - Parses the goal conditions (`task.goals`) to build mappings representing the desired
      block configuration:
      - `goal_config`: Maps each block to the block it should be directly on top of,
        or 'table' if it should be on the table.
      - `goal_above`: Maps each block to the block that should be directly on top of it,
        or None if nothing should be on top.
    - Identifies the set of all blocks involved in the goal configuration (`goal_blocks`).
    - Static facts (`task.static`) are not used as Blocksworld has no relevant static facts
      for this heuristic.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state (`node.state`):
    1. Parse the current state to determine the current block configuration:
       - `current_config`: Maps each block to the block it is currently directly on top of,
         or 'table' if it is on the table.
       - `current_above`: Maps each block to the block that is currently directly on top of it,
         or None if it is clear.
       - `holding_block`: Identifies the block currently held by the arm, or None if the arm is empty.
    2. Initialize the heuristic value `h` to 0.
    3. Iterate through each block that is part of the goal configuration (`self.goal_blocks`):
       a. Retrieve the block's required support (`below_goal`) from `self.goal_config`.
       b. Retrieve the block's current support (`below_current`) from `current_config`. Use `.get()`
          to handle cases where a goal block might not be found in `current_config` (e.g., if held).
       c. If `below_current` does not match `below_goal`, increment `h`. This block is in the wrong place.
       d. If `below_current` *does* match `below_goal` (the block is on the correct support),
          retrieve the block that should be on top of it in the goal (`above_goal`) from `self.goal_above`.
          Retrieve the block currently on top of it (`above_current`) from `current_above`. Use `.get()`.
          If `above_current` does not match `above_goal`, increment `h`. This block is correctly placed below,
          but is either blocked by the wrong block or should be clear but isn't.
    4. If the robot's arm is currently holding a block (`holding_block is not None`), increment `h`.
       This accounts for the action needed to place the held block.
    5. Return the final value of `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration.
        """
        super().__init__(task)

        self.goal_config = {}  # block -> block_below_it or 'table'
        self.goal_above = {}   # block -> block_above_it or None
        self.goal_blocks = set() # Set of all blocks involved in goal stacks

        # First pass to populate goal_config and identify all goal blocks
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'on':
                block, below = parts[1], parts[2]
                self.goal_config[block] = below
                self.goal_blocks.add(block)
                self.goal_blocks.add(below)
            elif parts and parts[0] == 'on-table':
                block = parts[1]
                self.goal_config[block] = 'table'
                self.goal_blocks.add(block)

        # Second pass to populate goal_above based on goal_config
        # Initialize all goal blocks to have nothing above them in the goal
        for block in list(self.goal_blocks): # Iterate over a copy as we might add 'table'
             if block != 'table': # 'table' cannot have something above it
                 self.goal_above[block] = None

        # Fill in goal_above based on 'on' relationships
        for block, below in self.goal_config.items():
            if below != 'table':
                self.goal_above[below] = block # 'block' is above 'below'

        # Ensure 'table' is not in goal_blocks set (it might have been added as a 'below' object)
        self.goal_blocks.discard('table')


    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state

        current_config = {}  # block -> block_below_it or 'table'
        current_above = {}   # block -> block_above_it or None
        holding_block = None

        # Populate maps and find held block
        for fact in state:
            parts = get_parts(fact)
            if parts:
                predicate = parts[0]
                if predicate == 'on':
                    block, below = parts[1], parts[2]
                    current_config[block] = below
                    current_above[below] = block
                elif predicate == 'on-table':
                    block = parts[1]
                    current_config[block] = 'table'
                elif predicate == 'holding':
                    block = parts[1]
                    holding_block = block
                # 'clear' and 'arm-empty' predicates don't define block relationships directly needed here

        h = 0

        # Count blocks not in correct position relative to below OR with wrong block above
        for block in self.goal_blocks:
            below_goal = self.goal_config.get(block)
            below_current = current_config.get(block) # Use .get() in case block is not in state

            if below_current != below_goal:
                h += 1
            else: # below_current == below_goal
                above_goal = self.goal_above.get(block)
                above_current = current_above.get(block) # Use .get()

                if above_current != above_goal:
                    h += 1

        # Add cost if arm is holding a block
        if holding_block is not None:
            h += 1

        return h
