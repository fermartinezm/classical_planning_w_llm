from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to extract parts of a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace and ensure it's a string
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
        # Not a valid PDDL fact string representation
        return []
    return fact[1:-1].split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    by counting the number of "misplaced adjacencies" in the current state
    compared to the goal state, plus a cost if the arm is holding a block.
    A block contributes to the heuristic if:
    1. It is currently on a different block or the table than it should be in the goal.
    2. The block currently on top of it is different from the block that should be on top of it in the goal.

    # Assumptions
    - The goal state consists of blocks stacked on other blocks or the table.
    - Any block not explicitly mentioned in an `(on ...)` or `(on-table ...)` goal predicate
      is assumed to have a goal of being on the table.
    - The heuristic counts deviations from the desired vertical structure.
    - Block names start with 'b'.

    # Heuristic Initialization
    - Parses the goal conditions to determine the desired base for each block
      (`goal_on_map`: block -> block_below or 'table') and the desired block
      on top of each block (`goal_block_on_top_map`: block_below -> block_on_top).
    - Identifies all blocks involved in the problem from the initial state and goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to determine:
       - Which block is on which other block (`current_on_map`).
       - Which blocks are on the table (`current_on_table_set`).
       - Which block (if any) the arm is holding (`current_holding_block`).
       - Which block is directly on top of which other block (`current_block_on_top_map`).
    2. Initialize the heuristic value `h` to 0.
    3. If the arm is holding a block, increment `h` by 1 (representing the cost to place it somewhere).
    4. Iterate through all blocks identified during initialization:
       - For each block `B`:
         - If `B` is currently being held, skip the base/top checks for this block as its contribution is already counted in step 3.
         - Determine `B`'s current base (the block it's on, or 'table').
         - Determine `B`'s goal base (the block it should be on, or 'table').
         - If the current base is not the goal base, increment `h` by 1.
         - Determine the block currently on top of `B`.
         - Determine the block that should be on top of `B` in the goal.
         - If the block currently on top is different from the goal block on top, increment `h` by 1.
    5. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and identifying all blocks.
        """
        self.goals = task.goals

        self.goal_on_map = {}  # block -> block_below (or 'table')
        self.goal_on_table_set = set() # {block}
        self.goal_block_on_top_map = {} # block_below -> block_on_top
        self.all_blocks = set() # Collect all block names

        # Parse goal facts
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip invalid facts

            predicate = parts[0]
            if predicate == "on":
                if len(parts) == 3:
                    block_on_top, block_below = parts[1], parts[2]
                    self.goal_on_map[block_on_top] = block_below
                    self.goal_block_on_top_map[block_below] = block_on_top
                    self.all_blocks.add(block_on_top)
                    self.all_blocks.add(block_below)
            elif predicate == "on-table":
                 if len(parts) == 2:
                    block = parts[1]
                    self.goal_on_table_set.add(block)
                    self.goal_on_map[block] = 'table'
                    self.all_blocks.add(block)
            # We ignore 'clear' goals as they are typically consequences of 'on' goals

        # Add any blocks from the initial state that weren't in the goals
        # This ensures we consider all blocks in the problem instance
        for fact in task.initial_state:
             parts = get_parts(fact)
             if not parts: continue
             predicate = parts[0]
             # Look for facts involving blocks (assuming block names start with 'b')
             if predicate in ["on", "on-table", "clear", "holding"]:
                 for obj in parts[1:]:
                     if obj.startswith('b'):
                         self.all_blocks.add(obj)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        current_on_map = {}  # block -> block_below
        current_on_table_set = set() # {block}
        current_holding_block = None
        current_block_on_top_map = {} # block_below -> block_on_top

        # Parse current state facts
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "on":
                if len(parts) == 3:
                    block_on_top, block_below = parts[1], parts[2]
                    current_on_map[block_on_top] = block_below
                    current_block_on_top_map[block_below] = block_on_top
            elif predicate == "on-table":
                if len(parts) == 2:
                    block = parts[1]
                    current_on_table_set.add(block)
            elif predicate == "holding":
                 if len(parts) == 2:
                    current_holding_block = parts[1]
            # We ignore 'clear' and 'arm-empty' for this heuristic calculation

        total_cost = 0

        # Cost for holding a block
        if current_holding_block is not None:
            total_cost += 1

        # Cost for misplaced adjacencies
        for block in self.all_blocks:
            # If the block is currently held, its base/top is not relevant in the stack structure
            # Its cost is accounted for by the 'holding' check above.
            if current_holding_block == block:
                continue

            # Determine current base
            current_base = None
            if block in current_on_table_set:
                current_base = 'table'
            else:
                # Find block_below in current_on_map
                current_base = current_on_map.get(block) # Will be None if not on another block or table/held

            # Determine goal base (default to 'table' if block not in goal_on_map)
            goal_base = self.goal_on_map.get(block, 'table')

            # Check if block is on the wrong base
            if current_base != goal_base:
                 total_cost += 1

            # Determine current block on top
            current_block_on_top = current_block_on_top_map.get(block) # Will be None if nothing on top

            # Determine goal block on top
            goal_block_on_top = self.goal_block_on_top_map.get(block) # Will be None if nothing should be on top

            # Check if the wrong block is on top
            if current_block_on_top != goal_block_on_top:
                 total_cost += 1

        return total_cost
