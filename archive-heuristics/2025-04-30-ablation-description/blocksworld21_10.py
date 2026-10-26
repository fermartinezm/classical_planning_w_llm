# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and handle empty fact string
    if not isinstance(fact, str) or not fact:
        return []
    # Remove outer parentheses if they exist
    if fact.startswith('(') and fact.endswith(')'):
        fact = fact[1:-1]
    # Split the remaining string by whitespace
    parts = fact.split()
    return parts

# Assumes a base class exists as per example:
# from heuristics.heuristic_base import Heuristic

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic function for the Blocksworld domain.

    # Summary
    This heuristic estimates the cost to reach the goal by counting the number
    of blocks that are not on their correct immediate base (the block or table
    they should be directly on according to the goal state). It adds an
    additional penalty for each such misplaced block that currently has another
    block stacked on top of it, as this requires an extra unstacking operation
    before the misplaced block can be moved.

    # Assumptions:
    - The goal state defines a specific configuration of blocks on top of each other or the table.
    - The heuristic focuses on achieving the correct relative placement of blocks.
    - The cost of moving a block is simplified: 1 unit for being misplaced, plus 1 unit
      if it's blocked by another block on top. This is a relaxed cost estimate.
    - The heuristic assumes well-formed Blocksworld goals where every block
      mentioned in an 'on' or 'on-table' goal fact has a defined target base.

    # Heuristic Initialization
    - Parses the goal state (`task.goals`) to determine the desired immediate base
      for each block (either another block or the string 'table'). This mapping
      is stored in `self.goal_below`.
    - Identifies the set of all blocks whose goal position is explicitly defined
      in the goal state. This set is stored in `self.goal_blocks_with_position`.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state (`node.state`) is exactly the goal state (`self.goals`).
       If they are identical, the heuristic is 0, as the goal is reached.
    2. Parse the current state to determine the current position of each block:
       - Create a dictionary `current_below` mapping each block to its immediate base
         ('table', another block, or 'holding').
       - Create a dictionary `current_above` mapping each block/table to the block
         currently stacked directly on top of it (if any).
       - Identify the block (if any) currently held by the arm.
    3. Initialize the heuristic cost counter (`cost`) to 0.
    4. Iterate through each block whose goal position is defined (`self.goal_blocks_with_position`).
    5. For the current block:
       - Retrieve its desired immediate base (`goal_base`) from `self.goal_below`.
       - Determine its current immediate base (`current_base`) from the parsed state,
         explicitly checking if the block is currently being held.
    6. Compare the `current_base` and `goal_base`. If they are different:
       - Increment the `cost` by 1. This accounts for the need to move the block.
       - Check if there is any block currently stacked directly on top of the current block
         (using the `current_above` mapping).
       - If there is a block on top, increment the `cost` by an additional 1. This accounts
         for the necessary unstacking operation before the current block can be moved.
    7. After iterating through all relevant blocks, return the total calculated `cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the goal configuration.
        """
        super().__init__(task) # Call the base class constructor

        # Parse goal facts to build the desired immediate base for each block.
        # goal_below[block] = base_block or 'table'
        self.goal_below = {}

        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts and parts[0] == 'on':
                # Fact is like '(on b1 b2)'
                if len(parts) == 3:
                    block, base = parts[1], parts[2]
                    self.goal_below[block] = base
            elif parts and parts[0] == 'on-table':
                # Fact is like '(on-table b1)'
                if len(parts) == 2:
                    block = parts[1]
                    self.goal_below[block] = 'table'
            # Ignore 'clear' and 'arm-empty' goals for this heuristic's structure

        # The set of blocks whose goal position is explicitly defined.
        self.goal_blocks_with_position = set(self.goal_below.keys())


    def __call__(self, node):
        """
        Compute the heuristic estimate for the given state.
        """
        state = node.state

        # If the state is the goal state, the heuristic is 0.
        if state == self.goals:
            return 0

        # Parse the current state to find block positions and what's on top.
        current_below = {} # block -> base_block or 'table' or 'holding'
        current_above = {} # base_block or 'table' -> block_on_top (only one per base)
        is_holding = None

        for fact in state:
            parts = get_parts(fact)
            if not parts: # Skip empty or malformed facts
                continue

            if parts[0] == 'on':
                # Fact is like '(on b1 b2)'
                if len(parts) == 3:
                    block, base = parts[1], parts[2]
                    current_below[block] = base
                    current_above[base] = block # Only one block can be directly on top
            elif parts[0] == 'on-table':
                # Fact is like '(on-table b1)'
                if len(parts) == 2:
                    block = parts[1]
                    current_below[block] = 'table'
            elif parts[0] == 'holding':
                # Fact is like '(holding b1)'
                if len(parts) == 2:
                    is_holding = parts[1]
                    # We don't set current_below[is_holding] here because
                    # it's not 'on' anything or 'on-table'. We handle 'holding' explicitly.
            # Ignore 'clear' and 'arm-empty' facts for this heuristic's structure

        cost = 0

        # Iterate through blocks whose goal position is defined.
        for block in self.goal_blocks_with_position:
            goal_base = self.goal_below[block] # Guaranteed to exist by goal_blocks_with_position definition

            # Determine the block's current base.
            # It's either on something, on the table, or being held.
            current_base = current_below.get(block) # Check if it's on something or on-table

            if is_holding == block:
                 current_base = 'holding' # Override if the block is being held

            # If the block is not on its correct immediate base
            if current_base != goal_base:
                cost += 1 # Cost for being misplaced

                # Check if the block is currently blocked by another block on top.
                # A block is blocked if something is on top of it in the current state.
                # We only care if the block itself is the base of another block.
                if block in current_above:
                    cost += 1 # Additional cost for needing to unstack something from it

        return cost
