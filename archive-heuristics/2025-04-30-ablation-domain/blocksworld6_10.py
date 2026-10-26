from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(on b1 b2)" -> ["on", "b1", "b2"]
    # Handle potential whitespace issues
    return fact[1:-1].strip().split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not in their correct
    position relative to the block directly below them, or are on the correct
    base but have the wrong block directly on top. It counts each such
    "misplaced" block as 1. The intuition is that each such block requires
    at least one operation (like unstack, pickup, or placing a block on it)
    to correct its local configuration, and these operations are often
    interdependent.

    # Assumptions
    - The goal state is defined by a set of (on X Y) and (on-table X) facts,
      implicitly defining stacks and positions on the table.
    - Blocks not explicitly mentioned in goal (on X Y) or (on-table X) facts
      are assumed to belong on the table and be clear in the goal state.
    - The heuristic considers the immediate relationship (base and direct top)
      for each block.
    - The cost of achieving the correct position for a block is assumed to be
      roughly proportional to the number of blocks that are currently in a
      "locally incorrect" configuration.

    # Heuristic Initialization
    - Parses the goal facts to build mappings:
        - `goal_base`: maps each block to the block it should be directly on top of, or 'table'.
        - `goal_top`: maps each block/table position to the block that should be directly on top of it, or 'clear' if nothing should be on it.
    - Identifies the set of all blocks mentioned in the goal.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the state facts to determine the current base and the current block
       directly on top for every block mentioned in the state. Also identify if
       any block is currently held by the arm.
        - `current_base`: maps each block to its current base (another block, 'table', or 'hand').
        - `current_top`: maps each block/table position to the block currently directly on top of it, or 'clear'.
    2. Identify all blocks relevant to the problem (those appearing in goals or the current state).
    3. Initialize the heuristic value `h` to 0.
    4. Iterate through each relevant block:
        a. Determine the block's goal base and goal top using the pre-calculated maps. If the block is not mentioned in the goal, its goal is assumed to be on the table and clear.
        b. Determine the block's current base and current top using the state maps (defaulting to 'clear' for current top if nothing is on it).
        c. If the block is currently held by the arm (`current_base == 'hand'`), increment `h`. This block is definitely not in its goal position.
        d. Else if the block's current base is different from its goal base (`current_base != goal_base`), increment `h`. The block is on the wrong base.
        e. Else (the block is on the correct base, `current_base == goal_base`), check the block directly on top:
            - If the block currently on top is different from the block that should be on top in the goal (`current_top != goal_top`), increment `h`. The block is correctly based but is blocked or should have something else on it.
    5. Return the total value of `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal base and top relationships.
        """
        self.goals = task.goals

        # Map block -> goal_base (block or 'table')
        self.goal_base = {}
        # Map block -> goal_top (block or 'clear')
        self.goal_top = {}
        # Set of all blocks mentioned in goals
        self.all_goal_blocks = set()

        # First pass: process 'on' and 'on-table' goals
        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == "on":
                block, base = parts[1], parts[2]
                self.goal_base[block] = base
                self.goal_top[base] = block # The base block 'base' should have 'block' on top
                self.all_goal_blocks.add(block)
                self.all_goal_blocks.add(base)
            elif predicate == "on-table":
                block = parts[1]
                self.goal_base[block] = 'table'
                self.all_goal_blocks.add(block)
            # (clear X) goals are implicitly handled by goal_top map

        # Second pass: identify blocks that should be clear in the goal
        # Any block mentioned in goals that is not a base for another block in a goal 'on' fact should be clear
        for block in list(self.all_goal_blocks):
             if block not in self.goal_top and block != 'table': # 'table' doesn't have a top in this map
                 self.goal_top[block] = 'clear'

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Map block -> current_base (block, 'table', or 'hand')
        current_base = {}
        # Map block -> current_top (block or 'clear')
        current_top = {}
        # Set of all blocks mentioned in the state
        state_blocks = set()
        held_block = None

        # First pass: process state facts to find bases and direct tops
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "on":
                block, base = parts[1], parts[2]
                current_base[block] = base
                current_top[base] = block # The base block 'base' currently has 'block' on top
                state_blocks.add(block)
                state_blocks.add(base)
            elif predicate == "on-table":
                block = parts[1]
                current_base[block] = 'table'
                state_blocks.add(block)
            elif predicate == "holding":
                block = parts[1]
                current_base[block] = 'hand'
                held_block = block
                state_blocks.add(block)
            # (clear X) and (arm-empty) are handled implicitly

        # Second pass: identify blocks that are currently clear
        # Any block mentioned in state that is not a base for another block in a state 'on' fact is clear
        for block in list(state_blocks):
             if block not in current_top:
                 current_top[block] = 'clear'

        # Combine all relevant blocks (those in goals or state)
        all_relevant_blocks = self.all_goal_blocks | state_blocks

        h = 0
        for block in all_relevant_blocks:
            # Determine goal state for this block (base and top)
            # If block is in all_relevant_blocks but NOT in all_goal_blocks, its goal is table/clear by default.
            if block in self.all_goal_blocks:
                # Block is explicitly mentioned in goals
                goal_b = self.goal_base.get(block) # Should exist if block in all_goal_blocks
                goal_t = self.goal_top.get(block, 'clear') # Default to 'clear' if not a base for another block
            else:
                # Block is in state but not in goals, assume goal is on-table and clear
                goal_b = 'table'
                goal_t = 'clear'

            # Get current state for this block (base and top)
            # If block is in all_relevant_blocks but not state_blocks, it's an issue with state representation.
            # Assuming all_relevant_blocks are present in state_blocks.
            current_b = current_base.get(block) # Should exist if block in state_blocks
            current_t = current_top.get(block, 'clear') # Default to 'clear' if nothing is on it

            if current_b == 'hand':
                # Block is held, it's definitely not in its goal position
                h += 1
            elif current_b != goal_b:
                # Block is on the wrong base
                h += 1
            else: # current_b == goal_b (Block is on the correct base)
                # Check if the block directly on top is correct
                if current_t != goal_t:
                    # Block is on correct base but has wrong block on top
                    h += 1

        return h
