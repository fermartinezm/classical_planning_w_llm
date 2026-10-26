from heuristics.heuristic_base import Heuristic

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
    This heuristic estimates the distance to the goal by counting the number of blocks
    that are not in their correct final position within their goal stack. A block is
    considered correctly placed if it is on the correct block (or the table) according
    to the goal, and the block below it is also correctly placed, recursively down
    to the table.

    # Assumptions
    - The goal specifies a configuration of blocks forming one or more stacks on the table.
    - The goal predicates primarily define the `on` and `on-table` relationships for the final configuration.
    - Every block involved in an `on` or `on-table` goal predicate is considered part of a goal stack structure.
    - The state representation is a frozenset of PDDL fact strings.
    - Only one block can be held at a time.

    # Heuristic Initialization
    - Parse the goal conditions (`task.goals`) to build the target stack structure. This involves identifying
      for each block `B` that appears in an `on` or `on-table` goal, what object (`UnderB` or 'table') it should be directly on top of.
    - Store this goal structure (e.g., in a dictionary `self.goal_below` mapping a block to the object it should be on).
    - Identify the set of all blocks (`self.goal_blocks`) that are part of this goal configuration (i.e., appear in an `on` or `on-table` goal).
    - Static facts (`task.static`) are not used as they are not relevant to the block positions in Blocksworld.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state (`node.state`) to determine the immediate support for each block. Create a dictionary `current_below` mapping each block to the object it is currently on (another block, 'table', or 'arm' if held).
    2. Define a recursive helper function, `is_correctly_placed(block)`, which determines if a block is in its final goal position relative to the table. This function uses memoization to store results for blocks already checked.
       - Base case: If the block's goal is to be on the table (`self.goal_below.get(block) == 'table'`), it is correctly placed if it is currently on the table (`current_below.get(block) == 'table'`).
       - Recursive case: If the block's goal is to be on another block `UnderB` (`self.goal_below.get(block) == UnderB`), it is correctly placed if it is currently on `UnderB` (`current_below.get(block) == UnderB`) AND `UnderB` is also correctly placed (recursive call `is_correctly_placed(UnderB)`).
       - If the block is not found in `current_below` (e.g., it's held or missing from the state) or if its current support doesn't match the goal support, it is not correctly placed.
    3. Initialize the heuristic value `h` to 0.
    4. Iterate through all blocks that are part of the goal configuration (`self.goal_blocks`).
    5. For each block, call `is_correctly_placed(block)`. If it returns False, increment `h`.
    6. Return the final value of `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the goal stack structure.
        """
        self.goals = task.goals

        # Build the goal stack structure: block -> object_below (or 'table')
        self.goal_below = {}
        self.goal_blocks = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip invalid facts

            predicate = parts[0]
            if predicate == 'on':
                if len(parts) == 3:
                    block, under_block = parts[1], parts[2]
                    self.goal_below[block] = under_block
                    self.goal_blocks.add(block)
                    self.goal_blocks.add(under_block)
            elif predicate == 'on-table':
                 if len(parts) == 2:
                    block = parts[1]
                    self.goal_below[block] = 'table' # Use a special string 'table'
                    self.goal_blocks.add(block)

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Build the current state structure: block -> object_below (or 'table' or 'arm')
        current_below = {}
        # Find held block first, as it's not on anything
        held_block = None
        for fact in state:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] == 'holding' and len(parts) == 2:
                 held_block = parts[1]
                 current_below[held_block] = 'arm' # Use a special string 'arm'
                 # Assuming only one block can be held in Blocksworld
                 break

        # Find blocks on other blocks or table
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block, under_block = parts[1], parts[2]
                current_below[block] = under_block
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                current_below[block] = 'table'


        # Compute correctly placed blocks using memoization
        correctly_placed = {} # Memoization dict: block -> True/False

        def is_correctly_placed(block):
            # If result is already computed, return it
            if block in correctly_placed:
                return correctly_placed[block]

            # A block is correctly placed if its current support matches the goal support
            # AND (if on another block) the block below is also correctly placed.

            is_correct = False
            goal_under = self.goal_below.get(block) # Get the object it should be on

            # Get the current object the block is on (or 'arm' or None if not found)
            current_under = current_below.get(block)

            if goal_under == 'table':
                # Goal is on the table
                if current_under == 'table':
                    is_correct = True
            elif goal_under is not None: # Goal is on another block
                if current_under == goal_under:
                    # It's on the right block, now check if the block below is correct
                    # The block below (goal_under) must also be in the goal structure
                    if goal_under in self.goal_blocks: # Safety check for valid goal structure
                         is_correct = is_correctly_placed(goal_under) # Recursive call
                    else:
                         # Should not happen in valid goals where goal stacks are complete
                         is_correct = False
                # else: block is not on the correct object or is held -> not correct

            correctly_placed[block] = is_correct
            return is_correct

        # Calculate heuristic
        h = 0
        # Iterate only over blocks that are part of the goal structure
        for block in self.goal_blocks:
            if not is_correctly_placed(block):
                h += 1

        return h
