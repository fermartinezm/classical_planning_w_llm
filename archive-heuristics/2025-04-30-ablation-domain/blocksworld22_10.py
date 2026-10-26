from heuristics.heuristic_base import Heuristic

# Helper function to parse facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed facts defensively
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    # Split by space, handling potential multiple spaces
    parts = fact[1:-1].split()
    return parts

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required by counting the number
    of blocks that are not yet part of a correctly built stack segment from the
    table upwards according to the goal configuration. A block is considered
    "correctly placed" if it is on the correct block or the table as specified
    in the goal, AND the block it is supposed to be on (if any) is also correctly
    placed. The heuristic value is the total number of blocks involved in the
    goal minus the number of blocks that are correctly placed.

    # Assumptions
    - The goal specifies a set of desired `on` and `on-table` relationships
      that form one or more stacks.
    - The heuristic is non-admissible and designed for greedy best-first search.
    - The set of blocks in the current state is assumed to be the same as the set
      of blocks relevant to the goal.

    # Heuristic Initialization
    - The heuristic analyzes the goal facts (`task.goals`) to determine the
      desired position for each block (which block or the table it should be
      directly on). This is stored in a mapping `self.goal_below`.
    - It also identifies the set of all blocks involved in the goal, stored
      in `self.goal_blocks`.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state (`node.state`):
    1. Parse the current state facts to determine the current position of each
       block (which block or the table it is directly on). This is stored in
       a mapping `current_below`.
    2. Initialize a set `correctly_placed_blocks` to store blocks that are
       currently in their correct goal position relative to the block below them,
       and the block below them is also correctly placed (recursively).
    3. Identify the base cases for correctly placed blocks: Any block `B`
       from `self.goal_blocks` that is supposed to be on the table
       (`self.goal_below.get(B) == 'table'`) and is currently on the table
       (`current_below.get(B) == 'table'`) is initially added to
       `correctly_placed_blocks`.
    4. Iteratively expand the set of `correctly_placed_blocks`:
       Repeat until no new blocks are added in an iteration:
         Initialize a flag `changed = False`.
         For each block `B` in `self.goal_blocks`:
           If `B` is not already in `correctly_placed_blocks`:
             # Get the block it should be on according to the goal
             target_below = self.goal_below.get(block)

             # Check if block is part of a goal stack (i.e., should be on another block)
             if target_below is not None and target_below != 'table':
               # Check if the block it should be on is already correctly placed
               if target_below in correctly_placed_blocks:
                 # Check if the block B is currently on the correct block target_below
                 # Use .get() with None default in case block is not in current_below (e.g., held)
                 if current_below.get(block) == target_below:
                   correctly_placed_blocks.add(block)
                   changed = True

         # If no new blocks were added in this iteration, we are done
         if not changed:
             break

    5. The heuristic value is the total number of blocks involved in the goal
       (`len(self.goal_blocks)`) minus the final count of blocks in
       `correctly_placed_blocks`. A value of 0 indicates the goal state has been reached.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration.
        """
        self.goals = task.goals

        self.goal_below = {}
        self.goal_blocks = set()

        # Parse goal facts to build the goal stack structure and identify blocks
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]
            if predicate == 'on':
                if len(parts) == 3:
                    block, support = parts[1], parts[2]
                    self.goal_below[block] = support
                    self.goal_blocks.add(block)
                    self.goal_blocks.add(support)
            elif predicate == 'on-table':
                 if len(parts) == 2:
                    block = parts[1]
                    self.goal_below[block] = 'table'
                    self.goal_blocks.add(block)
            # Ignore 'clear' or 'arm-empty' goals for the structural part of the heuristic

        # Remove 'table' if it was added as an object (it's a location, not a block)
        self.goal_blocks.discard('table')

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        current_below = {}

        # Parse current state facts to find block positions
        # We only care about 'on' and 'on-table' for the structure
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]
            if predicate == 'on':
                if len(parts) == 3:
                    block, support = parts[1], parts[2]
                    current_below[block] = support
            elif predicate == 'on-table':
                if len(parts) == 2:
                    block = parts[1]
                    current_below[block] = 'table'
            # Ignore other predicates

        # Identify blocks that are correctly placed from the bottom up
        correctly_placed_blocks = set()

        # Base cases: Blocks that should be on the table and are on the table
        # Iterate through blocks identified from the goal
        for block in self.goal_blocks:
            # Check if the block is supposed to be on the table in the goal
            if self.goal_below.get(block) == 'table':
                # Check if the block is currently on the table in the state
                # Use .get() with None default in case block is not in current_below (e.g., held)
                if current_below.get(block) == 'table':
                     correctly_placed_blocks.add(block)

        # Iteratively add blocks that are correctly placed on top of correctly placed blocks
        changed = True
        while changed:
            changed = False
            # Iterate through blocks identified from the goal
            for block in self.goal_blocks:
                # If the block is not already marked as correctly placed
                if block not in correctly_placed_blocks:
                    # Get the block it should be on according to the goal
                    target_below = self.goal_below.get(block)

                    # Check if block is part of a goal stack (i.e., should be on another block)
                    if target_below is not None and target_below != 'table':
                        # Check if the block it should be on is already correctly placed
                        if target_below in correctly_placed_blocks:
                            # Check if the block B is currently on the correct block target_below
                            # Use .get() with None default in case block is not in current_below (e.g., held)
                            if current_below.get(block) == target_below:
                                correctly_placed_blocks.add(block)
                                changed = True

         # If no new blocks were added in this iteration, we are done
        if not changed:
            break

        # The heuristic is the number of blocks involved in the goal
        # minus the number of blocks that are correctly placed.
        # If goal_blocks is empty (empty goal), heuristic is 0.
        return len(self.goal_blocks) - len(correctly_placed_blocks)
