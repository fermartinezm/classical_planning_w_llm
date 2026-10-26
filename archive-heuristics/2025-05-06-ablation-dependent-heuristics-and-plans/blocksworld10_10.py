from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by whitespace
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required by counting blocks
    that are not in their goal position relative to the block below them (cost 2),
    plus the number of blocks stacked on top of such misplaced blocks (cost 1 each).
    This captures the approximate cost of moving misplaced blocks and clearing blocks
    that are obstructing them or blocks below them.

    # Assumptions
    - The goal specifies the desired stack configuration for some or all blocks.
    - Blocks not explicitly mentioned in an (on X Y) goal fact are assumed to
      have a goal of being on the table.
    - The heuristic is non-admissible and designed for greedy best-first search.
    - Each action (pickup, putdown, stack, unstack) has a cost of 1.

    # Heuristic Initialization
    - Extract the goal location for each block from the task's goal conditions.
      A block's goal location is the block it should be directly on, or 'table'.
    - Identify all blocks (objects) in the problem from the initial state and goals.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Determine the current location for each block:
       - If `(on B Y)` is true, B is on Y.
       - If `(on-table B)` is true, B is on the table.
       - If `(holding B)` is true, B is in the arm.
       - A block can only be in one location at a time.
    2. Determine which block is currently directly on top of each block (if any).
    3. Initialize the heuristic value `h` to 0.
    4. Iterate through each block `B`:
       a. Find the block `B` should be on in the goal (`goal_loc[B]`).
       b. Find the block `B` is currently on (`current_loc[B]`).
       c. If `current_loc[B]` is not equal to `goal_loc[B]`:
          i. Add 2 to `h` (approximate cost to move B to its goal location: pickup/unstack + stack/putdown).
          ii. Find the block `A` currently on top of `B`.
          iii. While `A` exists (i.e., B is not clear):
              - Add 1 to `h` (approximate cost to move A out of the way: unstack/putdown).
              - Find the block `A'` currently on top of `A`. Set `A = A'`.
    5. Return the total heuristic value `h`.

    This heuristic counts a block if it's not on its correct immediate support,
    assigning a cost of 2 (representing pickup/unstack + stack/putdown).
    It then adds a cost of 1 for each block stacked above it, representing the
    cost to unstack those blocks to clear the one below. This provides a
    reasonable estimate of the work needed to dismantle incorrect stacks and
    place blocks correctly.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and identifying all blocks.
        """
        super().__init__(task)

        self.goals = task.goals
        # Static facts are empty in blocksworld, but we keep the structure.
        self.static = task.static

        # Collect all unique objects (blocks) from initial state and goal facts.
        self.blocks = set()
        # Helper to add objects from fact parts
        def add_objects_from_fact_parts(parts):
             if parts and parts[0] in ['on', 'on-table', 'clear', 'holding']:
                 # Objects are the arguments after the predicate
                 for obj in parts[1:]:
                     self.blocks.add(obj)

        for fact in task.initial_state:
            add_objects_from_fact_parts(get_parts(fact))

        for fact in task.goals:
             add_objects_from_fact_parts(get_parts(fact))

        # Map each block to its goal location (the block it should be on, or 'table').
        # Blocks not explicitly mentioned in an (on X Y) goal are assumed to be on the table.
        self.goal_loc = {block: 'table' for block in self.blocks} # Default to table
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'on':
                block, support = parts[1], parts[2]
                self.goal_loc[block] = support
            # If goal is (on-table X), the default 'table' is correct.
            # Other goal types like (clear X) or (arm-empty) don't define location.


    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # Determine current location and block above for each block.
        current_loc = {}
        current_above = {block: None for block in self.blocks}
        # We don't strictly need holding_block for this heuristic calculation,
        # but building current_loc handles it.

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts

            predicate = parts[0]
            if predicate == 'on':
                block, support = parts[1], parts[2]
                current_loc[block] = support
                # Ensure support is in blocks if it's a block, not 'table'
                if support in self.blocks:
                    current_above[support] = block # support is below block
            elif predicate == 'on-table':
                block = parts[1]
                current_loc[block] = 'table'
            elif predicate == 'holding':
                block = parts[1]
                current_loc[block] = 'arm'
                # A block in the arm is not on anything, so it doesn't set current_above


        heuristic_value = 0

        # Calculate heuristic based on misplaced blocks and blocks above them.
        for block in self.blocks:
            current_block_loc = current_loc.get(block) # Use .get for safety

            # If the block's current location is known and is not its goal location
            # A block might not be in current_loc if the state is malformed,
            # but assuming valid states, every block is either on something, on table, or held.
            if current_block_loc is not None:
                 # Get the goal location, defaulting to 'table' if the block isn't in goal_loc
                 # (This handles blocks whose final position isn't specified in goals,
                 # assuming they should be on the table).
                 goal_block_loc = self.goal_loc.get(block, 'table')

                 if current_block_loc != goal_block_loc:
                     heuristic_value += 2 # Penalty for the block itself being misplaced

                     # Add penalty for blocks stacked on top of this misplaced block
                     # Only applies if the block is currently on another block or table, not if held ('arm').
                     if current_block_loc != 'arm':
                         block_on_top = current_above.get(block)
                         while block_on_top is not None:
                             heuristic_value += 1 # Penalty for block on top
                             block_on_top = current_above.get(block_on_top)
            # else: Block location unknown - implies state is incomplete/malformed.
            # In a real planner, this might indicate an issue or require a default assumption.
            # For this problem, we assume valid states where all blocks have a location.


        # The heuristic is 0 iff current_loc[B] == goal_loc[B] for all blocks B,
        # and there are no blocks on top of any block B where current_loc[B] != goal_loc[B].
        # If current_loc[B] == goal_loc[B] for all B, the second condition is vacuously true.
        # So h=0 iff current_loc[B] == goal_loc[B] for all B.
        # This means the (on X Y) and (on-table X) facts match the goal.
        # This implies the goal stack structure is achieved.
        # (clear X) goals are implicitly handled: if (on A X) is true in state but (clear X) is goal,
        # then A must be moved. If X is in its goal_loc, A's goal_loc is likely not X, so A is misplaced.
        # If X is not in its goal_loc, X is penalized, and A on top is penalized.
        # (arm-empty) goal: If arm is not empty, the held block is likely misplaced (current_loc='arm' != goal_loc).
        # So h > 0 if arm is not empty and (arm-empty) is a goal (unless goal is to hold it, which is not blocksworld).
        # The heuristic seems to be 0 iff the relevant goal facts are met.

        return heuristic_value
