from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or invalid format defensively
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the distance to the goal by counting the number
    of blocks that are not in their correct position relative to the block
    immediately below them (or the table) according to the goal configuration.
    A block is considered "correctly stacked" if it is on the correct block
    (or table) as specified in the goal, AND the block below it is also
    correctly stacked.

    # Assumptions
    - The goal state specifies the position (on another block or on the table)
      for all blocks relevant to the problem.
    - The goal configuration forms one or more stacks resting on the table.
    - The arm can only hold one block at a time.
    - All blocks whose goal position is specified by (on X Y) or (on-table X)
      are considered relevant and are expected to be present in the state.

    # Heuristic Initialization
    - Parse the goal facts to determine the desired block-below relationship
      for each block (i.e., which block should be directly below it, or if
      it should be on the table). This mapping is stored in `self.goal_below`.

    # Step-By-Step Thinking for Computing Heuristic
    1.  For the current state, determine the immediate base for every block:
        Is it on another block, on the table, or currently held by the arm?
        Store this in a `state_below` mapping.
    2.  Initialize a counter for blocks that are not correctly stacked (`misplaced_count`).
    3.  Initialize a memoization dictionary to store results of the recursive
        `is_correctly_stacked` checks to avoid redundant computations.
    4.  For each block whose goal position is specified (i.e., each block
        that is a key in `self.goal_below`):
        Check if the block is "correctly stacked" using a recursive helper function.
        A block `X` is correctly stacked if:
        - It is currently on the same base (another block or the table) as specified in the goal (`state_below[X] == goal_below[X]`).
        - AND, if its goal base is another block `Y`, then `Y` must also be correctly stacked (`is_correctly_stacked(Y, ...)`).
        - If its goal base is the table, it is correctly stacked if it is currently on the table.
        - If it is currently held by the arm, it is not correctly stacked.
    5.  If the recursive check determines a block is *not* correctly stacked, increment `misplaced_count`.
    6.  The heuristic value is the final `misplaced_count`. This counts how many blocks are part of an incorrect stack structure, starting from the bottom.

    This heuristic is not admissible as it doesn't count the minimum actions
    precisely (e.g., clearing blocks above the target is not explicitly costed
    beyond marking the upper blocks as misplaced), but it provides a measure
    of how "far" the current state is from the goal structure. It guides the
    search towards building correct stacks from the bottom up.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals

        # Store goal locations for each block.
        # goal_below[block] = block_below or 'table'
        self.goal_below = {}

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue

            predicate = parts[0]
            if predicate == "on":
                if len(parts) == 3:
                    block, block_below = parts[1], parts[2]
                    self.goal_below[block] = block_below
            elif predicate == "on-table":
                 if len(parts) == 2:
                    block = parts[1]
                    self.goal_below[block] = 'table'
            # Ignore other goal predicates like (clear ?) or (arm-empty)

        # We only care about blocks whose goal position is explicitly defined
        # in goal_below.keys().

    def is_correctly_stacked(self, block, state_below, goal_below, memo):
        """
        Recursively check if a block is in its correct goal position relative
        to the block below it (or table), and if the base below it is also
        correctly stacked.

        Args:
            block (str): The block to check. This block is guaranteed to be a key in goal_below.
            state_below (dict): Mapping from block to its current base ('table', block_name, or 'arm').
            goal_below (dict): Mapping from block to its goal base ('table' or block_name).
            memo (dict): Memoization dictionary {block: bool}.

        Returns:
            bool: True if the block is correctly stacked, False otherwise.
        """
        if block in memo:
            return memo[block]

        # block is guaranteed to be in goal_below.keys()

        current_base = state_below.get(block)
        goal_base = goal_below[block] # Use [] as block is guaranteed to be a key

        # If the block is not found in state_below (not on anything, table, or held),
        # it's an invalid state representation for this domain. Treat as not correctly stacked.
        # This case should ideally not happen in a valid state representation.
        if current_base is None:
             memo[block] = False
             return False

        # If the block is held, it's not correctly stacked relative to its goal base.
        if current_base == 'arm':
            memo[block] = False
            return False

        # If the current base doesn't match the goal base, it's not correctly stacked.
        if current_base != goal_base:
            memo[block] = False
            return False

        # If the current base matches the goal base:
        # If the goal base is the table, the block is correctly stacked.
        if goal_base == 'table':
            memo[block] = True
            return True
        else:
            # If the goal base is another block, the block is correctly stacked
            # only if the block below it is correctly stacked.
            block_below = goal_base
            # For a valid goal stack ending on the table, block_below must also
            # be a key in goal_below or be 'table'. Since goal_base is not 'table',
            # block_below must be a block name. We assume block_below is also
            # a key in goal_below for a well-formed goal.
            if block_below not in goal_below:
                 # This indicates an issue with the goal structure definition
                 # relative to what the heuristic expects (stacks ending on table).
                 # In a strict sense, if the base isn't part of the goal structure
                 # we track, the block isn't correctly stacked *within that structure*.
                 # Treat as not correctly stacked in this heuristic.
                 memo[block] = False
                 return False

            memo[block] = self.is_correctly_stacked(block_below, state_below, goal_below, memo)
            return memo[block]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Determine the immediate base for every block in the current state.
        state_below = {}
        # We only need to track blocks that are keys in self.goal_below
        # or appear as bases for such blocks. A simpler approach is to
        # just map everything we see in the state that is a block.
        # We can identify blocks by checking terms in predicates like on, on-table, holding.
        # A more robust way might be to get objects from the task if available,
        # but it's not provided in the Task class definition.
        # Let's rely on parsing the state facts for blocks and their positions.
        all_blocks_in_state = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "on":
                if len(parts) == 3:
                    block, block_below = parts[1], parts[2]
                    state_below[block] = block_below
                    all_blocks_in_state.add(block)
                    all_blocks_in_state.add(block_below)
            elif predicate == "on-table":
                 if len(parts) == 2:
                    block = parts[1]
                    state_below[block] = 'table'
                    all_blocks_in_state.add(block)
            elif predicate == "holding":
                 if len(parts) == 2:
                    block = parts[1]
                    state_below[block] = 'arm' # Block is held
                    all_blocks_in_state.add(block)
            # clear and arm-empty predicates don't involve blocks as arguments in this way

        # 2. Count blocks not correctly stacked.
        misplaced_count = 0
        memo = {} # Memoization dictionary for recursive calls

        # 4. For each block whose goal position is specified (keys in self.goal_below):
        for block in self.goal_below.keys():
             # 5. Check if the block is correctly stacked.
             # If the block is not even present in the state_below mapping,
             # it means it's not on anything, on the table, or held. This
             # indicates an invalid state representation, but it's definitely
             # not correctly stacked.
             if block not in all_blocks_in_state:
                 # This case shouldn't happen in valid states, but handle defensively.
                 # A block required by the goal is missing from the state.
                 misplaced_count += 1
             elif not self.is_correctly_stacked(block, state_below, self.goal_below, memo):
                 misplaced_count += 1


        # 6. The heuristic value is the total count.
        return misplaced_count
