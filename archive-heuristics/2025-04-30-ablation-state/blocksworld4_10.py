def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
        # Not a valid PDDL fact string format we expect
        return []
    # Remove parentheses and split by whitespace
    # Use split() without arguments to handle multiple spaces
    return fact[1:-1].split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not in their
    correct final position as defined by the goal state. A block is
    considered correctly placed if it is on the table and the goal requires
    it to be on the table, OR if it is on another block and the goal requires
    it to be on that specific block, AND the block it is on is itself
    correctly placed.

    # Assumptions
    - The goal state specifies the desired position (on-table or on another block)
      for all blocks whose position matters for the goal configuration.
    - The goal state does not contain cycles in the 'on' relationships.
    - The initial state does not contain cycles in the 'on' relationships.
    - The arm being empty is typically required for the goal state to be fully achieved,
      and a block being held means it's not in its final resting place.
    - All blocks involved in goal 'on' or 'on-table' predicates exist in the state.

    # Heuristic Initialization
    - Parses the goal facts to build a mapping from each block to its desired
      position (either 'table' or the block it should be on). Only blocks
      explicitly mentioned in 'on' or 'on-table' goal facts are tracked.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Parse the current state to determine the current position of each block:
        - Iterate through state facts.
        - If `(on X Y)` is true, record that X is on Y.
        - If `(on-table X)` is true, record that X is on the table.
        - If `(holding X)` is true, record that X is held by the arm.
        - Store these in a dictionary `current_positions` mapping block name to its location ('table', another block name, or 'holding').
    2.  Initialize a counter `misplaced_blocks_count` to 0.
    3.  Initialize a memoization dictionary `is_correct_memo` to store the result
        of the correctness check for each block to avoid redundant calculations
        and infinite recursion in case of unexpected state structures (though cycles
        should not occur in valid Blocksworld states).
    4.  Define a recursive helper function `is_correctly_placed(block)`:
        - If the result for `block` is already in `is_correct_memo`, return it.
        - Get the desired goal position `goal_pos` for `block` from `self.goal_positions`.
        - If `block` is not in `self.goal_positions`, its position is not explicitly
          specified in the goal structure we are tracking. Consider it correctly placed
          with respect to this specific heuristic's focus. Store `True` in `is_correct_memo`
          and return `True`.
        - Get the current position `current_pos` for `block` from `current_positions`.
          If `block` is not found in `current_positions` (should not happen in valid states),
          treat it as misplaced. Store `False` in `is_correct_memo` and return `False`.
        - If the current position is 'holding' (`current_pos == 'holding'`), the block
          is not in its final resting place. Store `False` in `is_correct_memo` and return `False`.
        - If the desired goal position is 'table' (`goal_pos == 'table'`):
            - Check if the current position is also 'table' (`current_pos == 'table'`).
            - Store the result (`True` or `False`) in `is_correct_memo` and return it.
        - If the desired goal position is on another block (`goal_pos == under_block`):
            - Check if the current position is on that same block (`current_pos == under_block`).
            - If yes, recursively check if the block underneath (`under_block`) is correctly placed: `is_correctly_placed(under_block)`.
            - The result for `block` is the result of the recursive call. Store it in `is_correct_memo` and return it.
            - If no (current position is not `under_block`), `block` is not correctly placed.
              Store `False` in `is_correct_memo` and return `False`.
    5.  Iterate through each `block` that has a specified goal position (i.e., each key in `self.goal_positions`).
    6.  Call `is_correctly_placed(block)`. If it returns `False`, increment `misplaced_blocks_count`.
    7.  Return `misplaced_blocks_count`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal positions for each block.
        """
        self.goals = task.goals  # Goal conditions (frozenset of strings).
        # Blocksworld has no static facts relevant to position goals for this heuristic.
        # static_facts = task.static

        # Store goal locations for each block.
        # Maps block name -> 'table' or block name it should be on.
        self.goal_positions = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip invalid facts

            predicate = parts[0]
            if predicate == "on":
                # (on ?x ?y) goal means ?x should be on ?y
                if len(parts) == 3:
                    block, under_block = parts[1], parts[2]
                    self.goal_positions[block] = under_block
            elif predicate == "on-table":
                # (on-table ?x) goal means ?x should be on the table
                 if len(parts) == 2:
                    block = parts[1]
                    self.goal_positions[block] = 'table'
            # Ignore (clear ?x) and (arm-empty) goals for position tracking

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of strings).

        # Track where blocks are currently located.
        # Maps block name -> 'table', block name it's on, or 'holding'.
        current_positions = {}
        # No need to track holding_block separately, current_positions is enough

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            predicate = parts[0]
            if predicate == "on":
                if len(parts) == 3:
                    block, under_block = parts[1], parts[2]
                    current_positions[block] = under_block
            elif predicate == "on-table":
                 if len(parts) == 2:
                    block = parts[1]
                    current_positions[block] = 'table'
            elif predicate == "holding":
                 if len(parts) == 2:
                    block = parts[1]
                    current_positions[block] = 'holding'
            # Ignore (clear ?x) and (arm-empty) state facts for position tracking

        misplaced_blocks_count = 0
        is_correct_memo = {} # Memoization for recursive calls

        def is_correctly_placed(block):
            """
            Recursive helper to check if a block is in its goal position
            and its supporting stack (if any) is also correctly placed.
            """
            # Check memoization cache
            if block in is_correct_memo:
                return is_correct_memo[block]

            # Get the desired goal position for this block
            goal_pos = self.goal_positions.get(block)

            # If the block doesn't have a specific goal position defined
            # in the 'on' or 'on-table' goals, we consider it correctly placed
            # with respect to the stack structure we are tracking.
            if goal_pos is None:
                 is_correct_memo[block] = True
                 return True

            # Get the current position (on-table, on another block, or holding)
            current_pos = current_positions.get(block)

            # If the block is not found in current_positions, it's an invalid state
            # representation for this heuristic's assumptions. Treat as misplaced.
            if current_pos is None:
                 is_correct_memo[block] = False
                 return False

            # If the block is being held, it's not in its final resting place
            if current_pos == 'holding':
                is_correct_memo[block] = False
                return False

            # If the block is supposed to be on the table
            if goal_pos == 'table':
                result = (current_pos == 'table')
                is_correct_memo[block] = result
                return result

            # If the block is supposed to be on another block (under_block)
            under_block = goal_pos
            if current_pos == under_block:
                # It's on the right block, now check if the block underneath is correct
                # Recursive call. If under_block is not in goal_positions,
                # is_correctly_placed(under_block) will return True based on the logic above.
                result = is_correctly_placed(under_block)
                is_correct_memo[block] = result
                return result
            else:
                # It's not on the correct block
                is_correct_memo[block] = False
                return False

    # --- End of is_correctly_placed helper function ---

        # Iterate through all blocks that have a specified goal position
        for block in self.goal_positions.keys():
            if not is_correctly_placed(block):
                misplaced_blocks_count += 1

        return misplaced_blocks_count
