from heuristics.heuristic_base import Heuristic

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    Estimates the number of blocks that are not in their correct position
    relative to the block below them in the goal configuration, plus a penalty
    if the arm is not empty.

    # Summary
    Counts blocks that are part of a goal stack but are not currently positioned
    correctly within that stack, tracing down to the table. Adds 1 if the arm
    is holding a block.

    # Heuristic Initialization
    - Parses the goal state to determine the desired parent block or table
      for each block that has a specified position in the goal.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the goal facts to create a mapping `goal_parent` from each block
       to the block it should be on, or 'table' if it should be on the table.
    2. For a given state, parse the state facts to create a mapping
       `current_parent` from each block to the block it is currently on,
       or 'table' if it's on the table. Also, identify if the arm is holding a block.
    3. Initialize a counter `incorrectly_stacked_count` to 0.
    4. For each block that has a goal position (i.e., is a key in `goal_parent`):
       a. Check if this block is "correctly stacked" by tracing its desired
          position down the goal stack all the way to the table, and verifying
          that the current state matches this desired stack configuration.
       b. If the block is not correctly stacked, increment the counter.
    5. Add 1 to the counter if the arm is currently holding a block.
    6. Return the final counter value.

    A block is "correctly stacked" if:
    - It is currently on the block (or table) specified by its goal parent, AND
    - The block it is on (if not the table) is also correctly stacked.
    This check is implemented iteratively by tracing down the goal stack.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal configuration."""
        self.goals = task.goals

        # Build the goal_parent map: block -> parent_block or 'table'
        # This maps each block that has a goal position to what should be directly below it.
        self.goal_parent = {}
        for goal in self.goals:
            parts = self._get_parts(goal)
            predicate = parts[0]
            if predicate == 'on':
                block, parent = parts[1], parts[2]
                self.goal_parent[block] = parent
            elif predicate == 'on-table':
                block = parts[1]
                self.goal_parent[block] = 'table'
            # Ignore 'clear' goals for building the stack structure

    def _get_parts(self, fact_string):
        """Extract the components of a PDDL fact string."""
        # Remove parentheses and split by space
        return fact_string[1:-1].split()

    def _parse_state(self, state_facts):
        """Parse state facts to get current parent mapping and held block."""
        current_parent = {}
        held_block = None
        for fact in state_facts:
            parts = self._get_parts(fact)
            predicate = parts[0]
            if predicate == 'on':
                block, support = parts[1], parts[2]
                current_parent[block] = support
            elif predicate == 'on-table':
                block = parts[1]
                current_parent[block] = 'table'
            elif predicate == 'holding':
                held_block = parts[1]
            # Ignore 'clear' and 'arm-empty'
        return current_parent, held_block

    def _is_correctly_stacked_iterative(self, block, goal_parent, current_parent):
        """
        Checks if a block is correctly stacked according to the goal configuration
        by tracing the stack down to the table in both goal and current states.
        """
        # This function is designed to be called only for blocks that are keys in goal_parent.
        # If called otherwise, the logic might need adjustment.

        current_goal_block = block
        
        # Trace down the goal stack until we reach the block that should be on the table
        # For block A on B, goal_parent[A] = B. For B on table, goal_parent[B] = 'table'.
        # The loop continues as long as the current block in the goal chain is supposed to be ON something (not 'table').
        while current_goal_block in goal_parent and goal_parent[current_goal_block] != 'table':
            desired_p = goal_parent[current_goal_block]
            actual_p = current_parent.get(current_goal_block) # Get current parent, None if not found (e.g., held)

            # If the actual parent doesn't match the desired parent, the stack is broken.
            if actual_p != desired_p:
                return False

            # Move down the goal stack
            current_goal_block = desired_p

        # If the loop finished, current_goal_block is the block that should be the base
        # of this goal stack (its goal parent is 'table').
        # Check if this base block is actually on the table in the current state.
        final_goal_base = current_goal_block
        
        # Check if the block that is supposed to be the base of the stack
        # (i.e., the one whose goal parent is 'table', which is final_goal_base)
        # is actually on the table in the current state.
        if current_parent.get(final_goal_base) != 'table':
             return False # The base is not on the table

        # If we reached here, the entire goal stack from the original 'block' down to the table matches the goal.
        return True


    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        current_parent, held_block = self._parse_state(state)

        incorrectly_stacked_count = 0

        # Count blocks that have a goal position but are not correctly stacked
        # We only iterate through blocks that are explicitly mentioned as being ON something
        # or ON-TABLE in the goal.
        for block in self.goal_parent:
            if not self._is_correctly_stacked_iterative(block, self.goal_parent, current_parent):
                incorrectly_stacked_count += 1

        # Add a penalty if the arm is holding a block.
        # This encourages freeing the arm to perform other necessary actions.
        # A held block is effectively "misplaced" because it's not in its final position.
        # This also helps break ties between states that have the same stacking errors
        # but one has a free arm.
        if held_block is not None:
            incorrectly_stacked_count += 1

        return incorrectly_stacked_count
