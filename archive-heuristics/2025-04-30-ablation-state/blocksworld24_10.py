# Assume Heuristic base class is available from 'heuristics.heuristic_base'
# from heuristics.heuristic_base import Heuristic

# Helper function (defined outside the class)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Assumes fact is a string like "(predicate arg1 arg2)"
    fact = fact.strip()
    if not fact or fact[0] != '(' or fact[-1] != ')':
         # Handle unexpected format, though PDDL facts are standard
         return []
    return fact[1:-1].split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not yet in their correct position within the goal stack structure. A block is considered correctly stacked if it is on the correct block (or the table) according to the goal, and the block below it is also correctly stacked.

    # Assumptions
    - The goal specifies a desired configuration of blocks into stacks using `on` and `on-table` predicates.
    - The heuristic focuses on achieving the correct relative positions within these goal stacks.
    - The goal state implies all blocks involved in the goal stacks are in their final positions.
    - The heuristic value is 0 if and only if the state is a goal state.

    # Heuristic Initialization
    - The heuristic extracts the desired position for each block from the goal conditions (`on` and `on-table` predicates). This forms the `goal_pos` mapping, where `goal_pos[block]` is the block `block` should be on, or 'table'.
    - It also identifies all blocks that are explicitly mentioned as arguments in the goal's `on` or `on-table` predicates (`blocks_in_goal`).

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Check for Goal State: If the current state satisfies all goal conditions (`self.task.goal_reached(state)`), the heuristic value is 0. This ensures the heuristic is 0 only at goal states.
    2. Parse Current State: Determine the current position of every block present in the state. A block can be on another block (`on`), on the table (`on-table`), or held by the arm (`holding`). Store this information in the `current_pos` mapping, using 'held' as a special position indicator.
    3. Define "Correctly Stacked": A block `B` is defined as "correctly stacked" if it meets the following criteria based on the goal configuration (`self.goal_pos`):
       - If `B` is not in `self.goal_pos` (i.e., not part of the goal stacks we track), it is considered correctly stacked for the purpose of the heuristic count.
       - If `B` is in `self.goal_pos`, let `desired_pos = self.goal_pos[B]` and `actual_pos = current_pos.get(B)`.
       - If `desired_pos == 'table'`, `B` is correctly stacked if `actual_pos == 'table'`.
       - If `desired_pos` is another block `A`, `B` is correctly stacked if `actual_pos == A` AND block `A` is itself "correctly stacked".
    4. Compute Correctly Stacked Status: For each block identified as part of the goal configuration (`self.blocks_in_goal`), recursively evaluate whether it is correctly stacked using the definition from step 3. Memoization (`correctly_stacked_memo`) is used to store results for blocks as they are computed, preventing redundant calculations and handling dependencies efficiently.
    5. Count Misplaced Blocks: Initialize a counter to 0. Iterate through all blocks in `self.blocks_in_goal`. For each block, if the `is_correctly_stacked` function returns `False`, increment the counter.
    6. Return the Count: The final value of the counter is the heuristic estimate. This count represents the number of blocks that are part of the desired stack structure but are not yet in their correct place relative to that structure.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the goal stack structure.
        """
        self.task = task # Store task object to use goal_reached method
        self.goals = task.goals

        # Map blocks to the block they should be on in the goal, or 'table'.
        self.goal_pos = {}
        # Set of all blocks that are part of the goal configuration.
        self.blocks_in_goal = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "on" and len(parts) == 3:
                block, block_below = parts[1], parts[2]
                self.goal_pos[block] = block_below
                self.blocks_in_goal.add(block)
                self.blocks_in_goal.add(block_below)
            elif parts and parts[0] == "on-table" and len(parts) == 2:
                block = parts[1]
                self.goal_pos[block] = 'table'
                self.blocks_in_goal.add(block)
            # Ignore other goal predicates like (clear X) or (arm-empty) for this heuristic's core logic

        # 'table' is not a block, remove it if added
        self.blocks_in_goal.discard('table')


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # 1. Check for Goal State
        if self.task.goal_reached(state):
            return 0

        # 2. Parse Current State
        current_pos = {}
        # Find which block is on which, and which is on the table or held
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "on" and len(parts) == 3:
                block, block_below = parts[1], parts[2]
                current_pos[block] = block_below
            elif parts and parts[0] == "on-table" and len(parts) == 2:
                block = parts[1]
                current_pos[block] = 'table'
            elif parts and parts[0] == "holding" and len(parts) == 2:
                block = parts[1]
                current_pos[block] = 'held' # Special state for held block
            # Ignore other state predicates like (clear X) or (arm-empty) for this heuristic's core logic

        # 3. Compute correctly_stacked status using memoization
        correctly_stacked_memo = {}

        def is_correctly_stacked(block):
            """
            Recursively check if a block is in its correct position relative
            to the goal stack structure.
            """
            # Return memoized result if available
            if block in correctly_stacked_memo:
                return correctly_stacked_memo[block]

            # If the block is not part of the goal configuration we are tracking,
            # it doesn't contribute to the misplaced count.
            if block not in self.goal_pos:
                 correctly_stacked_memo[block] = True
                 return True

            desired_pos = self.goal_pos[block]
            actual_pos = current_pos.get(block) # Get current position, None if not found

            # Base case: Block should be on the table
            if desired_pos == 'table':
                result = (actual_pos == 'table')
            # Recursive case: Block should be on another block
            else: # desired_pos is another block
                # Check if it's currently on the desired block AND the block below is correctly stacked
                result = (actual_pos == desired_pos) and is_correctly_stacked(desired_pos)

            # Memoize the result before returning
            correctly_stacked_memo[block] = result
            return result

        # 4. Count Misplaced Blocks
        misplaced_count = 0
        # Only consider blocks that are part of the goal configuration
        for block in self.blocks_in_goal:
            if not is_correctly_stacked(block):
                misplaced_count += 1

        # 5. Return the count
        return misplaced_count
