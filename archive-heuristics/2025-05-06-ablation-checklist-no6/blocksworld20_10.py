from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Assumes fact is like '(predicate arg1 arg2)'
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    This heuristic estimates the cost to reach the goal state by considering:
    1. Blocks that are not in their correct goal stack position.
    2. Blocks that are obstructing correct goal stack segments.
    3. Whether the robot's arm is holding a block.

    The heuristic value is calculated as:
    2 * (misplaced_in_stack_count + direct_obstruction_count) + arm_holding_cost

    - misplaced_in_stack_count: Number of blocks in goal_blocks that are not
      part of their correct goal stack segment down to the table. A block B is
      in its correct goal stack segment if its current position matches its
      goal position (on the correct block or table), and the block below it
      (if any) is also in its correct goal stack segment, recursively down
      to the table.
    - direct_obstruction_count: Number of blocks that are sitting directly on
      top of a correctly placed goal stack segment but shouldn't be there
      according to the goal.
    - arm_holding_cost: 1 if the arm is holding a block, 0 otherwise. This
      represents the immediate cost of needing to put down or stack the held block.

    The multiplier of 2 is used because moving a block typically requires
    at least two actions (pickup/unstack and putdown/stack).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration.
        """
        self.goals = task.goals

        # Build the goal configuration mapping: block -> block_below or 'table'
        self.goal_on = {}
        # Keep track of all blocks mentioned in goal positions
        self.goal_blocks = set()

        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts[0] == 'on':
                block_on_top = parts[1]
                block_below = parts[2]
                self.goal_on[block_on_top] = block_below
                self.goal_blocks.add(block_on_top)
                self.goal_blocks.add(block_below)
            elif parts[0] == 'on-table':
                block_on_table = parts[1]
                self.goal_on[block_on_table] = 'table'
                self.goal_blocks.add(block_on_table)

        # Static facts are not used in this heuristic for Blocksworld.
        # self.static = task.static # Not needed

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Build current state mappings: block -> block_below or 'table'
        current_on = {}
        arm_empty = False

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                block_on_top = parts[1]
                block_below = parts[2]
                current_on[block_on_top] = block_below
            elif parts[0] == 'on-table':
                block_on_table = parts[1]
                current_on[block_on_table] = 'table'
            elif parts[0] == 'arm-empty':
                arm_empty = True
            # We don't need to explicitly track the 'holding' block,
            # just whether the arm is empty or not.

        # Memoization for the recursive check
        memo = {}
        def is_in_goal_stack_recursive(block):
            """
            Checks if a block is in its correct goal position relative to the block below it,
            and if the block below it is also in its correct goal stack position, recursively.
            """
            if block in memo:
                return memo[block]

            # If the block is not part of the goal configuration, it cannot be in a goal stack.
            if block not in self.goal_blocks:
                memo[block] = False
                return False

            goal_pos = self.goal_on.get(block)
            current_pos = current_on.get(block)

            # If block is not currently placed anywhere (e.g. holding), it's not in the goal stack.
            if current_pos is None:
                memo[block] = False
                return False

            # Check immediate position
            if current_pos != goal_pos:
                memo[block] = False
                return False

            # If on table, check if goal is on table
            if goal_pos == 'table':
                memo[block] = True
                return True

            # If on another block, check if that block is in its goal stack
            result = is_in_goal_stack_recursive(goal_pos)
            memo[block] = result
            return result

        # Part 1: Count blocks that are not in their correct goal stack segment
        misplaced_in_stack_count = 0
        for block in self.goal_blocks:
            if not is_in_goal_stack_recursive(block):
                misplaced_in_stack_count += 1

        # Part 2: Count blocks that are directly obstructing a correct goal stack segment
        direct_obstruction_count = 0
        # Iterate through all blocks that are currently on top of another block
        # We can get this from the keys of current_on whose value is not 'table'.
        blocks_on_top_of_something = {
            block_on_top for block_on_top, block_below in current_on.items()
            if block_below != 'table'
        }

        for block_on_top in blocks_on_top_of_something:
            block_below = current_on[block_on_top] # We know block_below is not 'table'

            # Check if the block below is part of a correct goal stack segment
            if block_below in self.goal_blocks and is_in_goal_stack_recursive(block_below):
                # The block below is correctly placed in its stack.
                # Check if the block on top is supposed to be there according to the goal.
                # If block_on_top is not in goal_on, it's definitely not supposed to be on block_below.
                if self.goal_on.get(block_on_top) != block_below:
                    # The block on top is NOT supposed to be on block_below. It's a direct obstruction.
                    direct_obstruction_count += 1

        # Heuristic calculation
        # Each misplaced block or direct obstruction roughly costs 2 actions (pickup/unstack + putdown/stack)
        h = 2 * (misplaced_in_stack_count + direct_obstruction_count)

        # Part 3: Add cost if the arm is not empty (needs to put down the block)
        if not arm_empty:
            h += 1 # At least one action (putdown or stack) is needed to free the arm

        return h
