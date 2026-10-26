# Assuming heuristic_base.py is in a directory named 'heuristics'
# and this file will be in the same directory or a subdirectory.
# If running as a standalone file, might need to adjust import.
# For the purpose of providing the code, assume the package structure.
from heuristics.heuristic_base import Heuristic


# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by considering blocks that are not in their correct goal position and blocks
    that are obstructing correctly positioned blocks.

    # Heuristic Calculation
    The heuristic value is calculated as the sum of:
    1.  Twice the number of blocks that are specified in the goal to be on a
        specific block or the table, but are currently on a different block,
        on the table when they should be on a block, or on a block when they
        should be on the table, or being held.
        (Estimated cost: 2 actions per block - pickup/unstack + putdown/stack)
    2.  Twice the number of blocks X that are currently on top of block Y
        (`(on X Y)` is true), where Y is currently on its correct goal support
        (block or table), but the goal state does *not* have X on Y.
        (Estimated cost: 2 actions per block - unstack + putdown/stack to clear Y)
    3.  One if the robot arm is currently holding a block.
        (Estimated cost: 1 action - putdown/stack to free the arm)

    This heuristic is non-admissible but aims to guide the search effectively
    by prioritizing fixing misplaced blocks and clearing necessary paths.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal positions for blocks.

        Args:
            task: The planning task object containing initial state, goals, etc.
        """
        self.goals = task.goals  # Keep the full set of goal facts

        # Map block to the block/table it should be directly on top of in the goal.
        # e.g., goal_support['b1'] = 'b2' if (on b1 b2) is a goal.
        # e.g., goal_support['b2'] = 'table' if (on-table b2) is a goal.
        self.goal_support = {}
        # Set of blocks whose position is explicitly specified in the goal
        # (i.e., blocks that appear as the first argument in an 'on' goal
        # or the argument in an 'on-table' goal).
        self.blocks_with_goal_position = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'on':
                block, support = parts[1], parts[2]
                self.goal_support[block] = support
                self.blocks_with_goal_position.add(block)
            elif parts[0] == 'on-table':
                block = parts[1]
                self.goal_support[block] = 'table'
                self.blocks_with_goal_position.add(block)

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer representing the estimated cost to reach the goal.
        """
        state = node.state

        # 1. Parse state to get current positions and blocks on top.
        current_support = {} # block -> support ('table', block, 'holding')
        blocks_on_top = {}   # block -> set of blocks on top
        arm_is_holding = False

        # Populate current_support and blocks_on_top
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                block, support = parts[1], parts[2]
                current_support[block] = support
                if support not in blocks_on_top:
                    blocks_on_top[support] = set()
                blocks_on_top[support].add(block)
            elif parts[0] == 'on-table':
                block = parts[1]
                current_support[block] = 'table'
            elif parts[0] == 'holding':
                block = parts[1]
                current_support[block] = 'holding' # Mark as held
                arm_is_holding = True
            # 'clear' and 'arm-empty' are state properties derived from on/holding

        h = 0

        # Calculate H_misplaced_support_actions
        # Count blocks that are in goal_support but not currently on their goal_support
        misplaced_support_count = 0
        for block in self.blocks_with_goal_position:
            current_sup = current_support.get(block)
            goal_sup = self.goal_support[block]

            # If block is not found in current_support (shouldn't happen for blocks
            # in goal_support in a valid state), or is held, or is on the wrong support.
            # A block being held means it's not on its goal support (unless goal is holding, which is not BW).
            if current_sup is None or current_sup != goal_sup:
                 misplaced_support_count += 1

        h += misplaced_support_count * 2

        # Calculate H_blocking_correct_support_actions
        # Count blocks X on Y where Y is on its correct support but (on X Y) is not a goal
        blocking_correct_support_count = 0
        # Iterate through blocks that have something on them in the current state
        # These are the keys in blocks_on_top.
        for support_block in blocks_on_top:
             # Check if the support_block is one whose position is specified in the goal
             if support_block in self.blocks_with_goal_position:
                 # Check if the support_block is currently on its correct goal support
                 current_sup_of_support = current_support.get(support_block)
                 goal_sup_of_support = self.goal_support[support_block]

                 if current_sup_of_support == goal_sup_of_support:
                     # The support_block is correctly placed relative to what's below it.
                     # Now check if the blocks on top of it are wrong.
                     for block_above in blocks_on_top[support_block]:
                         # Is (on block_above support_block) a goal fact?
                         goal_fact_on = f'(on {block_above} {support_block})'
                         if goal_fact_on not in self.goals:
                              blocking_correct_support_count += 1
                         # Note: This counts each block_above once if it's wrongly placed on a correctly supported block.

        h += blocking_correct_support_count * 2

        # Calculate H_holding_action
        # If the arm is holding a block, it needs at least one action (putdown or stack)
        # to free the arm and place the block.
        if arm_is_holding:
            h += 1

        return h

