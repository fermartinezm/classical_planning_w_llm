from collections import defaultdict
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

class blocksworld11Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goal by considering:
    - The number of blocks that need to be moved to their correct positions.
    - The number of blocks that are in the way of placing other blocks.
    - The number of blocks that need to be cleared to satisfy the goal conditions.

    # Assumptions
    - Each block movement (pickup, stack, unstack, putdown) takes 2 actions.
    - Blocks in the way of a target position must be moved, costing 2 actions per block.
    - Clear conditions require moving all blocks on top, costing 2 actions per block.

    # Heuristic Initialization
    - Extracts the goal conditions into three categories: 'on', 'on-table', and 'clear'.
    - Stores these in dictionaries and sets for quick lookup during heuristic calculation.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each 'on' goal, check if the block is correctly placed. If not, add 2 actions and check for blocks in the way.
    2. For each 'on-table' goal, check if the block is on the table. If not, add 2 actions and check for blocks in the way.
    3. For each 'clear' goal, check if the block is clear. If not, add 2 actions per block on top.
    4. Sum all actions to get the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic with goal conditions."""
        self.goal_on = {}  # Maps block to its goal base (another block)
        self.goal_on_table = set()  # Blocks that should be on the table
        self.goal_clear = set()  # Blocks that should be clear

        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'on':
                x, y = parts[1], parts[2]
                self.goal_on[x] = y
            elif parts[0] == 'on-table':
                x = parts[1]
                self.goal_on_table.add(x)
            elif parts[0] == 'clear':
                x = parts[1]
                self.goal_clear.add(x)

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        current_blocks_above = defaultdict(list)
        current_on_table = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                x, y = parts[1], parts[2]
                current_blocks_above[y].append(x)
            elif parts[0] == 'on-table':
                x = parts[1]
                current_on_table.add(x)
                current_blocks_above['table'].append(x)

        steps = 0

        # Process 'on' goals
        for x, y in self.goal_on.items():
            if y not in current_blocks_above or x not in current_blocks_above[y]:
                steps += 2
                # Blocks on y in current state
                blocks_on_y = current_blocks_above.get(y, [])
                steps += 2 * len(blocks_on_y)

        # Process 'on-table' goals
        for x in self.goal_on_table:
            if x not in current_on_table:
                steps += 2
                # Blocks on x in current state
                blocks_on_x = current_blocks_above.get(x, [])
                steps += 2 * len(blocks_on_x)

        # Process 'clear' goals
        for x in self.goal_clear:
            blocks_on_x = current_blocks_above.get(x, [])
            steps += 2 * len(blocks_on_x)

        return steps
