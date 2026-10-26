from heuristics.heuristic_base import Heuristic

class blocksworld19Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goal by considering:
    - The number of blocks that need to be moved to their correct positions.
    - The number of blocks that are currently on top of blocks that need to be clear.
    - The number of blocks that are on top of their target positions and need to be moved.

    # Assumptions
    - The goal consists of 'on', 'on-table', and 'clear' predicates.
    - Blocks not mentioned in the goal can be in any state.
    - Moving a block requires two actions (pickup and stack/putdown).
    - Each block on top of a misplaced block or a target position requires additional actions.

    # Heuristic Initialization
    - Extract 'on', 'on-table', and 'clear' conditions from the goal.
    - Build mappings for each block's target parent and required clear status.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each block required to be 'on' or 'on-table':
        a. If not in the correct position, add 2 actions for moving it.
        b. Add 1 action for each block stacked on top of it.
        c. If the target parent is a block, add 2 actions for each block currently on it.
    2. For each block required to be 'clear':
        a. Add 1 action for each block currently on top of it.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goal_parent = {}  # Maps each block to its target parent (block or 'table')
        self.goal_clear = set()  # Set of blocks that must be clear

        # Parse the goal conditions
        for fact in task.goals:
            parts = fact.strip('()').split()
            predicate = parts[0]
            if predicate == 'on':
                block, parent = parts[1], parts[2]
                self.goal_parent[block] = parent
            elif predicate == 'on-table':
                block = parts[1]
                self.goal_parent[block] = 'table'
            elif predicate == 'clear':
                block = parts[1]
                self.goal_clear.add(block)

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        current_parents = {}  # Maps each block to its current parent

        # Parse the current state to determine parent relationships
        for fact in state:
            parts = fact.strip('()').split()
            if parts[0] == 'on':
                block, parent = parts[1], parts[2]
                current_parents[block] = parent
            elif parts[0] == 'on-table':
                block = parts[1]
                current_parents[block] = 'table'

        cost = 0

        # Check blocks that need to be in specific positions
        for block, target_parent in self.goal_parent.items():
            current_parent = current_parents.get(block, 'table')
            if current_parent != target_parent:
                # Cost to move the block to its target position
                cost += 2

                # Blocks stacked on top of this block in the current state
                blocks_above = [b for b, p in current_parents.items() if p == block]
                cost += len(blocks_above)

                # If target is a block, check blocks currently on it
                if target_parent != 'table':
                    blocks_on_target = [b for b, p in current_parents.items() if p == target_parent]
                    cost += 2 * len(blocks_on_target)

        # Check blocks that need to be clear
        for block in self.goal_clear:
            blocks_on = [b for b, p in current_parents.items() if p == block]
            cost += len(blocks_on)

        return cost
