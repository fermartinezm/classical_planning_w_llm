from collections import defaultdict
from heuristics.heuristic_base import Heuristic

class blocksworld16Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goal by considering:
    - The number of blocks not in their correct positions.
    - The number of blocks that need to be moved to clear the way for each block's correct placement.

    # Assumptions
    - Each block can be moved directly once its current position and goal position are clear.
    - Moving a block requires two actions (pickup/unstack and stack/putdown).
    - Blocks above a target block must be moved first, contributing to the heuristic cost.

    # Heuristic Initialization
    - Extract the goal conditions to determine the desired parent (on or on-table) for each block.
    - Static facts are not used in this heuristic.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each block, determine its current parent (what it's on) and goal parent.
    2. For blocks not in the correct position:
        a. Count the number of blocks above it in the current state (need to be moved).
        b. Count the number of blocks above its goal parent in the current state (need to be moved).
        c. Add 2 actions for moving the block itself and 2 actions for each block counted in a and b.
    3. Sum the costs for all such blocks to get the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic with goal information."""
        self.goals = task.goals
        self.static = task.static  # Not used here, but part of the structure.

        # Extract goal parents for each block
        self.goal_parent = {}
        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'on' and len(parts) == 3:
                self.goal_parent[parts[1]] = parts[2]
            elif parts[0] == 'on-table' and len(parts) == 2:
                self.goal_parent[parts[1]] = 'table'

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # Check if the current state is a goal state
        if self.goals.issubset(state):
            return 0

        # Build current parent and children relationships
        current_parent = {}
        current_children = {}
        on_table = set()

        for fact in state:
            if fact.startswith('(on '):
                parts = fact[1:-1].split()
                x, y = parts[1], parts[2]
                current_parent[x] = y
                current_children[y] = x  # Each block can have only one child
            elif fact.startswith('(on-table '):
                parts = fact[1:-1].split()
                x = parts[1]
                current_parent[x] = 'table'
                on_table.add(x)

        # Build current_above for each block
        current_above = defaultdict(list)
        for block in current_parent:
            current_above[block] = []
            current_child = current_children.get(block, None)
            while current_child is not None:
                current_above[block].append(current_child)
                current_child = current_children.get(current_child, None)

        # Calculate heuristic cost
        heuristic_cost = 0

        for block in current_parent:
            current_p = current_parent[block]
            goal_p = self.goal_parent.get(block, 'table')

            if current_p != goal_p:
                # Blocks above the current block
                a = len(current_above[block])

                # Blocks above the goal parent
                if goal_p == 'table':
                    b = 0
                else:
                    b = len(current_above.get(goal_p, []))

                # Each block contributes 2*(a + b + 1) to the cost
                heuristic_cost += 2 * (a + b + 1)

        return heuristic_cost
