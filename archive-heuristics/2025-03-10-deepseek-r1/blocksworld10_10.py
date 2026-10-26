from collections import defaultdict
from heuristics.heuristic_base import Heuristic

class blocksworld10Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goal by considering:
    - Blocks not in their correct position require pickup and place actions.
    - Blocks above the current block that need to be moved.
    - Blocks required in the goal stack that are not yet in place.

    # Assumptions
    - Each block not in the correct position requires at least two actions (pickup and place).
    - Blocks above the current block must be moved first, adding one action per block.
    - Blocks needed in the goal stack add one action per missing block.

    # Heuristic Initialization
    - Extracts goal conditions to determine where each block should be.
    - Precomputes the number of blocks above each block in the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each block, check if it's in the correct position (on the right block or table).
    2. If not, add 2 actions (pickup and place) if on table or another block, or 1 action if held.
    3. Add the number of blocks above it in the current state (to be moved first).
    4. Add the number of blocks required above it in the goal (to be placed later).
    5. If in the correct position, add the difference between current and goal blocks above.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract all blocks from the initial state
        self.blocks = set()
        for fact in task.initial_state:
            parts = fact[1:-1].split()
            if parts[0] == 'on' and len(parts) == 3:
                self.blocks.add(parts[1])
                self.blocks.add(parts[2])
            elif parts[0] in ['on-table', 'clear', 'holding'] and len(parts) == 2:
                self.blocks.add(parts[1])
        self.blocks = sorted(self.blocks)

        # Extract goal structure
        self.goal_parent = {}
        for fact in self.goals:
            if fact.startswith('(on '):
                parts = fact[1:-1].split()
                self.goal_parent[parts[1]] = parts[2]
            elif fact.startswith('(on-table '):
                parts = fact[1:-1].split()
                self.goal_parent[parts[1]] = 'table'

        # Build reverse mapping for goal
        reverse_goal = defaultdict(list)
        for child, parent in self.goal_parent.items():
            if parent != 'table':
                reverse_goal[parent].append(child)

        # Compute goal_above for each block
        self.goal_above = {}
        memo = {}
        def compute_goal_above(block):
            if block in memo:
                return memo[block]
            count = 0
            for child in reverse_goal.get(block, []):
                count += 1 + compute_goal_above(child)
            memo[block] = count
            return count

        for block in self.blocks:
            self.goal_above[block] = compute_goal_above(block)

    def __call__(self, node):
        state = node.state

        # Determine current parent and if held
        current_parent = {}
        for block in self.blocks:
            current_parent[block] = None
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'on' and len(parts) == 3:
                current_parent[parts[1]] = parts[2]
            elif parts[0] == 'on-table' and len(parts) == 2:
                current_parent[parts[1]] = 'table'
            elif parts[0] == 'holding' and len(parts) == 2:
                current_parent[parts[1]] = 'held'

        # Build reverse mapping for current state
        reverse_current = defaultdict(list)
        for child, parent in current_parent.items():
            if parent not in ['table', 'held']:
                reverse_current[parent].append(child)

        # Compute current_above for each block
        current_above = {}
        memo_current = {}
        def compute_current_above(block):
            if block in memo_current:
                return memo_current[block]
            count = 0
            for child in reverse_current.get(block, []):
                count += 1 + compute_current_above(child)
            memo_current[block] = count
            return count

        for block in self.blocks:
            current_above[block] = compute_current_above(block)

        # Calculate heuristic value
        heuristic_value = 0
        for block in self.blocks:
            current_p = current_parent.get(block)
            goal_p = self.goal_parent.get(block)

            if current_p != goal_p:
                if current_p == 'held':
                    heuristic_value += 1
                else:
                    heuristic_value += 2
                heuristic_value += current_above[block] + self.goal_above[block]
            else:
                heuristic_value += abs(current_above[block] - self.goal_above[block])

        return heuristic_value
