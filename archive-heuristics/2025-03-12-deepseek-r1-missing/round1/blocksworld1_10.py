from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to rearrange blocks into their goal configuration. It calculates the sum of blocks above each block in the current state and the goal state, then doubles this sum.

    # Assumptions:
    - Each block's goal position is specified in the problem's goal conditions.
    - Blocks not mentioned in the goal can be ignored.
    - Moving a block requires two actions (pickup and putdown/stack), and each block above it must be moved first.

    # Heuristic Initialization
    - Extract goal conditions to determine each block's target position.
    - Precompute the number of blocks above each block in the goal stack.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each block in the goal, determine its target parent (another block or table).
    2. Build a hierarchy of blocks in the goal to compute how many blocks are above each block.
    3. For the current state, build a hierarchy of blocks to compute how many are above each block.
    4. For each block in the goal, sum the current and goal blocks above it.
    5. Multiply the total sum by 2 to estimate the required actions.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building goal hierarchies."""
        self.goal_parents = {}
        self.goal_children = defaultdict(list)
        self.goal_above = {}

        # Extract goal parents
        for fact in task.goals:
            parts = fact[1:-1].split()
            if parts[0] == 'on':
                block = parts[1]
                parent = parts[2]
                self.goal_parents[block] = parent
                self.goal_children[parent].append(block)
            elif parts[0] == 'on-table':
                block = parts[1]
                self.goal_parents[block] = 'table'

        # Precompute goal_above for each block in the goal
        for block in self.goal_parents:
            self.goal_above[block] = self._compute_goal_above(block)

    def _compute_goal_above(self, block):
        """Recursively compute the number of blocks above this block in the goal stack."""
        count = 0
        for child in self.goal_children.get(block, []):
            count += 1 + self._compute_goal_above(child)
        return count

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal from the current state."""
        state = node.state
        current_parents = {}
        current_children = defaultdict(list)

        # Parse current state to build parent-child relationships
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'on':
                child = parts[1]
                parent = parts[2]
                current_parents[child] = parent
                current_children[parent].append(child)
            elif parts[0] == 'on-table':
                block = parts[1]
                current_parents[block] = 'table'

        # Compute current_above for each block in the goal
        current_above = {}
        for block in self.goal_parents:
            count = 0
            stack = [block]
            while stack:
                current = stack.pop()
                for child in current_children.get(current, []):
                    count += 1
                    stack.append(child)
            current_above[block] = count

        # Calculate total heuristic value
        total = 0
        for block in self.goal_parents:
            total += current_above.get(block, 0)
            total += self.goal_above.get(block, 0)

        return total * 2
