from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class BlocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not in their correct positions as per the goal state. Each block that is not in the correct position contributes to the heuristic value.

    # Assumptions:
    - The goal state specifies the correct stacking of blocks.
    - Blocks not mentioned in the goal state are considered to be in their correct positions (on the table or not interfering with the goal structure).

    # Heuristic Initialization
    - Extract the goal conditions to build a target structure that maps each block to its target parent block or the table.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the goal conditions to determine the target parent for each block.
    2. Parse the current state to determine the current parent of each block.
    3. For each block in the target structure, compare its current parent with the target parent.
    4. Count the number of blocks where the current parent does not match the target parent.
    5. The heuristic value is the count of such mismatched blocks.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Build the target structure: maps each block to its target parent
        self.target = {}
        for goal in self.goals:
            if goal.startswith('(on '):
                parts = goal[4:-1].split(' ')
                x = parts[0]
                y = parts[2]
                self.target[x] = y

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Track which blocks are being held
        held_blocks = set()
        for fact in state:
            if fact.startswith('(holding '):
                x = fact.split()[1]
                held_blocks.add(x)

        # Track the current parent of each block
        current_parents = {}
        for fact in state:
            if fact.startswith('(on '):
                x = fact.split()[1]
                y = fact.split()[3]
                current_parents[x] = y

        count = 0
        for block in self.target:
            # Determine the current parent of the block
            if block in held_blocks:
                current_parent = 'held'
            elif block in current_parents:
                current_parent = current_parents[block]
            else:
                current_parent = 'table'

            # Compare with the target parent
            target_parent = self.target[block]
            if current_parent != target_parent:
                count += 1

        return count
