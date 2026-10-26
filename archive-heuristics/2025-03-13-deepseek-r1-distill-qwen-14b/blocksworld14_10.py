from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class BlocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not in their correct position in the goal stack. Each such block contributes 1 to the heuristic value.

    # Assumptions:
    - The goal is to achieve a specific stack configuration.
    - Each block has a designated position in the goal stack.
    - Blocks not in their correct position need to be moved.

    # Heuristic Initialization
    - Extract the goal conditions to determine the desired parent for each block.
    - Build a map of desired parents during initialization.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the desired parent for each block from the goal conditions.
    2. For each block in the current state, determine its current parent.
    3. Compare the current parent with the desired parent.
    4. If they differ, increment the heuristic value by 1.
    5. The total heuristic is the sum of all such discrepancies.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.desired_parent = {}
        for goal in self.goals:
            if goal.startswith('(on '):
                parts = goal[4:-1].split(' ')
                x = parts[1]
                y = parts[2]
                self.desired_parent[x] = y
            elif goal.startswith('(on-table '):
                parts = goal[9:-1].split(' ')
                x = parts[1]
                self.desired_parent[x] = 'table'

    def __call__(self, node):
        state = node.state
        current_parent = {}
        for fact in state:
            if fact.startswith('(on '):
                parts = fact[4:-1].split(' ')
                obj = parts[1]
                parent = parts[2]
                current_parent[obj] = parent
            elif fact.startswith('(on-table '):
                parts = fact[9:-1].split(' ')
                obj = parts[1]
                current_parent[obj] = 'table'
        heuristic = 0
        for block in current_parent:
            if block not in self.desired_parent:
                continue
            current_p = current_parent[block]
            desired_p = self.desired_parent[block]
            if current_p != desired_p:
                heuristic += 1
        return heuristic
