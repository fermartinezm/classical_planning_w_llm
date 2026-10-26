from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class BlocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not in their correct positions in the goal stack.

    # Assumptions:
    - Each block has a specific position in the goal stack.
    - The table is considered the base and is static.
    - A block is in the correct position if it is directly on top of its designated block or on the table if it's the base.

    # Heuristic Initialization
    - Extract the goal conditions to determine the correct parent for each block.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the goal state to build a dictionary mapping each block to its correct parent.
    2. Parse the current state to build a dictionary mapping each block to its current parent.
    3. For each block in the goal, check if its current parent matches the correct parent. If not, increment the heuristic.
    4. Check for any blocks in the current state that are not in the goal and increment the heuristic for each.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        static_facts = task.static

        # Extract correct parent for each block from the goal state.
        self.correct_parent = {}
        for fact in self.goals:
            if fact.startswith('(on '):
                parts = fact[4:-1].split(' ')
                obj = parts[1]
                parent = parts[2]
                self.correct_parent[obj] = parent

    def __call__(self, node):
        """Estimate the number of blocks not in their correct positions."""
        state = node.state

        # Extract current parent for each block from the current state.
        current_parent = {}
        for fact in state:
            if fact.startswith('(on '):
                parts = fact[4:-1].split(' ')
                obj = parts[1]
                parent = parts[2]
                current_parent[obj] = parent

        heuristic = 0

        # Check each block in the goal to see if it's correctly placed.
        for block in self.correct_parent:
            correct_p = self.correct_parent[block]
            current_p = current_parent.get(block, 'table')
            if current_p != correct_p:
                heuristic += 1

        # Check for extra blocks in the current state not present in the goal.
        for fact in state:
            if fact.startswith('(on '):
                parts = fact[4:-1].split(' ')
                block = parts[1]
                if block not in self.correct_parent:
                    heuristic += 1

        return heuristic
