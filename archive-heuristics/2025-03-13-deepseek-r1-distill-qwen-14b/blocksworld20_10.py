from heuristics.heuristic_base import Heuristic

class BlocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to reach the goal state by counting the number of blocks that are not in their correct positions.

    # Assumptions:
    - The goal state defines the correct structure of blocks.
    - Each block that is not in its correct position contributes to the heuristic value.
    - Extra blocks not present in the goal state are counted as needing to be moved.

    # Heuristic Initialization
    - Extract the goal structure and identify all goal blocks and their correct parents.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the goal state to build a dictionary mapping each block to its correct parent.
    2. Parse the current state to build a similar dictionary.
    3. For each block in the goal structure:
       a. If the block is not in the current state, increment the count.
       b. If the block is in the current state, check if its parent matches the goal parent. If not, increment the count.
    4. For each block in the current state that is not in the goal structure, increment the count.
    5. Return the total count as the heuristic value.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract goal structure
        self.goal_parent = self.build_parent_dict(task.goals)
        self.goal_blocks = set(self.goal_parent.keys())

    def build_parent_dict(self, facts):
        parent = {}
        for fact in facts:
            if fact.startswith('(on '):
                content = fact[4:-1]
                parts = content.split(' ')
                if len(parts) == 3:
                    obj = parts[1]
                    under = parts[2]
                    parent[obj] = under
            elif fact.startswith('(on-table '):
                content = fact[10:-1]
                parts = content.split(' ')
                if len(parts) == 1:
                    obj = parts[0]
                    parent[obj] = 'table'
        return parent

    def __call__(self, node):
        state = node.state

        current_parent = self.build_parent_dict(state)
        current_blocks = set(current_parent.keys())

        heuristic = 0

        # Count blocks in goal but not in current or parent mismatch
        for block in self.goal_blocks:
            if block not in current_parent:
                heuristic += 1
            else:
                if current_parent[block] != self.goal_parent.get(block, 'table'):
                    heuristic += 1

        # Count extra blocks in current but not in goal
        for block in current_blocks - self.goal_blocks:
            heuristic += 1

        return heuristic
