from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

class BlocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to reach the goal state by counting the number of blocks that are not in their correct positions and the number of blocks above them that are also incorrect.

    # Assumptions:
    - The goal is to achieve a specific stacking configuration.
    - Each block that is not in its correct position contributes to the heuristic.
    - Each block above a misplaced block that is also misplaced contributes an additional count.

    # Heuristic Initialization
    - Extract the goal conditions to determine the desired parent for each block.
    - Build a dictionary of desired parent relationships.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the goal facts to build the desired parent relationships for each block.
    2. Parse the current state to build the current parent relationships and the children structure.
    3. For each block in the goal:
        a. Check if it's misplaced (current parent doesn't match the desired parent).
        b. If misplaced, add 1 to the heuristic.
        c. Traverse all blocks above it (children in the current stack) and add 1 for each that is also misplaced.
    4. Sum all these counts to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        static_facts = task.static

        # Extract desired parent relationships from goal facts
        self.desired_parents = {}
        self.goal_blocks = set()
        for goal in self.goals:
            predicate, *args = goal[1:-1].split()
            if predicate == 'on':
                obj, parent = args
                self.desired_parents[obj] = parent
                self.goal_blocks.add(obj)
                self.goal_blocks.add(parent)

    def __call__(self, node):
        """Compute the heuristic value for the current state."""
        state = node.state

        # Check if the current state is the goal state
        if self.goals.issubset(state):
            return 0

        # Build current parent relationships and children structure
        current_parents = {}
        children = {}
        for fact in state:
            if fact.startswith('(on '):
                parts = fact[4:-1].split(' ', 2)
                obj = parts[0]
                parent = parts[1]
                current_parents[obj] = parent
                if parent not in children:
                    children[parent] = []
                children[parent].append(obj)

        heuristic = 0
        visited = set()

        for block in self.goal_blocks:
            if block not in current_parents and block != 'table':
                # Block is missing, consider as misplaced
                if block not in visited:
                    visited.add(block)
                    heuristic += 1
                    # Traverse all blocks above it
                    queue = deque(children.get(block, []))
                    while queue:
                        current = queue.popleft()
                        if current not in visited:
                            visited.add(current)
                            if current not in current_parents and current != 'table':
                                heuristic += 1
                            else:
                                desired_parent = self.desired_parents.get(current, 'table')
                                current_parent = current_parents.get(current, 'table')
                                if current_parent != desired_parent:
                                    heuristic += 1
                            queue.extend(children.get(current, []))
            else:
                if block != 'table':
                    desired_parent = self.desired_parents.get(block, 'table')
                    current_parent = current_parents.get(block, 'table')
                    if current_parent != desired_parent:
                        if block not in visited:
                            visited.add(block)
                            heuristic += 1
                            # Traverse all blocks above it
                            queue = deque(children.get(block, []))
                            while queue:
                                current = queue.popleft()
                                if current not in visited:
                                    visited.add(current)
                                    if current not in current_parents and current != 'table':
                                        heuristic += 1
                                    else:
                                        desired_parent_current = self.desired_parents.get(current, 'table')
                                        current_parent_current = current_parents.get(current, 'table')
                                        if current_parent_current != desired_parent_current:
                                            heuristic += 1
                                    queue.extend(children.get(current, []))

        return heuristic
