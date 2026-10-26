from fnmatch import fnmatch
from collections import defaultdict
from heuristics.heuristic_base import Heuristic

class blocksworld9Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goal by considering the number of blocks that need to be moved and the blocks above them in the current state. For each block not in its goal position, the cost is 2 times the number of blocks above it plus 2. Held blocks not in the goal position add 1 to the cost.

    # Assumptions
    - The goal specifies the required 'on' and 'on-table' predicates for certain blocks.
    - Blocks not mentioned in the goal do not affect the heuristic.
    - Moving a block requires unstacking all blocks above it first.
    - The arm can only carry one block at a time.

    # Heuristic Initialization
    - Extract the goal conditions to determine the target positions for each block.
    - Store the goal parent (block or 'table') for each block mentioned in the goal.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to determine the current parent (block or 'table') and children (blocks on top) for each block.
    2. Check if any block is currently being held.
    3. For each block mentioned in the goal:
        a. If the block is held and not in the goal position, add 1 action (putdown or stack).
        b. If the block is not held and not in the goal position, calculate the number of blocks above it and add 2 actions per block (unstack and putdown) plus 2 actions for the block itself.
    4. Sum all calculated costs to get the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goal_parent = {}
        for goal in task.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'on':
                block, parent = parts[1], parts[2]
                self.goal_parent[block] = parent
            elif parts[0] == 'on-table':
                block = parts[1]
                self.goal_parent[block] = 'table'

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        current_parent = {}
        current_children = defaultdict(list)
        held_block = None

        # Parse current state
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'on':
                child, parent = parts[1], parts[2]
                current_parent[child] = parent
                current_children[parent].append(child)
            elif parts[0] == 'on-table':
                block = parts[1]
                current_parent[block] = 'table'
            elif parts[0] == 'holding':
                held_block = parts[1]

        # Calculate cost
        cost = 0

        # Check held block
        if held_block is not None and held_block in self.goal_parent:
            current_pos = 'held'
            goal_pos = self.goal_parent[held_block]
            # If held block is not in goal position, add 1 action
            if (goal_pos == 'table' and current_pos != 'table') or \
               (goal_pos != 'table' and current_pos != goal_pos):
                cost += 1

        # Process each block in the goal
        for block in self.goal_parent:
            if block == held_block:
                continue  # Already handled

            current_parent_block = current_parent.get(block, 'table')
            goal_parent_block = self.goal_parent[block]

            if current_parent_block != goal_parent_block:
                # Calculate number of blocks above the current block
                def count_above(x):
                    cnt = 0
                    stack = [x]
                    while stack:
                        current = stack.pop()
                        for child in current_children.get(current, []):
                            cnt += 1
                            stack.append(child)
                    return cnt

                above = count_above(block)
                cost += 2 * (above + 1)

        return cost
