from fnmatch import fnmatch
from collections import defaultdict
from heuristics.heuristic_base import Heuristic

class blocksworld13Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goal by considering:
    - Misplaced goal blocks (2 actions each).
    - Blocks above correctly placed goal blocks (1 action each).
    - Non-goal blocks on top of goal blocks (1 action each).
    - Blocks currently being held (2 actions each).

    # Assumptions
    - The goal is a conjunction of 'on' and 'on-table' predicates.
    - Non-goal blocks do not need to be in specific positions but must be moved if they obstruct goal blocks.

    # Heuristic Initialization
    - Extract goal conditions to identify goal blocks and their target positions.
    - Static facts are not used as Blocksworld has no static predicates affecting movement.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the goal to identify which blocks are part of the goal structure and their required positions.
    2. For each block in the current state:
        a. If it's a goal block and misplaced, add 2 actions.
        b. If it's a correctly placed goal block, add 1 action for each block above it.
        c. If it's a non-goal block on a goal block or being held, add 1 or 2 actions respectively.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goal_blocks = set()
        self.goal_on = dict()  # Maps block to its goal parent (None for on-table)

        for goal in task.goals:
            parts = goal.strip('()').split()
            if parts[0] == 'on':
                child, parent = parts[1], parts[2]
                self.goal_blocks.add(child)
                self.goal_blocks.add(parent)
                self.goal_on[child] = parent
            elif parts[0] == 'on-table':
                block = parts[1]
                self.goal_blocks.add(block)
                self.goal_on[block] = None  # Indicates should be on-table

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal from the given state."""
        state = node.state
        current_on = {}
        blocks_above = defaultdict(list)
        all_blocks = set()

        # Parse current state to build current_on and blocks_above
        for fact in state:
            parts = fact.strip('()').split()
            if parts[0] == 'on':
                child, parent = parts[1], parts[2]
                current_on[child] = parent
                all_blocks.add(child)
                all_blocks.add(parent)
                blocks_above[parent].append(child)
            elif parts[0] == 'on-table':
                block = parts[1]
                current_on[block] = None
                all_blocks.add(block)
            elif parts[0] == 'holding':
                block = parts[1]
                current_on[block] = 'holding'
                all_blocks.add(block)

        heuristic = 0

        for block in all_blocks:
            if block in self.goal_blocks:
                # Handle goal blocks
                desired_parent = self.goal_on.get(block, None)
                current_parent = current_on.get(block, None)

                if current_parent == 'holding':
                    heuristic += 2
                else:
                    if desired_parent is None:
                        # Block should be on-table
                        if current_parent is not None:
                            heuristic += 2  # Not on-table
                        else:
                            # Correctly on-table, check blocks above
                            for above in blocks_above.get(block, []):
                                heuristic += 1
                    else:
                        # Block should be on desired_parent
                        if current_parent != desired_parent:
                            heuristic += 2
                        else:
                            # Correctly placed, check blocks above
                            for above in blocks_above.get(block, []):
                                heuristic += 1
            else:
                # Handle non-goal blocks
                current_parent = current_on.get(block, None)
                if current_parent == 'holding':
                    heuristic += 2
                else:
                    if current_parent in self.goal_blocks:
                        heuristic += 1

        return heuristic
