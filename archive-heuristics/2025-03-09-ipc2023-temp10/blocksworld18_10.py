from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class blocksworld18Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goal state in the Blocksworld domain.
    It considers the number of blocks that are not in their goal positions, the number of blocks that have
    blocks on top of them that should not be there, and the number of blocks that are clear but should not be.

    # Assumptions
    - Each block needs to be moved at most once to its correct position.
    - Moving a block involves picking it up, potentially putting down other blocks, and stacking it.
    - The arm can only hold one block at a time.

    # Heuristic Initialization
    - Extract the goal conditions from the task.
    - Identify the blocks that need to be on top of other blocks in the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize a counter for the heuristic value.
    2. Iterate through the goal conditions to identify the 'on' relationships that must be satisfied.
    3. For each 'on' goal condition, check if the relationship holds in the current state. If not, increment the counter.
    4. Iterate through the blocks in the current state and check if they are clear when they shouldn't be according to the goal. If so, increment the counter.
    5. Iterate through the blocks in the current state and check if they have blocks on top of them when they shouldn't according to the goal. If so, increment the counter.
    6. If the arm is empty and there are blocks that need to be moved, increment the counter.
    7. Return the final heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static
        self.on_goals = {}
        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'on':
                self.on_goals[parts[1]] = parts[2]

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        h = 0

        # Check if 'on' relationships are satisfied
        for block, underblock in self.on_goals.items():
            if f'(on {block} {underblock})' not in state:
                h += 1

        # Check if blocks are clear when they shouldn't be
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'clear':
                block = parts[1]
                if block in self.on_goals and f'(clear {block})' in state:
                    h += 1

        # Check if blocks have blocks on top of them when they shouldn't
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'on':
                block_above = parts[1]
                block_below = parts[2]
                if block_below not in self.on_goals.values():
                    h += 1

        # Check if arm is empty when there are blocks to move
        arm_empty = '(arm-empty)' in state
        if arm_empty and h > 0:
            h += 1

        # If the goal is reached, return 0
        goal_reached = True
        for goal in self.goals:
            if goal not in state:
                goal_reached = False
                break
        if goal_reached:
            return 0

        return h
