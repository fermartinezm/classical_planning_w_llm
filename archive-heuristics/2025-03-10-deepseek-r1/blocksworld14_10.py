from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class blocksworld14Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goal by considering:
    1. Each block that is not in its goal position.
    2. Each block that is above a misplaced block and needs to be moved first.

    # Assumptions
    - Moving a block requires two actions (pickup and putdown/stack).
    - Blocks above a misplaced block must be moved before the misplaced block can be addressed.

    # Heuristic Initialization
    - Extracts goal conditions to determine the correct positions for each block.
    - Maps each block to its goal under-block or on-table status.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each block, determine if it is in its goal position.
    2. For each misplaced block, add 2 to the heuristic value.
    3. For each block above a misplaced block, add 2 to the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract all blocks from the task's objects (assuming objects are of type 'object')
        self.blocks = set()
        for obj in task.objects:
            if obj.type_name == 'object':
                self.blocks.add(obj.name)

        # Build goal structure: maps each block to its goal under-block (None for on-table)
        self.goal_under = {}
        self.goal_on_table = set()
        for fact in self.goals:
            parts = fact[1:-1].split()
            if parts[0] == 'on':
                self.goal_under[parts[1]] = parts[2]
            elif parts[0] == 'on-table':
                self.goal_on_table.add(parts[1])

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal from the given state."""
        state = node.state

        # Build current on and on-table relations
        current_under = {}
        current_on_table = set()
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'on':
                current_under[parts[1]] = parts[2]
            elif parts[0] == 'on-table':
                current_on_table.add(parts[1])

        # Determine misplaced blocks and blocks above them
        misplaced = set()
        blocks_above_misplaced = set()

        for block in self.blocks:
            # Check if the block is part of the goal
            if block in self.goal_under or block in self.goal_on_table:
                # Determine correct position
                if block in self.goal_under:
                    correct_under = self.goal_under[block]
                    current_under_block = current_under.get(block, None)
                    if current_under_block != correct_under:
                        misplaced.add(block)
                else:  # block is supposed to be on-table
                    if block not in current_on_table:
                        misplaced.add(block)

        # For each misplaced block, find all blocks above it
        for block in misplaced:
            current = block
            while True:
                # Check all blocks above 'current'
                above = [b for b, under in current_under.items() if under == current]
                for b in above:
                    blocks_above_misplaced.add(b)
                if not above:
                    break
                current = above[0]  # Only one block can be on top of another

        # Each misplaced block contributes 2, each block above contributes 2
        total = 2 * (len(misplaced) + len(blocks_above_misplaced))
        return total
