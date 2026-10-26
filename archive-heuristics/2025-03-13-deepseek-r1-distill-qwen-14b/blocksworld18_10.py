from heuristics.heuristic_base import Heuristic

class BlocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to stack all blocks according to the goal configuration. It counts the number of blocks that are not in their correct positions and the number of blocks above them that are also incorrect.

    # Assumptions:
    - The goal state defines a specific stack configuration of blocks.
    - Each block has a target position in the goal stack, either on another block or the table.
    - Blocks above an incorrectly placed block must be moved, contributing to the heuristic value.

    # Heuristic Initialization
    - Extract the goal conditions to determine the correct stack structure for each block.
    - Build a mapping from each block to its target block in the goal stack.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the goal state to build the correct stack structure, mapping each block to its target.
    2. For each block in the current state, determine its current position.
    3. If a block is not in the goal stack, count it and all blocks above it in the current stack.
    4. If a block is in the goal stack but not on the correct target, count it and all blocks above it that are incorrect.
    5. Sum these counts to estimate the number of actions needed.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Build the correct 'on' mapping from the goal state
        self.correct_on = {}
        for fact in self.goals:
            if fact.startswith('(on '):
                parts = fact[4:-1].split(' ')
                block = parts[0]
                target = parts[2]
                self.correct_on[block] = target

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        current_on = {}
        for fact in state:
            if fact.startswith('(on '):
                parts = fact[4:-1].split(' ')
                obj = parts[0]
                target = parts[2]
                current_on[obj] = target

        heuristic = 0

        for block in current_on:
            if block not in self.correct_on:
                # This block is not part of the goal stack; count it and all above
                current = block
                count = 0
                while current in current_on:
                    count += 1
                    current = current_on[current]
                heuristic += count
            else:
                # Check if the block is on the correct target
                if current_on[block] != self.correct_on[block]:
                    current = block
                    count = 0
                    while current in current_on:
                        count += 1
                        current = current_on[current]
                    heuristic += count

        return heuristic
