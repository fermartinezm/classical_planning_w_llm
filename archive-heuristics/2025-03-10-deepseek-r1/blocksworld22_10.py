from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class blocksworld22Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goal by counting the number of blocks that are part of incorrect stacks. Each misplaced block contributes 2 actions (unstack and stack).

    # Assumptions
    - The goal specifies the desired state for all relevant blocks.
    - A block is considered misplaced if it or any block below it in the stack is not in the correct position.

    # Heuristic Initialization
    - Extract the target configuration from the goal conditions, specifically the 'on' and 'on-table' predicates.
    - Build a dictionary mapping each block to its target under-block (or None if it should be on the table).

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the goal to determine the target under-block for each block.
    2. For each block in the current state, determine its current under-block.
    3. For each block mentioned in the goal, check if the entire stack (from the block down to the table) matches the target configuration.
    4. Count the number of blocks that are part of incorrect stacks.
    5. Multiply the count by 2 to estimate the number of actions (unstack and stack per misplaced block).
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.target_under = {}
        for goal in task.goals:
            if isinstance(goal, str):
                if fnmatch(goal, '(on * *)'):
                    parts = goal[1:-1].split()
                    block = parts[1]
                    under = parts[2]
                    self.target_under[block] = under
                elif fnmatch(goal, '(on-table *)'):
                    parts = goal[1:-1].split()
                    block = parts[1]
                    self.target_under[block] = None

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        current_under = {}
        on_table = set()

        # Extract current block positions
        for fact in state:
            if fnmatch(fact, '(on * *)'):
                parts = fact[1:-1].split()
                block = parts[1]
                under = parts[2]
                current_under[block] = under
            elif fnmatch(fact, '(on-table *)'):
                parts = fact[1:-1].split()
                block = parts[1]
                current_under[block] = None
                on_table.add(block)

        # Check correctness of each block's stack
        misplaced = 0
        for block in self.target_under:
            current_block = block
            correct = True
            while current_block is not None:
                # Get target under-block
                target_under = self.target_under.get(current_block, None)
                # Get current under-block
                actual_under = current_under.get(current_block, None)
                if target_under != actual_under:
                    correct = False
                    break
                current_block = actual_under
            if not correct:
                misplaced += 1

        return 2 * misplaced
