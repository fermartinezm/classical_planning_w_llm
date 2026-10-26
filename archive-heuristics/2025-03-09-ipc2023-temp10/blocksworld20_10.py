from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class blocksworld20Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goal state in the Blocksworld domain.
    It counts the number of blocks that are not in their goal positions (either on the wrong block or on the table
    when they shouldn't be, or not clear when they should be). Each misplaced block contributes to the heuristic value.

    # Assumptions
    - Each block needs at most one move to reach its correct position.
    - The arm can only hold one block at a time.
    - Moving a block to its correct position might require moving other blocks first.

    # Heuristic Initialization
    - Extract the goal conditions from the task.
    - Store the goal 'on' relationships in a dictionary for easy access.
    - Store the goal 'clear' relationships in a set.
    - Store the blocks that should be on the table in a set.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize a counter for misplaced blocks.
    2. Iterate through the 'on' goal conditions and check if each block is in the correct position in the current state.
       - If a block is not on the correct block, increment the counter.
    3. Iterate through the 'on-table' goal conditions and check if each block is on the table in the current state.
       - If a block is not on the table, increment the counter.
    4. Iterate through the 'clear' goal conditions and check if each block is clear in the current state.
       - If a block is not clear, increment the counter.
    5. If the goal state is reached, return 0.
    6. Return the total count of misplaced blocks as the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        self.goal_on = {}  # Dictionary to store goal 'on' relationships: {block: block_underneath}
        self.goal_clear = set()  # Set to store goal 'clear' relationships: {block}
        self.goal_on_table = set() # Set to store goal 'on-table' relationships: {block}

        for goal in self.goals:
            if goal.startswith('(on '):
                parts = goal[1:-1].split()
                block, underneath = parts[1], parts[2]
                self.goal_on[block] = underneath
            elif goal.startswith('(clear '):
                block = goal[7:-1]
                self.goal_clear.add(block)
            elif goal.startswith('(on-table '):
                block = goal[10:-1]
                self.goal_on_table.add(block)

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        misplaced_blocks = 0

        # Check if the goal is reached
        if self.goals <= state:
            return 0

        # Check 'on' goal conditions
        for block, underneath in self.goal_on.items():
            on_correct_block = False
            for fact in state:
                if fact.startswith('(on '):
                    parts = fact[1:-1].split()
                    b1, b2 = parts[1], parts[2]
                    if b1 == block and b2 == underneath:
                        on_correct_block = True
                        break
            if not on_correct_block:
                misplaced_blocks += 1

        # Check 'on-table' goal conditions
        for block in self.goal_on_table:
            on_table = False
            for fact in state:
                if fact.startswith('(on-table '):
                    b = fact[10:-1]
                    if b == block:
                        on_table = True
                        break
            if not on_table:
                misplaced_blocks += 1

        # Check 'clear' goal conditions
        for block in self.goal_clear:
            is_clear = False
            for fact in state:
                if fact.startswith('(clear '):
                    b = fact[7:-1]
                    if b == block:
                        is_clear = True
                        break
            if not is_clear:
                misplaced_blocks += 1

        return misplaced_blocks
