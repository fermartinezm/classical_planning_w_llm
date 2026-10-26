from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class blocksworld7Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goal by considering:
    1. Blocks not in their goal positions (on/on-table).
    2. Blocks that need to be moved to satisfy 'clear' goals.
    3. Blocks currently held by the arm.

    # Assumptions
    - Each block not in its goal position requires at least two actions (pickup and putdown/stack).
    - Blocks on top of others that prevent 'clear' goals must be moved, regardless of their own goals.
    - Held blocks require one action to place.

    # Heuristic Initialization
    - Extract goal conditions for 'on', 'on-table', and 'clear' predicates.
    - Store static information about goal structures.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Process 'clear' goals**: For each block needing to be clear, move all blocks on top of it.
    2. **Process misplaced blocks**: For each block not in its goal position, add actions to move it and unstack blocks above.
    3. **Process held blocks**: Each held block adds one action to place it.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goal_on = {}        # {block: target_block}
        self.goal_on_table = set()  # Blocks that should be on-table
        self.clear_goals = set() # Blocks that should be clear

        for goal in task.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'on':
                self.goal_on[parts[1]] = parts[2]
            elif parts[0] == 'on-table':
                self.goal_on_table.add(parts[1])
            elif parts[0] == 'clear':
                self.clear_goals.add(parts[1])

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal from the given state."""
        state = node.state
        current_on = {}
        current_on_table = set()
        held_blocks = set()

        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'on':
                current_on[parts[1]] = parts[2]
            elif parts[0] == 'on-table':
                current_on_table.add(parts[1])
            elif parts[0] == 'holding' and parts[1] != 'empty':  # Assuming 'holding' fact has block as second part
                held_blocks.add(parts[1])

        all_blocks = set(current_on.keys()).union(current_on_table).union(held_blocks)
        processed_blocks = set()
        heuristic_value = 0

        # Process 'clear' goals first
        for clear_block in self.clear_goals:
            current = clear_block
            while True:
                next_block = None
                for y in current_on:
                    if current_on[y] == current:
                        next_block = y
                        break
                if next_block is None:
                    break
                if next_block not in processed_blocks:
                    processed_blocks.add(next_block)
                    if next_block in held_blocks:
                        heuristic_value += 1
                    else:
                        # Calculate blocks above next_block
                        blocks_above = 0
                        current_block = next_block
                        while True:
                            found = False
                            for y in current_on:
                                if current_on[y] == current_block:
                                    current_block = y
                                    blocks_above += 1
                                    found = True
                                    break
                            if not found:
                                break
                        heuristic_value += 2 + blocks_above
                current = next_block

        # Process other blocks not handled by clear goals and not held
        for block in all_blocks:
            if block in processed_blocks or block in held_blocks:
                continue

            # Check if block is in correct position
            correct = False
            if block in self.goal_on:
                target = self.goal_on[block]
                if current_on.get(block) == target:
                    correct = True
            elif block in self.goal_on_table:
                if block in current_on_table:
                    correct = True
            else:
                # Block not mentioned in on/on-table goals, consider correct
                correct = True

            if not correct:
                # Calculate blocks above
                blocks_above = 0
                current_block = block
                while True:
                    next_block = None
                    for y in current_on:
                        if current_on[y] == current_block:
                            next_block = y
                            break
                    if next_block is None:
                        break
                    blocks_above += 1
                    current_block = next_block
                heuristic_value += 2 + blocks_above

        # Add 1 for each held block not already processed
        for block in held_blocks:
            if block not in processed_blocks:
                heuristic_value += 1

        return heuristic_value
