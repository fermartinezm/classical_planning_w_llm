from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class BlocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to stack blocks according to the goal configuration.

    # Assumptions:
    - The goal is to achieve a specific stack of blocks.
    - Each block that is not in the correct position contributes to the heuristic.
    - Blocks not in the goal stack that are above goal blocks contribute one action each.
    - Each block in the goal stack that is not correctly positioned contributes two actions.

    # Heuristic Initialization
    - Extract the goal stack from the task's goals.
    - Build the current stack from the state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the goal facts to build the goal stack, representing the desired structure.
    2. Parse the current state to build the current stack, representing the current structure.
    3. For each block in the goal stack:
       a. Count the number of non-goal blocks above it in the current stack. Each contributes one action.
       b. If the block is not directly on top of its predecessor in the goal stack, add two actions.
    4. Sum all these actions to get the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract the goal stack
        goal_facts = [fact for fact in self.goals if fact.startswith('(on ')]
        goal_on = {}
        for fact in goal_facts:
            parts = fact[4:-1].split()
            obj = parts[1]
            base = parts[2] if len(parts) > 2 else None
            goal_on[obj] = base

        # Build the goal stack
        self.goal_stack = []
        table_blocks = [fact for fact in self.goals if fact.startswith('(on-table ')]
        for fact in table_blocks:
            block = fact[9:-1]
            current_block = block
            while current_block in goal_on:
                self.goal_stack.append(current_block)
                current_block = goal_on[current_block]
        self.goal_stack.reverse()

        # Create a set of goal blocks for quick lookup
        self.goal_set = set(self.goal_stack)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Function to count non-goal blocks above a given block
        def count_non_goal_above(current_on, goal_set):
            current_base_to_blocks = {}
            for block, base in current_on.items():
                if base is not None:
                    if base not in current_base_to_blocks:
                        current_base_to_blocks[base] = []
                    current_base_to_blocks[base].append(block)

            def count_above(block):
                count = 0
                stack = [block]
                visited = set()
                while stack:
                    current = stack.pop()
                    if current in visited:
                        continue
                    visited.add(current)
                    if current in current_base_to_blocks:
                        for b in current_base_to_blocks[current]:
                            if b not in goal_set:
                                count += 1
                            if b not in visited:
                                stack.append(b)
                return count

            return count_above

        # Build current_on for the current state
        current_on = {}
        for fact in state:
            if fact.startswith('(on '):
                parts = fact[4:-1].split()
                obj = parts[1]
                base = parts[2] if len(parts) > 2 else None
                current_on[obj] = base

        heuristic = 0

        # Count non-goal blocks above each block in the goal stack
        count_above = count_non_goal_above(current_on, self.goal_set)
        for block in self.goal_stack:
            heuristic += count_above(block)

        # Check each consecutive pair in the goal stack
        for i in range(len(self.goal_stack) - 1):
            current_block = self.goal_stack[i]
            next_block = self.goal_stack[i+1]
            # Check if next_block is on current_block in the current state
            if next_block not in current_on or current_on[next_block] != current_block:
                heuristic += 2

        return heuristic
