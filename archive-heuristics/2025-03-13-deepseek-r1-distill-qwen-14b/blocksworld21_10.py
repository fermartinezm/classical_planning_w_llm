from heuristics.heuristic_base import Heuristic

class BlocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to stack all blocks according to the goal configuration.

    # Assumptions:
    - The goal defines a specific stack of blocks.
    - Each block not in the correct position in the goal stack contributes to the heuristic value.

    # Heuristic Initialization
    - Extract the goal stack from the task's goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the goal to determine the target stack order.
    2. Parse the current state to determine the current stack order.
    3. Find the longest prefix where the current stack matches the goal stack.
    4. The heuristic is calculated as twice the number of blocks in the goal stack beyond the longest prefix, plus the number of extra blocks in the current stack.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting the goal stack."""
        self.goals = task.goals
        static_facts = task.static

        # Extract the goal stack
        self.goal_stack = self.build_stack(task.goals)

    def build_stack(self, facts):
        """Build the stack representation from the given facts."""
        on_dict = {}
        on_table = set()

        for fact in facts:
            parts = fact[1:-1].split()
            if parts[0] == 'on':
                on_dict[parts[1]] = parts[2]
            elif parts[0] == 'on-table':
                on_table.add(parts[1])

        # Find the bottom block (on table and not under any other block)
        bottom_block = None
        for block in on_table:
            if block not in on_dict.values():
                bottom_block = block
                break

        if bottom_block is None:
            return []  # No blocks on the table

        stack = [bottom_block]
        current = bottom_block
        while current in on_dict:
            current = on_dict[current]
            stack.append(current)

        return stack

    def __call__(self, node):
        """Compute the heuristic value."""
        state = node.state

        # Build current stack
        current_stack = self.build_stack(state)

        # Find the longest prefix
        goal_stack = self.goal_stack
        min_len = min(len(goal_stack), len(current_stack))
        longest_prefix = 0
        for i in range(min_len):
            if goal_stack[i] != current_stack[i]:
                break
            longest_prefix += 1

        # Calculate extra blocks
        extra_blocks = max(0, len(current_stack) - len(goal_stack))

        # Heuristic is twice the number of blocks beyond the prefix plus extra blocks
        heuristic = 2 * (len(goal_stack) - longest_prefix) + extra_blocks

        return heuristic
