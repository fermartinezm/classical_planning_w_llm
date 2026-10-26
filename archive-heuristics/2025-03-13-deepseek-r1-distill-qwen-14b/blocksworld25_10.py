from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class BlocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to stack blocks into the goal configuration by counting the number of blocks that are not in their correct positions in the goal stack. Each misplaced block contributes 2 actions (pickup and stack).

    # Assumptions:
    - The goal is to achieve a specific stack of blocks.
    - Each block that is not in its correct position in the goal stack requires two actions: one to pick it up and one to stack it correctly.
    - Blocks that are in their correct position do not contribute to the heuristic.

    # Heuristic Initialization
    - Extract the goal stack from the task's goals.
    - Build the current stack from the state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the goal stack from the task's goals.
    2. For the given state, build the current stack.
    3. Compare each block in the current stack with the goal stack.
    4. Count the number of blocks that are not in their correct positions.
    5. Multiply the count by 2 to estimate the number of actions needed.
    """

    @staticmethod
    def build_stack(facts):
        on_dict = {}
        on_table = None
        for fact in facts:
            if fact.startswith('(on '):
                parts = fact[1:-1].split()
                obj = parts[1]
                on = parts[2]
                on_dict[on] = obj
            elif fact.startswith('(on-table '):
                on_table = fact[1:-1].split()[1]
        if on_table is None:
            return []
        stack = []
        current = on_table
        stack.append(current)
        while current in on_dict:
            current = on_dict[current]
            stack.append(current)
        return stack

    def __init__(self, task):
        self.goals = task.goals
        self.goal_stack = self.build_stack(self.goals)

    def __call__(self, node):
        state = node.state
        current_stack = self.build_stack(state)
        count = 0
        for i in range(len(self.goal_stack)):
            if i >= len(current_stack) or current_stack[i] != self.goal_stack[i]:
                count += 1
        return count * 2
