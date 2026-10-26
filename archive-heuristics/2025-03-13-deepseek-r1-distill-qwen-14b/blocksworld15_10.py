from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class BlocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to rearrange blocks from their current state to the goal state. It does this by comparing the current stack of blocks with the goal stack and counting the number of blocks that are out of place. Each block that is not in its correct position contributes two actions (pickup and putdown) to the heuristic value.

    # Assumptions:
    - The goal is a specific arrangement of blocks on top of each other.
    - Each block that is not in its correct position requires two actions to move.

    # Heuristic Initialization
    - Extract the goal conditions and static facts from the task.
    - Build the goal stack from the goal facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to build the current stack of blocks from top to bottom.
    2. Parse the goal state to build the goal stack of blocks from top to bottom.
    3. Compare the current stack with the goal stack from top to bottom.
    4. Count the number of blocks that are not in their correct positions.
    5. The heuristic value is twice the count of mismatched blocks.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.goal_stack = self.build_stack(task.goals)

    def build_stack(self, facts):
        """Build the stack of blocks from the given facts."""
        on_map = {}
        table_blocks = set()
        for fact in facts:
            if fact.startswith('(on '):
                parts = get_parts(fact)
                obj = parts[1]
                under = parts[2]
                on_map[obj] = under
            elif fact.startswith('(on-table '):
                obj = get_parts(fact)[1]
                table_blocks.add(obj)
        stack = []
        for block in table_blocks:
            current = block
            while current in on_map:
                current = on_map[current]
                stack.append(current)
        stack.reverse()
        return stack

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        current_stack = self.build_stack(state)
        goal_stack = self.goal_stack
        count = 0
        for i in range(len(current_stack)):
            if i >= len(goal_stack) or current_stack[i] != goal_stack[i]:
                count += 1
        return count * 2
