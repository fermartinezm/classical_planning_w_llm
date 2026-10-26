from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(on a b)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class blocksworld8Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goal state in the Blocksworld domain.
    It considers the number of blocks that are not in their goal positions, the number of blocks that have
    incorrect blocks on top of them, and the number of blocks that are clear in the goal but not clear in the current state.

    # Assumptions
    - Each block needs to be moved at least once if it's not in the correct position.
    - Stacking or unstacking a block requires at least one action.
    - The arm can only hold one block at a time.

    # Heuristic Initialization
    - The heuristic initializes by extracting the goal state from the task definition.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize a counter for the heuristic value.
    2. Extract the goal state from the task.
    3. Identify blocks that are not in their goal positions.
    4. Identify blocks that have incorrect blocks on top of them.
    5. Identify blocks that are clear in the goal but not clear in the current state.
    6. Sum up the number of misplaced blocks, incorrectly stacked blocks, and blocks that need to be cleared.
    7. Return the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        heuristic_value = 0

        # Blocks that are on the wrong block
        misplaced_blocks = 0
        # Blocks that have a wrong block on top
        incorrectly_stacked_blocks = 0
        # Blocks that should be clear but are not
        blocks_to_clear = 0

        goal_on = {}
        goal_on_table = set()
        goal_clear = set()

        for goal in self.goals:
            if match(goal, "on", "*", "*"):
                parts = get_parts(goal)
                goal_on[parts[1]] = parts[2]
            elif match(goal, "on-table", "*"):
                parts = get_parts(goal)
                goal_on_table.add(parts[1])
            elif match(goal, "clear", "*"):
                parts = get_parts(goal)
                goal_clear.add(parts[1])

        current_on = {}
        current_on_table = set()
        current_clear = set()

        for fact in state:
            if match(fact, "on", "*", "*"):
                parts = get_parts(fact)
                current_on[parts[1]] = parts[2]
            elif match(fact, "on-table", "*"):
                parts = get_parts(fact)
                current_on_table.add(parts[1])
            elif match(fact, "clear", "*"):
                parts = get_parts(fact)
                current_clear.add(parts[1])

        # Check misplaced blocks
        for block, target in goal_on.items():
            if block not in current_on or current_on[block] != target:
                misplaced_blocks += 1

        # Check incorrectly stacked blocks
        for block, target in current_on.items():
            if block not in goal_on or goal_on[block] != target:
                incorrectly_stacked_blocks += 1

        # Check blocks to clear
        for block in goal_clear:
            if block not in current_clear:
                blocks_to_clear += 1

        heuristic_value = misplaced_blocks + incorrectly_stacked_blocks + blocks_to_clear

        return heuristic_value
