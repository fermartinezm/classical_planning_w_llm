from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class blocksworld24Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goal state in the Blocksworld domain.
    It considers the number of blocks that are not in their goal positions and the number of blocks that have blocks
    on top of them that should not be there according to the goal state.

    # Assumptions
    - The heuristic assumes that each misplaced block requires at least one 'unstack' and one 'stack' action.
    - It also assumes that blocks that are clear in the goal but not in the current state need to be cleared.
    - The heuristic does not consider the arm-empty predicate directly, assuming it will be handled by the planner.

    # Heuristic Initialization
    - The heuristic initializes by extracting the goal state and identifying the 'on' relationships, 'on-table' relationships,
      and 'clear' relationships that must hold in the goal.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the goal state information:
       - Identify the blocks that need to be on other blocks ('on' goals).
       - Identify the blocks that need to be on the table ('on-table' goals).
       - Identify the blocks that need to be clear ('clear' goals).

    2. Analyze the current state:
       - For each 'on' goal, check if the block is currently on the correct block. If not, increment the heuristic.
       - For each 'on-table' goal, check if the block is currently on the table. If not, increment the heuristic.
       - For each 'clear' goal, check if the block is currently clear. If not, increment the heuristic.

    3. Count Misplaced Blocks:
       - Count the number of blocks that are not in their goal positions according to the 'on' and 'on-table' goals.
       - Count the number of blocks that are not clear but should be according to the 'clear' goals.

    4. Estimate Actions:
       - Each misplaced block requires at least one 'unstack' and one 'stack' action, so multiply the number of misplaced blocks by 2.
       - Each block that needs to be cleared requires at least one 'unstack' action.

    5. Return the Heuristic Value:
       - The heuristic value is the sum of the estimated actions for misplaced blocks and blocks that need to be cleared.
       - If the current state is the goal state, the heuristic value is 0.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        self.goal_on = set()
        self.goal_on_table = set()
        self.goal_clear = set()

        for goal in self.goals:
            if match(goal, "on", "*", "*"):
                self.goal_on.add(goal)
            elif match(goal, "on-table", "*"):
                self.goal_on_table.add(goal)
            elif match(goal, "clear", "*"):
                self.goal_clear.add(goal)

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        if self.goals <= state:
            return 0

        misplaced_blocks = 0
        unclear_blocks = 0

        for goal in self.goal_on:
            if goal not in state:
                misplaced_blocks += 1

        for goal in self.goal_on_table:
            if goal not in state:
                misplaced_blocks += 1

        for goal in self.goal_clear:
            if goal not in state:
                unclear_blocks += 1

        heuristic_value = 2 * misplaced_blocks + unclear_blocks

        return heuristic_value
