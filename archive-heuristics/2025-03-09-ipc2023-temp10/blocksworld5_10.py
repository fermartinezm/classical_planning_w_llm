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


class blocksworld5Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goal state in the Blocksworld domain.
    It considers the number of blocks that are not in their goal positions and the number of blocks that have blocks on top of them that should not be there.

    # Assumptions
    - The heuristic assumes that each misplaced block requires at least one 'unstack' and one 'stack' action.
    - It also assumes that blocks that are on the table in the goal state are already on the table in the initial state, or can be put there with one 'putdown' action.

    # Heuristic Initialization
    - The heuristic initializes by extracting the goal state and identifying the 'on' relationships that must exist in the goal.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize a counter for the heuristic value.
    2. Extract the goal 'on' relationships from the goal state.
    3. Iterate through the current state and check the following:
        a. For each 'on' fact in the goal, check if it exists in the current state. If not, increment the counter.
        b. Check if a block has the wrong block on top of it. If so, increment the counter.
        c. Check if a block is on the table when it shouldn't be. If so, increment the counter.
    4. Return the final heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.static = task.static
        self.goal_ons = set()
        self.goal_clear = set()
        self.goal_on_table = set()

        for goal in self.goals:
            if match(goal, "on", "*", "*"):
                self.goal_ons.add(goal)
            elif match(goal, "clear", "*"):
                self.goal_clear.add(goal)
            elif match(goal, "on-table", "*"):
                self.goal_on_table.add(goal)

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        heuristic_value = 0

        # Check if goal is already reached
        if node.task.goal_reached(state):
            return 0

        # Check 'on' relationships
        for goal_on in self.goal_ons:
            if goal_on not in state:
                heuristic_value += 1

        # Check for blocks with wrong blocks on top
        for fact in state:
            if match(fact, "on", "*", "*"):
                block_above = get_parts(fact)[1]
                block_below = get_parts(fact)[2]
                correct_on = f"(on {block_above} {block_below})" in self.goals
                if not correct_on:
                    heuristic_value += 1

        # Check for blocks on the table when they shouldn't be
        for fact in state:
            if match(fact, "on-table", "*"):
                block = get_parts(fact)[1]
                correct_on_table = f"(on-table {block})" in self.goals
                if not correct_on_table:
                    heuristic_value += 1

        return heuristic_value
