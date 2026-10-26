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


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to reach the goal state in the Blocksworld domain.
    It counts the number of blocks that are not in their goal positions and adds penalties for blocks that are on top of blocks that should be clear in the goal, or blocks that are on top of blocks that should be on the table in the goal.

    # Assumptions:
    - The goal state is well-defined in terms of `on`, `on-table`, and `clear` predicates.

    # Heuristic Initialization
    - Extracts goal predicates from the task definition.
    - Preprocesses goal predicates to easily check against the current state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Extract goal predicates of type 'on', 'on-table', and 'clear'.
    3. For each goal predicate 'on(b1, b2)':
       - Check if 'on(b1, b2)' is true in the current state. If not, increment the heuristic.
       - Check if 'clear(b2)' is a goal and if 'clear(b2)' is not true in the current state. If not clear and should be clear in goal, increment heuristic.
       - Check if there is any block 'bx' such that 'on(bx, b1)' is true in the current state. If yes, increment heuristic.
    4. For each goal predicate 'on-table(b)':
       - Check if 'on-table(b)' is true in the current state. If not, increment the heuristic.
       - Check if there is any block 'bx' such that 'on(bx, b)' is true in the current state. If yes, increment heuristic.
    5. For each goal predicate 'clear(b)':
       - Check if 'clear(b)' is true in the current state. If not, increment the heuristic.
       - Check if there is any block 'bx' such that 'on(bx, b)' is true in the current state. If yes and 'clear(b)' is in goal, increment heuristic.
    6. Return the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals

    def __call__(self, node):
        """Estimate the cost to reach the goal state from the current state."""
        state = node.state
        heuristic_value = 0

        goal_on = set()
        goal_on_table = set()
        goal_clear = set()

        for goal in self.goals:
            if match(goal, 'on', '*', '*'):
                goal_on.add(goal)
            elif match(goal, 'on-table', '*'):
                goal_on_table.add(goal)
            elif match(goal, 'clear', '*'):
                goal_clear.add(goal)

        for goal in goal_on:
            if goal not in state:
                heuristic_value += 1
            block_under = get_parts(goal)[2]
            goal_clear_under_block = f'(clear {block_under})'
            if goal_clear_under_block in goal_clear and goal_clear_under_block not in state:
                 heuristic_value += 1

            block_over = get_parts(goal)[1]
            for fact in state:
                if match(fact, 'on', '*', block_over):
                    heuristic_value += 1

        for goal in goal_on_table:
            if goal not in state:
                heuristic_value += 1
            block = get_parts(goal)[1]
            for fact in state:
                if match(fact, 'on', '*', block):
                    heuristic_value += 1

        for goal in goal_clear:
            if goal not in state:
                heuristic_value += 1
            block = get_parts(goal)[1]
            for fact in state:
                if match(fact, 'on', '*', block):
                     heuristic_value += 1

        return heuristic_value
