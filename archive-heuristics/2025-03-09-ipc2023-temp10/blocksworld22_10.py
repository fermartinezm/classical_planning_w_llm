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


class blocksworld22Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goal state in the Blocksworld domain.
    It considers the number of blocks that are not in their goal positions and the number of blocks that have
    blocks on top of them that should not be there.

    # Assumptions
    - Each block needs to be moved at most once to its correct position.
    - The arm can only hold one block at a time.
    - Moving a block requires picking it up, potentially putting down the current holding block, stacking it, and potentially unstacking blocks.

    # Heuristic Initialization
    - The heuristic initializes by storing the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Iterate through each goal fact.
    3. For each goal fact, check if it is present in the current state.
    4. If a goal fact is not present in the current state, increment the heuristic value.
       - If the goal fact is an `on` predicate, it means a block is not on the correct block.
       - If the goal fact is a `clear` predicate, it means a block is not clear when it should be.
       - If the goal fact is an `on-table` predicate, it means a block is not on the table when it should be.
    5. Additionally, check for blocks that are on top of other blocks in the current state, but should not be according to the goal state.
    6. Return the final heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        heuristic_value = 0

        # Check for goal facts that are not in the current state
        for goal in self.goals:
            if goal not in state:
                heuristic_value += 1

        # Check for blocks that are incorrectly stacked
        for fact in state:
            if match(fact, "on", "*", "*"):
                block_above = get_parts(fact)[1]
                block_below = get_parts(fact)[2]
                correctly_stacked = False
                for goal in self.goals:
                    if match(goal, "on", block_above, block_below):
                        correctly_stacked = True
                        break
                if not correctly_stacked:
                    heuristic_value += 1

        # If the state is the goal state, return 0
        if task.goal_reached(state):
            return 0

        return heuristic_value
