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


class blocksworld17Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goal state in the Blocksworld domain.
    It considers the number of blocks that are not in their goal positions, and estimates the number of
    unstack, putdown, pickup, and stack operations required to move them to their correct positions.

    # Assumptions
    - The heuristic assumes that each block needs to be moved at most once.
    - It does not consider the arm-empty predicate directly, but assumes that picking and stacking actions
      are always possible when needed.
    - It simplifies the problem by ignoring the order in which blocks are moved, focusing only on the number
      of misplaced blocks.

    # Heuristic Initialization
    - The heuristic initializes by extracting the goal state and identifying the blocks that need to be in
      specific 'on' relationships.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Initialize a counter for the heuristic value to 0.
    2.  Iterate through each goal fact to identify 'on' relationships between blocks in the goal state.
    3.  For each 'on' goal fact, check if the relationship exists in the current state.
    4.  If the 'on' relationship does not exist in the current state, increment the heuristic counter. This
        represents the estimated cost of moving the block to its correct position.  We assume that each misplaced
        block requires at least one action (stack).
    5.  Additionally, for each block that is misplaced, we may need to clear the blocks above it (unstack),
        putdown the block in hand, and pickup the misplaced block. These are also counted as actions.
    6.  If the 'clear' goal fact does not exist in the current state, increment the heuristic counter.
    7.  If the 'on-table' goal fact does not exist in the current state, increment the heuristic counter.
    8.  If the current state satisfies all goal conditions, return 0.
    9.  Return the final heuristic value, which represents the estimated number of actions required to reach the goal state.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        heuristic = 0

        # Check if the goal is already reached
        if all(goal in state for goal in self.goals):
            return 0

        # Count the number of 'on' relationships that are not satisfied
        for goal in self.goals:
            if match(goal, "on", "*", "*"):
                if goal not in state:
                    heuristic += 1  # Estimate cost for moving the block

        # Count the number of 'clear' relationships that are not satisfied
        for goal in self.goals:
            if match(goal, "clear", "*"):
                if goal not in state:
                    heuristic += 1

        # Count the number of 'on-table' relationships that are not satisfied
        for goal in self.goals:
            if match(goal, "on-table", "*"):
                if goal not in state:
                    heuristic += 1

        return heuristic
