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
    This heuristic estimates the number of actions required to achieve the goal state in the Blocksworld domain.
    It counts the number of blocks that are not in their goal positions and estimates the minimum actions needed to correct their positions.
    Specifically, it considers the 'on', 'on-table', and 'clear' goal conditions and penalizes states where these conditions are not met.

    # Assumptions:
    - The heuristic assumes that each misplaced block requires at least one action to be moved to its correct position.
    - It focuses on achieving the 'on', 'on-table', and 'clear' goal predicates.
    - It does not explicitly consider the 'arm-empty' and 'holding' predicates in the heuristic calculation, assuming they will be managed by the planner to achieve the other goals.

    # Heuristic Initialization
    - The heuristic initializes by storing the goal predicates from the task definition.
    - No static facts are used in this heuristic for Blocksworld.

    # Step-By-Step Thinking for Computing Heuristic
    For each goal predicate in the goal state:
    1. Check if the goal predicate is satisfied in the current state.
    2. If a goal predicate of the form '(on block1 block2)' is not satisfied:
       - Increment the heuristic cost by 1. This accounts for the actions needed to stack 'block1' on 'block2'.
    3. If a goal predicate of the form '(on-table block)' is not satisfied:
       - Increment the heuristic cost by 1. This accounts for the actions needed to putdown 'block' on the table.
    4. If a goal predicate of the form '(clear block)' is not satisfied:
       - Increment the heuristic cost by 1. This accounts for the actions needed to unstack any block that might be on top of 'block'.
    5. The total heuristic value is the sum of increments for all unsatisfied goal predicates.
    This heuristic provides a simple estimate of the remaining actions by counting the number of goal conditions that are not yet met.
    It is domain-dependent as it directly uses the predicates 'on', 'on-table', and 'clear' specific to the Blocksworld domain.
    """

    def __init__(self, task):
        """Initialize the heuristic by storing the goal predicates."""
        self.goals = task.goals

    def __call__(self, node):
        """Estimate the cost to reach the goal state from the current state."""
        state = node.state
        heuristic_value = 0

        for goal in self.goals:
            if goal not in state:
                heuristic_value += 1  # Increment for each unsatisfied goal condition

        return heuristic_value
