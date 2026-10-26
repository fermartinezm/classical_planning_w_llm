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


class blocksworld23Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goal state in the Blocksworld domain.
    It considers the number of blocks that are not in their goal positions, the number of blocks that have blocks on top of them that should not be there, and the number of blocks that are clear but should not be.

    # Assumptions
    - Each block needs to be moved at most once to its correct position.
    - The arm can only hold one block at a time.
    - The heuristic is not admissible but aims to provide a reasonable estimate.

    # Heuristic Initialization
    - Extract the goal 'on' relationships between blocks.
    - Identify blocks that should be on the table in the goal state.
    - Identify blocks that should be clear in the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize a cost counter to 0.
    2. Extract the goal state information about the 'on', 'on-table', and 'clear' predicates.
    3. For each 'on' goal, check if the block is currently on the correct block. If not, increment the cost.
    4. For each 'on-table' goal, check if the block is currently on the table. If not, increment the cost.
    5. For each 'clear' goal, check if the block is currently clear. If not, increment the cost.
    6. Check for blocks that are on top of other blocks in the current state, but should not be according to the goal state. Increment the cost for each such misplaced block.
    7. Check for blocks that are clear in the current state, but should have another block on top of them according to the goal state. Increment the cost for each such misplaced block.
    8. Return the total cost, which represents the estimated number of actions needed to reach the goal state.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.static = task.static

        self.goal_on = set()
        self.goal_on_table = set()
        self.goal_clear = set()

        for goal in self.goals:
            if match(goal, "on", "*", "*"):
                self.goal_on.add(tuple(get_parts(goal)[1:]))
            elif match(goal, "on-table", "*"):
                self.goal_on_table.add(get_parts(goal)[1])
            elif match(goal, "clear", "*"):
                self.goal_clear.add(get_parts(goal)[1])

    def __call__(self, node):
        """Estimate the cost to reach the goal state from the current state."""
        state = node.state
        cost = 0

        # Check 'on' goals
        for block1, block2 in self.goal_on:
            found = False
            for fact in state:
                if match(fact, "on", block1, block2):
                    found = True
                    break
            if not found:
                cost += 1

        # Check 'on-table' goals
        for block in self.goal_on_table:
            found = False
            for fact in state:
                if match(fact, "on-table", block):
                    found = True
                    break
            if not found:
                cost += 1

        # Check 'clear' goals
        for block in self.goal_clear:
            found = False
            for fact in state:
                if match(fact, "clear", block):
                    found = True
                    break
            if not found:
                cost += 1

        # Check for misplaced blocks (blocks on top of others that shouldn't be)
        for fact in state:
            if match(fact, "on", "*", "*"):
                b1, b2 = get_parts(fact)[1:]
                if (b1, b2) not in self.goal_on:
                    cost += 1

        # Check for blocks that are clear but should have blocks on top
        for fact in state:
            if match(fact, "clear", "*"):
                block = get_parts(fact)[1]
                should_be_covered = False
                for b1, b2 in self.goal_on:
                    if b2 == block:
                        should_be_covered = True
                        break
                if should_be_covered:
                    cost += 1

        # Heuristic is 0 only for goal states
        if self.goals <= state:
            cost = 0

        return cost
