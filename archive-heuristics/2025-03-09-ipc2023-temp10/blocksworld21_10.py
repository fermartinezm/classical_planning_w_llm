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


class blocksworld21Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goal state in the Blocksworld domain.
    It considers the number of blocks that are not in their goal positions and estimates the cost of moving them to their correct positions.

    # Assumptions
    - Each block needs to be unstacked if it's not clear or if it's on the wrong block.
    - Each block needs to be stacked if it's not on the correct block.
    - The arm can only hold one block at a time.

    # Heuristic Initialization
    - Extract the goal 'on' relationships.
    - Identify the blocks involved.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Extract the goal state 'on' relationships into a dictionary.
    3. Iterate through the current state and check for goal violations.
    4. If a block is not on the correct block as specified in the goal state, increment the heuristic value.
       - If a block has another block on top of it in the goal state, but it is clear in the current state, increment the heuristic value.
    5. If a block is on the table in the goal state, but it is not on the table in the current state, increment the heuristic value.
    6. If a block is clear in the goal state, but it is not clear in the current state, increment the heuristic value.
    7. The heuristic value represents the estimated number of actions required to correct the violations.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        goal_on = {}
        goal_clear = set()
        goal_on_table = set()

        # Extract goal state information
        for goal in self.goals:
            if match(goal, "on", "*", "*"):
                parts = get_parts(goal)
                goal_on[parts[1]] = parts[2]
            elif match(goal, "clear", "*"):
                parts = get_parts(goal)
                goal_clear.add(parts[1])
            elif match(goal, "on-table", "*"):
                parts = get_parts(goal)
                goal_on_table.add(parts[1])

        h = 0

        # Check if the goal is already reached
        goal_reached = True
        for goal in self.goals:
            if goal not in state:
                goal_reached = False
                break
        if goal_reached:
            return 0

        # Check for blocks not in the correct 'on' position
        for fact in state:
            if match(fact, "on", "*", "*"):
                parts = get_parts(fact)
                block = parts[1]
                under = parts[2]
                if block in goal_on and goal_on[block] != under:
                    h += 1  # Misplaced block
            elif match(fact, "on-table", "*"):
                block = get_parts(fact)[1]
                if block not in goal_on_table and block in goal_on:
                    h += 1
            elif match(fact, "clear", "*"):
                block = get_parts(fact)[1]
                if block not in goal_clear and any(match(g, "on", "*", block) for g in self.goals):
                    h += 1

        # Check for blocks that should be clear but are not
        for block in goal_clear:
            clear_in_state = False
            for fact in state:
                if match(fact, "clear", block):
                    clear_in_state = True
                    break
            if not clear_in_state:
                h += 1

        return h
