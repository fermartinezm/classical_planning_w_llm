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


class blocksworld11Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goal state in the Blocksworld domain.
    It considers the number of blocks that are not in their goal positions, and estimates the number of pick-up, put-down, stack, and unstack actions required to move them to their correct positions.

    # Assumptions
    - The heuristic assumes that each block needs to be moved at most once to reach its final position.
    - It does not explicitly consider the arm-empty predicate, assuming that the robot arm can always be emptied or filled as needed.
    - It does not take into account the cost of clearing blocks that are on top of other blocks.

    # Heuristic Initialization
    - The heuristic initializes by extracting the goal state from the task definition.
    - It stores the goal 'on' relationships in a dictionary for efficient lookup.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize a counter for the heuristic value to 0.
    2. Extract the goal state information and store the 'on' relationships in a dictionary (goal_on).
    3. Iterate through the current state and compare the 'on' relationships with the goal state.
    4. For each block that is not in its goal position (either on the wrong block or on the table when it should be on a block, or vice versa), increment the heuristic counter.
    5. For each block that is clear in the goal but not clear in the current state, increment the heuristic counter.
    6. If the arm is not empty in the goal state, increment the heuristic counter.
    7. Return the final heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        self.goal_on = {}
        self.goal_clear = set()
        self.goal_on_table = set()
        self.goal_arm_empty = False

        for goal in self.goals:
            if match(goal, "on", "*", "*"):
                parts = get_parts(goal)
                self.goal_on[parts[1]] = parts[2]
            elif match(goal, "clear", "*"):
                parts = get_parts(goal)
                self.goal_clear.add(parts[1])
            elif match(goal, "on-table", "*"):
                parts = get_parts(goal)
                self.goal_on_table.add(parts[1])
            elif match(goal, "arm-empty"):
                self.goal_arm_empty = True

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        h = 0

        # Check 'on' relationships
        for fact in state:
            if match(fact, "on", "*", "*"):
                parts = get_parts(fact)
                block1 = parts[1]
                block2 = parts[2]
                if block1 not in self.goal_on or self.goal_on[block1] != block2:
                    h += 1
            elif match(fact, "on-table", "*"):
                parts = get_parts(fact)
                block = parts[1]
                if block in self.goal_on:
                    h += 1  # Block should be on another block, not the table
                elif block not in self.goal_on_table:
                    h += 1 # Block should not be on the table
            elif match(fact, "clear", "*"):
                parts = get_parts(fact)
                block = parts[1]
                if block not in self.goal_clear:
                    # Block should not be clear
                    h += 1
            elif match(fact, "holding", "*"):
                if self.goal_arm_empty:
                    h += 1
            elif match(fact, "arm-empty"):
                if not self.goal_arm_empty:
                    h += 1

        # Check if goal is reached
        if h == 0:
            return 0

        return h
