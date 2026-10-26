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


class blocksworld9Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goal state in the Blocksworld domain.
    It considers the number of blocks that are not in their goal positions and estimates the number of pick-up, put-down, stack, and unstack operations required to move them to their correct positions.

    # Assumptions
    - The heuristic assumes that each block needs to be moved at most once to reach its final position.
    - It does not explicitly consider the arm-empty condition, assuming that the robot can always pick up or put down a block when needed.
    - It does not account for the number of blocks that need to be cleared to perform an action.

    # Heuristic Initialization
    - The heuristic initializes by extracting the goal state from the task definition.
    - It stores the goal 'on' relationships between blocks for easy access during heuristic calculation.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Extract the goal 'on' relationships from the goal state.
    3. Iterate through the goal 'on' relationships and check if they are satisfied in the current state.
       - If a goal 'on' relationship is not satisfied, increment the heuristic value by 1 (representing the cost of a stack action).
    4. Iterate through the 'clear' goals and check if they are satisfied in the current state.
       - If a goal 'clear' relationship is not satisfied, increment the heuristic value by 1 (representing the cost of an unstack action).
    5. For each block that is on the table in the goal state, check if it is on the table in the current state.
       - If a block is on the table in the goal state but not in the current state, increment the heuristic value by 1 (representing the cost of a putdown action).
    6. Return the final heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.static = task.static

        self.goal_on = {}
        self.goal_clear = set()
        self.goal_on_table = set()

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

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        h = 0

        # Check 'on' goals
        for block, under in self.goal_on.items():
            on_goal_satisfied = False
            for fact in state:
                if match(fact, "on", block, under):
                    on_goal_satisfied = True
                    break
            if not on_goal_satisfied:
                h += 1  # Estimate cost for stack action

        # Check 'clear' goals
        for block in self.goal_clear:
            clear_goal_satisfied = False
            for fact in state:
                if match(fact, "clear", block):
                    clear_goal_satisfied = True
                    break
            if not clear_goal_satisfied:
                h += 1  # Estimate cost for unstack action

        # Check 'on-table' goals
        for block in self.goal_on_table:
            on_table_goal_satisfied = False
            for fact in state:
                if match(fact, "on-table", block):
                    on_table_goal_satisfied = True
                    break
            if not on_table_goal_satisfied:
                h += 1 # Estimate cost for putdown action

        # If the goal is reached, return 0
        if node.state >= self.goals:
            return 0

        return h
