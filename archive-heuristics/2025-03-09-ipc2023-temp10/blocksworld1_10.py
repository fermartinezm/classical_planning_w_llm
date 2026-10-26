from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class blocksworld1Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goal state in the Blocksworld domain.
    It counts the number of blocks that are not in their goal positions and estimates the number of pickup, putdown, stack, and unstack actions required to move them to their correct positions.

    # Assumptions
    - Each block needs to be moved at most once.
    - The arm can only hold one block at a time.
    - The heuristic does not consider the arm-empty predicate.

    # Heuristic Initialization
    - Extract the goal conditions from the task.
    - Store the goal 'on' relationships in a dictionary for easy access.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize a counter for the heuristic value.
    2. Iterate through the goal 'on' relationships and check if they are satisfied in the current state.
    3. If a block is not on the correct block in the current state, increment the heuristic counter.
       This accounts for 'stack' or 'on-table' actions.
    4. Iterate through the 'clear' goals and check if they are satisfied in the current state.
    5. If a block is not clear in the current state but should be, increment the heuristic counter.
       This accounts for 'unstack' or 'putdown' actions.
    6. If the arm is not empty in the initial state, increment the heuristic counter.
       This accounts for the 'putdown' action.
    7. Return the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.goal_on = {}
        self.goal_clear = set()

        for goal in self.goals:
            if goal.startswith('(on '):
                parts = goal[1:-1].split()
                self.goal_on[parts[1]] = parts[2]
            elif goal.startswith('(clear '):
                parts = goal[1:-1].split()
                self.goal_clear.add(parts[1])

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        heuristic = 0

        # Check 'on' goals
        for block, target in self.goal_on.items():
            current_on = None
            for fact in state:
                if fact.startswith('(on ' + block + ' '):
                    parts = fact[1:-1].split()
                    current_on = parts[2]
                    break
                elif fact.startswith('(on-table ' + block + ')'):
                    current_on = 'table'  # Represent on-table as 'table'

            if target == 'table':
                if current_on != 'table':
                    heuristic += 1
            else:
                if current_on != target:
                    heuristic += 1

        # Check 'clear' goals
        for block in self.goal_clear:
            is_clear = False
            for fact in state:
                if fact == '(clear ' + block + ')':
                    is_clear = True
                    break
            if not is_clear:
                heuristic += 1

        # Check if arm is empty
        arm_empty = True
        for fact in state:
            if fact.startswith('(holding '):
                arm_empty = False
                break
        if not arm_empty:
            heuristic += 1

        return heuristic
