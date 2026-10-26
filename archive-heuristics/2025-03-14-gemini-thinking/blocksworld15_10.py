from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_objects_from_fact(fact):
    """
    Extract the objects from a PDDL fact string.
    For example, from '(on b1 b2)' it returns ['b1', 'b2'].
    Ignores the predicate name.
    """
    parts = fact[1:-1].split()
    return parts[1:]

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not in their goal positions.
    For each block that is not correctly placed according to the goal state, it adds 1 to the heuristic value.
    A block is considered correctly placed if it satisfies the 'on', 'on-table', and 'clear' conditions specified in the goal.

    # Assumptions:
    - The goal state is defined by a set of 'on', 'on-table', and 'clear' predicates.
    - We are minimizing the number of actions to reach the goal state.

    # Heuristic Initialization
    - The heuristic initializes by parsing the goal conditions from the task.
    - It stores the goal 'on', 'on-table', and 'clear' predicates for efficient lookup during heuristic calculation.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. For each goal condition of type 'on (block1) (block2)':
       - Check if the 'on (block1) (block2)' predicate is present in the current state.
       - If not, increment the heuristic value by 1.
    3. For each goal condition of type 'on-table (block)':
       - Check if the 'on-table (block)' predicate is present in the current state.
       - If not, increment the heuristic value by 1.
    4. For each goal condition of type 'clear (block)':
       - Check if the 'clear (block)' predicate is present in the current state.
       - If not, increment the heuristic value by 1.
    5. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the blocksworld heuristic by extracting goal conditions.
        """
        self.goal_on = set()
        self.goal_on_table = set()
        self.goal_clear = set()

        for goal in task.goals:
            if fnmatch(goal, '(on * *)'):
                self.goal_on.add(goal)
            elif fnmatch(goal, '(on-table *)'):
                self.goal_on_table.add(goal)
            elif fnmatch(goal, '(clear *)'):
                self.goal_clear.add(goal)

    def __call__(self, node):
        """
        Calculate the heuristic value for the given state.
        The heuristic value is the number of goal predicates that are not satisfied in the current state.
        """
        state = node.state
        heuristic_value = 0

        for goal_fact in self.goal_on:
            if goal_fact not in state:
                heuristic_value += 1
        for goal_fact in self.goal_on_table:
            if goal_fact not in state:
                heuristic_value += 1
        for goal_fact in self.goal_clear:
            if goal_fact not in state:
                heuristic_value += 1

        return heuristic_value
