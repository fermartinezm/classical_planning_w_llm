from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_objects_from_fact(fact):
    """
    Extract the objects from a PDDL fact string.
    For example, from '(on b1 b2)' it returns ['b1', 'b2'].
    """
    parts = fact[1:-1].split()
    return parts[1:]  # Return objects, skipping the predicate name

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not in their goal positions.
    It counts the number of blocks that are either not in the correct 'on' or 'on-table' position as specified in the goal,
    and the number of blocks that are not 'clear' in the current state but are required to be 'clear' in the goal state.
    This heuristic is admissible if we consider each misplaced block requires at least one action to correct its position.

    # Assumptions:
    - The goal state is defined by a set of 'on', 'on-table', and 'clear' predicates.
    - We assume that each misplaced block or uncleared block (when it should be clear) requires at least one action to fix.

    # Heuristic Initialization
    - Extract the goal predicates related to 'on', 'on-table', and 'clear' from the task definition.
    - Store the goal 'on' and 'on-table' relations for easy lookup.
    - Store the blocks that are required to be 'clear' in the goal.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. For each goal predicate of the form '(on block1 block2)':
       - Check if '(on block1 block2)' is present in the current state.
       - If not, increment the heuristic value by 1.
    3. For each goal predicate of the form '(on-table block)':
       - Check if '(on-table block)' is present in the current state.
       - If not, increment the heuristic value by 1.
    4. For each goal predicate of the form '(clear block)':
       - Check if '(clear block)' is present in the current state.
       - If not, increment the heuristic value by 1.
    5. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the blocksworld heuristic.
        Extract goal predicates and static facts if needed (though static facts are not used in this heuristic).
        """
        self.goals = task.goals
        self.static_facts = task.static

    def __call__(self, node):
        """
        Calculate the heuristic value for a given state.
        The heuristic value is the number of goal predicates that are not satisfied in the current state,
        focusing on 'on', 'on-table', and 'clear' predicates.
        """
        state = node.state
        heuristic_value = 0

        goal_on_predicates = set()
        goal_on_table_predicates = set()
        goal_clear_predicates = set()

        for goal in self.goals:
            if goal.startswith('(on '):
                goal_on_predicates.add(goal)
            elif goal.startswith('(on-table '):
                goal_on_table_predicates.add(goal)
            elif goal.startswith('(clear '):
                goal_clear_predicates.add(goal)

        for goal_fact in goal_on_predicates:
            if goal_fact not in state:
                heuristic_value += 1

        for goal_fact in goal_on_table_predicates:
            if goal_fact not in state:
                heuristic_value += 1

        for goal_fact in goal_clear_predicates:
            if goal_fact not in state:
                heuristic_value += 1

        return heuristic_value
