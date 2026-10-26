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
    This heuristic estimates the number of actions needed to achieve the goal state in the Blocksworld domain.
    It counts the number of goal predicates that are not satisfied in the current state and adds penalties for blocks that are not clear when they should be in the goal.

    # Assumptions:
    - The problem is solvable in the Blocksworld domain using the given actions.
    - The heuristic aims to be efficient and guide the search effectively, not necessarily to be admissible.

    # Heuristic Initialization
    - The heuristic is initialized with the goal predicates from the task definition.
    - No static facts are used in this heuristic.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Iterate through each goal predicate in the task's goal description.
    3. For each goal predicate, check if it is satisfied in the current state.
       - If a goal predicate `(on b1 b2)` is not in the current state, increment the heuristic value by 1.
       - If a goal predicate `(on-table b)` is not in the current state, increment the heuristic value by 1.
    4. Identify blocks that should be clear in the goal state. A block `b` should be clear in the goal if it is not the 'underob' argument in any `(on ?ob ?underob)` goal predicate.
    5. For each block `b` that should be clear in the goal state, check if `(clear b)` is in the current state.
       - If `b` should be clear in the goal and `(clear b)` is not in the current state, increment the heuristic value by 1.
    6. Return the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by storing the goal predicates."""
        self.goals = task.goals

    def __call__(self, node):
        """Compute the heuristic value for a given state."""
        state = node.state
        heuristic_value = 0

        goal_on_predicates = []
        goal_on_table_predicates = []
        goal_clear_blocks = set()
        blocks_in_goal_on = set()

        for goal in self.goals:
            if match(goal, "on", "*", "*"):
                goal_on_predicates.append(goal)
                blocks_in_goal_on.add(get_parts(goal)[1]) # block on top
            elif match(goal, "on-table", "*"):
                goal_on_table_predicates.append(goal)

        for goal in goal_on_predicates:
            if goal not in state:
                heuristic_value += 1
        for goal in goal_on_table_predicates:
            if goal not in state:
                heuristic_value += 1

        all_blocks_in_goal = set()
        for goal in self.goals:
            parts = get_parts(goal)
            for part in parts[1:]:
                all_blocks_in_goal.add(part)

        blocks_to_be_clear_in_goal = set()
        for block in all_blocks_in_goal:
            is_under_block = False
            for goal in goal_on_predicates:
                if get_parts(goal)[2] == block:
                    is_under_block = True
                    break
            if not is_under_block:
                blocks_to_be_clear_in_goal.add(block)

        for block in blocks_to_be_clear_in_goal:
            clear_fact = f'(clear {block})'
            if clear_fact not in state:
                heuristic_value += 1

        return heuristic_value
