from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not in their goal positions.
    It counts the number of unsatisfied goal predicates related to block positions
    (`on`, `on-table`, `clear`). For each goal condition that is not met in the current state,
    the heuristic value is incremented by one. This heuristic is admissible in relaxed planning
    where delete effects are ignored, but not necessarily admissible in standard STRIPS planning.
    However, it is expected to guide the search effectively in practice for the Blocksworld domain.

    # Assumptions:
    - The goal is defined by a conjunction of `on`, `on-table`, and `clear` predicates.
    - The heuristic assumes that each unsatisfied goal predicate requires at least one action to be achieved.
    - It does not consider the cost of achieving preconditions for actions, focusing solely on the number of misplaced blocks and uncleared top blocks.

    # Heuristic Initialization
    - The heuristic is initialized with the goal predicates from the task definition.
    - No static facts are used in this heuristic.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Iterate through each goal predicate in the task's goal description.
    3. For each goal predicate, check if it is present in the current state.
       - A helper function `match_fact` is used to check if a fact in the state matches the goal predicate.
    4. If a goal predicate is not found in the current state, increment the heuristic value by 1.
    5. After checking all goal predicates, return the accumulated heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by storing the goal predicates."""
        self.goals = task.goals
        static_facts = task.static # Static facts are not used in this heuristic

    def __call__(self, node):
        """
        Estimate the number of actions needed to reach the goal state from the current state.
        This is done by counting the number of goal predicates that are not satisfied in the current state.
        """
        state = node.state
        heuristic_value = 0

        def get_predicate_name(fact_str):
            """Extracts the predicate name from a fact string."""
            return fact_str[1:].split(' ')[0]

        def get_objects_from_fact(fact_str):
            """Extracts the objects from a fact string, ignoring brackets."""
            parts = fact_str[1:-1].split(' ')
            return tuple(parts[1:])

        def match_fact(state_fact, goal_fact):
            """
            Checks if a fact from the state matches a goal fact.
            This is a simple string comparison for this heuristic.
            """
            return state_fact == goal_fact

        for goal_fact in self.goals:
            goal_predicate_name = get_predicate_name(goal_fact)
            goal_objects = get_objects_from_fact(goal_fact)
            goal_fact_str = str(goal_fact)

            is_goal_satisfied = False
            for state_fact in state:
                if match_fact(state_fact, goal_fact_str):
                    is_goal_satisfied = True
                    break
            if not is_goal_satisfied:
                heuristic_value += 1

        return heuristic_value
