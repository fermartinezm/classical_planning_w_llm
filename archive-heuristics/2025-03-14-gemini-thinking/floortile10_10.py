from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of paint actions required to achieve the goal state.
    It counts the number of goal tiles that are not yet painted with the correct color in the current state.
    This heuristic is admissible if we only consider paint actions and ignore move and color change actions.
    However, for greedy best-first search, it serves as a reasonable estimate of remaining work.

    # Assumptions:
    - The primary actions to reach the goal are `paint_up` and `paint_down`.
    - Move actions and color change actions are secondary and their costs are not explicitly considered in this simplified heuristic.
    - We assume that for each goal tile, a paint action is necessary if it's not yet painted correctly.

    # Heuristic Initialization
    - The heuristic initializes by storing the goal predicates from the task definition.
    - Static facts are not used in this version of the heuristic.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Iterate through each goal condition in the task's goal description.
    3. For each goal condition, check if it is a `painted` predicate.
    4. If it is a `painted` predicate, extract the tile and the target color from the goal condition.
    5. Check if the current state already satisfies this `painted` goal condition.
       - Iterate through the facts in the current state.
       - For each fact in the current state, check if it matches the `painted` predicate with the same tile and color as the goal condition.
    6. If the goal condition is not satisfied in the current state (i.e., the tile is not painted with the correct color), increment the heuristic value by 1. This represents the estimated cost of one `paint` action to satisfy this goal condition.
    7. After checking all goal conditions, the accumulated heuristic value is returned as the estimated cost to reach the goal state.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals

    def __call__(self, node):
        """Estimate the number of paint actions needed to reach the goal state."""
        state = node.state
        heuristic_value = 0

        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                goal_parts = get_parts(goal)
                goal_tile = goal_parts[1]
                goal_color = goal_parts[2]
                is_painted_correctly = False
                for state_fact in state:
                    if match(state_fact, "painted", "*", "*"):
                        state_parts = get_parts(state_fact)
                        state_tile = state_parts[1]
                        state_color = state_parts[2]
                        if state_tile == goal_tile and state_color == goal_color:
                            is_painted_correctly = True
                            break
                if not is_painted_correctly:
                    heuristic_value += 1

        return heuristic_value
