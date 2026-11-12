from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing surrounding parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern with wildcards.

    Args:
        fact (str): The PDDL fact string (e.g., "(on item1 item2)").
        *args: The pattern elements to match (e.g., "on", "*", "item2").

    Returns:
        bool: True if fact matches the pattern, False otherwise.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the `secret_domain` (blocksworld variant).

    Summary:
    This heuristic estimates the number of actions required to reach a goal state by analyzing
    the number of goal predicates not yet satisfied, focusing on the key predicate `(arm-empty ?v3 ?v4)`.
    
    The domain predicates and actions are specialized variants of blocksworld with:
    - `arm-empty` predicates relating two items (`item-3` and `item-2`),
    - `on`, `on-table`, and `holding` predicates managing stacking relations and manipulations.
    
    The heuristic reflects the number of necessary pickup, stack, and unstack actions to achieve
    all required `(arm-empty ...)` goal facts. Since the only goal predicates in the examples
    are `(arm-empty ...)` facts, the heuristic emphasizes reaching arm empty conditions on the correct objects.

    Assumptions:
    - The goal is given by a set of `(arm-empty ?v3 ?v4)` facts needing to be satisfied in the final state.
    - The heuristic is not guaranteed to be admissible but aims to guide greedy best-first search efficiently.
    - It counts roughly how many `(arm-empty ...)` goals are not yet achieved by counting how many 
      of these predicates in the goal are missing in the current state.
    - Each unsatisfied `(arm-empty ...)` predicate is estimated as requiring at least one action to achieve.
    - Additional penalties are added for inconsistencies and "holding" states, since having an object held 
      usually means an additional step is needed before the arm can be made empty again.

    Heuristic Initialization:
    - Extract the set of goal `(arm-empty ?v3 ?v4)` facts.
    - Extract key static relations from the static facts, if any are necessary (none critical here).
    - No complex static knowledge extraction is done because the domain is highly relational and action counts depend on the current arm-empty and holding facts.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify which `(arm-empty ?v3 ?v4)` facts are required by the goal.
    2. Count how many of these goal `(arm-empty ...)` facts are missing from the current state — each missing suggests an action is needed.
    3. Count how many `(holding ?v5 ?v6)` facts currently hold — since holding an object implies the arm is not empty and will require actions to stack/unstack before goal conditions are met.
    4. Penalty: For each held item, add estimated 1 to 2 steps (like put down or stack) because releasing the held object is necessary to get relevant arm-empty states.
    5. Combine above counts into a heuristic estimate representing a rough minimal number of individual pickup, stack, and unstack actions necessary to achieve all arm-empty goals.

    This heuristic is efficiently computable by simple set membership tests and counts on the current state facts.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal arm-empty facts and static information.

        Args:
            task (Task): The planning task instance.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract all goal arm-empty facts for direct comparison
        self.goal_arm_empty = frozenset(
            fact for fact in self.goals if match(fact, "arm-empty", "*", "*")
        )

    def __call__(self, node):
        """
        Compute the heuristic estimate for the given node's state.

        Args:
            node: The search node with attribute `state` (a frozenset of fact strings).

        Returns:
            int: Estimated number of actions to reach a goal state.
        """
        state = node.state

        # If state already satisfies all arm-empty goals => heuristic = 0
        if self.goal_arm_empty <= state:
            return 0

        # Count how many goal arm-empty facts are missing
        missing_arm_empty = len(self.goal_arm_empty - state)

        # Count how many "holding" facts are currently true
        holding_count = sum(1 for fact in state if match(fact, "holding", "*", "*"))

        # The heuristic estimate:
        # - Each missing arm-empty fact implies at least one action is needed
        # - Holding objects imply needing at least one action each to release them (stack or put down)
        # - Use a simple weighted sum with the heuristic geared to penalize holding because arm must be emptied again.
        heuristic_value = missing_arm_empty + 2 * holding_count

        return heuristic_value
