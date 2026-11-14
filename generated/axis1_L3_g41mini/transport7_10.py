from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(on-table item10 item6)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain (blocksworld).

    Summary:
    ----------
    This heuristic estimates the number of actions needed to achieve the goal, based primarily on
    how many "arm-empty" goal facts still need to be satisfied. Since the domain revolves around 
    managing multiple arms and items with complex typing, the heuristic measures the minimum number 
    of "arm-empty" conditions missing and factors in the manipulation of items assumed required 
    to reach those conditions.

    Assumptions:
    ------------
    - The goal conditions only impose requirements on (arm-empty ?arm ?item-2).
    - The heuristic is 0 iff all goal facts hold in the current state.
    - Actions used to achieve goals mainly involve preparing arms (making them empty or 
      "arm-empty" facts true).
    - Since multiple arms and items with layered typing exist, the heuristic approximates 
      progress by counting unsatisfied arm-empty goals.
    - Manipulation of "on", "on-table", "holding", and "clear" predicates is abstracted 
      out, but assumed bounded by the count of arms not in the goal-expected state.
    - The heuristic need not be admissible but should guide the search efficiently.

    Heuristic Initialization:
    --------------------------
    - Extract all goal facts of the form (arm-empty ?arm ?item-2) into a set `goal_arm_empties`.
    - Extract all static facts to support any potential domain-specific patterns (kept but not used explicitly here).
      For instance, static facts contain facts about "holding", "on", etc., but are not dynamically changing.
    - The static facts are stored for potential future use or extensions.

    Step-By-Step Thinking for Computing Heuristic:
    -----------------------------------------------
    1. For the current state, identify which goal (arm-empty) facts are already true.
    2. Count how many goal arm-empty facts remain unsatisfied.
       This number directly correlates with arms that need to be emptied or properly arranged.
    3. The heuristic value is at least the count of unsatisfied arm-empty goals, since each requires 
       at least one action to achieve.
    4. Since the domain actions relate to pickup, stack, and unstack with conditions on arm emptiness, 
       we multiply the number of unsatisfied arms by a small constant factor to approximate the expected 
       number of actions, as achieving each may require a few steps.
    5. If all goal arm-empty facts hold, the heuristic returns 0.
    6. This heuristic avoids complex search or reasoning over the full stack; its goal is efficient 
       computation and useful guidance.

    """

    def __init__(self, task):
        """
        Initialize by extracting goal arm-empty facts and static facts.

        Args:
            task (Task): The planning task instance.
        """
        self.goals = task.goals
        self.static = task.static  # might be useful for expanded heuristics

        # Extract all goal arm-empty facts to check their satisfaction status quickly
        self.goal_arm_empties = set(
            fact for fact in self.goals if fact.startswith("(arm-empty")
        )

    def __call__(self, node):
        """
        Compute heuristic estimate of actions to reach the goal from the current node.

        Args:
            node: Search node containing the current state.

        Returns:
            int: Estimated cost (number of actions) to reach the goal.
        """
        state = node.state

        # Quick check if goal is reached
        if self.goals <= state:
            return 0

        # Count how many goal arm-empty facts are currently satisfied
        satisfied_arm_empties = sum(1 for fact in self.goal_arm_empties if fact in state)
        unsatisfied_arm_empties = len(self.goal_arm_empties) - satisfied_arm_empties

        # Each unsatisfied arm-empty likely requires manipulating the arms/items,
        # approximated by a factor ~3 to guess the average number of actions needed per unsatisfied arm-empty
        heuristic_value = unsatisfied_arm_empties * 3

        # Ensure heuristic is positive for non-goal states
        return heuristic_value if heuristic_value > 0 else 1
