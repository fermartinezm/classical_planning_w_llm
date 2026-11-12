from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at locatable1 locatable2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the "secret_domain" transport domain.

    Summary:
    This heuristic estimates the minimum number of actions required to achieve
    the goal configuration of predicates (road, at, capacity-predecessor)
    given the current state. It is designed to guide greedy best-first search
    effectively, not necessarily admissible.

    Assumptions:
    - The domain models transport-like progressive state changes on "locatables".
    - Actions "drive", "pick-up", "drop", and "drive-2" transition facts related to:
        - road(?v)
        - capacity-predecessor(?v)
        - capacity (global flag)
        - in(?v)
        - at(?v ?loc)
    - The goal contains sets of road facts, at facts, and capacity-predecessor facts for specific locatables.
    - "capacity" and "in" facts operate similarly to a carrying capacity mechanic.
    - The heuristic focuses on how many objects differ from goal conditions and how their associated chaining of predicates imply sequences of moves/pick-ups/drops.
    - The domain structure suggests linear or dependent relations ("at" chaining) between locatables.
    - Static facts are empty, so heuristic does not use them.

    Heuristic Initialization:
    - Extract goal facts grouped by predicate: road-goals, at-goals, capacity-predecessor-goals.
    - Index goal facts by object and their requirements to check fulfillment.
    - No static facts exist, so no additional indexing.

    Step-By-Step Thinking for Computing Heuristic:
    1. Count how many goal predicates are not yet satisfied in the current state:
       - Number of goal "road" facts not true.
       - Number of goal "capacity-predecessor" facts not true.
       - Number of goal "at" facts not true.
    2. All these predicates represent parts of the final desired configuration and indicate
       objects that still need movement or adjustment.
    3. "in" and "capacity" predicates in preconditions and effects indicate that moving an object involves pick-up (in), hold capacity (capacity), and drop actions.
    4. Each unsatisfied "road" or "capacity-predecessor" fact roughly indicates at least one action needed to create it.
    5. "at" predicates chain locations of objects and must be arranged; each missing "at" fact suggests at least one movement or drop action.
    6. Due to the domain's nature, each missing goal fact likely demands at least one action among drive, pick-up, drop, or drive-2.
    7. The heuristic sums:
       - The number of unsatisfied road-goals,
       - Plus the number of unsatisfied capacity-predecessor-goals,
       - Plus the number of unsatisfied at-goals,
       as a rough estimate of actions still needed.
    8. This sum is admissible but also encourages the search towards fixing missing conditions efficiently.
    9. Since the heuristic is non-admissible, a small multiplier or constant bias could be applied to bias search, but here we keep it simple for efficiency.
    10. The heuristic returns zero exactly when all goal facts are true.

    This heuristic is efficiently computable by simple set membership tests and counts,
    and it summarizes the difference between current and goal states as a proxy to
    estimate remaining required actions.
    """

    def __init__(self, task):
        """
        Initialize by extracting goal predicates grouped by their names for quick membership checks.
        No static facts are used as the domain provides none.
        """
        self.goal_road = set()
        self.goal_capacity_pred = set()
        self.goal_at = set()

        for fact in task.goals:
            pred, *args = get_parts(fact)
            if pred == "road":
                self.goal_road.add(fact)
            elif pred == "capacity-predecessor":
                self.goal_capacity_pred.add(fact)
            elif pred == "at":
                self.goal_at.add(fact)

    def __call__(self, node):
        """
        Heuristic function estimating cost to goal from current state by counting
        unsatisfied goal facts in categories road, capacity-predecessor, and at.
        Returns 0 if goal reached.
        """
        state = node.state

        # Count unsatisfied goal road facts
        missing_road = len(self.goal_road - state)
        # Count unsatisfied goal capacity-predecessor facts
        missing_capacity_pred = len(self.goal_capacity_pred - state)
        # Count unsatisfied goal at facts
        missing_at = len(self.goal_at - state)

        heuristic_value = missing_road + missing_capacity_pred + missing_at
        return heuristic_value
