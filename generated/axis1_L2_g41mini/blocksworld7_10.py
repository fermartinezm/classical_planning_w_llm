from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate_name arg1 arg2)".
    - `args`: The expected pattern components (wildcards `*` allowed).
    - Returns True if the fact matches the pattern else False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain.

    Summary:
    ----------
    This heuristic estimates the minimal number of actions required to achieve the goal state
    from the current state by analyzing the mismatch between current and goal predicates,
    focusing particularly on the key predicates that describe the "chain" relationships
    (gaijkwef), the "bclhkf4r" status, and the fact zzar8qlk. It is derived from the domain's 
    cyclic state transformation pattern where objects "pass" or "transfer" states to each other.

    Assumptions:
    ------------
    - The domain encodes a token or state "zzar8qlk" that can be held or passed among objects marked bclhkf4r.
    - The gaijkwef predicate represents a directed relation from one object to another, describing a chain or pairing necessary for the goal.
    - The goal consists mainly of asserting certain gaijkwef relations and certain bclhkf4r and chxwa5j4 predicates.
    - Achieving the goal requires moving the zzar8qlk token among objects and altering their status predicates according to certain rules encoded by the operators.
    - The heuristic does not need to be admissible but should estimate the number of actions relative to how many predicate goals are still unmet.

    Heuristic Initialization:
    -------------------------
    - Extract and memorize the goal predicates, grouped by predicate name.
    - The heuristic focuses on predicates: gaijkwef(? ?), bclhkf4r(?), and chxwa5j4(?).
    - Static facts are not used as the domain declares none.
    - No other static or complex precomputation needed.

    Step-By-Step Thinking for Computing Heuristic:
    ----------------------------------------------
    1. If the current state meets all the goal conditions, return heuristic 0.
    2. Compute how many goal gaijkwef predicates are not yet satisfied.
    3. Compute how many goal bclhkf4r predicates are not yet satisfied.
    4. Compute how many goal chxwa5j4 predicates are not yet satisfied.
    5. Check if zzar8qlk is currently true in the state:
       - If not, at least one action is needed to restore it because many operators require zzar8qlk as precondition.
    6. These unsatisfied predicates correspond roughly to the number of needed object transformations or token passes.
       Each unmet gaijkwef likely requires one or more actions to set up.
    7. Use a weighted sum:
       - Missing gaijkwef goals are weighted highest, as they represent key pairwise relations.
       - Missing bclhkf4r and chxwa5j4 are weighted moderately.
       - Missing zzar8qlk adds a constant penalty if absent.
    8. The heuristic sums these contributions to produce a finite estimate guiding the search efficiently.

    This heuristic attempts to estimate "how many objects still need to be linked correctly" and
    whether the zzar8qlk token is present to enable further progress.

    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal predicates grouped by predicate names.
        """
        self.goals = task.goals  # frozenset of goal facts

        # Extract goal facts by predicate name for quick lookup
        self.goal_gaijkwef = set()
        self.goal_bclhkf4r = set()
        self.goal_chxwa5j4 = set()
        self.goal_zzar8qlk = False

        for fact in self.goals:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "gaijkwef":
                self.goal_gaijkwef.add(fact)
            elif predicate == "bclhkf4r":
                self.goal_bclhkf4r.add(fact)
            elif predicate == "chxwa5j4":
                self.goal_chxwa5j4.add(fact)
            elif predicate == "zzar8qlk":
                self.goal_zzar8qlk = True

    def __call__(self, node):
        """
        Compute the heuristic estimate.

        Returns an integer estimating the remaining actions to reach the goal.
        """
        state = node.state

        # If goal is reached, heuristic is zero.
        if self.goals <= state:
            return 0

        # Count missing gaijkwef goal facts
        missing_gaijkwef = sum(1 for fact in self.goal_gaijkwef if fact not in state)
        # Count missing bclhkf4r goal facts
        missing_bclhkf4r = sum(1 for fact in self.goal_bclhkf4r if fact not in state)
        # Count missing chxwa5j4 goal facts
        missing_chxwa5j4 = sum(1 for fact in self.goal_chxwa5j4 if fact not in state)
        # Check for zzar8qlk presence if required by goal
        missing_zzar8qlk = 0
        if self.goal_zzar8qlk and "(zzar8qlk)" not in state:
            missing_zzar8qlk = 1

        # Weights are chosen heuristically based on operator patterns:
        # gaijkwef changes require multiple actions (roughly ~2 each)
        # bclhkf4r and chxwa5j4 predicates changes cost about 1 action each to fix.
        # zzar8qlk absence prevents many operations, so a small penalty.

        heuristic_value = (
            2 * missing_gaijkwef
            + 1 * missing_bclhkf4r
            + 1 * missing_chxwa5j4
            + missing_zzar8qlk
        )

        # The heuristic is at least 1 if goal not reached (to steer search)
        return max(1, heuristic_value)
