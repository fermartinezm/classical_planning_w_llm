from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
    - `args`: The expected pattern with wildcards `*` allowed.
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (dnbnd6kg).

    Summary:
    This heuristic estimates the number of actions required to reach the goal by focusing on mismatches
    between the current state's relevant predicates and the goals. It models the flow of objects through
    states captured by predicates (bclhkf4r, mdawkki4, chxwa5j4, gaijkwef) and the global flag zzar8qlk,
    which controls whether some actions can be applied.

    Assumptions:
    - Objects go through states represented by predicates:
        * bclhkf4r(?obj): object in "bclhkf4r" state
        * mdawkki4(?obj): object in "mdawkki4" state
        * chxwa5j4(?obj): object in "chxwa5j4" state
        * gaijkwef(?obj1 ?obj2): a relation from obj1 to obj2
    - The global predicate zzar8qlk acts as a mode switch enabling certain transitions.
    - The goal predicates include some bclhkf4r, chxwa5j4, and gaijkwef facts.
    - Actions can move predicates between these states, possibly changing zzar8qlk.
    - The heuristic ignores static facts since none exist.

    Heuristic Initialization:
    - Extract the goal predicates partitioned by predicate name for fast checking.
    - Since no static facts exist, no static initialization is needed.

    Step-By-Step Thinking for Computing Heuristic:
    For the current state and the goal:
    1) Identify goal facts grouped by predicate.
       - goal_bclhkf4r: set of objects that must have (bclhkf4r ?obj)
       - goal_chxwa5j4: set of objects that must have (chxwa5j4 ?obj)
       - goal_gaijkwef: set of (obj1, obj2) pairs that must have (gaijkwef obj1 obj2)

    2) Extract current state facts similarly:
       - current_bclhkf4r, current_mdawkki4, current_chxwa5j4, current_gaijkwef
       - Also, note if zzar8qlk is true or false.

    3) Estimate missing facts for each predicate compared to goal:
       - missing_bclhkf4r = goal_bclhkf4r - current_bclhkf4r
       - missing_chxwa5j4 = goal_chxwa5j4 - current_chxwa5j4
       - missing_gaijkwef = goal_gaijkwef - current_gaijkwef

    4) To satisfy missing gaijkwef relations, objects involved must likely move from 
       mdawkki4 and bclhkf4r states through actions that consume or produce related predicates.

    5) Heuristic cost:
       - Each missing (gaijkwef) fact costs approx 2 actions (based on action effects: one to create, one to remove old facts).
       - Each missing (bclhkf4r) or (chxwa5j4) costs 1 action to produce or fix.
       - Toggle of zzar8qlk might add 1 action if not in required mode for needed transitions.

    6) Summing these estimates gives a heuristic cost estimate that is zero at the goal 
       (all goal facts present), positive otherwise, and finite.

    Note:
    This heuristic is not admissible but designed to provide useful guidance by counting missing
    goal-related facts and estimating minimal required actions considering the domain's structure.
    """

    def __init__(self, task):
        """
        Extract and index the goal facts for efficient comparison during heuristic evaluation.
        """
        self.goal_bclhkf4r = set()
        self.goal_chxwa5j4 = set()
        self.goal_gaijkwef = set()
        for fact in task.goals:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "bclhkf4r":
                self.goal_bclhkf4r.add(parts[1])
            elif pred == "chxwa5j4":
                self.goal_chxwa5j4.add(parts[1])
            elif pred == "gaijkwef":
                self.goal_gaijkwef.add((parts[1], parts[2]))

    def __call__(self, node):
        """
        Estimate the number of actions needed to reach the goal from the current state.
        
        This function:
        - Extracts predicates from the current state,
        - Compares to goal facts,
        - Estimates cost by summing missing goal facts weighted by action costs and considering zzar8qlk flag.
        """
        state = node.state

        current_bclhkf4r = set()
        current_mdawkki4 = set()
        current_chxwa5j4 = set()
        current_gaijkwef = set()
        zzar8qlk_true = False

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "bclhkf4r":
                current_bclhkf4r.add(parts[1])
            elif pred == "mdawkki4":
                current_mdawkki4.add(parts[1])
            elif pred == "chxwa5j4":
                current_chxwa5j4.add(parts[1])
            elif pred == "gaijkwef":
                current_gaijkwef.add((parts[1], parts[2]))
            elif pred == "zzar8qlk":
                zzar8qlk_true = True

        # Calculate missing facts from each predicate group
        missing_bclhkf4r = self.goal_bclhkf4r - current_bclhkf4r
        missing_chxwa5j4 = self.goal_chxwa5j4 - current_chxwa5j4
        missing_gaijkwef = self.goal_gaijkwef - current_gaijkwef

        # Heuristic action cost assumptions from analysis of domain actions:
        # - Creating or fixing a bclhkf4r or chxwa5j4 fact roughly costs 1 action each.
        # - Creating a missing gaijkwef fact involves a sequence of at least 2 actions.
        # - If zzar8qlk is false but needed for transitions, add 1 action cost to toggle it.

        # Check if zzar8qlk is needed:
        # From domain actions, many require zzar8qlk to be true in preconditions.
        # If missing goals include gaijkwef or bclhkf4r/chxwa5j4 that require zzar8qlk active, add toggle cost.
        # Since we do not know exact sequence, we heuristically add toggle cost if zzar8qlk True is needed but False now.

        need_zzar = False
        if missing_gaijkwef or missing_bclhkf4r or missing_chxwa5j4:
            # Since actions to achieve these facts depend on zzar8kl True mode,
            # assume toggle needed if currently false.
            if not zzar8qlk_true:
                need_zzar = True

        cost = 0
        cost += len(missing_bclhkf4r)  # 1 action each approx
        cost += len(missing_chxwa5j4)  # 1 action each approx
        cost += 2 * len(missing_gaijkwef)  # 2 actions each approx
        if need_zzar:
            cost += 1  # action to toggle zzar8qlk

        return cost
