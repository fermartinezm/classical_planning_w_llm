from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(bclhkf4r object1)".
    - `args`: The pattern to match (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, `False` otherwise.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain (dnbnd6kg).

    Summary:
    This heuristic estimates the number of actions required to reach a goal state in the secret_domain.
    The main difficulty is achieving all goal facts which are combinations of:
      - (bclhkf4r ?obj)
      - (chxwa5j4 ?obj)
      - (gaijkwef ?obj1 ?obj2)
    The domain features a boolean-like fluent (zzar8qlk) that restricts some actions and a state machine involving
    predicates mdawkki4, bclhkf4r, chxwa5j4, and gaijkwef that govern the transitions.

    Assumptions:
    - Each object satisfying goal predicates (bclhkf4r, chxwa5j4) can be "fixed" separately.
    - Achieving or fixing a (gaijkwef ?x ?y) goal fact requires a chain of precise actions involving zzar8qlk and predicates on the two objects.
    - The fluent (zzar8qlk) represents a resource or token that alternates states during actions.
    - Since the domain is quite abstract, the heuristic estimates the minimal "repairs" needed for goal predicates missing from the state.
    - Each needed fact that is false in the state costs at least 1, but setting (gaijkwef) facts likely costs more (≥2 actions).

    Heuristic Initialization:
    - Extract all goal facts and classify them by predicate.
    - Remember static facts (none in this domain) - no static info to extract here.
    - The heuristic uses the current state facts to detect gaps towards goals, counting missing facts by type.

    Step-By-Step Thinking for Computing Heuristic:
    1. If the current state already satisfies all goals, heuristic returns 0.
    2. For each goal fact of type (bclhkf4r ?obj), if missing in state, count +1 needed action.
    3. For each goal fact of type (chxwa5j4 ?obj), if missing, count +1.
    4. For each goal fact (gaijkwef ?obj1 ?obj2), if missing, count +2 (heuristically set as more complex).
    5. Consider the global fact (zzar8qlk):
       - Actions toggle (zzar8qlk) on/off; switching is needed to set gaijkwef and mdawkki4.
       - Each toggle costs at least 1 action.
       - If goal requires gaijkwef facts, heuristic adds possibly 1 extra toggle cost unless already set.
    6. Sum all estimated costs as heuristic value.
    7. The heuristic is finite for solvable states, and 0 only for goal states.

    This heuristic is designed for efficiency and pruning ability in greedy best-first search, not for admissibility.
    """

    def __init__(self, task):
        # Extract goal sets by predicate name for quick membership checks.
        self.goal_bclhkf4r = set()
        self.goal_chxwa5j4 = set()
        self.goal_gaijkwef = set()
        self.goal_zzar8qlk = False

        for goal in task.goals:
            parts = get_parts(goal)
            pred = parts[0]
            if pred == "bclhkf4r":
                self.goal_bclhkf4r.add(parts[1])
            elif pred == "chxwa5j4":
                self.goal_chxwa5j4.add(parts[1])
            elif pred == "gaijkwef":
                self.goal_gaijkwef.add((parts[1], parts[2]))
            elif pred == "zzar8qlk":
                self.goal_zzar8qlk = True

        # No static facts to process in this domain.
        self.static_facts = task.static

    def __call__(self, node):
        state = node.state

        # Quick check: if all goals are satisfied, heuristic=0
        if node.task.goal_reached(state):
            return 0

        # Collect facts currently true for quick membership tests
        state_bclhkf4r = set()
        state_chxwa5j4 = set()
        state_gaijkwef = set()
        state_zzar8qlk = False

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "bclhkf4r":
                state_bclhkf4r.add(parts[1])
            elif pred == "chxwa5j4":
                state_chxwa5j4.add(parts[1])
            elif pred == "gaijkwef":
                state_gaijkwef.add((parts[1], parts[2]))
            elif pred == "zzar8qlk":
                state_zzar8qlk = True

        # Count missing facts in each category
        missing_bclhkf4r = sum(1 for obj in self.goal_bclhkf4r if obj not in state_bclhkf4r)
        missing_chxwa5j4 = sum(1 for obj in self.goal_chxwa5j4 if obj not in state_chxwa5j4)
        missing_gaijkwef = sum(1 for pair in self.goal_gaijkwef if pair not in state_gaijkwef)

        # Base cost estimate for missing predicates:
        # - Each missing bclhkf4r or chxwa5j4: 1 action approx.
        # - Each missing gaijkwef: 2 actions approx. due to complexity of their achieving actions.

        cost = missing_bclhkf4r + missing_chxwa5j4 + 2 * missing_gaijkwef

        # zzar8qlk is a toggle-like predicate required for certain actions:
        # If the goal includes zzar8qlk but the state does not have it, count 1 extra toggle cost.
        # Also, if gaijkwef facts are missing (that require toggling zzar8qlk),
        # and zzar8qlk is currently in opposite state, add 1 to switch.
        if self.goal_zzar8qlk and not state_zzar8qlk:
            cost += 1
        elif missing_gaijkwef > 0:
            # Heuristic assumes that each needed gaijkwef might need at least one toggle of zzar8qlk
            # if the current zzar8kl state does not align with needed action.
            # Since we do not know exactly, add a small penalty per group of gaijkwef facts.
            # To keep heuristic finite and simple, add +1 only once.
            cost += 1

        # Return the heuristic cost estimate.
        return cost
