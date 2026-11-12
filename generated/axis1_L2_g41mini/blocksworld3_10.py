from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(bclhkf4r fk4o7pqr)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain domain.

    Summary
    -------
    This heuristic estimates the number of actions needed to reach the goal state by 
    considering the key predicates involved in actions and the goal conditions.
    It focuses on the relationships among facts involving (bclhkf4r ?x), (mdawkki4 ?x),
    (zzar8qlk), (gaijkwef ?x ?y), and (chxwa5j4 ?x), which are linked through the operators.

    Assumptions
    -----------
    - The goal is conjunctive involving facts over predicates bclhkf4r, chxwa5j4, and gaijkwef.
    - The fluents mdawkki4 and zzar8qlk represent "states" that toggle through actions but 
      are not part of the goal.
    - The "chain" of gaijkwef links pairs of objects and needs to be established per goal.
    - Actions move "ownership" or "state" across objects, with zzar8qlk as a global toggle.
    - The heuristic does not guarantee admissibility but aims to estimate the needed steps 
      while being efficient to compute.

    Heuristic Initialization
    ------------------------
    - Extract the goal facts and separate them by their predicate.
    - Collect the sets of goal bclhkf4r, chxwa5j4, and gaijkwef facts.
    - Since there are no static facts in this domain (empty static set), no static data 
      structures are required.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    1. If the state already satisfies the goal, return 0.

    2. Count how many goal bclhkf4r facts are missing in the current state.
       Each missing bclhkf4r fact will require at least one action to establish.

    3. Count how many goal chxwa5j4 facts are missing.
       Each missing chxwa5j4 fact requires at least one action, normally linked with an mdawkki4 toggle.

    4. Count how many goal gaijkwef facts are missing.
       Each missing gaijkwef(?x, ?y) represents a required "chain" relation to establish.
       Typically, this requires at least one operator call to create or modify.

    5. The domain enforces the global predicate zzar8qlk as a toggle (true or false),
       which can be changed by operators. If the state lacks zzar8qlk but needed in goal 
       (or vice versa), factor in one action to flip it.

    6. Aggregating these counts, the heuristic is a weighted sum of:
       - Number of missing bclhkf4r goal facts (weight 1)
       - Number of missing chxwa5j4 goal facts (weight 1)
       - Number of missing gaijkwef goal facts (weight 1)
       - One if zzar8qlk global predicate differs from goal (weight 1)

    7. Adjustment for mutual exclusivities:
       - Because operators toggle zzar8qlk while manipulating bclhkf4r and mdawkki4, 
         some actions will achieve multiple subgoals together.
       - To reflect this, use the maximum of the counts of missing bclhkf4r and chxwa5j4 
         as rough approximation to avoid overestimation.
       
    8. Return the sum of:
       - max(missing_bclhkf4r, missing_chxwa5j4)
       - missing gaijkwef facts
       - zzar8qlk mismatch cost (0 or 1)

    This heuristic can be computed by simple set membership queries and is efficient.
    """

    def __init__(self, task):
        """Initialize the heuristic by processing goal facts."""
        # Store the full goal facts
        self.goals = task.goals

        # Extract goal facts by predicate for fast lookup
        self.goal_bclhkf4r = set()
        self.goal_chxwa5j4 = set()
        self.goal_gaijkwef = set()
        self.goal_zzar8qlk = False

        for fact in self.goals:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "bclhkf4r":
                # single argument predicate
                self.goal_bclhkf4r.add(parts[1])
            elif pred == "chxwa5j4":
                self.goal_chxwa5j4.add(parts[1])
            elif pred == "gaijkwef":
                # two arguments predicate
                self.goal_gaijkwef.add((parts[1], parts[2]))
            elif pred == "zzar8qlk":
                self.goal_zzar8qlk = True  # present in goal

    def __call__(self, node):
        """
        Compute heuristic value estimating required number of actions to achieve goal.

        Parameters:
        - node: the current search node with attribute 'state' (a frozenset of facts).

        Returns:
        - int: Estimated number of actions to reach the goal from current state.
        """
        state = node.state

        # Heuristic is zero if goal is reached
        if self.goals <= state:
            return 0

        # Current sets of facts by predicate
        current_bclhkf4r = set()
        current_chxwa5j4 = set()
        current_gaijkwef = set()
        current_zzar8qlk = False

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "bclhkf4r":
                current_bclhkf4r.add(parts[1])
            elif pred == "chxwa5j4":
                current_chxwa5j4.add(parts[1])
            elif pred == "gaijkwef":
                current_gaijkwef.add((parts[1], parts[2]))
            elif pred == "zzar8qlk":
                current_zzar8qlk = True

        # Count missing goal facts
        missing_bclhkf4r = len(self.goal_bclhkf4r - current_bclhkf4r)
        missing_chxwa5j4 = len(self.goal_chxwa5j4 - current_chxwa5j4)
        missing_gaijkwef = len(self.goal_gaijkwef - current_gaijkwef)
        missing_zzar8qlk = 0 if current_zzar8qlk == self.goal_zzar8qlk else 1

        # Aggregate heuristic estimate:
        # Combine missing bclhkf4r and chxwa5j4 via maximum because these two are tightly coupled by actions,
        # while missing gaijkwef is additive, and zzar8klk mismatch is an additional toggle action.
        heuristic_val = max(missing_bclhkf4r, missing_chxwa5j4) + missing_gaijkwef + missing_zzar8qlk

        return heuristic_val
