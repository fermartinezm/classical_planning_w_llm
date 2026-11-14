from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(gaijkwef obj1 obj2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (dnbnd6kg).

    Summary:
    ----------
    This heuristic estimates the minimal number of actions required to reach the goal state by focusing on the
    achievement of goal predicates: (bclhkf4r ?obj), (chxwa5j4 ?obj), and (gaijkwef ?obj1 ?obj2) as per the goal.
    The domain's actions appear to define three object states/properties (bclhkf4r, mdawkki4, chxwa5j4), a global toggle (zzar8qlk),
    and a relationship predicate gaijkwef, which changes dynamically via the actions.

    Assumptions:
    -----------
    - The goal specifies some combination of (bclhkf4r), (chxwa5j4), and (gaijkwef) facts that must hold.
    - The global boolean fact (zzar8qlk) flips between true and false during execution and controls which actions can be applied.
    - Objects transit through states indicated by predicates: bclhkf4r (base state), mdawkki4 (intermediate state), chxwa5j4 (final state).
    - The relationship gaijkwef is directional and changes between pairs of objects.
    - Transition of one object or pair affects another object(s), inducing a chain of effects facilitating goal achievement.
    - Each goal fact that is false in the current state must be fixed by a sequence of domain actions, often requiring toggling (zzar8qlk) and state changes.

    Heuristic Initialization:
    -------------------------
    - The heuristic records the goal facts categorized by predicate:
      * goal_bclhkf4r: set of objects required to hold (bclhkf4r)
      * goal_chxwa5j4: set of objects required to hold (chxwa5j4)
      * goal_gaijkwef: set of (obj1,obj2) pairs required by (gaijkwef)
    - No static facts are used because the domain provides none.
    - The heuristic uses the current state to estimate how many steps it takes to fulfill missing goal facts.
    - Actions require (zzar8qlk) true or false, and certain predicates on objects, so the heuristic factors expected toggles of (zzar8qlk).

    Step-By-Step Thinking for Computing Heuristic:
    -----------------------------------------------
    1. Identify which goal facts are missing in the current state:
       - Missing (bclhkf4r ?obj) facts.
       - Missing (chxwa5j4 ?obj) facts.
       - Missing (gaijkwef ?obj1 ?obj2) pairs.

    2. Count missing goal predicates separately.

    3. Analyze how to achieve missing goal facts per predicate type:
       - For (bclhkf4r ?obj):
         + Actions can add this predicate only when (zzar8qlk) is true or by applying "mri4ci1h" (which adds bclhkf4r and chxwa5j4).
         + Usually requires moving from mdawkki4 state back to bclhkf4r and chxwa5j4 states, likely via a series of action cycles toggling zzar8qlk.
       - For (chxwa5j4 ?obj):
         + Usually achieved by "mri4ci1h" from mdawkki4 or "fqixw4x9" action when certain conditions hold.
         + May require toggling zzar8qlk and changing object predicates.
       - For (gaijkwef ?obj1 ?obj2):
         + The state of this predicate changes via the "inz39ca6" or "j2i3gtal" actions, which also depend on zzar8qlk status.
         + Setting gaijkwef to a goal pair may require removing gaijkwef from other pairs or reaching a particular intermediate object state.

    4. The heuristic approximates how many toggles of zzar8kl and related intermediate state changes are necessary to fix each missing goal fact.
       Because solving one goal fact often requires ~2-3 actions, including toggling (zzar8qlk) and predicate updates,
       the heuristic counts:
       - Each missing (bclhkf4r) or (chxwa5j4) goal fact roughly as 2 actions.
       - Each missing (gaijkwef) goal pair roughly as 3 actions due to required state changes and toggling.

    5. The heuristic returns the sum of these weighted counts as the estimate.

    6. In goal states, all goals hold, so heuristic returns 0.

    7. The heuristic guarantees finiteness as the domain and goal facts are finite. Overestimation is acceptable.

    This heuristic balances informative estimation with computational efficiency based on core domain dynamics.
    """

    def __init__(self, task):
        # Extract and categorize goal predicates by type.
        self.goal_bclhkf4r = set()
        self.goal_chxwa5j4 = set()
        self.goal_gaijkwef = set()
        for goal in task.goals:
            parts = get_parts(goal)
            if not parts:
                continue
            pred = parts[0]
            if pred == "bclhkf4r" and len(parts) == 2:
                self.goal_bclhkf4r.add(parts[1])
            elif pred == "chxwa5j4" and len(parts) == 2:
                self.goal_chxwa5j4.add(parts[1])
            elif pred == "gaijkwef" and len(parts) == 3:
                self.goal_gaijkwef.add((parts[1], parts[2]))
        # static facts are unused in this domain
        self.static = task.static

    def __call__(self, node):
        state = node.state

        # If goal is reached, heuristic = 0
        # Check if all goal predicates hold in state
        if (self.goal_bclhkf4r <= state and
            self.goal_chxwa5j4 <= state and
            all(f"(gaijkwef {o1} {o2})" in state for o1, o2 in self.goal_gaijkwef)):
            return 0

        # Count missing bclhkf4r goals
        missing_bclhkf4r = sum(1 for obj in self.goal_bclhkf4r if f"(bclhkf4r {obj})" not in state)

        # Count missing chxwa5j4 goals
        missing_chxwa5j4 = sum(1 for obj in self.goal_chxwa5j4 if f"(chxwa5j4 {obj})" not in state)

        # Count missing gaijkwef pairs
        missing_gaijkwef = 0
        for o1, o2 in self.goal_gaijkwef:
            if f"(gaijkwef {o1} {o2})" not in state:
                missing_gaijkwef += 1

        # Count how many toggling states (zzar8qlk) are currently true or false
        has_zzar8qlk = "(zzar8qlk)" in state

        # Basic action cost estimates per missing fact type:
        # - Fixing a missing bclhkf4r or chxwa5j4 requires approx 2 actions (state toggle + predicate change)
        # - Fixing a missing gaijkwef pair requires approx 3 actions (toggling and manipulation of pairs)
        cost_bclhkf4r = 2 * missing_bclhkf4r
        cost_chxwa5j4 = 2 * missing_chxwa5j4
        cost_gaijkwef = 3 * missing_gaijkwef

        # Additional minor adjustment:
        # If (zzar8qlk) is currently false and we have missing facts,
        # at least one toggle action is needed to allow actions that require (zzar8qlk).
        toggle_cost = 1 if (not has_zzar8qlk and (missing_bclhkf4r + missing_chxwa5j4 + missing_gaijkwef) > 0) else 0

        heuristic_value = cost_bclhkf4r + cost_chxwa5j4 + cost_gaijkwef + toggle_cost

        # Minimum heuristic value is 1 for non-goal states to avoid misleading zero cost
        return max(heuristic_value, 1)
