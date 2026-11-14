from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact: remove surrounding parentheses and split by spaces."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a fact matches the given pattern with optional wildcards.

    :param fact: String, e.g. "(predicate arg1 arg2)"
    :param args: Strings, e.g. "predicate", "*", "arg2"
    :return: True if matches, else False
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (dnbnd6kg).

    Summary:
    This heuristic estimates the number of actions required to achieve the goal
    state by reasoning about the status of objects in terms of three possible
    states represented by predicates: bclhkf4r(?obj), mdawkki4(?obj), chxwa5j4(?obj),
    and about the global flag zzar8qlk. The key relational predicate gaijkwef(?obj1, ?obj2)
    encodes links between objects.

    Assumptions:
    - The goal predicates include bclhkf4r, chxwa5j4 on specific objects, and possibly
      gaijkwef relations.
    - Objects cycle through states (bclhkf4r, mdawkki4, chxwa5j4) under the actions defined.
    - The zzar8qlk predicate toggles TRUE/FALSE and gates certain transitions.
    - The chain of gaijkwef pointers form a key structure to satisfy the goal.
    - Because the domain involves "passing" or "switching" states along object links,
      achieving the goal requires ensuring correct activation of these chains and 
      states simultaneously.

    Heuristic Initialization:
    - Extract the goal facts for quick membership checks.
    - Extract all involved objects from the current state and goals.
    - Identify the objects involved in the goal predicates of types bclhkf4r, chxwa5j4, gaijkwef.
    - Index current states of all objects for rapid lookup.
    - Gather current zzar8qlk state.

    Step-By-Step Thinking for Computing Heuristic:
    1. Quickly return 0 if state satisfies the goal.

    2. For each goal predicate, check if it is currently true in the state.
       - For each goal fact missing, estimate cost to achieve it by analyzing:
         a) Objects that must have predicate bclhkf4r or chxwa5j4 in goal but currently don't.
            This likely requires cycling through mdawkki4 state and flipping zzar8qlk flag properly.
            Each such state "fix" requires at least one or two actions:
             - One to make the object mdawkki4 to bclhkf4r or chxwa5j4 (via mri4ci1h or fqixw4x9).
             - Possibly one more action to set the global flag zzar8qlk to the correct value, if needed.

         b) For gaijkwef(?obj1,?obj2) goals not currently true:
            - Requires (?obj1) to be mdawkki4 and (?obj2) to be bclhkf4r with zzar8qlk true,
              so action inz39ca6 or j2i3gtal can establish gaijkwef.
            - If gaijkwef is missing, likely multiple steps of state-changing actions on the
              involved objects plus global flag toggling are needed.

    3. Count the minimal number of missing goals:
       - Number of missing bclhkf4r predicates in the goal.
       - Number of missing chxwa5j4 predicates in the goal.
       - Number of missing gaijkwef relations in the goal.

    4. As the global flag zzar8qlk can flip back and forth and actions require it to be true or false,
       add heuristic cost penalty proportional to how many zzar8qlk flips might be required,
       approximated by the number of objects that must change state (since most actions toggle it).

    5. Sum up these costs to produce a heuristic estimate.

    This heuristic is not admissible but aims to be tight and efficient to compute,
    helping greedy best-first search prune unpromising branches intelligently.

    """

    def __init__(self, task):
        """Initialize by extracting goal predicates and their objects."""
        self.goals = task.goals
        self.static = task.static  # empty in this domain, but kept for completeness

        # Extract goal requirements by predicate:
        # Sets of objects for bclhkf4r(?obj), chxwa5j4(?obj), and tuples for gaijkwef(?obj1, ?obj2)
        self.goal_bclhkf4r = set()
        self.goal_chxwa5j4 = set()
        self.goal_gaijkwef = set()

        for fact in self.goals:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "bclhkf4r":
                self.goal_bclhkf4r.add(parts[1])
            elif pred == "chxwa5j4":
                self.goal_chxwa5j4.add(parts[1])
            elif pred == "gaijkwef":
                self.goal_gaijkwef.add((parts[1], parts[2]))

    def __call__(self, node):
        state = node.state
        # If goal already reached, heuristic is zero
        if self.goals <= state:
            return 0

        # Index current facts of interest:
        # Sets of objects currently with specific predicates
        current_bclhkf4r = set()
        current_chxwa5j4 = set()
        current_mdawkki4 = set()
        current_gaijkwef = set()
        zzar8qlk_true = False

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "bclhkf4r":
                current_bclhkf4r.add(parts[1])
            elif pred == "chxwa5j4":
                current_chxwa5j4.add(parts[1])
            elif pred == "mdawkki4":
                current_mdawkki4.add(parts[1])
            elif pred == "gaijkwef":
                current_gaijkwef.add((parts[1], parts[2]))
            elif pred == "zzar8qlk":
                zzar8qlk_true = True

        # Count missing predicates from the goal:
        missing_bclhkf4r = self.goal_bclhkf4r - current_bclhkf4r
        missing_chxwa5j4 = self.goal_chxwa5j4 - current_chxwa5j4
        missing_gaijkwef = self.goal_gaijkwef - current_gaijkwef

        # Estimate cost for missing bclhkf4r facts:
        # To establish bclhkf4r(?obj) if missing,
        # minimal actions: either mri4ci1h (from mdawkki4 to bclhkf4r+zzar8qlk),
        # or fqixw4x9 (to chxwa5j4 if precondition met),
        # or requires cycling through states using actions that flip zzar8qlk.
        # We approximate 2 actions per missing bclhkf4r to cover state changes and zzar8qlk flips.
        cost_bclhkf4r = 2 * len(missing_bclhkf4r)

        # Similarly for missing chxwa5j4 facts:
        # Requires mdawkki4 -> chxwa5j4 and clearing zzar8qlk, approx 2 actions as well.
        cost_chxwa5j4 = 2 * len(missing_chxwa5j4)

        # For missing gaijkwef(?o1,?o2), establishing it requires:
        # - ?o1 in mdawkki4
        # - ?o2 in bclhkf4r
        # - zzar8qlk true
        # Might take several state changes and flag toggles.
        # Approximate cost as 3 actions per missing link to cover these transitions.
        cost_gaijkwef = 3 * len(missing_gaijkwef)

        # Add cost for potentially flipping zzar8kl multiple times:
        # Number of objects involved that are currently in mdawkki4 or must change state.
        # Count how many objects are in mdawkki4 but need different states:
        potentially_to_fix = (missing_bclhkf4r | missing_chxwa5j4)
        n_flag_flips = len(potentially_to_fix)
        cost_flag_flips = n_flag_flips  # approx 1 extra action per object needing flag flip

        # Sum total heuristic cost estimate
        heuristic_cost = cost_bclhkf4r + cost_chxwa5j4 + cost_gaijkwef + cost_flag_flips

        # Minimum heuristic should be at least 1 for non-goal states
        return max(1, heuristic_cost)
