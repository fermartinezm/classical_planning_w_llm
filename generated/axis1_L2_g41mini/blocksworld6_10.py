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
    Domain-dependent heuristic for the secret_domain (dnbnd6kg).

    Summary
    -------
    This heuristic estimates the number of actions required to reach the goal from a given state by analyzing
    the status of facts involved (bclhkf4r, mdawkki4, gaijkwef, chxwa5j4, zzar8qlk) and their relations.
    It focuses on how many "chains" or "links" between objects need to be rearranged or fixed to satisfy the goal,
    considering the unique interplay between the zzar8qlk flag and the object predicates.

    Assumptions
    -----------
    - The goal consists of certain (bclhkf4r ?obj), (chxwa5j4 ?obj), and (gaijkwef ?obj1 ?obj2) facts.
    - The predicate zzar8qlk acts like a global toggle that enables specific actions and affects possible transitions.
    - Objects can exist in "states" reflected by predicates mdawkki4, bclhkf4r, chxwa5j4, and relations gaijkwef.
    - The heuristic does not require admissibility, but aims at guiding greedy best-first search effectively.

    Heuristic Initialization
    ------------------------
    - Extract all goal facts and index them by predicate for quick lookup.
    - Extract the set of objects appearing in the goal to limit the heuristic calculation scope.
    - No static facts are used explicitly, as the domain does not define static predicates.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Check if the current state satisfies all goal facts; if yes, heuristic is 0.

    2. For each object involved in the goal:
       - Check if the goal requires (bclhkf4r ?obj), (chxwa5j4 ?obj), or (mdawkki4 ?obj).
       - Check whether these are satisfied currently.
       - Count how many of these predicates mismatch current state versus goal state.

    3. For each gaijkwef relation (pair) required by the goal:
       - Check if it exists in the current state.
       - Count as one unmet goal if missing.

    4. Consider the global flag zzar8qlk:
       - Since many actions depend on it being true or false, estimate a cost of toggling this flag as needed.
       - If the flag state in the current state differs from a convenient state for progressing toward the goal, add a small penalty.

    5. Combine counts:
       - Number of mismatched predicates on objects + number of missing gaijkwef relations + cost to toggle zzar8qlk.
       - Multiply each count by small weights approximating the action cost (usually 1 per fix).

    The heuristic is the sum of these costs and estimates the minimum number of changes (actions) required
    to reach the goal.

    This heuristic sacrifices admissibility for informativeness and computational efficiency,
    guiding the greedy best-first search effectively.
    """

    def __init__(self, task):
        # Extract and index goals by predicate
        self.goal_bclhkf4r = set()
        self.goal_mdawkki4 = set()
        self.goal_gaijkwef = set()
        self.goal_chxwa5j4 = set()
        self.goal_zzar8qlk = False

        for fact in task.goals:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "bclhkf4r":
                self.goal_bclhkf4r.add(parts[1])
            elif pred == "mdawkki4":
                self.goal_mdawkki4.add(parts[1])
            elif pred == "gaijkwef":
                self.goal_gaijkwef.add((parts[1], parts[2]))
            elif pred == "chxwa5j4":
                self.goal_chxwa5j4.add(parts[1])
            elif pred == "zzar8qlk":
                self.goal_zzar8qlk = True

        # Derive the set of all objects referenced in the goal
        self.goal_objects = set()
        self.goal_objects.update(self.goal_bclhkf4r)
        self.goal_objects.update(self.goal_mdawkki4)
        self.goal_objects.update(self.goal_chxwa5j4)
        for o1, o2 in self.goal_gaijkwef:
            self.goal_objects.add(o1)
            self.goal_objects.add(o2)

    def __call__(self, node):
        state = node.state

        # Quick check: if goal is reached, heuristic is 0
        if (self.goal_zzar8qlk and ("(zzar8qlk)" not in state)) or \
           (not self.goal_zzar8qlk and ("(zzar8qlk)" in state)):
            # zzar8qlk flag mismatch is counted below;
            # we only do this to continue with heuristic calculation
            pass

        if (
            self.goal_bclhkf4r <= {get_parts(f)[1] for f in state if f.startswith("(bclhkf4r")}
            and self.goal_mdawkki4 <= {get_parts(f)[1] for f in state if f.startswith("(mdawkki4")}
            and self.goal_chxwa5j4 <= {get_parts(f)[1] for f in state if f.startswith("(chxwa5j4")}
            and self.goal_gaijkwef <= {(get_parts(f)[1], get_parts(f)[2]) for f in state if f.startswith("(gaijkwef")}
            and ((self.goal_zzar8kl and "(zzar8qlk)" in state) or (not self.goal_zzar8kl))
        ):
            return 0

        # Current objects that have each predicate true:
        bclhkf4r_true = {get_parts(f)[1] for f in state if f.startswith("(bclhkf4r")}
        mdawkki4_true = {get_parts(f)[1] for f in state if f.startswith("(mdawkki4")}
        chxwa5j4_true = {get_parts(f)[1] for f in state if f.startswith("(chxwa5j4")}
        gaijkwef_true = {(get_parts(f)[1], get_parts(f)[2]) for f in state if f.startswith("(gaijkwef")}
        zzar8qlk_true = ("(zzar8qlk)" in state)

        cost = 0

        # Weight factors for each kind of fix (approximate action cost)
        W_OBJECT_PREDICATE = 1
        W_RELATION = 1
        W_FLAG_TOGGLE = 1

        # Count mismatches on bclhkf4r predicate for goal objects
        for obj in self.goal_bclhkf4r:
            if obj not in bclhkf4r_true:
                cost += W_OBJECT_PREDICATE
        # Count objects that should NOT have the predicate but currently have it.
        # This helps because actions often toggle predicates on objects.
        for obj in bclhkf4r_true:
            if obj not in self.goal_bclhkf4r and obj in self.goal_objects:
                cost += W_OBJECT_PREDICATE // 2  # smaller penalty for unnecessary predicate

        # Count mismatches on mdawkki4 predicate for goal objects
        for obj in self.goal_mdawkki4:
            if obj not in mdawkki4_true:
                cost += W_OBJECT_PREDICATE
        for obj in mdawkki4_true:
            if obj not in self.goal_mdawkki4 and obj in self.goal_objects:
                cost += W_OBJECT_PREDICATE // 2

        # Count mismatches on chxwa5j4 predicate for goal objects
        for obj in self.goal_chxwa5j4:
            if obj not in chxwa5j4_true:
                cost += W_OBJECT_PREDICATE
        for obj in chxwa5j4_true:
            if obj not in self.goal_chxwa5j4 and obj in self.goal_objects:
                cost += W_OBJECT_PREDICATE // 2

        # Count missing gaijkwef relations in goal
        for pair in self.goal_gaijkwef:
            if pair not in gaijkwef_true:
                cost += W_RELATION
        # Count gaijkwef pairs in current state that should not be there:
        for pair in gaijkwef_true:
            if pair not in self.goal_gaijkwef and (pair[0] in self.goal_objects or pair[1] in self.goal_objects):
                cost += W_RELATION // 2

        # Consider the zzar8qlk flag (global toggle)
        if zzar8kl_true != self.goal_zzar8kl:
            cost += W_FLAG_TOGGLE

        # Return heuristic estimate (int)
        return int(cost)
