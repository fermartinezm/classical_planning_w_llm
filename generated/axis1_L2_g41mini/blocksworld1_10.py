from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components from a PDDL fact string by stripping parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: A PDDL fact string, e.g. "(bclhkf4r obj1)"
    - `args`: Pattern tokens, wildcards '*' supported.
    Returns True if all parts match, else False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain domain.

    Summary
    -------
    Estimates the number of actions needed to achieve all goal facts from the
    current state for the secret_domain. The main complexity comes from correctly
    arranging or transferring objects through a stateful predicate `(zzar8qlk)`
    and using the chain of predicates `(bclhkf4r ?obj)`, `(mdawkki4 ?obj)`,
    `(chxwa5j4 ?obj)`, and `(gaijkwef ?obj1 ?obj2)` as key markers of progress.

    Assumptions
    -----------
    - The `(zzar8qlk)` predicate acts like a "token" or "turn" indicator that enables some actions.
    - Objects can be in one of three main unary states: `(bclhkf4r)`, `(mdawkki4)`, `(chxwa5j4)`.
    - The `(gaijkwef ?x ?y)` predicate encodes a directed relation between objects.
    - The goal is composed of some `(bclhkf4r ?obj)`, `(chxwa5j4 ?obj)`, and `(gaijkwef ?obj1 ?obj2)` facts to be satisfied.
    - The problem mainly revolves around forming or preserving specific `(gaijkwef ?x ?y)` links and object unary predicates.
    - The heuristic does not need to be admissible, but should be efficiently computable and reduce node expansions.

    Heuristic Initialization
    ------------------------
    - Extract goal facts and separate them by predicate:
      * Goal sets for `(bclhkf4r)`, `(mdawkki4)`, `(chxwa5j4)`, `(gaijkwef)`, and for the token `(zzar8qlk)`.
    - Extract static facts if available (none in this domain).
    - The heuristic will count the number of mismatched goal facts comparing current and goal state,
      with special consideration of `(gaijkwef)` relations.
    - Use the presence or absence of `(zzar8qlk)` in current state as a factor,
      since many actions require it - managing it may add extra steps.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. For unary predicates `(bclhkf4r)`, `(mdawkki4)`, and `(chxwa5j4)`, count how many objects need to be added or removed to achieve the goal.
       Each such missing or extra object unary fact implies at least one action.
    2. For binary predicates `(gaijkwef ?x ?y)` that are in the goal but missing in the current state,
       each missing link usually requires a sequence of at least one action (like `inz39ca6` or `j2i3gtal`)
       that rearranges these links, so count each missing link as at least one action.
    3. Check if the token `(zzar8qlk)` is present if needed to perform actions; if missing but needed,
       add 1 extra step to recover it (e.g., `mri4ci1h`).
    4. Since the domain actions toggle states and transfer the token, some overhead steps are needed:
       approximate this by adding the maximum between the count of missing `(zzar8qlk)` and zero,
       plus the counts of mismatches.
    5. Return the sum of these estimates as the heuristic value.
    6. If all goals are met, heuristic returns 0.

    This heuristic is not admissible, but tries to be informative and fast to compute.
    """

    def __init__(self, task):
        # Extract goal facts by predicate for quick membership tests
        self.goal_bclhkf4r = set()
        self.goal_mdawkki4 = set()
        self.goal_chxwa5j4 = set()
        self.goal_gaijkwef = set()
        self.goal_zzar8qlk = False

        for fact in task.goals:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "bclhkf4r":
                self.goal_bclhkf4r.add(parts[1])
            elif pred == "mdawkki4":
                self.goal_mdawkki4.add(parts[1])
            elif pred == "chxwa5j4":
                self.goal_chxwa5j4.add(parts[1])
            elif pred == "gaijkwef":
                self.goal_gaijkwef.add((parts[1], parts[2]))
            elif pred == "zzar8qlk":
                self.goal_zzar8qlk = True

        # No static facts to extract in this domain
        self.static = task.static

    def __call__(self, node):
        state = node.state

        # Helper sets for current state predicates
        current_bclhkf4r = set()
        current_mdawkki4 = set()
        current_chxwa5j4 = set()
        current_gaijkwef = set()
        current_zzar8qlk = False

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
                current_zzar8qlk = True

        # If goal already reached, return 0
        goal_facts = set()
        goal_facts |= {"(bclhkf4r %s)" % obj for obj in self.goal_bclhkf4r}
        goal_facts |= {"(mdawkki4 %s)" % obj for obj in self.goal_mdawkki4}
        goal_facts |= {"(chxwa5j4 %s)" % obj for obj in self.goal_chxwa5j4}
        goal_facts |= {"(gaijkwef %s %s)" % (x, y) for (x, y) in self.goal_gaijkwef}
        if self.goal_zzar8qlk:
            goal_facts.add("(zzar8qlk)")

        if self.goal_zzar8qlk == current_zzar8qlk and \
           self.goal_bclhkf4r == current_bclhkf4r and \
           self.goal_mdawkki4 == current_mdawkki4 and \
           self.goal_chxwa5j4 == current_chxwa5j4 and \
           self.goal_gaijkwef == current_gaijkwef:
            return 0

        # Count mismatches: missing goal facts plus extra facts not wanted
        # Missing unary preds in goal:
        missing_bclhkf4r = self.goal_bclhkf4r - current_bclhkf4r
        extra_bclhkf4r = current_bclhkf4r - self.goal_bclhkf4r

        missing_mdawkki4 = self.goal_mdawkki4 - current_mdawkki4
        extra_mdawkki4 = current_mdawkki4 - self.goal_mdawkki4

        missing_chxwa5j4 = self.goal_chxwa5j4 - current_chxwa5j4
        extra_chxwa5j4 = current_chxwa5j4 - self.goal_chxwa5j4

        # Missing and extra binary (gaijkwef)
        missing_gaijkwef = self.goal_gaijkwef - current_gaijkwef
        extra_gaijkwef = current_gaijkwef - self.goal_gaijkwef

        # zzar8qlk token penalty if needed (present or missing)
        zzar8qlk_mismatch = 0
        if self.goal_zzar8qlk != current_zzar8qlk:
            # Usually the token must be toggled using actions like mri4ci1h or fqixw4x9
            # Count at least 1 step if mismatch
            zzar8qlk_mismatch = 1

        # Sum up heuristic cost components carefully.

        # Each missing unary fact usually requires at least one action to produce,
        # and each extra unary fact likely requires at least one action to remove.
        unary_cost = (
            len(missing_bclhkf4r) + len(extra_bclhkf4r) +
            len(missing_mdawkki4) + len(extra_mdawkki4) +
            len(missing_chxwa5j4) + len(extra_chxwa5j4)
        )

        # Each missing `(gaijkwef ?x ?y)` link requires at least one action to create,
        # while extra links normally must be destroyed, adding to cost.
        binary_cost = len(missing_gaijkwef) + len(extra_gaijkwef)

        # Estimate: some unary predicate transitions require two actions minimum,
        # but we add a moderation factor by counting all unary costs fully.
        # Likewise, binary links are harder, but count as one each.

        # The zzar8qlk token is essential for most transitions, so consider if it is mismatched.

        # Large mismatches may indicate many reorderings, so this heuristic is "optimistic conservative".

        heuristic_value = unary_cost + binary_cost + zzar8qlk_mismatch

        # Return at least 1 if heuristic is non-zero to avoid search stagnation
        return max(heuristic_value, 1)
