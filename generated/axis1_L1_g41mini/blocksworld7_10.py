from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(yfm3frbn b1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain (z1spq3ra).

    Summary:
    This heuristic estimates the minimum number of actions required to reach a goal state from a given state
    in the secret_domain. The domain revolves around manipulating three unary predicates on objects
    (yfm3frbn, a4ha3ujl, jjtnwk3y), a global predicate (ug241uf7), and the binary predicate s0899d5u,
    which relates pairs of objects. The actions allow toggling these predicates on single objects
    or transferring the "state" between pairs of objects maintaining consistent predicate combinations.

    Assumptions:
    - Each goal fact involves a unary predicate (yfm3frbn or a4ha3ujl) on some object or a binary relation s0899d5u.
    - The predicate ug241uf7 can be either true or false and is required for some actions, influencing state transitions.
    - The key difficulty lies in achieving the correct distribution of these predicates on objects simultaneously satisfying s0899d5u relations.
    - Since no static facts are relevant, heuristic relies solely on analyzing current and goal predicates.

    Heuristic Initialization:
    - Extract the sets of objects that must have predicates yfm3frbn, a4ha3ujl, jjtnwk3y, and pairs for predicate s0899d5u, from the goal.
    - No static knowledge is used since static is empty.
    - No domain-specific graph or advanced structure is needed due to the limited complexity of predicates and relations.

    Step-By-Step Thinking for Computing Heuristic:
    1. If the current state already satisfies all goal facts, heuristic is 0.
    2. Count how many goal unary predicates (yfm3frbn, a4ha3ujl) are missing on their respective objects.
    3. Count how many goal binary predicates s0899d5u are missing.
    4. Count how many objects that must have jjtnwk3y in the goal currently lack it.
    5. Consider that the global predicate ug241uf7 may be required to change to enable some actions; changing it requires 1 action.
    6. Due to the domain's cyclic toggling nature:
       - Each object might require a pair of actions to flip between yfm3frbn/a4ha3ujl and jjtnwk3y states.
       - The s0899d5u relations correspond to transfers of state between objects, which require at least one action per missing relation.
    7. Heuristic estimates as sum of estimated fixes:
       - Sum of missing unary predicates corrected by 1 action each (assuming flipping).
       - Missing binary s0899d5u relationships corrected by at least 1 action each.
       - If ug241uf7 is missing but required for action, estimate 1 action to toggle.
    8. To avoid overestimating, use max of counts related to unary predicates and relation fixes plus the toggle of ug241uf7 if needed.
    9. This heuristic is not guaranteed admissible but aims to guide the search efficiently by measuring discrepancies in predicate distributions.

    The heuristic output is always finite for solvable states and zero exactly for goal states.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static  # empty for this domain, not used

        # Sets to hold goal facts by predicate and involved objects
        self.goal_yfm3frbn = set()
        self.goal_a4ha3ujl = set()
        self.goal_jjtnwk3y = set()
        self.goal_s0899d5u = set()

        # From the goal, extract which objects must hold which predicates, and which pairs for s0899d5u
        for fact in self.goals:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "yfm3frbn":
                self.goal_yfm3frbn.add(parts[1])
            elif pred == "a4ha3ujl":
                self.goal_a4ha3ujl.add(parts[1])
            elif pred == "jjtnwk3y":
                self.goal_jjtnwk3y.add(parts[1])
            elif pred == "s0899d5u":
                self.goal_s0899d5u.add((parts[1], parts[2]))

    def __call__(self, node):
        state = node.state

        # Quick check if goal reached
        if self.goals <= state:
            return 0

        # Helper: check presence of unary predicate on object
        def has_pred(pred, obj):
            return f"({pred} {obj})" in state

        # Helper: check presence of binary predicate on pair
        def has_s0899d5u(obj1, obj2):
            return f"(s0899d5u {obj1} {obj2})" in state

        # Check if ug241uf7 is currently true or false
        ug241uf7_in_state = "(ug241uf7)" in state
        ug241uf7_in_goal = any("ug241uf7" in g for g in self.goals) or any(
            "ug241uf7" in op for op in self.goals
        )

        # Count mismatches for unary predicates in goal
        missing_yfm3frbn = sum(1 for obj in self.goal_yfm3frbn if not has_pred("yfm3frbn", obj))
        missing_a4ha3ujl = sum(1 for obj in self.goal_a4ha3ujl if not has_pred("a4ha3ujl", obj))
        missing_jjtnwk3y = sum(1 for obj in self.goal_jjtnwk3y if not has_pred("jjtnwk3y", obj))

        # Count mismatches for binary s0899d5u predicates in goal
        missing_s0899d5u = sum(
            1 for (o1, o2) in self.goal_s0899d5u if not has_s0899d5u(o1, o2)
        )

        # Changing ug241uf7 likely costs one toggle action if required and states differ
        ug241uf7_cost = 0
        # The domain actions manipulate ug241uf7 state (enable/disable) for actions cb5rcmgi and dezigdo8 and others.
        # If goal demands relations that require ug241uf7 true and it's currently false, we add 1 to toggle
        # We assume that toggle of ug241uf7 might be needed if there are missing facts that require it.
        # Heuristically, check if any missing facts require ug241uf7
        # We conservatively add cost if ug241uf7 current and goal differ and some predicates depending on it are off
        # Since we have no direct info which facts require ug241uf7 in goal, we estimate this cost if missing > 0 and ug241uf7 off
        if (missing_yfm3frbn + missing_a4ha3ujl + missing_jjtnwk3y + missing_s0899d5u) > 0:
            if not ug241uf7_in_state:
                ug241uf7_cost = 1

        # Combine heuristic estimates:
        # Each missing unary predicate fixed via 1 or 2 actions (flip states)
        # Each missing s0899d5u requires at least 1 action (to create relation)
        # Use max of unary counts because flipping jjtnwk3y vs yfm3frbn/a4ha3ujl is done on objects
        unary_fix = max(missing_yfm3frbn + missing_a4ha3ujl, missing_jjtnwk3y)

        # Sum up all costs with ug241uf7 toggle cost included
        heuristic_value = unary_fix + missing_s0899d5u + ug241uf7_cost

        # Ensure heuristic is at least 1 if not goal to guide search away from zero heuristic except at goal.
        return max(heuristic_value, 1)
