from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting the string.
    """
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern with wildcards '*' allowed.
    
    :param fact: The fact string, e.g., "(predicate arg1 arg2)"
    :param args: Pattern parts to match against the fact parts.
    :return: True if all parts match, else False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the `secret_domain`.

    Summary
    -------
    This heuristic estimates the number of actions required to reach the goal from a given state in the secret_domain,
    which revolves around manipulating predicates `(yfm3frbn ?x)`, `(a4ha3ujl ?x)`, `(jjtnwk3y ?x)`, and `(s0899d5u ?x ?y)`,
    alongside the global boolean flag `(ug241uf7)`. Actions mainly switch objects between states characterized by `yfm3frbn` and
    `jjtnwk3y` predicates, and connect these objects via `s0899d5u`.

    Assumptions
    -----------
    - Each object can be in one of two mutually exclusive states: `yfm3frbn`+`a4ha3ujl` or `jjtnwk3y`.
    - The global predicate `ug241uf7` acts like a toggle facilitating certain transitions.
    - Reaching the goal involves having particular objects satisfy `yfm3frbn`, `a4ha3ujl` and `s0899d5u` relations.
    - The `s0899d5u` predicate forms links between objects and must be arranged as in the goal.
    - All goal predicates must be simultaneously satisfied.
    - Actions usually involve flipping an object's state and rearranging connections in at most 3-4 steps per predicate discrepancy.
    - Heuristic need not be admissible but should efficiently reflect effort to fix differences in object states and links.

    Heuristic Initialization
    ------------------------
    The constructor extracts:
    - The set of goal predicates (facts).
    - The sets of objects mentioned in the goal for predicates `yfm3frbn`, `a4ha3ujl` and links `s0899d5u`.
    - No static facts are used since the domain static set is empty.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a state, we estimate the number of needed actions by comparing the current state's facts with the goal facts:
    
    1. Check if the current state already satisfies the goal; if yes, heuristic = 0.
    2. Extract all objects currently having predicates `(yfm3frbn ?o)`, `(a4ha3ujl ?o)`, `(jjtnwk3y ?o)` and current `(s0899d5u ?o1 ?o2)` links.
       We call these the current "states" of objects and current links.
    3. Determine for each goal object whether it currently has the right state predicates:
       - For every object that should have `(yfm3frbn ?o)` in goal but does not, count 1 needed fix.
       - For every object that should have `(a4ha3ujl ?o)` in goal but does not, count 1 needed fix.
       - For every object that should have `(jjtnwk3y ?o)` in goal but does not, count 1 needed fix.
       Typically, an object is either `(yfm3frbn & a4ha3ujl)` or `(jjtnwk3y)`.
       Flipping an object state requires at least one action (`cb5rcmgi` or `dezigdo8`) but sometimes re-linking involved.
    4. Compare the links: for each `(s0899d5u ?o1 ?o2)` in the goal but missing in state, count 1 fix.
       Similarly, links present in state but not in goal should be removed; count 1 per such extra link.
    5. Since the `ug241uf7` predicate toggles enabling actions, count how many flips of this predicate might be required
       roughly proportional to the total number of state changes and links edits (we approximate).
    6. Combine counts of required state fixes and link fixes.
    7. Multiply by a small multiplier (~2) to estimate multiple action steps needed to fix these issues,
       considering domain actions often remove facts before adding new ones.
    8. Return the computed heuristic as a finite integer estimate.

    This heuristic trades strict admissibility for reasonable accuracy and efficiency by focusing on predicate mismatches
    between current and goal states, accounting for state toggles and link adjustments.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static  # Unused for this domain.

        # Extract goal objects by predicate for fast reference.
        self.goal_yfm3frbn = set()
        self.goal_a4ha3ujl = set()
        self.goal_jjtnwk3y = set()
        self.goal_links = set()  # (obj1, obj2)

        for fact in self.goals:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "yfm3frbn":
                self.goal_yfm3frbn.add(parts[1])
            elif pred == "a4ha3ujl":
                self.goal_a4ha3ujl.add(parts[1])
            elif pred == "jjtnwk3y":
                self.goal_jjtnwk3y.add(parts[1])
            elif pred == "s0899d5u" and len(parts) == 3:
                self.goal_links.add((parts[1], parts[2]))

    def __call__(self, node):
        state = node.state

        # Shortcut: if goal reached, heuristic is 0.
        if self.goals <= state:
            return 0

        # Extract current predicates.
        current_yfm3frbn = set()
        current_a4ha3ujl = set()
        current_jjtnwk3y = set()
        current_links = set()
        ug241uf7_present = False

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "yfm3frbn":
                current_yfm3frbn.add(parts[1])
            elif pred == "a4ha3ujl":
                current_a4ha3ujl.add(parts[1])
            elif pred == "jjtnwk3y":
                current_jjtnwk3y.add(parts[1])
            elif pred == "s0899d5u" and len(parts) == 3:
                current_links.add((parts[1], parts[2]))
            elif pred == "ug241uf7":
                ug241uf7_present = True

        # Calculate mismatch counts.
        # Objects missing required predicates:
        missing_yfm3frbn = self.goal_yfm3frbn - current_yfm3frbn
        missing_a4ha3ujl = self.goal_a4ha3ujl - current_a4ha3ujl
        missing_jjtnwk3y = self.goal_jjtnwk3y - current_jjtnwk3y

        # Objects with predicates not in goal (to be flipped back):
        extra_yfm3frbn = current_yfm3frbn - self.goal_yfm3frbn
        extra_a4ha3ujl = current_a4ha3ujl - self.goal_a4ha3ujl
        extra_jjtnwk3y = current_jjtnwk3y - self.goal_jjtnwk3y

        # Links differences:
        missing_links = self.goal_links - current_links
        extra_links = current_links - self.goal_links

        # Count total predicate fixes needed for objects.
        # Each misaligned predicate likely requires at least one action to fix.
        state_mismatches = (
            len(missing_yfm3frbn)
            + len(missing_a4ha3ujl)
            + len(missing_jjtnwk3y)
            + len(extra_yfm3frbn)
            + len(extra_a4ha3ujl)
            + len(extra_jjtnwk3y)
        )

        # Count link fixes needed.
        link_mismatches = len(missing_links) + len(extra_links)

        # Estimating ug241uf7 flips roughly proportional to number of state changes,
        # since actions alternate this predicate usage.
        ug_flip_estimate = max(1, (state_mismatches + link_mismatches) // 2)

        # Final heuristic estimate:
        # Each predicate fix might require about 2 actions,
        # each link fix about 2 actions,
        # plus an estimated number of ug241uf7 toggles.
        heuristic_value = 2 * state_mismatches + 2 * link_mismatches + ug_flip_estimate

        # Enforce minimum heuristic of 1 if not goal.
        return max(1, heuristic_value)
