from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string (e.g., "(yfm3frbn b2)").
    - `args`: The pattern to match, may contain wildcards '*' (e.g. "yfm3frbn", "*").
    - Returns True if the fact matches the pattern, else False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain.

    Summary:
    This heuristic estimates the number of actions required to reach the goal
    state by analyzing the presence and configuration of key predicates in the current state
    relative to the goal conditions. The domain models transformations of objects through
    toggling of predicates `yfm3frbn`, `a4ha3ujl`, `jjtnwk3y`, and relations `s0899d5u`
    between objects, with a global flag `ug241uf7` influencing action applicability.

    Assumptions:
    - Objects cycle through states defined by predicates `yfm3frbn`, `a4ha3ujl`, and `jjtnwk3y`.
    - The predicate `s0899d5u` defines a kind of one-way link or relation between objects.
    - `ug241uf7` is a global boolean that gates some actions.
    - Goal predicates involve a conjunction of specific properties on objects and relations between them.

    Heuristic Initialization:
    - Extract the goal conditions and organize them by predicate.
    - Store the set of objects involved in each goal predicate.
    - The heuristic uses no static facts as `static` is empty.
    - Prepare utility structures to quickly check goal requirements for objects and pairs.

    Step-By-Step Thinking for Computing Heuristic:
    1. If the current state already satisfies all goal predicates, the heuristic is 0.

    2. Extract, from the goal, sets of objects required to:
       - Have predicate `yfm3frbn`
       - Have predicate `a4ha3ujl`
       - Have predicate `jjtnwk3y`
       - Have pairs `(obj1,obj2)` in relation `s0899d5u`

    3. Determine the current sets of objects (and pairs) that satisfy these predicates in the state.

    4. Calculate:
       - How many objects must change their state for `yfm3frbn`, `a4ha3ujl`, and `jjtnwk3y` to match the goal.
       - How many pairs for `s0899d5u` must be established or broken.

    5. Use domain knowledge of the actions:
       - The action `cb5rcmgi` toggles `yfm3frbn`, `a4ha3ujl`, and sets `jjtnwk3y` on the same object, consuming `ug241uf7`.
       - The action `dezigdo8` toggles those back.
       - The pair relation `s0899d5u` is created and removed by `ig1fewq3` and `i57yke4t` actions on pairs of objects toggling `yfm3frbn`, `jjtnwk3y`, and `ug241uf7`.
       - Changing one object's state typically requires at least one action.
       - Creating or removing a link requires at least one action.

    6. Since `ug241uf7` is a global toggle, and many actions consume or reset it,
       count toggles to set it appropriately.

    7. The heuristic sums up:
       - Approximate action costs to fix predicate mismatches on single objects.
       - Plus action costs to fix relation pairs `s0899d5u`.
       - Plus estimated costs for `ug241uf7` toggling to enable these actions.

    8. The heuristic is not guaranteed to be admissible but aims to guide search efficiently.

    """

    def __init__(self, task):
        self.goals = task.goals  # goal facts as strings
        self.static = task.static

        # Parse goal predicates into categorized structures for quick reference
        self.goal_yfm3frbn = set()
        self.goal_a4ha3ujl = set()
        self.goal_jjtnwk3y = set()
        self.goal_s0899d5u = set()  # Set of (obj1, obj2)

        for fact in self.goals:
            parts = get_parts(fact)
            p = parts[0]
            if p == "yfm3frbn":
                self.goal_yfm3frbn.add(parts[1])
            elif p == "a4ha3ujl":
                self.goal_a4ha3ujl.add(parts[1])
            elif p == "jjtnwk3y":
                self.goal_jjtnwk3y.add(parts[1])
            elif p == "s0899d5u":
                self.goal_s0899d5u.add((parts[1], parts[2]))

    def __call__(self, node):
        state = node.state

        # If goal already reached, heuristic is zero
        if self.goals <= state:
            return 0

        # Extract current objects for each predicate
        current_yfm3frbn = set()
        current_a4ha3ujl = set()
        current_jjtnwk3y = set()
        current_s0899d5u = set()
        ug241uf7_present = "(ug241uf7)" in state

        for fact in state:
            parts = get_parts(fact)
            p = parts[0]
            if p == "yfm3frbn":
                current_yfm3frbn.add(parts[1])
            elif p == "a4ha3ujl":
                current_a4ha3ujl.add(parts[1])
            elif p == "jjtnwk3y":
                current_jjtnwk3y.add(parts[1])
            elif p == "s0899d5u":
                current_s0899d5u.add((parts[1], parts[2]))

        # Count mismatches for singleton predicates
        yfm3frbn_to_add = self.goal_yfm3frbn - current_yfm3frbn
        yfm3frbn_to_remove = current_yfm3frbn - self.goal_yfm3frbn

        a4ha3ujl_to_add = self.goal_a4ha3ujl - current_a4ha3ujl
        a4ha3ujl_to_remove = current_a4ha3ujl - self.goal_a4ha3ujl

        jjtnwk3y_to_add = self.goal_jjtnwk3y - current_jjtnwk3y
        jjtnwk3y_to_remove = current_jjtnwk3y - self.goal_jjtnwk3y

        # Count mismatches for pair predicate
        s0899d5u_to_add = self.goal_s0899d5u - current_s0899d5u
        s0899d5u_to_remove = current_s0899d5u - self.goal_s0899d5u

        # Heuristic components estimation:

        # 1. Fixing singleton predicates on objects:
        #
        # The domain actions `cb5rcmgi` and `dezigdo8` toggle the triple of predicates (yfm3frbn, a4ha3ujl, jjtnwk3y) on *one* object
        # but require ug241uf7 to be present / consumed.
        #
        # Changing an object to have these predicates matching the goal requires at least 1 action per object in mismatch.
        #
        # Since all three predicates flip together in the actions, we approximate by counting the number of objects that are mismatched in any predicate of the triple.
        # So union all objects that mismatch any of the three singleton predicates.

        # Objects that do not match goal triple:
        objs_goal = set().union(self.goal_yfm3frbn, self.goal_a4ha3ujl, self.goal_jjtnwk3y)
        objs_current = set().union(current_yfm3frbn, current_a4ha3ujl, current_jjtnwk3y)
        objects_to_fix = set()

        # Consider all objects appearing in either current or goal relevant sets
        for obj in objs_goal.union(objs_current):
            # For each of the three predicates check if they match goal
            match_yfm3frbn = (obj in self.goal_yfm3frbn) == (obj in current_yfm3frbn)
            match_a4ha3ujl = (obj in self.goal_a4ha3ujl) == (obj in current_a4ha3ujl)
            match_jjtnwk3y = (obj in self.goal_jjtnwk3y) == (obj in current_jjtnwk3y)
            # Because these predicates are tightly coupled by the actions, we consider an object mismatched if *any* of these predicates mismatch for it.
            if not (match_yfm3frbn and match_a4ha3ujl and match_jjtnwk3y):
                objects_to_fix.add(obj)

        # 2. Fixing pairs in s0899d5u predicate:

        # The actions ig1fewq3 and i57yke4t can be used to create or remove pairs s0899d5u between two objects,
        # and toggle yfm3frbn and jjtnwk3y on those objects while toggling ug241uf7 globally.
        #
        # We approximate the cost to fix pairs as at least 1 action per pair that differs from goal.
        pair_changes = len(s0899d5u_to_add) + len(s0899d5u_to_remove)

        # 3. ug241uf7 toggling cost:

        # The global predicate ug241uf7 is required for most actions changing predicates or pairs.
        # ug241uf7 is created by `dezigdo8` and consumed by `cb5rcmgi` and similarly by `ig1fewq3` and `i57yke4t`.
        # Sometimes the heuristic may require toggling ug241uf7 if not present when needed.

        # Rough estimate: each object fix or pair change requires ug241uf7 to be set before the action.
        # If ug241uf7 is currently false, we assume at least one extra action to set it before starting fixes.

        ug241uf7_toggle_cost = 0
        if not ug241uf7_present and (len(objects_to_fix) > 0 or pair_changes > 0):
            # We pay one action cost to get ug241uf7 set before operations.
            # This is a rough approximation since ug241uf7 is mutually exclusive (consumed or created).
            ug241uf7_toggle_cost = 1

        # Final heuristic estimate:
        # Each object fix requires 1 action (one toggle action).
        # Each pair fix requires 1 action.
        # Add potential cost for ug241uf7 toggling.

        cost = len(objects_to_fix) + pair_changes + ug241uf7_toggle_cost

        # Heuristic is at least 1 if we have any mismatch, never zero here unless goal.
        if cost == 0:
            # Defensive fallback: if no mismatch found but goal not reached, return 1.
            return 1

        return cost
