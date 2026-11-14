from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The fact string, e.g. "(on-table item1 item6)".
    - `args`: pattern elements with wildcards allowed.
    - Returns True if all parts match the pattern.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (blocksworld) domain.

    Summary:
    This heuristic estimates the number of actions needed to achieve the goal
    conditions involving 'arm-empty' predicates on certain pairs of
    (item-3, item-2) objects. As the domain actions affect 'arm-empty', 'on-table',
    'on', 'holding', and 'clear' predicates, the heuristic focuses primarily on
    how many goal arm-empty relations are currently false.

    Assumptions:
    - The goal predicates are conjunctions of 'arm-empty' facts on pairs of objects.
    - The heuristic evaluates how many goal arm-empty pairs are *not* currently achieved.
    - Since the actions redefine the arm-empty relations in a structural way,
      achieving each required 'arm-empty' fact often requires at least one action.
    - The heuristic does not attempt to perfectly estimate the complex manipulations
      between items, but uses the count of missing 'arm-empty' predicates plus
      estimated cost to "fix" arm-empty relations.
    - The static facts 'holding' and 'on' can be used to deduce some dependencies.
    - The heuristic does not attempt to reorder blocks or evaluate exact moves for stacks,
      but counts missing arm-empty facts and estimates minimal work needed.

    Heuristic Initialization:
    - Extract all goal 'arm-empty' predicates as a set.
    - Extract static 'holding' facts as a lookup to identify held items.
    - Extract static 'on' relations to understand which items are on which items.
    - Extract 'on-table' facts from static for additional context.
    - These static relations are mainly for potential further extensions or pruning.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if current state satisfies all goal predicates; if so, return 0.
    2. Count how many 'arm-empty' goal predicates are currently true in the state.
    3. Compute missing 'arm-empty' predicates needed to achieve the goal.
    4. Each missing 'arm-empty' fact generally requires at least one action (pickup, stack, or unstack).
    5. Account for the fact that some actions might fix multiple arm-empty predicates simultaneously:
       - When moving one block away, you fix the arm state of two positions.
    6. We estimate the heuristic as the count of missing goal arm-empty pairs,
       possibly multiplied by a small factor (e.g., 1.5) to be slightly informed
       about extra work (not admissible, just guidance).
    7. Return the heuristic cost as the estimate of required actions to reach goal arm-empty conditions.

    This heuristic remains efficiently computable by just counting facts in the current state and comparing
    to the goal arm-empty facts.
    """

    def __init__(self, task):
        # Store goal arm-empty predicates required
        self.goal_arm_empty = {
            fact
            for fact in task.goals
            if fact.startswith("(arm-empty ")
        }
        # Store static holding facts (item -> item) for potential further reasoning
        self.static_holding = {
            fact
            for fact in task.static
            if fact.startswith("(holding ")
        }
        # Store static on relations for possible insights
        self.static_on = {
            fact
            for fact in task.static
            if fact.startswith("(on ")
        }
        # Store static on-table relations
        self.static_on_table = {
            fact
            for fact in task.static
            if fact.startswith("(on-table ")
        }

    def __call__(self, node):
        """
        Estimate number of actions to achieve goal arm-empty predicates.

        Returns 0 if state is already goal.

        Otherwise, counts how many goal arm-empty predicates are not true in
        current state, then uses that count times a factor (1.5) to reflect
        that some actions may fix multiple arm-empty facts.
        """
        state = node.state

        # Quick check: if goal reached, heuristic is 0
        if self.goal_arm_empty <= state:
            return 0

        # Count how many goal arm-empty predicates hold in current state
        arm_empty_true = sum(1 for fact in self.goal_arm_empty if fact in state)

        # Number of missing arm-empty predicates
        missing = len(self.goal_arm_empty) - arm_empty_true

        # Heuristic estimate:
        # Each missing arm-empty predicate likely needs at least one action to fix.
        # We multiply by 1.5 to roughly estimate overhead from stack/unstack dependencies
        # This is a coarse estimate to guide GBFS.
        heuristic_value = missing * 1.5

        # Return heuristic as integer (number of actions)
        return int(heuristic_value)
