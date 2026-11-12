from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by stripping parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a fact matches a given pattern with possible wildcards.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain ("blocksworld") domain.

    Summary
    -------
    Estimates the number of required actions to reach a goal state by approximating how many 
    "arm-empty" facts need to be achieved and how many "on" relations must be re-arranged.

    Assumptions
    -----------
    - Goal conditions target some "arm-empty" facts: the heuristic is 0 if all goal arm-empty
      facts are present in the current state, otherwise > 0.
    - The domain features actions pickup, stack, and unstack, manipulating arm emptiness and 
      on/on-table relations.
    - Each action fixes or achieves one of these conditions.
    - Static facts include stable "holding" and "on" relations which inform needed rearrangements.
    - Clear and holding predicates are less explicitly used in preconditions, so heuristic focuses 
      mainly on "arm-empty" and "on"/"on-table" predicates.
    - The heuristic is not necessarily admissible but should guide search efficiently.

    Heuristic Initialization
    -------------------------
    - Collect goal arm-empty facts, to track which must be true for the goal.
    - Extract static "holding" facts (items held by something) — these indicate relations not changed.
    - Extract all "on" and "on-table" static relations encoding the block stacking structure.
    - Identify all objects involved in relevant predicates.
    - Build mappings to quickly check if items are currently placed as required by the goal.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Check if all goal arm-empty facts hold in the current state — if yes, return heuristic 0.
    2. Count how many goal arm-empty facts are missing in the current state. Each missing 
       arm-empty fact typically requires at least one action to achieve (pickup/unstack or stack).
    3. Compare the current "on" and "on-table" relations with the static ones to identify misplaced blocks.
       - Each misplaced block stack relation potentially indicates at least one action needed to unstack or stack.
    4. Since the heuristic need not be admissible, approximate the cost as the sum of:
       - Number of missing arm-empty facts (actions to empty arms or pick/place blocks).
       - Number of incorrect "on"/"on-table" relations compared to statics (disrupted stacks).
    5. Return that sum as the heuristic estimate.

    This heuristic thus balances between arm manipulations and block rearrangements to guide 
    greedy best-first search efficiently.
    """

    def __init__(self, task):
        self.goals = task.goals  # The goal facts (including arm-empty)
        self.static = task.static  # Static facts (including holding and on relations)

        # Extract goal arm-empty facts (each is like "(arm-empty ?v3 ?v4)")
        self.goal_arm_empty = {
            fact for fact in self.goals if fact.startswith("(arm-empty ")
        }

        # Extract static "on" and "on-table" relations
        self.static_on = {
            fact for fact in self.static if fact.startswith("(on ")
        }
        self.static_on_table = {
            fact for fact in self.static if fact.startswith("(on-table ")
        }

    def __call__(self, node):
        state = node.state

        # Check if all goal arm-empty facts are in state (goal reached)
        if self.goal_arm_empty.issubset(state):
            return 0

        # Count missing arm-empty facts from goal
        missing_arm_empty = sum(1 for fact in self.goal_arm_empty if fact not in state)

        # Count how many static "on" relations are missing or incorrect in current state
        # i.e., count how many "on" facts in static are not currently true
        misplaced_on = sum(1 for fact in self.static_on if fact not in state)

        # Similarly check for "on-table" facts that should hold but do not
        misplaced_on_table = sum(1 for fact in self.static_on_table if fact not in state)

        # Heuristic estimate: sum of missing arm-empty plus misplaced relations
        # This estimates number of actions needed to fix arms plus rearrange blocks
        heuristic_value = missing_arm_empty + misplaced_on + misplaced_on_table

        # Ensure heuristic is at least 1 if not zero, to not get stuck
        if heuristic_value == 0:
            heuristic_value = 1

        return heuristic_value
