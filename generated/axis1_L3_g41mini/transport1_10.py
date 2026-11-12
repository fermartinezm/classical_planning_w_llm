from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(on-table item1 item6)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (blocksworld variant).

    Summary:
    This heuristic estimates the number of actions required to achieve the goal,
    which consists solely of (arm-empty ?item-4 ?item-2) facts. The heuristic 
    measures how many arm-empty conditions from the goal are currently unsatisfied.
    Since the goal conditions involve the arm being empty for specific (item-4, item-2) pairs,
    the heuristic estimates how soon these conditions can be reached based on the 
    current state's facts, particularly focusing on arm-empty pairs and the locations 
    of items on tables or stacked.

    Assumptions:
    - The overall goal is a conjunction of (arm-empty ?item-4 ?item-2) predicates.
    - Actions affecting arm-empty typically move items between item-2 objects.
    - Each unsatisfied (arm-empty ?i4 ?i2) goal condition likely requires at least one
      action to achieve, possibly more depending on the "on-table" and "on" predicates.
    - The domain uses "on", "on-table", "arm-empty", and "holding" predicates.
    - Only (arm-empty ?item-4 ?item-2) appear as goals (from instance examples).
    - Achieving an (arm-empty ?i4 ?i2) goal requires that the arm is empty on that pair,
      meaning no currently held object blocks it.

    Heuristic Initialization:
    - Extract all goal (arm-empty ?item-4 ?item-2) pairs.
    - Extract static facts relevant for reasoning about object hierarchies:
      "on" and "on-table" relations.
    - Build mappings to quickly check relationships between items.
    - No complex search or relaxed planning graph is constructed for efficiency.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify which arm-empty goal pairs are already satisfied in the current state.
       These require zero cost.
    2. For each unsatisfied arm-empty goal pair:
       - Check if there is a conflict related to the item-4 and item-2 involved.
         For example, if the arm is not empty with the required item-2 and item-4,
         it indicates an object is held or blocking.
       - Count at least one action is needed to free the arm-empty condition:
         - Unstacking an item or picking up an item to free the arm.
         - Stacking or placing back to allow arm to be empty with that pair.
    3. Estimate the minimal number of actions by summing 1 per unsatisfied arm-empty pair,
       giving a simple but informed lower bound.
    4. Optionally, penalize states where the holding predicate indicates something is held
       by an arm that blocks achieving arm-empty conditions.
    5. Return the total count as heuristic: number of unsatisfied arm-empty pairs + penalty
       if something is currently held that blocks achieving arm-empty conditions.

    This heuristic is not admissible but is efficient and reduces expanded nodes by focusing on
    unsatisfied arm-empty goals and current holdings.
    """

    def __init__(self, task):
        # Extract goal arm-empty pairs.
        self.goal_arm_empty = set()
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == "arm-empty":
                # (arm-empty ?item-4 ?item-2)
                self.goal_arm_empty.add((parts[1], parts[2]))

        # Extract static "on"/"on-table" relations to understand stacking (object hierarchy).
        self.on_predicates = set()
        self.on_table_predicates = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "on":
                # (on ?item-2 ?item-2)
                self.on_predicates.add((parts[1], parts[2]))
            elif parts[0] == "on-table":
                # (on-table ?item-4 ?item)
                self.on_table_predicates.add((parts[1], parts[2]))

        # No additional complex precomputation done here for performance.

    def __call__(self, node):
        state = node.state

        # Quickly check which arm-empty goals are satisfied in the current state.
        current_arm_empty = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "arm-empty":
                current_arm_empty.add((parts[1], parts[2]))

        # Count how many goal arm-empty pairs are unsatisfied.
        unsatisfied_arm_empty = self.goal_arm_empty - current_arm_empty

        # Extract holding facts to check if arms hold something.
        # holding ?item ?item
        holding_pairs = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "holding":
                holding_pairs.add((parts[1], parts[2]))

        # If any arm is holding something, that likely blocks arm-empty predicates.
        # Count how many arms hold something that are relevant to goal arm-empty pairs.
        # Penalty: +1 for each holding relation that blocks an arm-empty goal currently unsatisfied.
        # We approximate blocking if the arm (item-3) holding something relates to any goal arm-empty ?item-4.
        # Since item types are complicated with hierarchies, use a heuristic assumption:
        # If an item-4 in goal arm-empty matches the arm part in arm-empty, and currently holding something,
        # heuristic cost increases (because arm must be freed to achieve arm-empty).

        # Collect all arm parts of goal_arm_empty (item-4)
        goal_arms = {arm for arm, block in self.goal_arm_empty}

        penalty = 0
        # We do not have explicit arm identifiers separate from item-4/item-2 in holding,
        # but from domain, holding is (holding ?item ?item), no arm involved explicitly.
        # The arm-empty predicate is (arm-empty ?item-4 ?item-2).
        # From domain and examples, arm-empty facts have two parameters:
        # ?item-4 (first param) and ?item-2 (second param).
        # From holding facts we have (holding ?v5 ?v6) both items.
        # We know from domain definition:
        # holding ?v5 ?v6 - item
        # arm-empty ?v3 - item-3 ?v4 - item-2
        # So holding facts relate to items; arms are item-3 type.
        # We don't have direct mapping arms holding items in holding facts, so instead:
        # If any holding exists, assume it blocks all arm-empty pairs currently unsatisfied.
        # We count penalty as count of unique holding facts.

        if holding_pairs:
            # Penalize proportional to how many holding facts exist, multiplied by
            # the number of unsatisfied arm-empty goals to reflect blocking complexity.
            penalty = len(holding_pairs) * len(unsatisfied_arm_empty)

        # The heuristic cost is at least the number of unsatisfied arm-empty pairs,
        # plus additional penalty for currently held items blocking the arm-empty conditions.
        heuristic_value = len(unsatisfied_arm_empty) + penalty

        # The heuristic is zero if and only if goal reached (no unsatisfied arm-empty).
        # This corresponds with heuristic 0 on goal states.

        return heuristic_value
