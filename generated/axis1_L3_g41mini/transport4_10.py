from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern with optional wildcards.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (blocksworld variant).

    Summary:
    This heuristic estimates the number of actions required to reach the goal,
    where the goal is solely specified by (arm-empty ?item-4 ?item-2) facts.
    The domain actions revolve around managing the "arm-empty" predicate and
    the locations of items (on, on-table, holding). 

    The heuristic focuses on how many arm-empty conditions specified by the goal
    are currently unsatisfied and estimates the minimal number of unstack,
    pickup, or stack operations required to achieve them.

    Assumptions:
    - Each goal fact is always of the form (arm-empty ?item-4 ?item-2).
    - Achieving each arm-empty goal fact mostly requires making sure the arm is empty on the specified parameters.
    - The heuristic assumes each unsatisfied arm-empty goal corresponds to roughly one action, possibly more if the arm is currently holding something.
    - The domain's unusual typing and predicates are considered via pattern matching the facts.
    - The heuristic does not assume to be admissible but aims to quickly guide greedy best-first search.

    Heuristic Initialization:
    - Extract all arm-empty goal facts.
    - Extract static "holding" facts (holding relations never change) for quick lookup.
    - Store goal arm-empty pairs as tuples for efficient checking.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify which arm-empty goals are already satisfied in the current state.
    2. For each unsatisfied arm-empty goal (arm-empty x y), check if the arm is currently holding something (using static "holding" facts):
       - If the arm referred to in the goal is currently holding something, at least one action is needed to free it (unstack or pickup).
       - If arm is not holding anything but the arm-empty is still unsatisfied, it likely represents a short transitional state.
    3. Sum one action per unsatisfied arm-empty goal to estimate minimum required actions.
    4. Additionally, count how many "on" or "on-table" facts contradict the postcondition of arm-empty, i.e., if any item occupies the arm or interferes with it, increment the cost slightly.
    5. Return 0 if all goals are satisfied.
    """

    def __init__(self, task):
        """
        Initialize the heuristic.

        - Extract arm-empty goal facts as tuples (item4, item2).
        - Extract static holding facts as a set of (item, item) pairs.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract goal arm-empty pairs: (item-4, item-2)
        self.arm_empty_goals = set()
        for fact in self.goals:
            if fact.startswith("(arm-empty "):
                parts = get_parts(fact)
                if len(parts) == 3:
                    _, item4, item2 = parts
                    self.arm_empty_goals.add((item4, item2))

        # Extract static holding pairs (item, item) for quick test of holding relation
        self.holding_pairs = set()
        for fact in self.static:
            if fact.startswith("(holding "):
                parts = get_parts(fact)
                if len(parts) == 3:
                    _, item1, item2 = parts
                    self.holding_pairs.add((item1, item2))

    def __call__(self, node):
        """
        Estimate the number of actions required to reach the goal from the current state.

        The heuristic counts how many arm-empty goals are currently not satisfied, 
        assuming at least one action per unsatisfied arm-empty fact is necessary,
        and factors in if the arm is currently holding something (indicating extra work).
        """
        state = node.state

        # Current arm-empty facts present
        current_arm_empty = set()
        for fact in state:
            if fact.startswith("(arm-empty "):
                parts = get_parts(fact)
                if len(parts) == 3:
                    _, item4, item2 = parts
                    current_arm_empty.add((item4, item2))

        # If all arm-empty goal facts are satisfied => heuristic = 0
        if self.arm_empty_goals <= current_arm_empty:
            return 0

        # Count unsatisfied arm-empty goals and cost estimate
        unsatisfied_goals = self.arm_empty_goals - current_arm_empty

        cost = 0
        for item4, item2 in unsatisfied_goals:
            # Check if the arm (item4) is holding something currently
            # Meaning: is there a holding fact involving item4 (either as holder or held?)
            # Because from domain, holding is (holding ?v5 ?v6 - item)
            # We try to check if item4 is holding or held (item4 might be ?v5 or ?v6)
            # More precisely, from domain signature arm-empty is (arm-empty ?v3 - item-3 ?v4 - item-2)
            # But holding involves item and item, types may differ; anyway check for holding pairs involving item4

            holding_involved = False
            # Check holding facts where arm-empty's first param or second param matches any item in holding
            # Because arm-empty ?v3 ?v4 where ?v3 - item-3 and ?v4 - item-2,
            # from examples arm-empty itemX itemY means arm on something or empty on something?
            # We conservatively check if item4 or item2 appears in any holding pair to guess holding.
            for h1, h2 in self.holding_pairs:
                if item4 == h1 or item4 == h2 or item2 == h1 or item2 == h2:
                    holding_involved = True
                    break

            # If arm is holding something => likely an unstack or pickup is needed to free it
            if holding_involved:
                cost += 2  # count as 2 actions for freeing and restacking
            else:
                cost += 1  # likely only one action required

        # Additional rough estimate:
        # Count how many clear predicates contradict arm-empty goals, i.e. stacks that block arm-empty state.
        # From domain, "clear" predicate exists, but is typed confusingly.
        # Since arm-empty expresses the arm/tool state, 
        # We count how many "on" and "on-table" facts involve the same item-2 objects as in arm-empty goals,
        # as blocking those positions might require clearing actions.
        blocking_count = 0
        arm_empty_item2s = {item2 for item4, item2 in unsatisfied_goals}

        for fact in state:
            if fact.startswith("(on "):
                parts = get_parts(fact)
                if len(parts) == 3:
                    _, top, bottom = parts
                    if bottom in arm_empty_item2s:
                        blocking_count += 1
            elif fact.startswith("(on-table "):
                parts = get_parts(fact)
                if len(parts) >= 3:
                    # Format: (on-table ?item4 ?item)
                    _, item4_ft, item_ft = parts
                    if item4_ft in arm_empty_item2s or item_ft in arm_empty_item2s:
                        blocking_count += 1

        # Weight blocking count less heavily
        cost += blocking_count // 2  # half the blocking count as approximate extra cost

        return cost if cost > 0 else 1  # heuristic should never be zero unless at goal state

