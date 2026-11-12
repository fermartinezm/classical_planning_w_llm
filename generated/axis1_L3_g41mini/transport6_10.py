from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    Args:
        fact (str): A fact string, e.g., "(on item3 item4)"
        args (str): Pattern parts with wildcard (*) support.

    Returns:
        bool: True if fact matches the pattern, else False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))



class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (blocksworld variant).

    Summary:
    This heuristic estimates the minimum number of actions required to achieve the goal by focusing on
    the "arm-empty" goal conditions as indicators of required manipulations. Since the domain involves
    stacks of items and arms that must be emptied, the heuristic counts how many goal "arm-empty" facts
    are currently not satisfied in the given state and estimates the cost to reach them. The heuristic
    accounts for which items need to be manipulated and how many moves must be performed based on 
    the current "arm-empty" and "holding" facts as well as positions of items on the table.

    Assumptions:
    - The primary goal is reaching certain "arm-empty" facts.
    - Items involved in goal arm-empty facts may currently hold other items or be obstructed.
    - Each "arm-empty" goal fact not yet true likely requires at least one action to fix.
    - A simple linear estimate based on differences in arm states and items on table approximates action cost.
    - The domain's complex predicates and typing imply some indirect relations; we rely on observed facts only.
    - We do not need full admissibility; heuristic may overestimate but should efficiently guide search.

    Heuristic Initialization:
    - Extract all goal "arm-empty" facts and store them as a set of (arm, item) tuples.
    - Extract from the static facts the "holding" facts which describe items being held and never change.
    - No complex structural knowledge inferred beyond static "holding" facts and goal arm-empty facts.

    Step-By-Step Thinking for Computing Heuristic:
    1. Extract the current "arm-empty" facts from the given state.
    2. Identify how many goal "arm-empty" facts are still unsatisfied.
       Each unsatisfied is assumed to require at least one action.
    3. Extract all current "holding" facts from both static and current state.
       Holding an item may indicate that arm is not empty, increasing estimated effort.
    4. Count how many items are currently "on-table" in the current state but need repositioning for goal arms.
       Every misplaced item on the table is assumed to require at least one action.
    5. The heuristic value is roughly:
       number of missing goal arm-empty facts +
       number of currently held items (which block arm-empty states) +
       number of misplaced "on-table" items relevant to these arms.
    6. This sum is a rough estimate of steps needed to achieve all goal arm-empty facts.
    7. The heuristic returns 0 if the current state meets all goal arm-empty conditions (goal state),
       ensuring heuristic is 0 for goal states and finite otherwise.

    This heuristic leverages domain structure, focusing on arm and item configurations central to the domain's goal.
    """

    def __init__(self, task):
        # Store goal arm-empty facts as (arm, item) tuples
        self.goal_arm_empty = set()
        for fact in task.goals:
            if match(fact, "arm-empty", "*", "*"):
                parts = get_parts(fact)
                # fact format: (arm-empty arm item)
                self.goal_arm_empty.add((parts[1], parts[2]))

        # Extract static holding facts (which never change)
        self.static_holding = set()
        for fact in task.static:
            if match(fact, "holding", "*", "*"):
                parts = get_parts(fact)
                self.static_holding.add((parts[1], parts[2]))

    def __call__(self, node):
        state = node.state

        # Current arm-empty facts as set of (arm, item)
        current_arm_empty = set()
        for fact in state:
            if match(fact, "arm-empty", "*", "*"):
                parts = get_parts(fact)
                current_arm_empty.add((parts[1], parts[2]))

        # Count how many goal arm-empty facts are not satisfied
        missing_arm_empty = len(self.goal_arm_empty - current_arm_empty)

        if missing_arm_empty == 0:
            # All goal arm-empty facts are satisfied - heuristic is zero at goal
            return 0

        # Count currently held items: combine static holding facts and current holding facts in state
        # Holding may indicate items blocking arm-empty states
        current_holding = set()
        for fact in state:
            if match(fact, "holding", "*", "*"):
                parts = get_parts(fact)
                current_holding.add((parts[1], parts[2]))
        # Union current holding and static holding to capture all held items
        held_items = current_holding | self.static_holding
        num_held_items = len(held_items)

        # Identify on-table facts relevant to arms involved in goal arm-empty,
        # counting items that are on-table but not consistent with arm-empty states
        # For heuristic simplicity, count the number of on-table facts in the state
        # This approximates the necessary moves of items to clear arms.
        # (Since arms are typed as item-4 and item-5, we consider all on-table facts.)
        on_table_items = set()
        for fact in state:
            if match(fact, "on-table", "*", "*"):
                parts = get_parts(fact)
                on_table_items.add((parts[1], parts[2]))
        num_on_table = len(on_table_items)

        # Simple weighted sum heuristic estimate:
        # - each missing arm-empty fact: 1 action to correct
        # - each held item: 1-2 actions to place down/release
        # - each on-table item: 1 action to possibly rearrange
        # To keep heuristic optimistic enough, weight missing arm-empty by 2,
        # held items by 1, and on-table items by 1.
        heuristic_value = 2 * missing_arm_empty + num_held_items + num_on_table

        return heuristic_value
