from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(on item3 item4)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (blocksworld) domain.

    Summary
    -------
    This heuristic estimates the number of actions required to achieve a goal state,
    where the goal conditions are expressed as conjunctions of (arm-empty ?item3 ?item-2) predicates.
    The domain models blocks on tables with conditions about the arm's emptiness and holding states.

    Assumptions
    -----------
    - The primary goal is to ensure that certain arms (typed item-3 and item-2 pairs) are empty.
    - The state holds facts about blocks being on tables, arms being empty relative to certain items, and holding facts.
    - The heuristic does NOT have to be admissible but should be efficient and guide search well.
    - We assume each goal (arm-empty X Y) can be achieved either by unstacking or stacking operations that rearrange blocks.
    - The main cost contributor is how many arms must be emptied and whether the arm is currently holding something.
    - Holding facts and arm-empty facts are key in estimating progress.

    Heuristic Initialization
    ------------------------
    - Extract all arm-empty goal facts into a set for quick membership tests.
    - Store static relations about which items are "on" others, and which arms hold which objects,
      to help estimate how many moves/actions are needed to free the arms.
    - No extensive precomputation is done beyond capturing the goal arm-empty pairs and related static facts.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the current state satisfies all goal arm-empty facts, heuristic is 0.

    2. Identify all goal arm-empty pairs (arm-empty X Y) that are not currently true in the state.

    3. For each such missing (arm-empty X Y) goal:
       - Check if arm X is currently holding any object.
         * If yes, estimate at least 1 action needed to "put down" or stack/unstack to get empty.
       - If arm X is currently not empty, we count 1 per such arm.

    4. Additionally, estimate how many "stack" or "unstack" actions are likely required to restore arm emptiness conditions:
       - For each arm that should be empty, count how many blocks are on or above that arm's item that prevent emptiness.
       - These blocks will require unstack or stack actions, roughly one action per blocking block.

    5. Sum these rough counts to produce a heuristic cost estimate.

    6. Return the total as the heuristic value.

    This heuristic thus balances:
     - The number of arms that are not empty as required by the goal.
     - The number of holding facts blocking empty arms.
     - The number of blocks currently stacked in ways that contradict the arm-empty goal conditions.

    While not guaranteed admissible, the heuristic aims to efficiently guide search by estimating
    the minimal required actions related to emptiness of arms and the rearrangement of blocks.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal arm-empty facts.
        - Static facts indicating relations 'holding' and 'on' to estimate blocking.
        """
        self.goals = task.goals  # The goal facts.

        # Extract all goal arm-empty facts, format: (arm-empty ?item3 ?item-2)
        self.goal_arm_empty = {
            fact for fact in self.goals if match(fact, "arm-empty", "*", "*")
        }

        # Store static facts for quick reference (holding and on)
        self.static_holding = set()
        self.static_on = set()

        for fact in task.static:
            if match(fact, "holding", "*", "*"):
                self.static_holding.add(fact)
            elif match(fact, "on", "*", "*"):
                self.static_on.add(fact)

    def __call__(self, node):
        """
        Compute heuristic estimate of the number of actions to reach a goal from the given state.

        Parameters
        ----------
        node: search node containing current state as a frozenset of facts.

        Returns
        -------
        int: Estimated number of actions needed to reach a goal state from this state.
        """

        state = node.state

        # If goal reached, heuristic is zero.
        if self.goals <= state:
            return 0

        # Find goal arm-empty facts missing in the current state.
        missing_arm_empty = self.goal_arm_empty - state
        if not missing_arm_empty:
            # If no missing arm-empty goals but other goals present, heuristic 0 suffices here.
            return 0

        # Parse holding facts in current state:
        # Map from arm (?item-3) to what it is holding (?item)
        arm_holding = {}
        for fact in state:
            if match(fact, "holding", "*", "*"):
                parts = get_parts(fact)
                # (holding ?item ?item)
                # According to domain: (holding ?v5 ?v6 - item)
                # We interpret arm as the first argument ?v5? or second? Actually uncertain.
                # The predicate is (holding ?v5 ?v6 - item)
                # From example, we see (holding item5 item6), both items. From domain, holding is defined on ("holding ?v5 ?v6 - item").
                # The arm seems to be the ?v5 item holding ?v6 item.
                # but domain examples do not indicate arm in holding, but items.
                # Since "arm-empty ?v3 - item-3 ?v4 - item-2" and holding "?v5 ?v6 - item"
                # likely the arm corresponds to some item-3, holding some item.
                # To relate holding to arms, we guess arms are the first argument of arm-empty (item-3), and holding maps ?v5 to ?v6.
                # We'll assume that the first parameter in arm-empty (?v3) corresponds to an item that could hold an item in a holding fact.
                # So to link holding to arm-empty, treat the first argument of holding as the arm.

                arm = get_parts(fact)[1]  # first arg of holding
                held = get_parts(fact)[2]  # second arg
                arm_holding[arm] = held

        # Parse on-table facts in current state: items on tables of ?item-4 on ?item
        # and "on" facts (from static) representing items on other items.
        # To estimate blocking stacks, gather for each item-2 (second argument in arm-empty) 
        # which item-4 or item-5/3 are on it or above.

        # We create a map item->set of items on it to approximate stacks.
        on_map = {}
        # Add facts from current state that are (on ?v1 ?v2 - item-2)
        for fact in state:
            if match(fact, "on", "*", "*"):
                parts = get_parts(fact)
                upper, lower = parts[1], parts[2]
                on_map.setdefault(lower, set()).add(upper)

        # Add static "on" facts (these represent the known static stacking relations).
        # Usually static facts might be invariant stacking relations, but in this domain "on" is only static
        # if it appears in task.static - it's possible, so we merge both.
        for fact in self.static_on:
            parts = get_parts(fact)
            upper, lower = parts[1], parts[2]
            on_map.setdefault(lower, set()).add(upper)

        # We also consider "on-table" facts that relate item-4 on item:
        # from the domain: (on-table ?v4 ?v5 - item)
        # We'll consider those as bases of stacks.

        on_table_map = {}
        for fact in state:
            if match(fact, "on-table", "*", "*"):
                parts = get_parts(fact)
                block, table = parts[1], parts[2]
                on_table_map.setdefault(table, set()).add(block)

        # The goal arm-empty facts are for pairs (arm item3, item-2).
        # We estimate blocking by counting how many blocks are above the item-2 component in these pairs.

        def count_blocks_above(item2, visited=None):
            """
            Recursively count number of blocks stacked above item2.
            This counts number of blocks on item2 and their stacked blocks.
            """
            if visited is None:
                visited = set()
            if item2 in visited:
                return 0  # Avoid cycles
            visited.add(item2)
            blocks_on = on_map.get(item2, set())
            count = len(blocks_on)
            for b in blocks_on:
                count += count_blocks_above(b, visited)
            return count

        heuristic_value = 0

        for arm_empty_fact in missing_arm_empty:
            parts = get_parts(arm_empty_fact)
            arm = parts[1]  # item-3
            item2 = parts[2]  # item-2

            # Cost 1 if arm currently holds something (to put down)
            if arm in arm_holding:
                heuristic_value += 1

            # Count how many blocks are stacked above item2 to free it
            # Each block approx corresponds to an unstack/stack action needed
            blocks_above = count_blocks_above(item2)
            heuristic_value += blocks_above

            # Check if item2 itself is on something else (means may need stacking/unstacking)
            # If item2 is on another block, we could add 1 step for unstacking.

            # Find if item2 is on-table or on another item.
            # If item2 is on something else, at least 1 action needed to rearrange.

            is_on_something = False
            # Check on-table in state -> item2 is on a table: no extra unstack needed here
            for fact in state:
                if match(fact, "on-table", item2, "*"):
                    is_on_something = False
                    break
                # Check on fact for item2 as upper block, i.e. if item2 is on something
            else:
                # If no on-table facts for item2 found, check if item2 is on something in on_map keys
                for lower_item in on_map:
                    if item2 in on_map[lower_item]:
                        is_on_something = True
                        break

            if is_on_something:
                heuristic_value += 1

        # Limit heuristic to at least 1 for non-goal states to avoid zero trapping except actual goal.
        if heuristic_value == 0:
            heuristic_value = 1

        return heuristic_value
