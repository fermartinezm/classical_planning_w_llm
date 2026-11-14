from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract elements of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    Wildcards '*' are supported.

    - fact: String representation of a fact, e.g. "(on item1 item2)"
    - args: Pattern parts, e.g. ("on", "*", "item2")
    Returns True if all parts match, else False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (blocksworld-like) domain.

    Summary
    -------
    This heuristic estimates the number of actions needed to reach a goal state from a given state.
    The domain involves "arm-empty", "on", "on-table", "holding", and "clear" predicates with several types
    of hierarchical objects and involves pickup, stack, and unstack operations.
    The heuristic estimates progress primarily by counting how many goal-required (arm-empty) conditions
    are not met and how many block relations (on/on-table) are inconsistent with goal arm-empty facts.

    Assumptions
    -----------
    - The goal conditions are only of the form (arm-empty ?v3 ?v4).
    - To achieve (arm-empty vA vB), certain 'on' and 'on-table' relations must hold consistently.
    - Every incorrect "on" or "on-table" fact requiring rearrangement roughly corresponds to at least one action.
    - An arm that is not empty but required to be empty implies at least one action to make it empty.
    - The heuristic does not need to be admissible but should estimate the number of required actions efficiently.

    Heuristic Initialization
    ------------------------
    - Extract the set of goal arm-empty predicates.
    - Extract all static "on", "on-table", and "holding" facts and organize them for fast lookup of object relations.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify which arm-empty goals are currently true in the state and which are still missing.
       - Each missing arm-empty goal contributes at least 1 action (to clear that arm).
    2. For each arm in a goal (arm-empty arm object), check if the arm is currently holding an object.
       - If holding something, it must be put down or stacked, incurring further actions.
    3. Check the "on" and "on-table" relations relevant to the objects involved in arm-empty goals.
       - If the objects are not stable (i.e., a block is not on the expected block/table), count the actions needed.
    4. Combine these counts into one heuristic estimate:
       - Actions needed to clear arms not currently empty as required by the goal.
       - Actions needed to fix incorrect "on"/"on-table" relations related to the arm-empty goals.
    5. Return 0 if all arm-empty goal conditions hold (goal state).

    Because the domain is complex and nested, this heuristic avoids full state reconstruction, focusing on the arm-empty predicates
    which directly represent the goal, and penalizes states deviating from these.

    This heuristic encourages the greedy best-first search to prioritize achieving goal arm-empty states and clearing
    obstructing blocks efficiently.

    """

    def __init__(self, task):
        # The goals (arm-empty facts that must hold at the end).
        self.goal_arm_empty = {
            fact
            for fact in task.goals
            if match(fact, "arm-empty", "*", "*")
        }

        # Extract static facts for quick lookup.
        self.static_holding = {
            (parts[1], parts[2])
            for fact in task.static
            if match(fact, "holding", "*", "*")
            for parts in [get_parts(fact)]
        }

        self.static_on = {
            (parts[1], parts[2])
            for fact in task.static
            if match(fact, "on", "*", "*")
            for parts in [get_parts(fact)]
        }

        self.static_on_table = {
            (parts[1], parts[2])
            for fact in task.static
            if match(fact, "on-table", "*", "*")
            for parts in [get_parts(fact)]
        }

        # No static data needed for arm-empty or clear, as these may change.

    def __call__(self, node):
        state = node.state

        # Quick goal test: if all goal arm-empty facts hold, heuristic is zero.
        if self.goal_arm_empty <= state:
            return 0

        cost = 0

        # Find current arm-empty facts in state.
        current_arm_empty = {
            fact for fact in state if match(fact, "arm-empty", "*", "*")
        }

        # 1. Penalty for each goal arm-empty fact not yet achieved (at least one action to clear that arm).
        missing_arm_empty = self.goal_arm_empty - current_arm_empty
        cost += len(missing_arm_empty)

        # 2. For each arm in the goal arm-empty, check if that arm is currently holding an object.
        # This means that to achieve arm-empty, the held object must be put down or stacked.
        # Count one action per such holding.
        # Holding facts are of form (holding ?v5 ?v6).
        holding_current = {
            (parts[1], parts[2])
            for fact in state
            if match(fact, "holding", "*", "*")
            for parts in [get_parts(fact)]
        }

        # Arms involved in goals: extract arm and related object from goal arm-empty facts.
        # arm-empty has form (arm-empty ?arm ?obj)
        arms_in_goal = {
            (get_parts(fact)[1], get_parts(fact)[2]) for fact in self.goal_arm_empty
        }

        # If arm is holding an object, it needs to be emptied (count action)
        for arm, obj in arms_in_goal:
            # Find if arm is holding some object currently
            for held_arm, held_obj in holding_current:
                if held_arm == arm:
                    cost += 1  # action needed to put this object down or stack it

        # 3. Check objects on-table or on relations that are inconsistent with arm-empty goals.
        # For each goal arm-empty (arm, obj), check if obj is on correct blocks or table.
        # Count one action per "wrong" relation suspected to require fixing.

        # Extract current on and on-table facts:
        current_on = {
            (parts[1], parts[2])
            for fact in state
            if match(fact, "on", "*", "*")
            for parts in [get_parts(fact)]
        }
        current_on_table = {
            (parts[1], parts[2])
            for fact in state
            if match(fact, "on-table", "*", "*")
            for parts in [get_parts(fact)]
        }

        # For each arm-empty goal (arm, obj), the arm must be empty holding that obj.
        # Since arm-empty means the arm is free from holding, probably the obj is placed correctly.
        # We heuristically penalize obj that is not on a clear block or not on table properly.
        # Because domain is complex, we approximate by:
        #   If obj is not on any block or table (not in current_on or current_on_table),
        #   or if obj is on a block that is not arm-empty (thus probably obstructed), count penalty.

        # Extract current clear facts (which indicate blocks can be stacked upon)
        current_clear = {
            (parts[1], parts[2])
            for fact in state
            if match(fact, "clear", "*", "*")
            for parts in [get_parts(fact)]
        }

        # Build set of objects that are currently clear:
        clear_objects = {obj for _, obj in current_clear}

        # For each obj in goals, check if it is properly "clear" or on table
        for arm, obj in arms_in_goal:
            # If object is not on something, we assume it needs at least one action to be placed correctly
            on_any = False
            if any(t[0] == obj for t in current_on) or any(t[0] == obj for t in current_on_table):
                on_any = True
            if not on_any:
                # Object is floating? Penalize
                cost += 1
            else:
                # If on another block, and that block is not clear, penalize fixing that
                # Find what obj is on:
                on_target = None
                for (top, bottom) in current_on:
                    if top == obj:
                        on_target = bottom
                        break
                if on_target is None:
                    for (top, bottom) in current_on_table:
                        if top == obj:
                            on_target = bottom
                            break

                # If on_target exists and is not clear, penalize
                if on_target and on_target not in clear_objects:
                    cost += 1

        return cost
