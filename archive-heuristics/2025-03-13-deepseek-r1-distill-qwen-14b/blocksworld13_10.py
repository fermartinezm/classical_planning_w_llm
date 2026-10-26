from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class BlocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to stack blocks in the desired order by counting the number of blocks that are out of place and need to be moved.

    # Assumptions:
    - The goal is to have a specific stack of blocks.
    - Each block that is not in its correct position contributes to the heuristic value.
    - Moving a block from one position to another requires a series of actions (pickup, stack, etc.).

    # Heuristic Initialization
    - Extract the goal conditions to determine the target stack of blocks.
    - Parse the current state to determine the current stack of blocks.

    # Step-By-Step Thinking for Computing Heuristic
    1. Build the target stack from the goal conditions.
    2. Build the current stack from the state.
    3. Compare the two stacks to identify misplaced blocks.
    4. Count the number of actions needed to correct the misplacements.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        """Estimate the minimum number of actions needed to reach the goal state."""
        state = node.state

        def get_parts(fact):
            """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
            return fact[1:-1].split()

        def match(fact, *args):
            """
            Check if a PDDL fact matches a given pattern.
            - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
            - `args`: The expected pattern (wildcards `*` allowed).
            - Returns `True` if the fact matches the pattern, else `False`.
            """
            parts = get_parts(fact)
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        # Build the target stack from the goal conditions
        goal_stack = []
        for goal in self.goals:
            if match(goal, "on", "*", "*"):
                _, obj, under = get_parts(goal)
                goal_stack.append((obj, under))

        # Build the current stack from the state
        current_stack = {}
        for fact in state:
            if match(fact, "on", "*", "*"):
                _, obj, under = get_parts(fact)
                current_stack[obj] = under

        # If there are no goal stacks, return 0 (already at goal)
        if not goal_stack:
            return 0

        # Determine the base block (the one on the table)
        base_goal = None
        for goal in self.goals:
            if match(goal, "on-table", "*"):
                base_goal = get_parts(goal)[1]
                break

        # If the base block is not on the table, add to cost
        if base_goal not in current_stack or current_stack.get(base_goal) != base_goal:
            return 1  # At least one action needed to fix the base

        # Now, count the number of blocks that are out of place
        total_cost = 0
        current_order = []
        goal_order = []

        # Build current order
        current_under = base_goal
        while current_under in current_stack:
            current_block = current_stack[current_under]
            current_order.append(current_block)
            current_under = current_block

        # Build goal order
        goal_under = base_goal
        while goal_under in current_stack:  # Assuming all goal blocks are present
            goal_block = None
            for goal in self.goals:
                if match(goal, "on", "*", goal_under):
                    goal_block = get_parts(goal)[1]
                    break
            if goal_block is None:
                break
            goal_order.append(goal_block)
            goal_under = goal_block

        # Compare current and goal order to count mismatches
        min_length = min(len(current_order), len(goal_order))
        for i in range(min_length):
            if current_order[i] != goal_order[i]:
                total_cost += (min_length - i)  # Count all blocks above as needing moves
                break
        else:
            # If all blocks up to the minimum length match, check if one is longer than the other
            if len(current_order) != len(goal_order):
                total_cost = abs(len(current_order) - len(goal_order))

        return total_cost
