from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to rearrange blocks into their target configuration by counting the necessary stack and unstack operations for each block that is not in the correct position.

    # Assumptions:
    - Each block that is not in its correct position requires two actions: one to unstack it and one to stack it in the correct place.
    - The target configuration is determined by the goal state.
    - Blocks that are already in their correct position do not contribute to the heuristic value.

    # Heuristic Initialization
    - Extract the target stack configuration from the goal conditions.
    - Store the current stack configuration for each block in the initial state.

    # Step-by-Step Thinking for Computing Heuristic Value
    1. Parse the goal state to determine the target stack configuration.
    2. Parse the current state to determine the current stack configuration.
    3. For each block in the target stack:
        a. If the block is not in the correct position, increment the action count by 2 (one for unstacking, one for stacking).
        b. If the block is in the correct position, check if all blocks above it are also correctly placed. If not, increment the action count accordingly.
    4. Sum the required actions for all blocks to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
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

        current_stacks = {}
        goal_stacks = {}

        # Extract current stack information
        for fact in state:
            if match(fact, "on", "*", "*"):
                obj, under = get_parts(fact)
                current_stacks[obj] = under
            elif match(fact, "on-table", "*"):
                obj = get_parts(fact)[1]
                current_stacks[obj] = None  # Indicates the block is on the table

        # Extract goal stack information
        for goal in self.goals:
            if match(goal, "on", "*", "*"):
                obj, under = get_parts(goal)
                goal_stacks[obj] = under
            elif match(goal, "on-table", "*"):
                obj = get_parts(goal)[1]
                goal_stacks[obj] = None  # Indicates the block should be on the table

        total_actions = 0
        goal_blocks = set(goal_stacks.keys())

        # Process each block in the goal stack
        for block in goal_blocks:
            current_under = current_stacks.get(block, None)
            goal_under = goal_stacks.get(block, None)

            if current_under != goal_under:
                total_actions += 2  # Each move requires two actions: unstack and stack

        return total_actions
