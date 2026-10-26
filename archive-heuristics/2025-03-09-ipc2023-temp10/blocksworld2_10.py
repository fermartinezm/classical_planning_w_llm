from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class blocksworld2Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goal state in the Blocksworld domain.
    It considers the number of blocks that are not in their goal positions, the number of blocks that have
    blocks on top of them that should not be there, and the number of blocks that are clear but should not be.

    # Assumptions
    - The heuristic assumes that each misplaced block requires at least one unstack and one stack operation.
    - It also assumes that each block that needs to be clear requires at least one unstack operation.
    - The arm-empty condition is implicitly considered when counting actions.

    # Heuristic Initialization
    - The heuristic initializes by extracting the goal state from the task definition.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the goal state from the task.
    2. Identify blocks that are not in their goal positions.
    3. Identify blocks that have incorrect blocks on top of them.
    4. Identify blocks that are clear in the current state but should not be clear in the goal state.
    5. For each misplaced block, add a cost of 2 (one unstack and one stack).
    6. For each block with an incorrect block on top, add a cost of 1 (one unstack).
    7. For each block that is clear but should not be, add a cost of 1 (one stack).
    8. Return the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        def match(fact, *args):
            """
            Utility function to check if a PDDL fact matches a given pattern.
            - `fact`: The fact as a string (e.g., "(on b1 b2)").
            - `args`: The pattern to match (e.g., "on", "*", "b2").
            - Returns `True` if the fact matches the pattern, `False` otherwise.
            """
            parts = fact[1:-1].split()  # Remove parentheses and split into individual elements.
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        # If the state is the goal state, return 0
        if self.goal_reached(state):
            return 0

        misplaced_blocks = 0
        incorrect_blocks_on_top = 0
        incorrectly_clear_blocks = 0

        # Goal state information
        goal_on = {}  # Maps block to the block it should be on
        goal_on_table = set()  # Blocks that should be on the table
        goal_clear = set()  # Blocks that should be clear

        for goal in self.goals:
            if match(goal, "on", "*", "*"):
                parts = goal[1:-1].split()
                block1 = parts[1]
                block2 = parts[2]
                goal_on[block1] = block2
            elif match(goal, "on-table", "*"):
                parts = goal[1:-1].split()
                block = parts[1]
                goal_on_table.add(block)
            elif match(goal, "clear", "*"):
                parts = goal[1:-1].split()
                block = parts[1]
                goal_clear.add(block)

        # Current state information
        current_on = {}  # Maps block to the block it is on
        current_on_table = set()  # Blocks that are on the table
        current_clear = set()  # Blocks that are clear

        for fact in state:
            if match(fact, "on", "*", "*"):
                parts = fact[1:-1].split()
                block1 = parts[1]
                block2 = parts[2]
                current_on[block1] = block2
            elif match(fact, "on-table", "*"):
                parts = fact[1:-1].split()
                block = parts[1]
                current_on_table.add(block)
            elif match(fact, "clear", "*"):
                parts = fact[1:-1].split()
                block = parts[1]
                current_clear.add(block)

        # Calculate misplaced blocks
        for block, correct_underblock in goal_on.items():
            if block not in current_on or current_on.get(block) != correct_underblock:
                misplaced_blocks += 1

        for block in goal_on_table:
            if block not in current_on_table:
                misplaced_blocks += 1

        # Calculate blocks with incorrect blocks on top
        for block, underblock in current_on.items():
            if block in goal_on and goal_on[block] != underblock:
                incorrect_blocks_on_top += 1
            elif block in goal_on_table:
                incorrect_blocks_on_top += 1

        # Calculate blocks that are clear but shouldn't be
        for block in current_clear:
            if block in goal_on or block in goal_on_table:
                incorrectly_clear_blocks += 1

        # Estimate the cost
        cost = 2 * misplaced_blocks + incorrect_blocks_on_top + incorrectly_clear_blocks

        return cost

    def goal_reached(self, state):
        """Check if all goal conditions are satisfied in the given state."""
        return self.goals <= state
