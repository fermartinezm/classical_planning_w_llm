from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class blocksworld7Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goal state in the Blocksworld domain.
    It considers the number of blocks that are not in their goal positions, the number of blocks that have
    blocks on top of them that should not be there, and the number of blocks that need to be clear to be moved.

    # Assumptions
    - Each block can only be moved one at a time.
    - The arm can only hold one block at a time.
    - The heuristic does not take into account the cost of returning blocks to the table.

    # Heuristic Initialization
    - Extract the goal conditions from the task.
    - No static facts are used in this heuristic.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Iterate through each goal condition.
    3. For each goal condition, check if it is satisfied in the current state.
    4. If a goal condition is not satisfied, increment the heuristic value based on the type of goal:
       - `on(x, y)`: If block `x` is not on block `y`, increment the heuristic. Also, if `x` has blocks on top of it, increment the heuristic for each block that needs to be moved.
       - `on_table(x)`: If block `x` is not on the table, increment the heuristic. Also, if `x` has blocks on top of it, increment the heuristic for each block that needs to be moved.
       - `clear(x)`: If block `x` is not clear, increment the heuristic.
    5. Return the final heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        heuristic = 0

        # Helper function to extract objects from a fact string
        def get_objects(fact):
            fact = fact[1:-1]  # Remove parentheses
            return fact.split()[1:]  # Split and return objects

        # Helper function to check if a fact is in the state
        def is_fact_in_state(fact, state):
            return fact in state

        # Helper function to check if a block is clear in the state
        def is_clear(block, state):
            return f"(clear {block})" in state

        # Helper function to check if a block is on another block in the state
        def is_on(block1, block2, state):
            return f"(on {block1} {block2})" in state

        # Helper function to check if a block is on the table in the state
        def is_on_table(block, state):
            return f"(on-table {block})" in state

        # Check if the goal is already reached
        if node.task.goal_reached(state):
            return 0

        # Iterate through each goal condition
        for goal in self.goals:
            if not is_fact_in_state(goal, state):
                predicate = goal[1:].split()[0]  # Extract predicate name

                if predicate == "on":
                    block_x, block_y = get_objects(goal)
                    if not is_on(block_x, block_y, state):
                        heuristic += 1

                        # Check if block_x has blocks on top of it that need to be moved
                        for fact in state:
                            if fact.startswith("(on ") and get_objects(fact)[1] == block_x:
                                heuristic += 1  # Increment for each block on top
                elif predicate == "on-table":
                    block_x = get_objects(goal)[0]
                    if not is_on_table(block_x, state):
                        heuristic += 1

                        # Check if block_x has blocks on top of it that need to be moved
                        for fact in state:
                            if fact.startswith("(on ") and get_objects(fact)[1] == block_x:
                                heuristic += 1  # Increment for each block on top
                elif predicate == "clear":
                    block_x = get_objects(goal)[0]
                    if not is_clear(block_x, state):
                        heuristic += 1

        return heuristic
