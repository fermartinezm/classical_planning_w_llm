from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class blocksworld12Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goal state in the Blocksworld domain.
    It considers the number of blocks that are not in their goal positions, the number of blocks that have
    blocks on top of them that should not be there, and the number of blocks that are clear but should not be.

    # Assumptions
    - The arm can only hold one block at a time.
    - The heuristic assumes that each block needs to be moved at least once if it is not in its goal position.
    - The heuristic is not admissible, as it may underestimate the true cost.

    # Heuristic Initialization
    - The heuristic initializes by extracting the goal conditions from the task.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize a counter for the heuristic value.
    2. Iterate through the goal conditions to identify the desired block positions.
    3. For each block, check if it is in the correct position according to the goal state.
       - If a block is not in the correct position (either on the wrong block or on the table when it shouldn't be),
         increment the heuristic counter.
    4. Check for blocks that have other blocks on top of them in the current state, but should be clear in the goal.
       - For each such block, increment the heuristic counter.
    5. Check for blocks that are clear in the current state, but should have another block on top of them in the goal.
       - For each such block, increment the heuristic counter.
    6. Return the final heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        heuristic_value = 0

        # Create sets to store goal state information
        goal_on = set()
        goal_on_table = set()
        goal_clear = set()

        # Extract goal information from the goal state
        for goal in self.goals:
            if goal.startswith('(on '):
                parts = goal[1:-1].split()
                goal_on.add((parts[1], parts[2]))
            elif goal.startswith('(on-table '):
                parts = goal[1:-1].split()
                goal_on_table.add(parts[1])
            elif goal.startswith('(clear '):
                parts = goal[1:-1].split()
                goal_clear.add(parts[1])

        # Check if the current state satisfies the goal state
        if self.is_goal_state(state, goal_on, goal_on_table, goal_clear):
            return 0

        # Count blocks that are not in their goal positions
        for fact in state:
            if fact.startswith('(on '):
                parts = fact[1:-1].split()
                block1, block2 = parts[1], parts[2]
                if (block1, block2) not in goal_on:
                    heuristic_value += 1
            elif fact.startswith('(on-table '):
                parts = fact[1:-1].split()
                block = parts[1]
                if block not in goal_on_table:
                    heuristic_value += 1

        # Count blocks that have blocks on top of them but should be clear
        for block in goal_clear:
            for fact in state:
                if fact.startswith('(on '):
                    parts = fact[1:-1].split()
                    if parts[2] == block:
                        heuristic_value += 1
                        break

        # Count blocks that are clear but should have blocks on top of them
        for fact in state:
            if fact.startswith('(clear '):
                parts = fact[1:-1].split()
                block = parts[1]
                
                found = False
                for goal_fact in self.goals:
                    if goal_fact.startswith('(on '):
                        goal_parts = goal_fact[1:-1].split()
                        if goal_parts[2] == block:
                            found = True
                            break
                if found:
                    heuristic_value += 1

        return heuristic_value

    def is_goal_state(self, state, goal_on, goal_on_table, goal_clear):
        """Check if the current state satisfies the goal state."""
        current_on = set()
        current_on_table = set()
        current_clear = set()

        for fact in state:
            if fact.startswith('(on '):
                parts = fact[1:-1].split()
                current_on.add((parts[1], parts[2]))
            elif fact.startswith('(on-table '):
                parts = fact[1:-1].split()
                current_on_table.add(parts[1])
            elif fact.startswith('(clear '):
                parts = fact[1:-1].split()
                current_clear.add(parts[1])

        return (
            goal_on <= current_on and
            goal_on_table <= current_on_table and
            goal_clear <= current_clear
        )
