from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class blocksworld19Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goal state in the Blocksworld domain.
    It considers the number of blocks that are not in their goal positions, the number of blocks that have blocks on top of them that should not be there, and the number of blocks that are clear but should not be.

    # Assumptions
    - Each block needs to be moved at least once if it is not in the correct position.
    - Moving a block requires picking it up, potentially putting down another block, and stacking it.
    - The arm can only hold one block at a time.

    # Heuristic Initialization
    - Extract the goal `on` relations.
    - Identify the blocks that should be on the table in the goal.
    - Identify the clear blocks in the goal.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize a counter for the heuristic value.
    2. Extract the current `on` relations from the state.
    3. Extract the blocks currently on the table.
    4. Extract the clear blocks in the current state.
    5. Identify blocks that are in the wrong `on` position.
    6. Identify blocks that have blocks on top of them that should not be there.
    7. Identify blocks that are clear but should not be.
    8. For each block in the wrong `on` position, increment the heuristic counter.
    9. For each block that has blocks on top of them that should not be there, increment the heuristic counter.
    10. For each block that is clear but should not be, increment the heuristic counter.
    11. Return the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        self.goal_on = set()
        self.goal_on_table = set()
        self.goal_clear = set()

        for goal in self.goals:
            fact = goal[1:-1].split()
            if fact[0] == 'on':
                self.goal_on.add((fact[1], fact[2]))
            elif fact[0] == 'on-table':
                self.goal_on_table.add(fact[1])
            elif fact[0] == 'clear':
                self.goal_clear.add(fact[1])

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        if self.goal_reached(state):
            return 0

        h = 0

        current_on = set()
        current_on_table = set()
        current_clear = set()

        for fact in state:
            fact_parts = fact[1:-1].split()
            if fact_parts[0] == 'on':
                current_on.add((fact_parts[1], fact_parts[2]))
            elif fact_parts[0] == 'on-table':
                current_on_table.add(fact_parts[1])
            elif fact_parts[0] == 'clear':
                current_clear.add(fact_parts[1])

        # Blocks in wrong on position
        for block1, block2 in current_on:
            if (block1, block2) not in self.goal_on:
                h += 1

        # Blocks that have blocks on top of them that should not be there
        for block1, block2 in current_on:
            clear_block2 = block2 in current_clear
            if clear_block2 and block2 not in self.goal_clear:
                h += 1

        # Blocks that are clear but should not be
        for block in current_clear:
            if block not in self.goal_clear and block not in current_on_table:
                
                is_part_of_goal = False
                for goal_block1, goal_block2 in self.goal_on:
                    if goal_block2 == block:
                        is_part_of_goal = True
                        break
                if is_part_of_goal:
                    h += 1

        return h

    def goal_reached(self, state):
        """Check if the goal has been reached."""
        return self.goals <= state
