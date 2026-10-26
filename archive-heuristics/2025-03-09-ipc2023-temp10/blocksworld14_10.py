from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class blocksworld14Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goal state in the Blocksworld domain.
    It considers the number of blocks that are not in their goal positions, and the number of blocks that have blocks
    on top of them that should not be there.

    # Assumptions
    - The heuristic assumes that each misplaced block requires at least one unstack and one stack operation.
    - It also assumes that blocks that are in the correct stack but have incorrect blocks on top require at least one unstack operation.
    - The arm-empty and clear predicates are implicitly considered when counting actions.

    # Heuristic Initialization
    - The heuristic initializes by extracting the goal conditions from the task.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the goal 'on' and 'clear' predicates from the goal state.
    2. Identify blocks that are not in their goal positions according to the 'on' predicates.
    3. Identify blocks that have incorrect blocks stacked on top of them.
    4. Count the number of 'unstack' and 'stack' actions required to move misplaced blocks to their goal positions.
    5. Add a penalty for blocks that have incorrect blocks on top of them, as these will require 'unstack' actions.
    6. If the current state is the goal state, return 0.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        # If the current state is the goal state, return 0.
        if task.goal_reached(state):
            return 0

        goal_ons = {}
        goal_clears = set()

        for goal in self.goals:
            if goal.startswith('(on'):
                parts = goal[1:-1].split()
                goal_ons[parts[1]] = parts[2]
            elif goal.startswith('(clear'):
                parts = goal[1:-1].split()
                goal_clears.add(parts[1])

        misplaced_blocks = 0
        incorrect_blocks_on_top = 0

        for block, target in goal_ons.items():
            found = False
            for fact in state:
                if fact.startswith('(on'):
                    parts = fact[1:-1].split()
                    if parts[1] == block and parts[2] == target:
                        found = True
                        break
            if not found:
                misplaced_blocks += 1

        for fact in state:
            if fact.startswith('(on'):
                parts = fact[1:-1].split()
                block = parts[1]
                top_block = parts[0]
                
                correct_top_block = None
                for goal_block, goal_target in goal_ons.items():
                    if goal_target == block:
                        correct_top_block = goal_block
                        break
                
                if correct_top_block is not None and top_block != correct_top_block:
                    incorrect_blocks_on_top += 1

        # Each misplaced block requires at least one unstack and one stack operation.
        # Blocks with incorrect blocks on top require at least one unstack operation.
        return misplaced_blocks * 2 + incorrect_blocks_on_top
