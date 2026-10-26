from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class blocksworld25Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goal state in the Blocksworld domain.
    It considers the number of blocks that are not in their goal positions, the number of blocks that have blocks on top of them that should not be there, and the number of blocks that are clear but should not be.

    # Assumptions
    - The heuristic assumes that each misplaced block requires at least one unstack and one stack operation.
    - It also assumes that each block that needs to be clear requires at least one unstack operation.
    - The arm-empty condition is implicitly considered when counting actions.

    # Heuristic Initialization
    - The heuristic initializes by extracting the goal conditions from the task.
    - It also initializes data structures to store the on and clear relationships in the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Initialize counters for misplaced blocks, incorrect stacks, and incorrect clear states.
    2.  Iterate through the goal state and extract the 'on' and 'clear' relationships.
    3.  For each 'on' goal, check if the relationship holds in the current state. If not, increment the misplaced block counter.
    4.  For each 'clear' goal, check if the block is clear in the current state. If not, increment the incorrect clear state counter.
    5.  Iterate through the current state and identify blocks that are 'on' other blocks, but this relationship is not in the goal state. Increment the incorrect stack counter for each such block.
    6.  The heuristic value is the sum of the misplaced block count, the incorrect stack count, and the incorrect clear state count. This represents a lower bound on the number of actions required to achieve the goal.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static
        self.goal_on = {}
        self.goal_clear = set()

        for goal in self.goals:
            if goal.startswith('(on'):
                parts = goal[1:-1].split()
                self.goal_on[parts[1]] = parts[2]
            elif goal.startswith('(clear'):
                parts = goal[1:-1].split()
                self.goal_clear.add(parts[1])

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        misplaced_blocks = 0
        incorrect_stacks = 0
        incorrect_clear = 0

        # Check 'on' goals
        for block, on_block in self.goal_on.items():
            if not f'(on {block} {on_block})' in state:
                misplaced_blocks += 1

        # Check 'clear' goals
        for block in self.goal_clear:
            if not f'(clear {block})' in state:
                incorrect_clear += 1

        # Check for incorrect stacks
        for fact in state:
            if fact.startswith('(on'):
                parts = fact[1:-1].split()
                block_above = parts[1]
                block_below = parts[2]
                if block_above in self.goal_on and self.goal_on[block_above] != block_below:
                    incorrect_stacks += 1
                elif block_above not in self.goal_on:
                    incorrect_stacks += 1

        return misplaced_blocks + incorrect_stacks + incorrect_clear
