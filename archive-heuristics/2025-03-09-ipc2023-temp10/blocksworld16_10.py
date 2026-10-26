from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class blocksworld16Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goal state in the Blocksworld domain.
    It considers the number of blocks that are not in their goal positions, and estimates the number of moves
    required to put them in the correct positions.

    # Assumptions
    - Each block needs to be moved at most once to its correct position.
    - Moving a block involves picking it up, potentially putting down another block to clear the destination,
      and then stacking the block onto its destination.
    - The arm-empty predicate is always true after stacking or putdown.

    # Heuristic Initialization
    - Extract the goal `on` and `clear` predicates from the task.
    - Store the goal `on` relations in a dictionary for easy access.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Initialize the heuristic value to 0.
    2.  Extract the goal state information, focusing on `on` relationships between blocks.
    3.  Iterate through the current state's `on` relationships.
    4.  For each `on` relationship in the goal state, check if it exists in the current state.
        - If the `on` relationship is not in the current state, increment the heuristic value.
    5.  Iterate through the current state's `clear` predicates.
    6.  For each `clear` predicate in the goal state, check if it exists in the current state.
        - If the `clear` predicate is not in the current state, increment the heuristic value.
    7.  If the current state is the goal state, return 0.
    8.  Return the calculated heuristic value.
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
        heuristic = 0

        # Check if the current state satisfies the goal state
        if node.state == self.goals:
            return 0

        # Count misplaced blocks based on 'on' predicates
        for block, target in self.goal_on.items():
            goal_fact = f'(on {block} {target})'
            if goal_fact not in state:
                heuristic += 1

        # Count blocks that are not clear when they should be
        for block in self.goal_clear:
            goal_fact = f'(clear {block})'
            if goal_fact not in state:
                heuristic += 1

        return heuristic
