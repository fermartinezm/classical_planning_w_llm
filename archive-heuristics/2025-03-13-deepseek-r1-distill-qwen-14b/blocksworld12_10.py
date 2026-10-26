from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class BlocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to stack all blocks in the desired order as specified by the goal state.

    # Assumptions:
    - The goal state specifies a specific stack of blocks.
    - Each block that is not in the correct position requires two actions: one to remove it and one to place it correctly.
    - The arm starts empty and must pick up blocks to move them.

    # Heuristic Initialization
    - Extract the goal conditions to determine the desired stack structure.
    - Store the desired parent block for each block in the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the desired stack structure from the goal state.
    2. For each block in the current state, determine its current parent block.
    3. Compare the current parent with the desired parent:
       - If they match, no action is needed.
       - If they do not match, count two actions (one to remove the block, one to place it correctly).
    4. Sum the required actions for all blocks to get the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract the desired parent for each block from the goal state
        self.goal_parents = {}
        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'on':
                block = parts[1]
                parent = parts[2]
                self.goal_parents[block] = parent

    def __call__(self, node):
        """Estimate the minimum number of actions needed to reach the goal state."""
        state = node.state

        def match(fact, *args):
            """Utility function to check if a PDDL fact matches a given pattern."""
            parts = fact[1:-1].split()
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        current_parents = {}
        for fact in state:
            if match(fact, 'on', '*', '*'):
                block = fact[1:-1].split()[1]
                parent = fact[1:-1].split()[2]
                current_parents[block] = parent

        total_actions = 0
        for block in self.goal_parents:
            desired_parent = self.goal_parents[block]
            current_parent = current_parents.get(block, None)
            if current_parent != desired_parent:
                total_actions += 2  # One action to remove, one to place

        return total_actions
