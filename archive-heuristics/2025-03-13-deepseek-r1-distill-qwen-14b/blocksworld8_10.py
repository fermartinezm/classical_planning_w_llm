from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to stack blocks into their goal configuration.

    # Assumptions:
    - Blocks can be on the table, on top of another block, or being held by the arm.
    - The arm can hold only one block at a time.
    - Each block must be moved individually to its correct position.

    # Heuristic Initialization
    - Extract the goal configuration of the blocks from the task's goals.
    - Store the current state's block positions for quick lookup.

    # Step-by-Step Thinking for Computing Heuristic
    1. Identify the goal configuration of the blocks.
    2. For each block in the goal configuration, check if it is already in the correct position.
    3. If a block is not in the correct position, count the necessary actions to move it:
       - If the block is being held, it needs to be put down.
       - If the block is on the table but needs to be stacked, count the required stack actions.
       - If the block is on the wrong block, count the necessary unstack and stack actions.
    4. Sum the actions for all blocks to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract goal positions for each block
        self.goal_positions = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "on":
                obj, under_obj = args
                self.goal_positions[obj] = under_obj if under_obj != '-table' else None
            elif predicate == "on-table":
                obj = args[0]
                self.goal_positions[obj] = None

    def __call__(self, node):
        """Estimate the minimum number of actions to reach the goal state."""
        state = node.state
        current_positions = {}

        # Parse current state to get block positions
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "on":
                obj, under_obj = args
                current_positions[obj] = under_obj if under_obj != '-table' else None
            elif predicate == "on-table":
                obj = args[0]
                current_positions[obj] = None
            elif predicate == "holding":
                obj = args[0]
                current_positions[obj] = 'held'

        # If all goals are already achieved, return 0
        if self.goal_reached(state):
            return 0

        total_actions = 0

        # Check each block in the goal configuration
        for block in self.goal_positions:
            goal_pos = self.goal_positions[block]
            current_pos = current_positions.get(block, None)

            if current_pos == goal_pos:
                continue  # Block is already in the correct position

            if current_pos is None:
                # Block is on the table but needs to be stacked
                if goal_pos is not None:
                    total_actions += 1  # Need to pick up and stack
            elif current_pos == 'held':
                # Block is being held but needs to be placed
                if goal_pos is None:
                    total_actions += 1  # Need to put down
                else:
                    # Need to unstack and stack
                    total_actions += 2
            else:
                # Block is on another block but needs to be moved
                if goal_pos is None:
                    # Need to unstack and put down
                    total_actions += 2
                else:
                    # Need to unstack and stack
                    total_actions += 2

        return total_actions

    def goal_reached(self, state):
        """Check if all goal facts are satisfied in the current state."""
        return all(fact in state for fact in self.goals)
