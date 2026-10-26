from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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

class BlocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to stack blocks into their target configuration.

    # Assumptions:
    - Each block has a specific target position in the goal state.
    - A block is considered correctly positioned if it is on its target block or on the table as required.
    - Any block not in its target position requires two actions: one to unstack and one to stack.

    # Heuristic Initialization
    - Extract the target positions for each block from the goal conditions.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. Parse the goal conditions to determine the target position for each block.
    2. For each block, determine its current position in the state.
    3. If a block's current position does not match its target position, add two actions to the heuristic.
    4. Sum the actions for all blocks to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting target positions for each block."""
        self.goals = task.goals
        self.static = task.static

        # Extract target positions for each block from the goal conditions
        self.goal_positions = {}
        for goal in self.goals:
            if match(goal, "on", "*", "*"):
                obj, target = get_parts(goal)
                self.goal_positions[obj] = target
            elif match(goal, "on-table", "*"):
                obj = get_parts(goal)[1]
                self.goal_positions[obj] = "table"

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Dictionary to hold the current position of each block
        current_positions = {}

        # Parse the state to determine the current position of each block
        for fact in state:
            if match(fact, "on", "*", "*"):
                obj, on_obj = get_parts(fact)
                current_positions[obj] = on_obj
            elif match(fact, "on-table", "*"):
                obj = get_parts(fact)[1]
                current_positions[obj] = "table"
            elif match(fact, "holding", "*"):
                obj = get_parts(fact)[1]
                current_positions[obj] = "held"

        heuristic_value = 0

        # For each block with a target position, check if it's correctly placed
        for obj, target in self.goal_positions.items():
            current = current_positions.get(obj, None)
            if current != target:
                heuristic_value += 2  # Two actions per mispositioned block

        return heuristic_value
