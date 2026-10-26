from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class BlocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to stack blocks into their goal configuration. It counts the number of blocks that are out of place and the blocks above them that need to be moved.

    # Assumptions:
    - The goal is to achieve a specific stack configuration.
    - Each block not in its correct position contributes to the heuristic value based on its position in the current stack.

    # Heuristic Initialization
    - Extract the goal conditions to determine the target stack configuration.
    - Build a structure mapping each block to its target position.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the goal facts to build the target stack structure.
    2. Represent the current state as a stack structure.
    3. For each block in the current state, check if it is in the correct position as per the goal.
    4. If a block is not in the correct position, count the number of blocks above it that also need to be moved.
    5. Sum these counts to estimate the minimal number of actions needed.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building target structures."""
        self.goals = task.goals
        static_facts = task.static

        # Build target position map from goal facts
        self.target = {}
        for fact in self.goals:
            if fact.startswith('(on '):
                parts = fact[4:-1].split(' ')
                obj = parts[0]
                target = parts[2]
                self.target[obj] = target
            elif fact == '(on-table b1)':
                self.target['b1'] = 'table'

        # Build reverse map for target positions
        self.inv_target = {}
        for obj, pos in self.target.items():
            if pos != 'table':
                if pos not in self.inv_target:
                    self.inv_target[pos] = []
                self.inv_target[pos].append(obj)

    def __call__(self, node):
        """Estimate the minimal number of actions needed to reach the goal state."""
        state = node.state

        # Build current stack structure
        current = {}
        for fact in state:
            if fact.startswith('(on '):
                parts = fact[4:-1].split(' ')
                obj = parts[0]
                if parts[2] == 'table':
                    current[obj] = 'table'
                else:
                    current[obj] = parts[2]
            elif fact == '(arm-empty)':
                pass  # No block being held

        # Calculate the heuristic value
        heuristic = 0
        visited = set()

        # Process each block to determine if it's in the correct position
        for fact in state:
            if fact.startswith('(on ') and fact not in visited:
                parts = fact[4:-1].split(' ')
                obj = parts[0]
                pos = parts[2]
                if pos == 'table':
                    continue  # Skip table blocks as they are leaves
                if obj in self.target and self.target[obj] != pos:
                    # This block is not in the correct position
                    stack = []
                    current_obj = obj
                    while current_obj in current and current[current_obj] != 'table':
                        stack.append(current_obj)
                        current_obj = current[current_obj]
                    # Count all blocks above that need to be moved
                    for block in stack:
                        if block in self.target and self.target[block] != current[block]:
                            heuristic += 1
                    visited.update(frozenset(stack))

        return heuristic
