from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class blocksworld4Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state by counting each misplaced block (including those being held) as requiring two actions. A block is considered misplaced if it is not on the correct block or table as specified in the goal.

    # Assumptions
    - Each block not in its goal position requires at least two actions (pickup and place/stack).
    - Blocks held by the arm are considered misplaced and need to be placed, adding two actions.
    - The heuristic does not account for blocks that need to be moved to access other blocks, focusing instead on direct positional correctness.

    # Heuristic Initialization
    - Extracts goal conditions to determine the correct position (on-table or on another block) for each block.
    - Collects all blocks from the initial state and goals to ensure coverage.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each block, check if it is currently being held by the arm. If so, add 2 actions.
    2. For blocks not being held, compare their current position (on-table or on another block) with their goal position.
    3. If the current position does not match the goal position, add 2 actions for that block.
    4. Sum the actions for all misplaced and held blocks to get the heuristic value.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract goal information: map each block to its goal parent (another block or 'table')
        self.goal_parent = {}
        for fact in self.goals:
            parts = fact[1:-1].split()
            if parts[0] == 'on':
                self.goal_parent[parts[1]] = parts[2]
            elif parts[0] == 'on-table':
                self.goal_parent[parts[1]] = 'table'

        # Collect all blocks from the initial state and goals
        self.blocks = set()
        for fact in task.initial_state:
            if fact.startswith('(on ') or fact.startswith('(on-table '):
                self.blocks.add(fact.split()[1])
        for fact in self.goals:
            if fact.startswith('(on ') or fact.startswith('(on-table '):
                self.blocks.add(fact.split()[1])

    def __call__(self, node):
        state = node.state
        current_parent = {}
        holding = set()

        # Parse current state to determine block positions and held blocks
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'on':
                current_parent[parts[1]] = parts[2]
            elif parts[0] == 'on-table':
                current_parent[parts[1]] = 'table'
            elif parts[0] == 'holding':
                holding.add(parts[1])

        heuristic_value = 0
        for block in self.blocks:
            if block in holding:
                # Block is being held, needs to be placed (2 actions)
                heuristic_value += 2
            else:
                # Determine current and goal positions
                current_p = current_parent.get(block, 'table')
                goal_p = self.goal_parent.get(block, 'table')
                if current_p != goal_p:
                    # Block is misplaced, requires at least 2 actions
                    heuristic_value += 2

        return heuristic_value
