from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class Blocksworld12Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goal by checking:
    1. Blocks that are not on their correct target (as per the goal) contribute 2 actions each.
    2. Blocks that are on top of a goal block which needs to be clear contribute 2 actions each.

    # Assumptions
    - The goal is a conjunction of 'on', 'on-table', and 'clear' predicates.
    - Blocks not mentioned in the goal can be in any state, except if they block a goal-required clear condition.
    - Moving a block requires two actions (pickup/putdown or unstack/stack).

    # Heuristic Initialization
    - Extract goal conditions into dictionaries and sets for quick lookup:
        - `goal_on`: Maps each block to its target block.
        - `goal_on_table`: Set of blocks that must be on the table.
        - `goal_clear`: Set of blocks that must be clear.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Parse Goal Conditions**: Identify which blocks need to be on specific blocks, on the table, or clear.
    2. **Analyze Current State**: Track where each block is located and what is on top of it.
    3. **Check Misplaced Blocks**:
        - For each block in `goal_on`, if not on its target, add 2 actions.
        - For each block in `goal_on_table`, if not on the table, add 2 actions.
    4. **Check Clear Conditions**:
        - For each block in `goal_clear`, count blocks on top and add 2 actions per block.
    """

    def __init__(self, task):
        """Extract goal conditions into suitable data structures."""
        self.goal_on = {}        # Maps block to its target block in the goal
        self.goal_on_table = set()  # Blocks that must be on the table
        self.goal_clear = set()  # Blocks that must be clear

        for goal in task.goals:
            parts = goal[1:-1].split()  # Remove parentheses and split
            predicate = parts[0]
            if predicate == 'on':
                block, target = parts[1], parts[2]
                self.goal_on[block] = target
            elif predicate == 'on-table':
                block = parts[1]
                self.goal_on_table.add(block)
            elif predicate == 'clear':
                block = parts[1]
                self.goal_clear.add(block)

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        current_on = {}          # Maps block to what it's on
        current_on_table = set() # Blocks currently on the table

        for fact in state:
            if fact.startswith('(on '):
                parts = fact[1:-1].split()
                block, under = parts[1], parts[2]
                current_on[block] = under
            elif fact.startswith('(on-table '):
                parts = fact[1:-1].split()
                block = parts[1]
                current_on_table.add(block)

        total_cost = 0

        # Check blocks that should be on another block
        for block, target in self.goal_on.items():
            if current_on.get(block) != target:
                total_cost += 2

        # Check blocks that should be on the table
        for block in self.goal_on_table:
            if block not in current_on_table:
                total_cost += 2

        # Check blocks that should be clear (nothing on top)
        for block in self.goal_clear:
            # Count how many blocks are on top of 'block'
            on_top = [b for b, under in current_on.items() if under == block]
            total_cost += 2 * len(on_top)

        return total_cost
