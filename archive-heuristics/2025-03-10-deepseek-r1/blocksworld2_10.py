from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()

def match(fact, *args):
    """Check if a PDDL fact matches a pattern with wildcards."""
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class blocksworld2Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goal by counting
    the number of blocks that are not in their correct stack configuration. Each misplaced
    block contributes 2 actions (pickup and putdown/stack).

    # Assumptions
    - Each block must be part of a stack that matches the goal configuration from the block down to the table.
    - A block is misplaced if its current stack does not exactly match the goal stack.
    - A block being held by the robot is considered misplaced.

    # Heuristic Initialization
    - Extract goal conditions to determine the correct parent (on-table or on another block) for each block.
    - Precompute the goal stack for each block, which is the sequence of blocks from the block down to the table.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each block, determine its goal stack based on the goal conditions.
    2. For the current state, build each block's current stack by following 'on' and 'on-table' relationships.
    3. A block is misplaced if its current stack does not match the goal stack or if it is being held.
    4. Count all misplaced blocks and multiply by 2 to estimate the required actions.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building goal stacks."""
        self.goal_parent = {}  # Maps each block to its goal parent (block or 'table')
        self.goal_stacks = {}  # Maps each block to its goal stack (list from parent down to table)
        self.blocks = set()

        # Extract goal parents from task.goals
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'on':
                child, parent = parts[1], parts[2]
                self.goal_parent[child] = parent
                self.blocks.update([child, parent])
            elif parts[0] == 'on-table':
                child = parts[1]
                self.goal_parent[child] = 'table'
                self.blocks.add(child)

        # Build goal stacks for each block
        for block in self.blocks:
            current = block
            stack = []
            visited = set()
            while True:
                if current in visited:
                    break  # Prevent infinite loops from malformed goals
                visited.add(current)
                parent = self.goal_parent.get(current, 'table')
                if parent == 'table':
                    stack.append('table')
                    break
                else:
                    stack.append(parent)
                    current = parent
            self.goal_stacks[block] = stack

    def __call__(self, node):
        """Estimate the number of actions needed as twice the number of misplaced blocks."""
        state = node.state
        misplaced = 0

        # Check all relevant blocks (those mentioned in the goal)
        for block in self.blocks:
            # Check if the block is being held
            if f'(holding {block})' in state:
                misplaced += 1
                continue

            # Build current stack for the block
            current_stack = []
            current = block
            visited = set()
            valid_stack = True

            while True:
                if current in visited:
                    valid_stack = False  # Cyclic dependency, invalid state
                    break
                visited.add(current)

                # Find parent in current state
                parent = None
                for fact in state:
                    if match(fact, 'on', current, '*'):
                        parent = get_parts(fact)[2]
                        break
                if not parent:
                    if f'(on-table {current})' in state:
                        parent = 'table'
                    else:
                        valid_stack = False  # Block is neither on another nor on table
                        break

                current_stack.append(parent)
                if parent == 'table':
                    break
                current = parent

            if not valid_stack or tuple(current_stack) != tuple(self.goal_stacks.get(block, [])):
                misplaced += 1

        return 2 * misplaced
