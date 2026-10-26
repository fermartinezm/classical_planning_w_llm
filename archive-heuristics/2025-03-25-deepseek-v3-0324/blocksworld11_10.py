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
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to reach the goal state by:
    1. Counting mismatched blocks (blocks not in their goal position)
    2. Considering the dependencies between blocks (a block can't be moved until blocks above it are moved)
    3. Accounting for the arm state (whether we need to pick up or put down blocks)

    # Assumptions:
    - The arm can hold only one block at a time.
    - Blocks can only be stacked on other blocks or on the table.
    - The goal specifies a complete configuration (all blocks must be in their specified positions).

    # Heuristic Initialization
    - Extract the goal conditions to determine the desired configuration.
    - Build a mapping of which block should be on which other block/table in the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each block, check if it's in its correct position:
       - If on table in goal but not in current state: needs to be put down
       - If on another block in goal but not in current state: needs to be stacked
    2. For blocks not in correct position:
       - If the block is clear (can be moved directly): count 1 action
       - If the block has blocks above it: count actions to unstack each block above it
    3. If the arm is holding a block:
       - If the held block is in correct position: count 1 action to put it down
       - If not: count 1 action to put it down somewhere temporarily
    4. The total heuristic is the sum of:
       - Actions needed to clear blocks above mismatched blocks
       - Actions needed to move mismatched blocks
       - Actions needed to handle the arm state
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.static = task.static
        
        # Build goal structure: {block: (on-table/on-block, target)}
        self.goal_structure = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "on":
                block, under_block = args
                self.goal_structure[block] = ("on", under_block)
            elif predicate == "on-table":
                block = args[0]
                self.goal_structure[block] = ("on-table", None)

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        holding_block = None
        current_structure = {}
        clear_blocks = set()

        # Parse current state
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "on":
                block, under_block = args
                current_structure[block] = ("on", under_block)
            elif predicate == "on-table":
                block = args[0]
                current_structure[block] = ("on-table", None)
            elif predicate == "clear":
                clear_blocks.add(args[0])
            elif predicate == "holding":
                holding_block = args[0]

        total_cost = 0

        # Check each block's position
        for block, (goal_type, goal_target) in self.goal_structure.items():
            current_pos = current_structure.get(block, (None, None))
            
            # Block is in correct position
            if current_pos == (goal_type, goal_target):
                continue
                
            # Block needs to be moved
            if block in clear_blocks:
                # Can be moved directly
                total_cost += 1
            else:
                # Need to unstack blocks above first
                # Find how many blocks are above this one
                above_count = 0
                current = block
                while True:
                    found = False
                    for other_block, (other_type, other_under) in current_structure.items():
                        if other_under == current:
                            above_count += 1
                            current = other_block
                            found = True
                            break
                    if not found:
                        break
                total_cost += above_count + 1  # unstack all above + move this block

        # Handle the arm state
        if holding_block:
            # Check if held block is in correct position
            goal_type, goal_target = self.goal_structure.get(holding_block, (None, None))
            current_type, current_target = current_structure.get(holding_block, (None, None))
            
            if (goal_type, goal_target) == (current_type, current_target):
                total_cost += 1  # put down in correct position
            else:
                total_cost += 2  # put down somewhere + pick up again later

        return total_cost
