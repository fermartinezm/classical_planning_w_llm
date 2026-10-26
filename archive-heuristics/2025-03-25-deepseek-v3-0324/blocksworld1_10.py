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
    2. Considering the dependencies between blocks (a block must be moved before its supporting block)
    3. Accounting for the arm state (whether a block is being held or not)

    # Assumptions:
    - The arm can hold only one block at a time.
    - Blocks must be clear to be moved (no blocks on top of them).
    - The goal specifies exact positions for blocks (on-table or on another block).

    # Heuristic Initialization
    - Extract the goal conditions to identify desired block positions.
    - No static facts are used in this heuristic.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each block, check if its current position matches the goal position:
       - If not, it will need to be moved (at least 1 action per mismatched block).
    2. For blocks that are on other blocks in the goal:
       - If the supporting block is not in place, additional moves may be needed.
    3. If the arm is holding a block, it must either:
       - Place it correctly (if possible) or place it temporarily (adding actions).
    4. The total heuristic is the sum of:
       - 1 for each block not in its goal position.
       - 1 for each block that needs to be moved to free a supporting block.
       - 1 if the arm is holding a block that's not in its goal position.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.static = task.static

        # Extract goal conditions: maps each block to its goal position (either '(on X Y)' or '(on-table X)')
        self.goal_on = {}
        self.goal_on_table = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "on":
                block, under_block = args
                self.goal_on[block] = under_block
            elif predicate == "on-table":
                block = args[0]
                self.goal_on_table.add(block)

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        # Check if the goal is already reached
        if self.goals <= state:
            return 0

        # Track current block positions
        current_on = {}  # block -> under_block
        current_on_table = set()  # blocks on table
        holding = None  # block being held, if any
        clear_blocks = set()  # blocks with nothing on top

        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "on":
                block, under_block = args
                current_on[block] = under_block
            elif predicate == "on-table":
                block = args[0]
                current_on_table.add(block)
            elif predicate == "holding":
                holding = args[0]
            elif predicate == "clear":
                clear_blocks.add(args[0])

        # Count mismatched blocks
        mismatched = 0
        for block in self.goal_on_table | set(self.goal_on.keys()):
            # Check if block is in correct position
            if block in self.goal_on_table:
                if block not in current_on_table:
                    mismatched += 1
            else:  # block should be on another block
                if current_on.get(block) != self.goal_on[block]:
                    mismatched += 1

        # Count blocks that need to be moved to free their supporting block
        dependency_moves = 0
        for block, under_block in self.goal_on.items():
            if under_block in current_on and current_on[under_block] != block:
                # Supporting block is under something else
                dependency_moves += 1

        # If holding a block that's not in goal position, need to place it
        holding_penalty = 1 if holding and holding not in self.goal_on_table and (
            holding not in self.goal_on or current_on.get(holding) != self.goal_on[holding]
        ) else 0

        return mismatched + dependency_moves + holding_penalty
