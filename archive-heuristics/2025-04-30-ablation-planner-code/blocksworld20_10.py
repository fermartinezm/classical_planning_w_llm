from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions used by the heuristic class
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not isinstance(fact, str) or len(fact) < 2:
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    by counting blocks that are not in their correct goal position relative
    to the block below them, plus blocks that are currently blocking others
    from reaching their goal positions.

    # Assumptions
    - The goal is a set of desired (on ?x ?y) and (on-table ?x) predicates,
      forming one or more stacks.
    - Blocks not mentioned as the subject of goal (on ?x ?y) or (on-table ?x)
      predicates do not have a specific target location, but might still need
      to be moved if they are blocking blocks that do have goal positions.
    - The heuristic counts two types of "misplacedness":
      1. A block whose final position is specified in the goal is not currently
         on the correct block or table it should be on according to the goal.
      2. A block is currently on top of another block, and it is not the block
         that should be directly on top in the goal configuration.

    # Heuristic Initialization
    - Parses the goal conditions to build a mapping of which block should be
      directly on which other block, or on the table.
    - Identifies all blocks that are the subject of a goal (on ?x ?y) or
      (on-table ?x) predicate.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the goal facts to create:
       - `goal_on`: A dictionary mapping a block `b` to the block `a` it should be
         directly on in the goal (e.g., `{'b1': 'b2'}` if goal is `(on b1 b2)`).
       - `goal_table`: A set of blocks that should be on the table in the goal
         (e.g., `{'b2'}` if goal is `(on-table b2)`).
       - `goal_blocks_subject`: A set of all blocks that are the subject of
         an `on` or `on-table` goal predicate (i.e., the blocks whose final
         position is explicitly specified in the goal).

    2. Parse the current state facts to create:
       - `current_on`: A dictionary mapping a block `c` to the block `b` it is
         currently directly on (e.g., `{'b1': 'b2'}` if state has `(on b1 b2)`).
       - `current_table`: A set of blocks currently on the table.
       - `current_holding`: The block currently held by the arm, or None.
       - `current_pos`: A dictionary mapping a block to its current location
         ('table', 'holding', or the block it's on).

    3. Calculate the first component of the heuristic (misplaced blocks):
       - Initialize `misplaced_count = 0`.
       - For each block `b` in `goal_blocks_subject`:
         - Determine the desired position below `b` (`desired_below`). This is
           `goal_on[b]` if `b` in `goal_on`, or 'table' if `b` in `goal_table`.
         - Determine the current position below `b` (`current_below`). If `b`
           is in `current_pos`, it's `current_pos[b]`. If `b` is not in `current_pos`
           (meaning it's not on anything, on table, or held according to the state),
           it is considered misplaced relative to its goal.
         - If `b` is not in `current_pos` OR `current_pos[b]` is different from
           `desired_below`, increment `misplaced_count`.

    4. Calculate the second component of the heuristic (blocking blocks):
       - Initialize `blocking_count = 0`.
       - Iterate through all blocks `c` that are currently on top of some other
         block `b` (i.e., `c` is a key in `current_on`).
       - For each such pair `(c, b)` where `current_on[c] == b`:
         - Check if `c` is the block that should be directly on `b` in the goal.
           This is true if `goal_on.get(c) == b`.
         - If `c` is *not* the block that should be directly on `b` in the goal
           (`goal_on.get(c) != b`), then `c` is blocking `b`. Increment
           `blocking_count`.

    5. The total heuristic value is `misplaced_count + blocking_count`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration.
        """
        self.goals = task.goals

        self.goal_on = {}
        self.goal_table = set()
        self.goal_blocks_subject = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue

            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, block_below = parts[1], parts[2]
                self.goal_on[block] = block_below
                self.goal_blocks_subject.add(block)
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                self.goal_table.add(block)
                self.goal_blocks_subject.add(block)
            # (clear ?) and (arm-empty) goals are not used by this heuristic

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        current_on = {}
        current_table = set()
        current_holding = None
        current_pos = {}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, block_below = parts[1], parts[2]
                current_on[block] = block_below
                current_pos[block] = block_below
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                current_table.add(block)
                current_pos[block] = 'table'
            elif predicate == "holding" and len(parts) == 2:
                block = parts[1]
                current_holding = block
                current_pos[block] = 'holding'
            # (clear ?) and (arm-empty) facts are not used by this heuristic

        # Part 1: Count blocks not in their goal position relative to below
        misplaced_count = 0
        for block in self.goal_blocks_subject:
            # Determine the desired position below the block
            desired_below = None
            if block in self.goal_on:
                desired_below = self.goal_on[block]
            elif block in self.goal_table:
                desired_below = 'table'
            # Note: A block in goal_blocks_subject must be in either goal_on or goal_table,
            # so desired_below will not be None if goal is well-formed.

            # Determine the current position below the block
            # If the block is not found in current_pos, it's not on anything, on table, or held.
            # This indicates an issue with the state representation or it's simply not in the state.
            # In either case, it's not in its goal position.
            if block not in current_pos or current_pos[block] != desired_below:
                 misplaced_count += 1

        # Part 2: Count blocking blocks
        blocking_count = 0
        # Iterate through blocks that are currently on top of something
        for block_on_top, block_below in current_on.items():
            # Check if block_on_top is the block that should be directly on block_below in the goal
            # goal_on.get(block_on_top) will be None if block_on_top is not the subject of any 'on' goal
            if self.goal_on.get(block_on_top) != block_below:
                 # block_on_top is currently on block_below, but goal says it should be on something else (or table)
                 # or block_on_top is not even in goal_on keys (meaning it doesn't need to be on anything specific)
                 # In either case, it's blocking block_below if block_below needs to be moved or cleared.
                 blocking_count += 1

        return misplaced_count + blocking_count
