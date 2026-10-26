from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming Heuristic base class exists

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle cases like '(arm-empty)' which have no arguments
    content = fact[1:-1]
    if not content:
        return [fact[1:-1]] # Return just the predicate name
    return content.split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are currently in a "wrong"
    configuration relative to the goal state. A block is considered "wrong" if
    it is not on its correct goal support (another block or the table) OR if it
    has the wrong block on top of it (including having any block on top when it
    should be clear according to the goal).

    # Assumptions
    - The heuristic focuses on the positions and clear status of blocks specified in the goal state.
    - Blocks not mentioned in the goal state do not contribute to the heuristic value.
    - The heuristic counts each "wrongly placed or obstructed" block once.

    # Heuristic Initialization
    - Parses the goal state to build data structures representing the desired
      configuration:
        - `goal_support`: Maps each block to the block it should be on, or 'table'.
        - `goal_on_top`: Maps each block to the block that should be directly on top of it.
        - `goal_clear_blocks`: Set of blocks that should be clear in the goal.
        - `all_goal_blocks`: Set of all blocks mentioned in any goal predicate.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to determine the current support for each block
       (the block it's on, the table, or the arm) and the block currently on top
       of each block.
    2. Initialize the heuristic value to 0.
    3. Iterate through each block that is mentioned in the goal state (`all_goal_blocks`).
    4. For the current block `B`:
       a. Determine its goal support (`goal_pos`) and the block that should be on top of it (`goal_block_on_top`) from the pre-parsed goal structures.
       b. Determine its current support (`current_pos`) and the block currently on top of it (`current_block_on_top`) from the parsed state structures.
       c. Check if the block's position is wrong: If `goal_pos` is specified (i.e., the block's position is part of an `on` or `on-table` goal) AND `current_pos` is different from `goal_pos`, mark the block as positionally wrong.
       d. Check if the block on top is wrong:
          - If `goal_block_on_top` is specified (i.e., the block should have a specific block on top) AND `current_block_on_top` is different from `goal_block_on_top`, mark the block as having the wrong block on top.
          - If the block should be clear (`B` is in `goal_clear_blocks` AND `goal_block_on_top` is NOT specified) AND it is *not* clear (`current_block_on_top` is not None), mark the block as having the wrong block on top (specifically, *any* block).
       e. If the block is marked as positionally wrong OR having the wrong block on top, increment the heuristic value by 1.
    5. Return the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals

        # Pre-parse goal conditions for faster lookup during heuristic calculation
        self.goal_support = {}
        self.goal_on_top = {}
        self.goal_clear_blocks = set()
        self.all_goal_blocks = set()

        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            args = parts[1:]
            self.all_goal_blocks.update(args)

            if predicate == "on":
                x, y = args
                self.goal_support[x] = y
                self.goal_on_top[y] = x
            elif predicate == "on-table":
                x = args[0]
                self.goal_support[x] = 'table'
            elif predicate == "clear":
                x = args[0]
                self.goal_clear_blocks.add(x)
            # Note: 'arm-empty' goal is ignored by this heuristic,
            # as it focuses on block configurations.

    def _parse_state(self, state):
        """
        Parses the current state to determine block positions and stacking.
        Returns: current_support, current_on_top, current_holding
        """
        current_support = {}
        current_on_top = {}
        current_holding = None

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            args = parts[1:]

            if predicate == "on":
                x, y = args
                current_support[x] = y
                current_on_top[y] = x
            elif predicate == "on-table":
                x = args[0]
                current_support[x] = 'table'
            elif predicate == "holding":
                x = args[0]
                current_holding = x
                current_support[x] = 'arm' # Represent being held as being on the arm

        return current_support, current_on_top, current_holding

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # If the goal is already reached, the heuristic is 0.
        if self.goals <= state:
             return 0

        current_support, current_on_top, current_holding = self._parse_state(state)

        heuristic_cost = 0

        # Iterate over all blocks mentioned in the goal
        for block in self.all_goal_blocks:
            goal_pos = self.goal_support.get(block)
            goal_block_on_top = self.goal_on_top.get(block)
            current_pos = current_support.get(block)
            current_block_on_top = current_on_top.get(block)

            is_wrong = False

            # Condition 1: Block is not on its correct support (if goal specifies support)
            if goal_pos is not None and current_pos != goal_pos:
                 is_wrong = True

            # Condition 2: Block has the wrong block on top (if goal specifies block on top)
            # This includes having any block when a specific one is required, or having
            # the wrong specific block.
            if not is_wrong and goal_block_on_top is not None and current_block_on_top != goal_block_on_top:
                 is_wrong = True

            # Condition 3: Block should be clear but isn't
            # This applies if the goal specifies the block should be clear AND
            # the goal does NOT specify a block on top (which is redundant with clear).
            if not is_wrong and goal_block_on_top is None and block in self.goal_clear_blocks and current_block_on_top is not None:
                 is_wrong = True

            if is_wrong:
                 heuristic_cost += 1

        return heuristic_cost
