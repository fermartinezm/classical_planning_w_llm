from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle facts like '(arm-empty)' which have no arguments
    if len(fact) <= 2: # e.g., "()"
        return []
    return fact[1:-1].split()

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
    This heuristic estimates the number of blocks that are misplaced
    either relative to their support (the block or table below them)
    or relative to the block currently on top of them, plus a penalty
    if the arm needs to be empty but isn't. It aims to guide the search
    towards states where blocks are forming correct segments of the
    goal stacks.

    # Assumptions
    - The goal specifies the desired arrangement of blocks into stacks
      or on the table using `on` and `on-table` predicates.
    - The goal may also require certain blocks to be `clear` and the
      `arm-empty`.
    - The heuristic counts blocks that are "wrong" in position or
      have a "wrong" block on top, as these require actions to fix.

    # Heuristic Initialization
    - Parses the goal conditions to build:
        - `goal_config`: A dictionary mapping each block to its required support
          (another block or 'table').
        - `goal_above`: A dictionary mapping each block/table to the block
          that should be directly on top of it according to the goal (or None
          if it should be clear).
        - `goal_blocks`: A set of all blocks mentioned in the goal `on` or
          `on-table` predicates.
    - Collects all objects (blocks) present in the initial state and goal state
      to ensure all relevant objects are considered during state parsing.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state facts to determine:
       - `current_config`: A dictionary mapping each block to its current support
         (another block, 'table', or 'arm' if held).
       - `current_above`: A dictionary mapping each block/table to the block
         currently directly on top of it (or None if clear).
       - `holding_block`: The block currently held, or None.
       - `is_arm_empty`: True if the arm is empty.
    2. Initialize the heuristic value `h` to 0.
    3. Iterate through each block `b` that is part of the goal configuration (`goal_blocks`).
    4. For each block `b`, compare its current state to its goal state:
       - Get the required support: `goal_sup = self.goal_config.get(b)`.
       - Get the current support: `current_sup = current_config.get(b)`.
       - If `current_sup` is different from `goal_sup`, the block is misplaced relative to its support. Increment `h`.
       - If `current_sup` is the same as `goal_sup`, check the block above:
         - Get the block that should be above `b` in the goal: `goal_ab = self.goal_above.get(b)`.
         - Get the block currently above `b`: `current_ab = current_above.get(b)`.
         - If `current_ab` is different from `goal_ab`, the block `b` has the wrong block on top (or should be clear but isn't, or shouldn't be clear but is). Increment `h`.
    5. Check the arm state: If `(arm-empty)` is a goal condition and the arm is not currently empty (`is_arm_empty` is False), increment `h` by 1.
    6. Return the final heuristic value `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        # Static facts are not needed for this heuristic in Blocksworld.
        # static_facts = task.static

        self.goal_config = {} # block -> support (block or 'table')
        self.goal_above = {}  # support -> block (or None if should be clear)
        self.goal_blocks = set()

        # Parse goal facts to build goal_config and goal_blocks
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip empty facts like "()"

            predicate = parts[0]
            if predicate == "on":
                b, y = parts[1], parts[2]
                self.goal_config[b] = y
                self.goal_blocks.add(b)
                self.goal_blocks.add(y)
            elif predicate == "on-table":
                b = parts[1]
                self.goal_config[b] = 'table'
                self.goal_blocks.add(b)
            # Note: 'clear' and 'arm-empty' goals are handled separately in __call__

        # Build goal_above from goal_config
        # Initialize all goal blocks as potentially clear in the goal
        for b in self.goal_blocks:
             self.goal_above[b] = None
        # Add entries for blocks that should have something on them
        for b, y in self.goal_config.items():
            if y != 'table':
                self.goal_above[y] = b # b should be on y

        # Collect all objects from initial state and goal state
        self.all_objects = set()
        for fact in task.initial_state:
             self.all_objects.update(get_parts(fact)[1:])
        self.all_objects.update(self.goal_blocks)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        current_config = {} # block -> support (block or 'table' or 'arm')
        current_above = {}  # support -> block (or None if clear)
        holding_block = None
        is_arm_empty = False

        # Initialize current_above: assume all objects are clear unless proven otherwise
        for obj in self.all_objects:
            current_above[obj] = None

        # Parse current state facts
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts

            predicate = parts[0]
            if predicate == "on":
                b, y = parts[1], parts[2]
                current_config[b] = y
                current_above[y] = b # y is not clear
            elif predicate == "on-table":
                b = parts[1]
                current_config[b] = 'table'
            elif predicate == "holding":
                b = parts[1]
                holding_block = b
                current_config[b] = 'arm' # Block is held
            elif predicate == "arm-empty":
                is_arm_empty = True
            # 'clear' facts are implicitly handled by initializing current_above to None
            # and then setting it for blocks that have something on them via 'on' facts.

        # Heuristic calculation
        h = 0

        # Count blocks that are misplaced or have the wrong block on top
        for b in self.goal_blocks:
            goal_sup = self.goal_config.get(b)
            current_sup = current_config.get(b) # Will be None if block is not on/on-table/holding

            # If a goal block is not in the state at all (shouldn't happen in valid PDDL)
            # or is not on/on-table/holding, it's definitely misplaced.
            # current_config.get(b) will be None in this case.
            # goal_sup will be 'table' or another block. None != 'table' or None != block.
            # So the first condition `current_sup != goal_sup` handles this correctly.

            if current_sup != goal_sup:
                h += 1
            else: # current_sup == goal_sup
                goal_ab = self.goal_above.get(b) # None if b should be clear
                current_ab = current_above.get(b) # None if b is clear

                if current_ab != goal_ab:
                     h += 1

        # Add cost for the arm state if arm-empty is a goal
        if '(arm-empty)' in self.goals and not is_arm_empty:
            h += 1 # Need 1 action (putdown) to make arm empty

        return h
