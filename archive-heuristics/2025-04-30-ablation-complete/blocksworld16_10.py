# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure input is treated as a string and handle standard PDDL fact format
    fact_str = str(fact).strip()
    if fact_str.startswith('(') and fact_str.endswith(')'):
        return fact_str[1:-1].split()
    # Handle facts without parentheses if necessary, though PDDL facts usually have them
    return fact_str.split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    by counting discrepancies in the current state compared to the goal state.
    It sums three components:
    1. Blocks that are not in their correct goal position (on the wrong block/table or held).
    2. Blocks that are currently on top of a block that is needed as a base
       in a goal stack, but the block on top is not the one required by the goal.
    3. Blocks that are required to be clear in the goal state but are not clear
       in the current state.

    # Assumptions
    - The goal state specifies the desired arrangement of blocks using `on`
      and `on-table` predicates, and potentially `clear` for the top blocks.
    - The goal configuration forms one or more stacks.
    - The heuristic value is 0 if and only if the state is a goal state.

    # Heuristic Initialization
    - Parses the goal facts (`task.goals`) to extract the desired structure:
      - `self.goal_pos`: A dictionary mapping each block that appears in an
        `(on X Y)` or `(on-table X)` goal fact to its required base (`Y` or 'table').
      - `self.goal_on`: A dictionary mapping each block `Y` that appears as a
        base in an `(on X Y)` goal fact to the block `X` that should be directly
        on top of it.
      - `self.goal_clear`: A set of blocks that must have the `(clear X)` predicate
        true in the goal state.
    - Static facts (`task.static`) are not used as they are empty in Blocksworld.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state (`node.state`):
    1. Parse the current state facts to build the current configuration:
       - `current_pos`: A dictionary mapping each block to its current base
         ('table' or another block), if it's on something.
       - `current_on`: A dictionary mapping each block `Y` to the block `X`
         currently directly on top of it, if any.
       - `holding_block`: The block currently held by the arm, or `None`.
       - `current_clear`: A set of blocks that currently have the `(clear X)`
         predicate true.
    2. Initialize the heuristic value `h` to 0.
    3. Calculate the first component (Misplaced Blocks):
       For each block `block` that has a specified goal position (i.e., `block`
       is a key in `self.goal_pos`):
       If the `block` is currently being held by the arm (`holding_block == block`),
       OR if the `block` is not present in `current_pos` (meaning it's held
       or somehow missing),
       OR if the `block` is in `current_pos` but its current base
       (`current_pos[block]`) is not equal to its required goal base
       (`self.goal_pos[block]`):
       Increment `h` by 1.
    4. Calculate the second component (Blocking Blocks):
       For each block `base` that is required to have a specific block
       `goal_block_on_base` directly on top of it in the goal state
       (i.e., `base` is a key in `self.goal_on`, and `goal_block_on_base = self.goal_on[base]`):
       Check if there is currently a block `current_block_on_base` on top of `base`
       (i.e., `base` is a key in `current_on`).
       If there is a block on top (`base in current_on`) AND this block
       (`current_block_on_base = current_on[base]`) is NOT the block that
       should be there according to the goal (`current_block_on_base != goal_block_on_base`):
       Increment `h` by 1.
    5. Calculate the third component (Not Clear Goal Blocks):
       For each block `block` that is required to be clear in the goal state
       (i.e., `block` is in `self.goal_clear`):
       If the `block` is not currently clear in the state (i.e., `block` is
       NOT in `current_clear`):
       Increment `h` by 1.
    6. The final heuristic value is the sum `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the goal structure.
        """
        self.goals = task.goals

        # Parse goal facts to build the desired structure
        self.goal_pos = {}  # Map block -> required base ('table' or another block)
        self.goal_on = {}   # Map base -> block that should be on top
        self.goal_clear = set() # Set of blocks that must be clear

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip empty facts if any
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block, base = parts[1], parts[2]
                self.goal_pos[block] = base
                self.goal_on[base] = block
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                self.goal_pos[block] = 'table'
            elif predicate == 'clear' and len(parts) == 2:
                block = parts[1]
                self.goal_clear.add(block)
            # Ignore (arm-empty) and other potential goal facts not related to block positions/clearness

    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # Parse current state
        current_pos = {}  # Map block -> current base ('table' or another block)
        current_on = {}   # Map base -> block currently on top
        holding_block = None
        current_clear = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts if any
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block, base = parts[1], parts[2]
                current_pos[block] = base
                current_on[base] = block
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                current_pos[block] = 'table'
            elif predicate == 'holding' and len(parts) == 2:
                holding_block = parts[1]
            elif predicate == 'clear' and len(parts) == 2:
                block = parts[1]
                current_clear.add(block)
            # Ignore (arm-empty) and other potential state facts

        h = 0

        # 1. Count blocks not in their goal position
        for block, goal_base in self.goal_pos.items():
            if holding_block == block:
                h += 1
            elif block not in current_pos or current_pos[block] != goal_base:
                h += 1

        # 2. Count blocks blocking a required goal base
        for base, goal_block_on_base in self.goal_on.items():
            if base in current_on:
                current_block_on_base = current_on[base]
                if current_block_on_base != goal_block_on_base:
                    h += 1
            # If base is not in current_on, it means nothing is on it, which is fine if the goal is to put something on it.

        # 3. Count blocks that should be clear but aren't
        for block in self.goal_clear:
            if block not in current_clear:
                 h += 1

        return h
